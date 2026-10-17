// SPDX-License-Identifier: AGPL-3.0-only
//! Checkpoint-derived GLM history budget; tail slots retain physical arena shape.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;

#[derive(Clone, Copy, Debug)]
pub(crate) struct GlmIndexBudget {
    history_tokens: u32,
    dense_threshold: u32,
    logical_width: u32,
    physical_width: u32,
}

impl GlmIndexBudget {
    pub(crate) fn from_config(config: &ModelConfig, native: bool) -> Result<Self> {
        Self::new(
            config.index_topk,
            config.index_kpool,
            config.index_kpool_always_select_tail,
            native,
        )
    }

    pub(crate) fn new(topk: usize, pool: usize, always_tail: bool, native: bool) -> Result<Self> {
        ensure!(
            topk > 0 && pool > 0 && topk.is_multiple_of(pool),
            "GLM index budget requires a positive whole-pool top-k"
        );
        ensure!(
            !native || (always_tail && topk > pool),
            "native EXL3 GLM index requires explicit always-select-tail and at least two pools"
        );
        let physical = topk
            .checked_add(pool - 1)
            .context("GLM index width overflow")?;
        let history = if native { topk - pool } else { topk };
        Ok(Self {
            history_tokens: u32::try_from(history).context("GLM history budget exceeds u32")?,
            dense_threshold: u32::try_from(if native { topk - 1 } else { topk })
                .context("GLM dense threshold exceeds u32")?,
            logical_width: u32::try_from(history + pool - 1)
                .context("GLM logical width exceeds u32")?,
            physical_width: u32::try_from(physical).context("GLM physical width exceeds u32")?,
        })
    }

    pub(crate) fn history_tokens(self) -> u32 {
        self.history_tokens
    }
    pub(crate) fn dense_threshold(self) -> u32 {
        self.dense_threshold
    }
    pub(crate) fn physical_width(self) -> u32 {
        self.physical_width
    }
    pub(crate) fn logical_width(self) -> u32 {
        self.logical_width
    }
    pub(crate) fn dense_is_exact(self, length: usize) -> bool {
        length > 0 && length <= self.dense_threshold as usize
    }
}

impl super::Qwen3AttentionLayer {
    /// Called while the loader still owns resources, before transfer into this layer.
    pub(crate) fn configure_exl3_index_budget(
        &mut self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        config: &ModelConfig,
    ) -> Result<()> {
        let budget = GlmIndexBudget::from_config(config, true)?;
        ensure!(
            config.model_type == "glm5_next"
                && budget.history_tokens() == 2044
                && budget.logical_width() == 2047
                && budget.physical_width() == 2051,
            "native EXL3 dynamic index supports configured topk2048/pool4 with reserved tail"
        );
        if super::glm_multi_seq_sparse_graphs_enabled(&config.model_type)? {
            self.glm_index_topk_dynamic_k =
                gpu.kernel("glm_indexer", "glm_index_topk_expand_dynamic_tail_reserved")?;
        }
        Ok(())
    }

    pub(super) fn glm_index_weight_element_bytes(&self) -> usize {
        if self.has_exl3_mla() { 4 } else { 2 }
    }

    pub(super) fn glm_index_budget(&self, config: &ModelConfig) -> Result<GlmIndexBudget> {
        GlmIndexBudget::from_config(config, self.has_exl3_mla())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_reserves_one_pool_for_tail_without_shrinking_arena() {
        let budget = GlmIndexBudget::new(2048, 4, true, true).unwrap();
        assert_eq!(budget.history_tokens(), 2044);
        assert_eq!(budget.dense_threshold(), 2047);
        assert_eq!(budget.logical_width(), 2047);
        assert_eq!(budget.physical_width(), 2051);
        for length in [2044, 2045, 2046, 2047] {
            assert!(budget.dense_is_exact(length));
        }
        for length in [2048, 2049, 2050, 2051, 8193] {
            assert!(!budget.dense_is_exact(length));
        }
        assert_eq!(
            crate::layer::prefill_attention_pieces(2040, 16, budget.dense_threshold() as usize),
            vec![(0, 7), (7, 9)]
        );
    }

    #[test]
    fn legacy_budget_and_boundary_are_unchanged() {
        for tail in [false, true] {
            let budget = GlmIndexBudget::new(2048, 4, tail, false).unwrap();
            assert_eq!(budget.history_tokens(), 2048);
            assert_eq!(budget.dense_threshold(), 2048);
            assert_eq!(budget.logical_width(), 2051);
            assert_eq!(budget.physical_width(), 2051);
            assert!(budget.dense_is_exact(2048));
            assert!(!budget.dense_is_exact(2049));
        }
    }

    #[test]
    fn unsupported_native_tail_and_invalid_geometry_fail_closed() {
        assert!(GlmIndexBudget::new(2048, 4, false, true).is_err());
        for (topk, pool) in [(0, 4), (2048, 0), (2047, 4), (4, 4), (usize::MAX, 1)] {
            assert!(GlmIndexBudget::new(topk, pool, true, true).is_err());
        }
    }
}
