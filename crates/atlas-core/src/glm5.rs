// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side GLM-5.3-Flash execution and persistent-state contract.
//!
//! This module is intentionally independent of CUDA. It gives admission,
//! state allocation, and future kernel construction one checked source of
//! truth instead of letting each layer infer a slightly different layout.

use anyhow::{Context, Result, bail};

use crate::config::{LayerType, ModelConfig};

pub mod dsa_reference;
pub mod kda_reference;

/// Exact execution plan derived from a validated GLM-5.3-Flash config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glm53FlashPlan {
    pub kda_layers: Vec<usize>,
    pub dsa_layers: Vec<usize>,
    pub dense_mlp_layers: Vec<usize>,
    pub moe_layers: Vec<usize>,
    pub kda_heads: usize,
    pub kda_head_dim: usize,
    pub kda_conv_kernel: usize,
    pub mla_latent_dim: usize,
    pub index_key_dim: usize,
    pub index_topk: usize,
    pub index_kpool: usize,
}

impl Glm53FlashPlan {
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        if config.model_type != "glm5_next" {
            bail!("GLM-5.3 execution plan requires model_type=glm5_next");
        }
        let kda_layers = indices(config, LayerType::LinearAttention);
        let dsa_layers = indices(config, LayerType::FullAttention);
        if kda_layers.len() != 34 || dsa_layers.len() != 11 {
            bail!("GLM-5.3 execution plan requires 34 KDA and 11 DSA layers");
        }
        if config.linear_num_key_heads != config.linear_num_value_heads
            || config.linear_key_head_dim != config.linear_value_head_dim
        {
            bail!("GLM-5.3 KDA requires matching key/value head geometry");
        }
        if config.index_kpool == 0 || !config.index_topk.is_multiple_of(config.index_kpool) {
            bail!("GLM-5.3 index_topk must be divisible by index_kpool");
        }
        let dense_mlp_layers = config.mlp_only_layers.clone();
        let moe_layers = (0..config.num_hidden_layers)
            .filter(|layer| !dense_mlp_layers.contains(layer))
            .collect();
        Ok(Self {
            kda_layers,
            dsa_layers,
            dense_mlp_layers,
            moe_layers,
            kda_heads: config.linear_num_key_heads,
            kda_head_dim: config.linear_key_head_dim,
            kda_conv_kernel: config.linear_conv_kernel_dim,
            mla_latent_dim: config.kv_lora_rank,
            index_key_dim: config.index_head_dim,
            index_topk: config.index_topk,
            index_kpool: config.index_kpool,
        })
    }

    /// FP32 recurrent matrix bytes across all KDA layers for one sequence.
    pub fn kda_recurrent_bytes_per_sequence(&self) -> Result<usize> {
        checked_product(&[
            self.kda_layers.len(),
            self.kda_heads,
            self.kda_head_dim,
            self.kda_head_dim,
            4,
        ])
    }

    /// FP32 Q/K/V convolution histories across all KDA layers for one sequence.
    /// A width-4 causal convolution retains the preceding 3 values; the
    /// current token is input, not persistent history.
    pub fn kda_conv_bytes_per_sequence(&self) -> Result<usize> {
        let history = self
            .kda_conv_kernel
            .checked_sub(1)
            .context("GLM-5.3 KDA convolution kernel must be non-zero")?;
        checked_product(&[
            self.kda_layers.len(),
            3,
            self.kda_heads,
            self.kda_head_dim,
            history,
            4,
        ])
    }

    pub fn kda_state_bytes_per_sequence(&self) -> Result<usize> {
        self.kda_recurrent_bytes_per_sequence()?
            .checked_add(self.kda_conv_bytes_per_sequence()?)
            .context("GLM-5.3 KDA state size overflow")
    }

    /// Live DSA bytes across all sparse layers at one exact sequence length.
    ///
    /// The main MLA cache stores one 512-wide latent per token. The semantic
    /// index stores one 128-wide learned pooled key per complete group of four,
    /// plus raw key+gate+valid lanes for the incomplete tail (at most 3).
    pub fn dsa_cache_bytes_for_context(
        &self,
        context_tokens: usize,
        element_bytes: usize,
    ) -> Result<usize> {
        let completed_pools = context_tokens / self.index_kpool;
        let tail = context_tokens % self.index_kpool;
        let latent = context_tokens
            .checked_mul(self.mla_latent_dim)
            .context("GLM-5.3 MLA cache size overflow")?;
        let pooled = completed_pools
            .checked_mul(self.index_key_dim)
            .context("GLM-5.3 pooled-index size overflow")?;
        let tail_vector_width = self
            .index_key_dim
            .checked_mul(2)
            .context("GLM-5.3 index-tail width overflow")?;
        let tail_state = tail
            .checked_mul(tail_vector_width)
            .and_then(|elements| elements.checked_mul(element_bytes))
            // The native DSA ABI owns one fixed int32[4] control record per
            // layer: first_valid, valid_count, pool_count, tail_len. It exists
            // even when the current tail is empty.
            .and_then(|bytes| bytes.checked_add(4 * size_of::<i32>()))
            .context("GLM-5.3 index-tail size overflow")?;
        let vector_bytes = latent
            .checked_add(pooled)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .context("GLM-5.3 DSA cache dimension overflow")?;
        let per_layer = vector_bytes
            .checked_add(tail_state)
            .context("GLM-5.3 DSA cache dimension overflow")?;
        checked_product(&[self.dsa_layers.len(), per_layer])
    }

    /// Fixed-address DSA capacity for any length up to `context_tokens`.
    ///
    /// A context ceiling divisible by four still passes through lengths with
    /// a three-token incomplete pool. A reusable sequence slot must therefore
    /// reserve the maximum tail, not the remainder at the ceiling.
    pub fn dsa_cache_capacity_bytes(
        &self,
        context_tokens: usize,
        element_bytes: usize,
    ) -> Result<usize> {
        let completed_pools = context_tokens / self.index_kpool;
        let latent = checked_product(&[context_tokens, self.mla_latent_dim, element_bytes])?;
        let pooled = checked_product(&[completed_pools, self.index_key_dim, element_bytes])?;
        let max_tail = self.index_kpool - 1;
        let tail_vectors = checked_product(&[max_tail, self.index_key_dim, 2, element_bytes])?;
        // SSOT with `glm53_dsa_pool_append`: one int32[4] control record,
        // not one metadata tuple per tail lane.
        let tail_metadata = 4 * size_of::<i32>();
        let per_layer = latent
            .checked_add(pooled)
            .and_then(|bytes| bytes.checked_add(tail_vectors))
            .and_then(|bytes| bytes.checked_add(tail_metadata))
            .context("GLM-5.3 DSA capacity overflow")?;
        checked_product(&[self.dsa_layers.len(), per_layer])
    }

    /// Conservative persistent bytes for one admitted sequence.
    pub fn persistent_bytes_per_sequence(
        &self,
        context_tokens: usize,
        cache_element_bytes: usize,
    ) -> Result<usize> {
        let context = self.dsa_cache_capacity_bytes(context_tokens, cache_element_bytes)?;
        self.kda_state_bytes_per_sequence()?
            .checked_add(context)
            .context("GLM-5.3 persistent sequence size overflow")
    }

    /// Hard state-slot cap for admission after reserving weights and scratch.
    pub fn max_sequences_for_state_budget(
        &self,
        state_budget_bytes: usize,
        reserved_context_tokens: usize,
        cache_element_bytes: usize,
    ) -> Result<usize> {
        let per_sequence =
            self.persistent_bytes_per_sequence(reserved_context_tokens, cache_element_bytes)?;
        Ok(state_budget_bytes / per_sequence)
    }

    /// Bytes for a fixed-address sequence-state pool, including padding slots
    /// that collective batches may address but admission never owns.
    pub fn state_pool_bytes(
        &self,
        claimable_sequences: usize,
        padding_sequences: usize,
        reserved_context_tokens: usize,
        cache_element_bytes: usize,
    ) -> Result<usize> {
        let slots = claimable_sequences
            .checked_add(padding_sequences)
            .context("GLM-5.3 state slot count overflow")?;
        self.persistent_bytes_per_sequence(reserved_context_tokens, cache_element_bytes)?
            .checked_mul(slots)
            .context("GLM-5.3 state pool size overflow")
    }

    /// Appliance topology: pure TP2 across the two Sparks. Every rank executes
    /// every routed expert using one half of its EXL3 intermediate dimension.
    pub fn validate_bootstrap_topology(&self, ep_size: usize, tp_size: usize) -> Result<()> {
        if ep_size != 1 || tp_size != 2 {
            bail!("GLM-5.3-Flash two-Spark appliance requires TP=2 and EP=1");
        }
        Ok(())
    }
}

fn indices(config: &ModelConfig, kind: LayerType) -> Vec<usize> {
    config
        .layer_types
        .iter()
        .enumerate()
        .filter_map(|(index, actual)| (*actual == kind).then_some(index))
        .collect()
}

fn checked_product(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, factor| {
        product
            .checked_mul(*factor)
            .context("GLM-5.3 state size overflow")
    })
}

#[cfg(test)]
mod tests {
    use super::Glm53FlashPlan;
    use crate::config::{LayerType, ModelConfig};

    fn config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".to_string();
        config.num_hidden_layers = 45;
        config.layer_types = (0..45)
            .map(|index| {
                if (index + 1) % 4 == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        config.linear_num_key_heads = 64;
        config.linear_num_value_heads = 64;
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;
        config.linear_conv_kernel_dim = 4;
        config.kv_lora_rank = 512;
        config.index_head_dim = 128;
        config.index_topk = 2048;
        config.index_kpool = 4;
        config.mlp_only_layers = vec![0, 1, 2];
        config
    }

    #[test]
    fn state_accounting_matches_official_geometry() {
        let plan = Glm53FlashPlan::from_config(&config()).unwrap();
        assert_eq!(plan.kda_layers.len(), 34);
        assert_eq!(plan.dsa_layers.len(), 11);
        assert_eq!(plan.moe_layers.len(), 42);
        assert_eq!(
            plan.kda_recurrent_bytes_per_sequence().unwrap(),
            142_606_336
        );
        assert_eq!(plan.kda_conv_bytes_per_sequence().unwrap(), 10_027_008);
        assert_eq!(plan.kda_state_bytes_per_sequence().unwrap(), 152_633_344);
        assert_eq!(
            plan.dsa_cache_bytes_for_context(32_768, 2).unwrap(),
            392_167_600
        );
        assert_eq!(
            plan.dsa_cache_capacity_bytes(32_768, 2).unwrap(),
            392_184_496
        );
        assert_eq!(
            plan.persistent_bytes_per_sequence(32_768, 2).unwrap(),
            544_817_840
        );
    }

    #[test]
    fn admission_uses_state_and_context_not_request_count_alone() {
        let plan = Glm53FlashPlan::from_config(&config()).unwrap();
        let one = plan.persistent_bytes_per_sequence(32_768, 2).unwrap();
        assert_eq!(
            plan.max_sequences_for_state_budget(one * 4, 32_768, 2)
                .unwrap(),
            4
        );
        assert_eq!(
            plan.max_sequences_for_state_budget(one * 4, 65_536, 2)
                .unwrap(),
            2
        );
    }

    #[test]
    fn appliance_requires_pure_tp2() {
        let plan = Glm53FlashPlan::from_config(&config()).unwrap();
        assert!(plan.validate_bootstrap_topology(1, 2).is_ok());
        assert!(plan.validate_bootstrap_topology(2, 2).is_err());
        assert!(plan.validate_bootstrap_topology(1, 1).is_err());
    }

    #[test]
    fn pool_reserve_includes_collective_padding_slot() {
        let plan = Glm53FlashPlan::from_config(&config()).unwrap();
        let per_sequence = plan.persistent_bytes_per_sequence(32_768, 2).unwrap();
        assert_eq!(
            plan.state_pool_bytes(4, 1, 32_768, 2).unwrap(),
            per_sequence * 5
        );
    }
}
