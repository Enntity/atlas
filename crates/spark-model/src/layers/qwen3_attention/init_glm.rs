// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 kernel selection for `Qwen3AttentionLayer::new_with_gating`: the
//! env-selected semantic-index / sparse-attention variants and the exact GLM
//! MLA paged-decode kernel.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::init_arch_gates::{ArchProbes, gated as gate};

/// Env-selected GLM semantic-index / sparse-attention kernel variants.
pub(super) struct GlmIndexSelection {
    pub(super) glm_sparse_attn_heads_per_cta: u32,
    pub(super) glm_sparse_attn_fn: &'static str,
    pub(super) glm_index_logits_rows_per_cta: u32,
    pub(super) glm_index_logits_pools_per_cta: u32,
    pub(super) glm_index_wmma: bool,
    pub(super) glm_sparse_graphs: bool,
    pub(super) glm_index_logits_fn: &'static str,
}

impl GlmIndexSelection {
    pub(super) fn from_config(config: &ModelConfig, probes: &ArchProbes) -> Result<Self> {
        // Multi-head sparse MLA reuses GLM's shared compressed K/V rows. Keep
        // the original one-head kernel available as an operational fallback.
        let glm_sparse_attn_heads_per_cta =
            if std::env::var("ATLAS_GLM_SPARSE_HEAD_GROUP").ok().as_deref() == Some("1") {
                1
            } else {
                8
            };
        let glm_sparse_attn_fn = if glm_sparse_attn_heads_per_cta == 1 {
            "glm_sparse_mla_prefill_bf16"
        } else {
            "glm_sparse_mla_prefill_bf16_head8"
        };
        // Tile semantic-index scoring over eight query rows during prefill so
        // both queries and pooled keys are reused. Decode retains the original
        // eight-pool kernel because it has only one live row.
        let env_on = |name| std::env::var(name).ok().as_deref() == Some("1");
        let row_group = !env_on("ATLAS_GLM_INDEX_ROW_GROUP");
        let wmma_shape = probes.glm_kpool_indexer
            && config.index_n_heads == 32
            && config.index_head_dim == 128
            && config.index_kpool == 4
            && row_group;
        // Bit-identical to the WMMA scorer (same module and contract), faster.
        let glm_index_logits_v2 = wmma_shape && env_on("ATLAS_GLM_INDEX_LOGITS_V2");
        let glm_index_wmma = wmma_shape && (env_on("ATLAS_GLM_INDEX_WMMA") || glm_index_logits_v2);
        let glm_sparse_graphs = probes.glm_kpool_indexer
            && super::glm_multi_seq_sparse_graphs_enabled(&config.model_type)?;
        let (glm_index_logits_fn, glm_index_logits_rows_per_cta, glm_index_logits_pools_per_cta) =
            if glm_index_logits_v2 {
                let pools = crate::layers::ops::GLM_INDEX_LOGITS_V2_POOLS;
                ("glm_index_logits_bf16_mma_v2", 4, pools)
            } else if glm_index_wmma {
                ("glm_index_logits_bf16_wmma_row8_pool32", 8, 32)
            } else if row_group {
                ("glm_index_logits_bf16_row8", 8, 8)
            } else {
                ("glm_index_logits_bf16", 1, 8)
            };
        Ok(Self {
            glm_sparse_attn_heads_per_cta,
            glm_sparse_attn_fn,
            glm_index_logits_rows_per_cta,
            glm_index_logits_pools_per_cta,
            glm_index_wmma,
            glm_sparse_graphs,
            glm_index_logits_fn,
        })
    }
}

/// `paged_decode_mla_k`. `#[track_caller]` so the boot audit still names the
/// constructor's field, as with `gated`.
#[track_caller]
pub(super) fn paged_decode_mla(
    probes: &ArchProbes,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    mla_decode_mod: &str,
) -> Result<KernelHandle> {
    // GLM's latent width is 512, not the inherited DeepSeek 576.
    // Require its exact kernel; a missing module must not fall back.
    Ok(if config.model_type == "glm5_next" {
        gpu.kernel(mla_decode_mod, "paged_decode_attn")?
    } else {
        gate(probes.mla, gpu, mla_decode_mod, "paged_decode_attn")
    })
}
