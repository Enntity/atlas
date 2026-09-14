// SPDX-License-Identifier: AGPL-3.0-only
//! Narrow startup admission for the experimental repaired-K3 BF16 attention lane.
use super::GlmMtpBuildMode;
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::{gpu::GpuBackend, kv_cache::KvCacheDtype};

pub(super) struct BuildPolicy<'a> {
    pub config: &'a ModelConfig,
    pub mode: GlmMtpBuildMode,
    pub speculative: bool,
    pub self_speculative: bool,
    pub drafts: usize,
    pub owners: usize,
    pub context: usize,
    pub block_size: usize,
    pub kv_dtype: KvCacheDtype,
    pub layer_dtypes: &'a [KvCacheDtype],
    pub alternate_owner: bool,
}

impl BuildPolicy<'_> {
    fn validate(&self, repaired: bool, long_context: bool) -> Result<()> {
        ensure!(
            self.config.model_type == "glm5_next"
                && self.config.tp_world_size == 2
                && self.config.ep_world_size == 2
                // Serving topology has already converted this to local heads.
                && self.config.num_attention_heads == 32
                && self.mode == GlmMtpBuildMode::Legacy
                && self.speculative
                && !self.self_speculative
                && self.drafts == 2
                && self.owners == 4
                && (2049..=32768).contains(&self.context)
                && self.block_size == 16
                && self.kv_dtype == KvCacheDtype::Bf16
                && self.layer_dtypes.iter().all(|d| *d == KvCacheDtype::Bf16)
                && !self.alternate_owner
                && repaired
                && long_context,
            "GLM sparse decode TC requires repaired long-context MTP2, four owners, TP2/EP2 local32 heads and BF16 block16 caches"
        );
        Ok(())
    }
}

pub(super) fn initialize(gpu: &dyn GpuBackend, policy: BuildPolicy<'_>) -> Result<()> {
    let split = crate::layers::ops::glm_sparse_decode_split_enabled(&policy.config.model_type)?;
    let tc = crate::layers::ops::glm_sparse_decode_tc_enabled(&policy.config.model_type)?;
    ensure!(
        !split || tc,
        "GLM split requires ATLAS_GLM_SPARSE_DECODE_TC=1"
    );
    if !tc {
        return Ok(());
    }
    policy.validate(
        crate::speculative::glm_repair_policy::enabled(),
        crate::speculative::glm_repair_policy::long_context_enabled(),
    )?;
    crate::layers::ops::initialize_glm_sparse_decode_tc(gpu, policy.config)?;
    tracing::info!(
        "GLM sparse decode TC initialized before KV sizing; repaired K3 rows1, BF16 identical K/V"
    );
    if split {
        crate::layers::ops::initialize_glm_sparse_decode_split(gpu, policy.config)?;
        tracing::info!(
            "GLM sparse decode split initialized before KV sizing; fixed S8, repaired K3 rows1"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm_sparse_decode_tests.rs"]
mod tests;
