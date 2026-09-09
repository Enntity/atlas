// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit constructor selection, not server/supervisor admission authority.
use crate::layers::{Glm5MtpHead, MtpQuantization};
use crate::weight_loader::glm5::Glm5MtpModule;
use crate::weight_map::{DenseWeight, QuantizedWeight};
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlmMtpBuildMode {
    Legacy,
    Paired,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate(
    mode: GlmMtpBuildMode,
    config: &ModelConfig,
    max_batch_tokens: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    mtp_quant: MtpQuantization,
    use_speculative: bool,
    self_speculative: bool,
    num_drafts: usize,
    kv_dtype: KvCacheDtype,
    layer_dtypes: &[KvCacheDtype],
    comm: Option<&dyn spark_comm::CommBackend>,
    hss: Option<u32>,
    dflash: bool,
    lora: bool,
) -> Result<()> {
    ensure!(
        mode == GlmMtpBuildMode::Paired || crate::model::glm_c2_pair_policy::requested()?.is_none(),
        "joint verification requires explicitly selected paired factory"
    );
    if mode == GlmMtpBuildMode::Legacy {
        return Ok(());
    }
    ensure!(
        config.model_type == "glm5_next"
            && config.hidden_size == 4096
            && config.kv_lora_rank == 512
            && config.qk_rope_head_dim == 0
            && config.tp_world_size == 2
            && config.ep_world_size == 2
            && config.tp_rank == config.ep_rank
            && config.ep_rank < 2,
        "paired factory requires actual GLM4096 NoPE512 TP2/EP2"
    );
    let comm = comm.ok_or_else(|| anyhow::anyhow!("paired factory communicator missing"))?;
    ensure!(
        comm.world_size() == 2 && comm.rank() == config.ep_rank,
        "paired factory communicator rank/world mismatch"
    );
    ensure!(
        use_speculative
            && !self_speculative
            && num_drafts == 4
            && matches!(mtp_quant, MtpQuantization::Bf16)
            && config.num_mtp_modules > 0
            && (2..=4).contains(&max_batch_size)
            && max_batch_tokens >= 5
            && (2..=2044).contains(&max_seq_len),
        "paired factory requires two to four owners, BF16 MTP4 and bounded target capacity"
    );
    ensure!(
        kv_dtype == KvCacheDtype::Bf16
            && layer_dtypes.iter().all(|d| *d == KvCacheDtype::Bf16)
            && !dflash
            && !lora
            && hss.is_none()
            && config.adapter_max_rank == 0
            && config.vision.is_none()
            && config.dflash_capture_layers.is_empty(),
        "paired factory requires base BF16 KV without alternate model owners"
    );
    ensure!(
        std::env::var("ATLAS_EP_PROTOCOL").as_deref() == Ok("v2")
            && crate::layers::glm5_mtp::distributed_enabled(),
        "paired factory requires actual EPv2 distributed GLM policy"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_head(
    mode: GlmMtpBuildMode,
    module: Glm5MtpModule,
    embed: DenseWeight,
    lm_head: DenseWeight,
    nvfp4: Option<QuantizedWeight>,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    vocabulary: u32,
    context: usize,
    max_batch_size: usize,
) -> Result<Glm5MtpHead> {
    match mode {
        GlmMtpBuildMode::Legacy => Glm5MtpHead::new(
            module, embed, lm_head, nvfp4, config, gpu, vocabulary, context,
        ),
        GlmMtpBuildMode::Paired => Glm5MtpHead::new_paired_with_owner_capacity(
            module,
            embed,
            lm_head,
            nvfp4,
            config,
            gpu,
            vocabulary,
            context,
            max_batch_size,
        ),
    }
}

/// Own the already-assembled target across appended-head construction errors.
#[allow(clippy::too_many_arguments)]
pub(super) fn install_head(
    model: crate::model::TransformerModel,
    mode: GlmMtpBuildMode,
    module: Option<Glm5MtpModule>,
    embed: DenseWeight,
    lm_head: DenseWeight,
    nvfp4: Option<QuantizedWeight>,
    vocabulary: u32,
    context: usize,
    max_batch_size: usize,
    distributed: bool,
) -> Result<crate::model::TransformerModel> {
    let mut model = super::ColdOwner::new(model, mode == GlmMtpBuildMode::Paired);
    ensure!(
        mode != GlmMtpBuildMode::Paired || module.is_some(),
        "paired factory requires an actual appended GLM module on this rank"
    );
    if let Some(module) = module {
        match build_head(
            mode,
            module,
            embed,
            lm_head,
            nvfp4,
            model.config_ref(),
            model.gpu_backend(),
            vocabulary,
            context,
            max_batch_size,
        ) {
            Ok(head) => {
                model.set_dflash_proposer(std::sync::Arc::new(head));
                tracing::info!("GLM-5 MTP speculative decoding: ENABLED (single module)");
            }
            Err(error) if distributed || mode == GlmMtpBuildMode::Paired => {
                return Err(error.context(format!(
                    "distributed GLM MTP proposer construction failed on rank {}",
                    model.config_ref().ep_rank
                )));
            }
            Err(error) => tracing::warn!(
                "Failed to build GLM-5 MTP proposer: {error:#}. Speculative decoding disabled."
            ),
        }
    }
    if mode == GlmMtpBuildMode::Paired {
        model
            .configure_glm_pair_verification()
            .inspect_err(|error| {
                // ColdOwner must retain fatal cleanup semantics, but it exits before
                // the caller can print a propagated error. Diagnose before its drop.
                tracing::error!(
                    rank = model.config_ref().ep_rank,
                    "paired target verification admission failed: {error:#}"
                );
            })?;
    }
    Ok(model.into_inner())
}

#[cfg(test)]
#[path = "glm_paired_tests.rs"]
mod tests;
