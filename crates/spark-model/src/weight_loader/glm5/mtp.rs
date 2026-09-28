// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 Flash next-token-prediction module loader.
//!
//! The checkpoint stores its single MTP module as decoder layer
//! `num_hidden_layers`, with an MTP-specific input combiner and shared-head
//! norm. Its body is a full-attention MLA + MoE layer with ordinary residuals
//! (the target's mHC tensors are intentionally absent from the MTP layer).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::weight_map::{
    DenseWeight, QuantizeCtx, QuantizedWeight, dense_auto, detect_nvfp4_variant, quantize_to_nvfp4,
};

/// Loaded GLM MTP module. Embedding and LM head are shared with the target.
pub struct Glm5MtpModule {
    pub body: Box<dyn TransformerLayer>,
    pub enorm: DenseWeight,
    pub hnorm: DenseWeight,
    /// Fused `[embed || target_hidden] -> hidden` projection.
    pub eh_proj: DenseWeight,
    /// Optional decode-native copy of `eh_proj`.  Prompt KV prefill retains
    /// the BF16 matrix above; the autoregressive proposer can stream this
    /// compact copy once per draft on GB10.
    pub eh_proj_nvfp4: Option<QuantizedWeight>,
    pub norm: DenseWeight,
}

/// Load the appended GLM MTP layer as a rank-0 proposer, or replicate that
/// exact body on both ranks when split-vocabulary MTP is explicitly enabled.
pub fn load_glm5_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<Glm5MtpModule>> {
    if config.num_mtp_modules == 0 {
        return Ok(None);
    }
    anyhow::ensure!(
        config.num_mtp_modules == 1,
        "GLM-5 Atlas MTP currently supports exactly one appended predictor layer"
    );
    let layer_idx = config.num_hidden_layers;
    let lp = config.layer_prefix(layer_idx);
    if !store.contains(&format!("{lp}.enorm.weight")) {
        tracing::info!("GLM-5: checkpoint has no appended MTP layer; speculative decode off");
        return Ok(None);
    }

    let distributed = std::env::var("ATLAS_GLM_MTP_DISTRIBUTED").ok().as_deref() == Some("1");

    // Keep the appended body identical to the proven rank-0 proposer on both
    // ranks: full TP dimensions and all experts, with no body collectives.
    // Distributed mode uses rank 1 only to split the dominant vocabulary
    // projection, preserving draft arithmetic and therefore acceptance.
    let mut draft_config = config.clone();
    if distributed {
        anyhow::ensure!(
            draft_config.tp_world_size == 2 && draft_config.ep_world_size == 2,
            "ATLAS_GLM_MTP_DISTRIBUTED=1 requires GLM overlapping TP=EP=world=2 \
             (got TP={}, EP={})",
            draft_config.tp_world_size,
            draft_config.ep_world_size,
        );
    }
    let target_tp = draft_config.tp_world_size.max(1);
    draft_config.num_attention_heads *= target_tp;
    draft_config.num_key_value_heads *= target_tp;
    draft_config.tp_rank = 0;
    draft_config.tp_world_size = 1;
    draft_config.ep_rank = 0;
    draft_config.ep_world_size = 1;
    // The appended layer keeps whole experts (see the loader's replication
    // prefixes), so it never takes the expert-TP slices.
    draft_config.expert_tp = false;

    let variant = detect_nvfp4_variant(store, &draft_config);
    let qctx = QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    };
    let input_norm = dense_auto(store, &format!("{lp}.input_layernorm.weight"), gpu)?;
    let post_attn_norm = dense_auto(store, &format!("{lp}.post_attention_layernorm.weight"), gpu)?;
    let ffn = super::components::load_ffn(
        store,
        &lp,
        layer_idx,
        &draft_config,
        gpu,
        variant,
        qctx,
        false, // decode-only proposer: do not materialize full-rank prefill twins
    )?;
    let body = super::layers::load_mla_layer(
        store,
        &lp,
        layer_idx,
        0, // private MTP cache contains exactly this one attention layer
        input_norm,
        post_attn_norm,
        ffn,
        None, // GLM's appended predictor uses ordinary residuals, not mHC
        &draft_config,
        gpu,
        KvCacheDtype::Bf16,
        true,
        None, // Replicated TP1 body retains its checkpoint projections.
    )?;

    let eh_proj = dense_auto(store, &format!("{lp}.eh_proj.weight"), gpu)?;
    let eh_proj_nvfp4 = if std::env::var("ATLAS_GLM_MTP_NVFP4_EH").ok().as_deref() == Some("1") {
        let q = quantize_to_nvfp4(
            &eh_proj,
            draft_config.hidden_size,
            2 * draft_config.hidden_size,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        )?;
        tracing::info!(
            "GLM-5 MTP eh_proj: built decode-native NVFP4 copy (BF16 retained for prefill)"
        );
        Some(q)
    } else {
        None
    };

    let module = Glm5MtpModule {
        body,
        enorm: dense_auto(store, &format!("{lp}.enorm.weight"), gpu)?,
        hnorm: dense_auto(store, &format!("{lp}.hnorm.weight"), gpu)?,
        eh_proj,
        eh_proj_nvfp4,
        norm: dense_auto(store, &format!("{lp}.shared_head.norm.weight"), gpu)?,
    };
    tracing::info!("GLM-5 MTP module loaded: appended layer {layer_idx} (full MLA + MoE)");
    Ok(Some(module))
}
