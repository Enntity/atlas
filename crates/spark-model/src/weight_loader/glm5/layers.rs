// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::{GlmIndexerWeights, MlaWeights, Qwen3AttentionLayer};
use crate::layers::{FfnComponent, Glm5KdaLayer};
use crate::tp_shard::shard_dense_bf16;
use crate::weight_map::{
    AttentionWeights, DenseWeight, QuantizeCtx, QuantizedWeight, dense_auto, detect_nvfp4_variant,
};

pub(super) fn load_all(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    anyhow::ensure!(
        config.max_position_embeddings >= 2048 && config.index_topk == 2048,
        "GLM-5 Atlas bring-up expects index_topk=2048"
    );
    let variant = detect_nvfp4_variant(store, config);
    let qctx = QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    };
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    let mut attn_idx = 0usize;
    for layer_idx in 0..config.num_hidden_layers {
        let lp = config.layer_prefix(layer_idx);
        let input_norm = dense_auto(store, &format!("{lp}.input_layernorm.weight"), gpu)?;
        let post_attn_norm =
            dense_auto(store, &format!("{lp}.post_attention_layernorm.weight"), gpu)?;
        let ffn =
            super::components::load_ffn(store, &lp, layer_idx, config, gpu, variant, qctx, true)?;
        let hc = super::components::load_hc(store, &lp, config, gpu)?;
        match config.layer_type(layer_idx) {
            LayerType::LinearAttention => {
                let weights = super::components::load_kda_weights(store, &lp, config, gpu, qctx)?;
                layers.push(Box::new(Glm5KdaLayer::new(
                    input_norm,
                    post_attn_norm,
                    weights,
                    ffn,
                    hc,
                    layer_idx,
                    config,
                    gpu,
                )?));
            }
            LayerType::FullAttention => {
                let kv_dtype = layer_kv_dtypes
                    .get(attn_idx)
                    .copied()
                    .unwrap_or(KvCacheDtype::Bf16);
                let layer = load_mla_layer(
                    store,
                    &lp,
                    layer_idx,
                    attn_idx,
                    input_norm,
                    post_attn_norm,
                    ffn,
                    Some(hc),
                    config,
                    gpu,
                    kv_dtype,
                    false,
                )?;
                layers.push(layer);
                attn_idx += 1;
            }
            other => anyhow::bail!("GLM-5 unsupported layer type {other:?}"),
        }
        if layer_idx < 5 || (layer_idx + 1).is_multiple_of(10) {
            tracing::info!(
                "GLM-5 loaded layers 0..{} — {:.1} GiB free",
                layer_idx + 1,
                gpu.free_memory()? as f64 / 1_073_741_824.0,
            );
        }
    }
    Ok(layers)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_mla_layer(
    store: &WeightStore,
    lp: &str,
    layer_idx: usize,
    attn_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    hc: Option<crate::layers::qwen3_attention::HcWeights>,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    kv_dtype: KvCacheDtype,
    force_dimension_overrides: bool,
) -> Result<Box<dyn TransformerLayer>> {
    let p = format!("{lp}.self_attn");
    let tp = super::tp::MlaTpPlan::from_config(config);
    let load_tp = |name: &str, shape: (usize, usize, crate::tp_shard::TpShardKind)| {
        let source = dense_auto(store, &format!("{p}.{name}.weight"), gpu)?;
        let (local, _, _) = shard_dense_bf16(
            source.weight,
            shape.0,
            shape.1,
            shape.2,
            tp.tp_rank,
            tp.tp_size,
            gpu,
        )?;
        if local != source.weight {
            gpu.free(source.weight)?;
        }
        Ok::<DenseWeight, anyhow::Error>(DenseWeight { weight: local })
    };
    let wq_a = dense_auto(store, &format!("{p}.q_a_proj.weight"), gpu)?;
    let wq_b = load_tp("q_b_proj", tp.q_b())?;
    let wkv_a = dense_auto(store, &format!("{p}.kv_a_proj_with_mqa.weight"), gpu)?;
    let wkv_b = load_tp("kv_b_proj", tp.kv_b())?;
    let wo = load_tp("o_proj", tp.o())?;
    let wq_b_shape = tp.local_q_b_shape();
    let wkv_b_shape = tp.local_kv_b_shape();
    let (w_uk_t, w_uv, wq_b_rope, _) = super::super::deepseek_v4::compute::build_per_head_views(
        &wkv_b,
        &wkv_b_shape,
        &wq_b,
        &wq_b_shape,
        config,
        gpu,
    )?;
    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    let indexer_prefix = format!("{p}.indexer");
    let glm_indexer = GlmIndexerWeights {
        wq_b: dense_auto(store, &format!("{indexer_prefix}.wq_b.weight"), gpu)?,
        wk: dense_auto(store, &format!("{indexer_prefix}.wk.weight"), gpu)?,
        weights_proj: dense_auto(store, &format!("{indexer_prefix}.weights_proj.weight"), gpu)?,
        kpool_gate: dense_auto(
            store,
            &format!("{indexer_prefix}.index_kpool_compress_gate"),
            gpu,
        )?,
        kpool_ape: dense_auto(
            store,
            &format!("{indexer_prefix}.index_kpool_compress_ape"),
            gpu,
        )?,
        k_norm_weight: dense_auto(store, &format!("{indexer_prefix}.k_norm.weight"), gpu)?,
        k_norm_bias: dense_auto(store, &format!("{indexer_prefix}.k_norm.bias"), gpu)?,
    };
    let mla = MlaWeights {
        wq_a,
        wq_a_nvfp4: None,
        wq_a_fp8: None,
        wq_b,
        wq_b_nvfp4: None,
        wq_b_fp8: None,
        q_a_norm: dense_auto(store, &format!("{p}.q_a_layernorm.weight"), gpu)?,
        wkv_a,
        wkv_a_nvfp4: None,
        wkv_a_fp8: None,
        wkv_b,
        kv_a_norm: dense_auto(store, &format!("{p}.kv_a_layernorm.weight"), gpu)?,
        // Zero-width RoPE: the forward path skips this projection.
        wkv_a_rope: null,
        wkv_a_merged: wkv_a,
        wo,
        wo_nvfp4: None,
        wo_a: null,
        wo_a_nvfp4: None,
        wo_a_fp8: None,
        wo_b: wo,
        wo_b_nvfp4: None,
        wo_b_fp8: None,
        w_uk_t,
        w_uv,
        wq_b_rope,
        // These historical precomputes are not read by Atlas's active MLA
        // paths; leaving them null avoids multi-GiB sparse block diagonals.
        w_qk_absorbed: null,
        w_uk_block_diag: null,
        w_uv_block_diag: null,
        yarn_inv_freq: DevicePtr::NULL,
        main_inv_freq: DevicePtr::NULL,
        q_lora_rank: config.q_lora_rank,
        kv_lora_rank: config.kv_lora_rank,
        o_lora_rank: 0,
        nope: config.qk_nope_head_dim,
        rope: 0,
        v_dim: config.v_head_dim,
        glm_indexer: Some(glm_indexer),
        compressor: None,
        attn_sink: DevicePtr::NULL,
    };
    let dummy = AttentionWeights {
        q_proj: null,
        k_proj: null,
        v_proj: null,
        o_proj: QuantizedWeight::null(),
        q_norm: null,
        k_norm: null,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        dummy,
        post_attn_norm,
        ffn,
        attn_idx,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        0,
        config,
    )?;
    layer.set_block_idx(layer_idx);
    layer.set_mla_weights(mla);
    if let Some(hc) = hc {
        layer.set_hc_weights(hc);
    }
    if force_dimension_overrides {
        // A model-specific MTP proposer is replicated on rank 0 and therefore
        // uses the checkpoint's full attention dimensions even when the target
        // model is TP-sharded.  Runtime ForwardContext still carries the
        // target's local head counts, so pin the full dimensions on the layer.
        layer.set_dimension_overrides(
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
        );
    }
    Ok(Box::new(layer))
}
