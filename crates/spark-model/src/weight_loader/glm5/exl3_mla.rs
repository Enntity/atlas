// SPDX-License-Identifier: AGPL-3.0-only

//! GLM zero-RoPE MLA with native EXL3 projections and ordinary F16 KV-B.

use anyhow::{Context, Result, ensure};
use atlas_core::{config::ModelConfig, scope::ModelResource};
use half::{bf16, f16};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;
use spark_runtime::weights::exl3::dense::Exl3CheckpointDtype as Dtype;

use super::{exl3::Exl3ProjectionSpec, exl3_projection::Glm5Exl3Projection};
use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::layers::qwen3_attention::{
    GlmIndexerWeights, HcWeights, MlaWeights, Qwen3AttentionLayer,
};
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight, dense_auto};

/// Explicit model owner for packed projections and local ordinary KV views.
/// Slots are Qa, Qb, KvA, O, and indexer Q, in that order.
pub struct Glm5Exl3MlaResources {
    pub(crate) projections: Vec<Glm5Exl3Projection>,
    pub(crate) mhc: Option<crate::layers::glm5_exl3_mhc::NativeExl3Mhc>,
    derived: Vec<DevicePtr>,
    index_params: Option<super::exl3_index::Glm5Exl3IndexParams>,
    index_exec: Option<super::exl3_index_exec::Glm5Exl3IndexExec>,
    /// Opt-in reference-comparison capture (`ATLAS_EXL3_MLA_DUMP_*`).
    pub(crate) trace: Option<crate::layers::qwen3_attention::exl3_mla_trace::MlaTrace>,
}
impl Glm5Exl3MlaResources {
    pub fn index_exec(&self) -> Result<&super::exl3_index_exec::Glm5Exl3IndexExec> {
        self.index_exec
            .as_ref()
            .context("native MLA index execution state absent")
    }
    /// F32 pointers remain separate from legacy BF16 indexer weights.
    pub fn index_f32_params(&self) -> Result<super::exl3_index::Glm5Exl3IndexF32Params> {
        self.index_params
            .as_ref()
            .context("native MLA index parameters absent")?
            .f32_params()
    }
}
impl<'a> ModelResource<dyn GpuBackend + 'a> for Glm5Exl3MlaResources {
    fn label(&self) -> &'static str {
        "GLM native EXL3 MLA resources"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        let mut first = None;
        if let Some(exec) = &mut self.index_exec {
            if let Err(error) = exec.release(gpu) {
                first = Some(error);
            }
        }
        if let Some(params) = &mut self.index_params {
            if let Err(error) = params.release(gpu)
                && first.is_none()
            {
                first = Some(error);
            }
        }
        for projection in &mut self.projections {
            if let Err(error) = projection.release(gpu)
                && first.is_none()
            {
                first = Some(error);
            }
        }
        for ptr in self.derived.iter_mut().rev() {
            if ptr.is_null() {
                continue;
            }
            match gpu.free(*ptr) {
                Ok(()) => *ptr = DevicePtr::NULL,
                Err(error) => {
                    if first.is_none() {
                        first = Some(error);
                    }
                }
            }
        }
        first.map_or(Ok(()), Err)
    }
}

/// Build a production attention layer; no EXL3 weight is reconstructed or
/// converted. Ordinary KV-B supplies the existing per-head absorption views.
#[allow(clippy::too_many_arguments)]
pub fn load_exl3_mla_layer(
    store: &WeightStore,
    config: &ModelConfig,
    layer_idx: usize,
    attn_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    hc: Option<HcWeights>,
    gpu: &dyn GpuBackend,
    kv_dtype: KvCacheDtype,
    capacity: usize,
    stream: u64,
) -> Result<Box<dyn TransformerLayer>> {
    load_exl3_mla_layer_with_numerics(
        store,
        config,
        layer_idx,
        attn_idx,
        input_norm,
        post_attn_norm,
        ffn,
        hc,
        gpu,
        kv_dtype,
        capacity,
        stream,
        super::exl3_numerics::glm53_tp2_packed_numerics,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn load_exl3_mla_layer_with_numerics(
    store: &WeightStore,
    config: &ModelConfig,
    layer_idx: usize,
    attn_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    hc: Option<HcWeights>,
    gpu: &dyn GpuBackend,
    kv_dtype: KvCacheDtype,
    capacity: usize,
    stream: u64,
    select: super::exl3_numerics::Exl3NumericsSelector,
) -> Result<Box<dyn TransformerLayer>> {
    ensure!(
        config.model_type == "glm5_next" && config.qk_rope_head_dim == 0,
        "native EXL3 MLA requires zero-RoPE glm5_next"
    );
    ensure!(
        config.ep_world_size == 1 && !config.expert_tp,
        "native EXL3 MLA requires pure TP"
    );
    ensure!(
        config.num_attention_heads == config.num_key_value_heads,
        "native EXL3 MLA requires matching local Q/KV heads"
    );
    let checkpoint = store
        .exl3_checkpoint()
        .context("native MLA missing EXL3 checkpoint router")?;
    let p = format!("{}.self_attn", config.layer_prefix(layer_idx));
    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    for (name, shape) in [
        ("wk.weight", [config.index_head_dim, config.hidden_size]),
        (
            "weights_proj.weight",
            [config.index_n_heads, config.hidden_size],
        ),
        (
            "index_kpool_compress_gate",
            [config.index_head_dim, config.hidden_size],
        ),
    ] {
        super::exl3_index::validate_bf16_index_matrix(
            store,
            &format!("{p}.indexer.{name}"),
            &shape,
        )?;
    }
    let mut resources = Glm5Exl3MlaResources {
        mhc: hc
            .as_ref()
            .map(|hc| {
                crate::layers::glm5_exl3_mhc::NativeExl3Mhc::load(
                    gpu,
                    config.hidden_size,
                    hc.hc_mult,
                    hc.sinkhorn_iters,
                    config.rms_norm_eps as f32,
                    hc.hc_eps,
                    crate::layers::glm5_exl3_mhc::MhcNumericalPolicy::GLM53_TP2_REFERENCE,
                    config.hc_post_multiplier,
                )
            })
            .transpose()?,
        projections: Vec::new(),
        derived: Vec::new(),
        index_params: None,
        index_exec: None,
        trace: crate::layers::qwen3_attention::exl3_mla_trace::MlaTrace::from_env(
            config, layer_idx,
        )?,
    };
    let result = (|| -> Result<Qwen3AttentionLayer> {
        resources.index_params = Some(super::exl3_index::Glm5Exl3IndexParams::load(
            store,
            &format!("{p}.indexer"),
            config.index_head_dim,
            config.index_kpool,
            gpu,
        )?);
        resources.index_exec = Some(super::exl3_index_exec::Glm5Exl3IndexExec::new(
            gpu,
            capacity,
            config.index_n_heads,
            config.hidden_size,
            stream,
        )?);
        for name in [
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "o_proj",
            "indexer.wq_b",
        ] {
            let prefix = format!("{p}.{name}");
            let spec = Exl3ProjectionSpec::from_rank_local_config(config, &prefix)?;
            resources
                .projections
                .push(spec.load_projection(checkpoint, gpu, capacity, stream, select)?);
        }
        let (kv_b, uk, uv) = load_kv_views(checkpoint, &p, config, gpu, &mut resources.derived)?;
        let dense = |name: &str| dense_auto(store, &format!("{p}.{name}"), gpu);
        let indexer = GlmIndexerWeights {
            wq_b: null,
            wk: dense("indexer.wk.weight")?,
            weights_proj: dense("indexer.weights_proj.weight")?,
            kpool_gate: dense("indexer.index_kpool_compress_gate")?,
            kpool_ape: dense("indexer.index_kpool_compress_ape")?,
            k_norm_weight: dense("indexer.k_norm.weight")?,
            k_norm_bias: dense("indexer.k_norm.bias")?,
        };
        let mla = MlaWeights {
            wq_a: null,
            wq_a_nvfp4: None,
            wq_a_fp8: None,
            wq_b: null,
            wq_b_nvfp4: None,
            wq_b_fp8: None,
            q_a_norm: dense("q_a_layernorm.weight")?,
            wkv_a: null,
            wkv_a_nvfp4: None,
            wkv_a_fp8: None,
            wkv_b: kv_b,
            kv_a_norm: dense("kv_a_layernorm.weight")?,
            wkv_a_rope: null,
            wkv_a_merged: null,
            wo: null,
            wo_nvfp4: None,
            wo_a: null,
            wo_a_nvfp4: None,
            wo_a_fp8: None,
            wo_b: null,
            wo_b_nvfp4: None,
            wo_b_fp8: None,
            w_uk_t: uk,
            w_uv: uv,
            wq_b_rope: null,
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
            glm_indexer: Some(indexer),
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
        layer.configure_exl3_index_budget(gpu, config)?;
        layer.set_block_idx(layer_idx);
        layer.set_mla_weights(mla);
        if let Some(hc) = hc {
            layer.set_hc_weights(hc);
        }
        Ok(layer)
    })();
    match result {
        Ok(mut layer) => {
            // This new layer has no converted caches or prior native owner.
            layer.set_exl3_mla_resources(resources)?;
            Ok(Box::new(layer))
        }
        Err(error) => match resources.release(gpu) {
            Ok(()) => Err(error),
            Err(cleanup) => {
                Err(error.context(format!("EXL3 MLA cleanup also failed: {cleanup:#}")))
            }
        },
    }
}

fn upload_owned(
    gpu: &dyn GpuBackend,
    data: &[u8],
    owned: &mut Vec<DevicePtr>,
) -> Result<DenseWeight> {
    let ptr = gpu.alloc(data.len())?;
    owned.push(ptr);
    gpu.copy_h2d(data, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

fn load_kv_views(
    checkpoint: &spark_runtime::weights::exl3::Exl3CheckpointRouter,
    prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    owned: &mut Vec<DevicePtr>,
) -> Result<(DenseWeight, DenseWeight, DenseWeight)> {
    let heads = config.num_key_value_heads;
    let rank = config.tp_rank;
    let world = config.tp_world_size;
    ensure!(world > 0 && rank < world, "EXL3 MLA invalid TP geometry");
    let nope = config.qk_nope_head_dim;
    let v = config.v_head_dim;
    let latent = config.kv_lora_rank;
    ensure!(
        heads > 0 && nope > 0 && v > 0 && latent > 0,
        "EXL3 MLA KV dimensions must be explicit"
    );
    let full_rows = heads
        .checked_mul(world)
        .and_then(|h| h.checked_mul(nope + v))
        .context("EXL3 MLA KV dimensions overflow")?;
    let source = checkpoint.dense_view(
        &format!("{prefix}.kv_b_proj.weight"),
        Dtype::F16,
        &[full_rows, latent],
    )?;
    let row_bytes = latent * 2;
    let local_rows = heads * (nope + v);
    let start = rank * local_rows * row_bytes;
    let bytes = &source.data()[start..start + local_rows * row_bytes];
    let (kv, uk, uv) = kv_view_bytes(bytes, heads, nope, v, latent)?;
    Ok((
        upload_owned(gpu, &kv, owned)?,
        upload_owned(gpu, &uk, owned)?,
        upload_owned(gpu, &uv, owned)?,
    ))
}

/// Ordinary F16 conversion and layout only; EXL3 Q-B is never read here.
fn kv_view_bytes(
    bytes: &[u8],
    heads: usize,
    nope: usize,
    v: usize,
    latent: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let elements = heads
        .checked_mul(nope + v)
        .and_then(|n| n.checked_mul(latent))
        .context("EXL3 MLA KV view dimensions overflow")?;
    ensure!(
        bytes.len() == elements * 2,
        "EXL3 MLA KV view byte length mismatch"
    );
    let kv: Vec<u8> = bytes
        .chunks_exact(2)
        .flat_map(|pair| {
            bf16::from_f32(f16::from_le_bytes([pair[0], pair[1]]).to_f32()).to_le_bytes()
        })
        .collect();
    let mut uk = vec![0u8; heads * latent * nope * 2];
    let mut uv = vec![0u8; heads * v * latent * 2];
    for h in 0..heads {
        for p in 0..nope {
            for l in 0..latent {
                let src = ((h * (nope + v) + p) * latent + l) * 2;
                let dst = ((h * latent + l) * nope + p) * 2;
                uk[dst..dst + 2].copy_from_slice(&kv[src..src + 2]);
            }
        }
        let src = (h * (nope + v) + nope) * latent * 2;
        let dst = h * v * latent * 2;
        uv[dst..dst + v * latent * 2].copy_from_slice(&kv[src..src + v * latent * 2]);
    }
    Ok((kv, uk, uv))
}

#[cfg(test)]
#[path = "exl3_mla_tests.rs"]
mod tests;
