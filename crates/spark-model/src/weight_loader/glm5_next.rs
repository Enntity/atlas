// SPDX-License-Identifier: AGPL-3.0-only

//! Mixed-dtype GLM-5.3-Flash EXL3 checkpoint loader.

use anyhow::{Context, Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
use std::{mem::size_of, sync::Arc};

use crate::layer::TransformerLayer;
use crate::layers::glm5::{
    DsaWeights, Glm5Layer, GlmAttentionWeights, GlmDenseFfnWeights, GlmExl3MoeWeights, GlmFfn,
    GlmHcWeights, GlmSharedExpertSchedule, KdaWeights,
};
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{DenseWeight, MtpWeights, QuantWeight};

const ROOT: &str = "model.language_model";
pub struct Glm5NextWeightLoader;

mod exl3;

fn tensor<'a>(store: &'a WeightStore, name: &str, dtype: WeightDtype) -> Result<&'a WeightTensor> {
    let value = store.get(name)?;
    ensure!(
        value.dtype == dtype,
        "GLM tensor `{name}` has {:?}, expected {dtype:?}",
        value.dtype
    );
    Ok(value)
}

fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: tensor(store, name, WeightDtype::BF16)?.ptr,
    })
}

fn dense_or_f32(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    let value = store.get(name)?;
    ensure!(
        matches!(value.dtype, WeightDtype::BF16 | WeightDtype::FP32),
        "GLM tensor `{name}` has {:?}, expected BF16 or FP32",
        value.dtype
    );
    Ok(DenseWeight { weight: value.ptr })
}

fn quant_dense(store: &WeightStore, name: &str) -> Result<QuantWeight> {
    Ok(QuantWeight::Dense(dense(store, name)?))
}

fn widen_bf16_bytes(values: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        values.len().is_multiple_of(2),
        "BF16 tensor has an odd byte count"
    );
    Ok(values
        .chunks_exact(2)
        .flat_map(|bytes| {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]) as u32;
            (bits << 16).to_le_bytes()
        })
        .collect())
}

fn widen_hc_function(
    value: &WeightTensor,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let expected = [
        (2 + config.hc_mult) * config.hc_mult,
        config.hc_mult * config.hidden_size,
    ];
    ensure!(
        value.shape == expected,
        "GLM mHC function has shape {:?}, expected {expected:?}",
        value.shape
    );
    let elements = expected[0] * expected[1];
    let mut bf16 = vec![0u8; elements * 2];
    gpu.copy_d2h(value.ptr, &mut bf16)?;
    let fp32 = widen_bf16_bytes(&bf16)?;
    let destination = gpu.alloc(fp32.len())?;
    gpu.copy_h2d(&fp32, destination)?;
    Ok(destination)
}

fn hc(
    store: &mut WeightStore,
    root: &str,
    kind: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<GlmHcWeights> {
    let function_name = format!("{root}.hc_{kind}_fn");
    let function = tensor(store, &function_name, WeightDtype::BF16)?;
    let function_ptr = function.ptr;
    let function_shape = function.shape.clone();
    let function_tf32_ptr = widen_hc_function(function, config, gpu)
        .with_context(|| format!("widening `{function_name}` for tensor-core mHC"))?;
    let derived_name = format!("{function_name}.__atlas_tf32");
    if let Err(error) = store.insert_tensor(
        derived_name,
        WeightTensor {
            ptr: function_tf32_ptr,
            shape: function_shape,
            dtype: WeightDtype::FP32,
        },
    ) {
        gpu.free(function_tf32_ptr)?;
        return Err(error).with_context(|| format!("registering widened `{function_name}`"));
    }
    Ok(GlmHcWeights {
        function: DenseWeight {
            weight: function_ptr,
        },
        function_tf32: DenseWeight {
            weight: function_tf32_ptr,
        },
        base: dense_or_f32(store, &format!("{root}.hc_{kind}_base"))?,
        scale: dense_or_f32(store, &format!("{root}.hc_{kind}_scale"))?,
        use_tensor_core: std::env::var("ATLAS_GLM_MHC_SCALAR").ok().as_deref() != Some("1"),
    })
}

fn dense_ffn(
    store: &WeightStore,
    root: &str,
    intermediate_size: usize,
) -> Result<GlmDenseFfnWeights> {
    Ok(GlmDenseFfnWeights {
        gate: dense(store, &format!("{root}.gate_proj.weight"))?,
        up: dense(store, &format!("{root}.up_proj.weight"))?,
        down: dense(store, &format!("{root}.down_proj.weight"))?,
        intermediate_size,
    })
}

fn concat_bf16_rows(
    parts: &[(DenseWeight, usize)],
    columns: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let row_bytes = columns
        .checked_mul(size_of::<u16>())
        .context("GLM merged KDA row size overflow")?;
    let total_bytes = parts.iter().try_fold(0usize, |total, (_, rows)| {
        total
            .checked_add(
                rows.checked_mul(row_bytes)
                    .context("GLM merged KDA part overflow")?,
            )
            .context("GLM merged KDA allocation overflow")
    })?;
    let merged = gpu.alloc(total_bytes)?;
    let mut offset = 0usize;
    for (weight, rows) in parts {
        let bytes = rows * row_bytes;
        gpu.copy_d2d(weight.weight, merged.offset(offset), bytes)?;
        offset += bytes;
    }
    Ok(DenseWeight { weight: merged })
}

fn kda_local_geometry(config: &ModelConfig) -> Result<(usize, usize)> {
    // Topology construction has already converted the global 64-head config
    // to this rank's 32-head view before the model loader runs. Dividing here
    // again would incorrectly describe a 16-head tensor.
    let local_heads = config.linear_num_key_heads;
    let local_width = local_heads * config.linear_key_head_dim;
    ensure!(
        local_heads == 32 && local_width == 4096 && config.linear_key_head_dim == 128,
        "GLM merged KDA projection requires official TP2 geometry (32x128 local), got {local_heads}x{}",
        config.linear_key_head_dim
    );
    Ok((local_heads, local_width))
}

fn load_kda(
    store: &mut WeightStore,
    root: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<KdaWeights> {
    let (local_heads, local_width) = kda_local_geometry(config)?;
    let source_names = [
        format!("{root}.q_proj.weight"),
        format!("{root}.k_proj.weight"),
        format!("{root}.v_proj.weight"),
        format!("{root}.b_proj.weight"),
        format!("{root}.f_a_proj.weight"),
        format!("{root}.g_a_proj.weight"),
    ];
    let query = dense(store, &source_names[0])?;
    let key = dense(store, &source_names[1])?;
    let value = dense(store, &source_names[2])?;
    let beta = dense(store, &source_names[3])?;
    let forget_a = dense(store, &source_names[4])?;
    let gate_a = dense(store, &source_names[5])?;
    let input_merged = concat_bf16_rows(
        &[
            (query, local_width),
            (key, local_width),
            (value, local_width),
            (beta, local_heads),
            (forget_a, 128),
            (gate_a, 128),
        ],
        config.hidden_size,
        gpu,
    )?;
    let merged_rows = 3 * local_width + local_heads + 256;
    store.insert_tensor(
        format!("{root}.__atlas_input_merged.weight"),
        WeightTensor {
            ptr: input_merged.weight,
            shape: vec![merged_rows, config.hidden_size],
            dtype: WeightDtype::BF16,
        },
    )?;
    for name in &source_names {
        let source = store.take_tensor(name)?;
        gpu.free(source.ptr)
            .with_context(|| format!("releasing source tensor `{name}` after KDA merge"))?;
    }
    Ok(KdaWeights {
        input_merged: QuantWeight::Dense(input_merged),
        query_conv: dense(store, &format!("{root}.q_conv1d.weight"))?,
        key_conv: dense(store, &format!("{root}.k_conv1d.weight"))?,
        value_conv: dense(store, &format!("{root}.v_conv1d.weight"))?,
        forget_b: quant_dense(store, &format!("{root}.f_b_proj.weight"))?,
        dt_bias: dense_or_f32(store, &format!("{root}.dt_bias"))?,
        a_log: dense_or_f32(store, &format!("{root}.A_log"))?,
        gate_b: quant_dense(store, &format!("{root}.g_b_proj.weight"))?,
        output_norm: dense(store, &format!("{root}.o_norm.weight"))?,
        output: quant_dense(store, &format!("{root}.o_proj.weight"))?,
    })
}

fn load_dsa(store: &WeightStore, root: &str) -> Result<DsaWeights> {
    let index = format!("{root}.indexer");
    Ok(DsaWeights {
        query_a: quant_dense(store, &format!("{root}.q_a_proj.weight"))?,
        query_a_norm: dense(store, &format!("{root}.q_a_layernorm.weight"))?,
        query_b: quant_dense(store, &format!("{root}.q_b_proj.weight"))?,
        kv_a: quant_dense(store, &format!("{root}.kv_a_proj_with_mqa.weight"))?,
        kv_a_norm: dense(store, &format!("{root}.kv_a_layernorm.weight"))?,
        kv_b: dense(store, &format!("{root}.kv_b_proj.weight"))?,
        output: quant_dense(store, &format!("{root}.o_proj.weight"))?,
        index_query: quant_dense(store, &format!("{index}.wq_b.weight"))?,
        index_key: quant_dense(store, &format!("{index}.wk.weight"))?,
        index_key_norm: dense(store, &format!("{index}.k_norm.weight"))?,
        index_key_bias: dense(store, &format!("{index}.k_norm.bias"))?,
        index_head_weights: quant_dense(store, &format!("{index}.weights_proj.weight"))?,
        index_ape: dense(store, &format!("{index}.index_kpool_compress_ape"))?,
        index_gates: quant_dense(store, &format!("{index}.index_kpool_compress_gate"))?,
    })
}

fn load_ffn(
    store: &WeightStore,
    root: &str,
    layer: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<GlmFfn> {
    if config.mlp_only_layers.contains(&layer) {
        return Ok(GlmFfn::Dense(dense_ffn(
            store,
            root,
            config.intermediate_size / config.tp_world_size,
        )?));
    }
    let pointers = exl3::pointer_tables(store, root, config, gpu)
        .with_context(|| format!("loading GLM EXL3 pointers for layer {layer}"))?;
    let (local_expert_start, local_expert_end) = config.local_expert_range();
    Ok(GlmFfn::Exl3(GlmExl3MoeWeights {
        router: dense(store, &format!("{root}.gate.weight"))?,
        correction_bias: dense_or_f32(store, &format!("{root}.gate.e_score_correction_bias"))?,
        shared: dense_ffn(
            store,
            &format!("{root}.shared_experts"),
            config.shared_expert_intermediate_size / config.tp_world_size,
        )?,
        pointers,
        intermediate_size: config.moe_intermediate_size / config.tp_world_size,
        local_expert_start,
        local_expert_end,
    }))
}

impl ModelWeightLoader for Glm5NextWeightLoader {
    fn supports_tp(&self) -> bool {
        true
    }

    fn load_layers(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        anyhow::bail!("GLM appliance construction requires mutable weight packing")
    }

    fn load_layers_mut(
        &self,
        store: &mut WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let quant_method = config
            .quantization_config
            .as_ref()
            .map(|quant| quant.quant_method.as_str());
        ensure!(
            quant_method == Some("exl3"),
            "GLM native loader currently supports the routed-expert EXL3 checkpoint, got {:?}",
            quant_method
        );
        ensure!(config.tp_world_size == 2, "GLM appliance requires TP2");
        ensure!(config.ep_world_size == 1, "GLM appliance requires pure TP2");
        let shared_expert_schedule = Arc::new(GlmSharedExpertSchedule::new(gpu)?);
        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let root = format!("{ROOT}.layers.{layer}");
            let attention_root = format!("{root}.self_attn");
            let attention = match config.layer_type(layer) {
                LayerType::LinearAttention => {
                    GlmAttentionWeights::Kda(load_kda(store, &attention_root, config, gpu)?)
                }
                LayerType::FullAttention => {
                    GlmAttentionWeights::Dsa(load_dsa(store, &attention_root)?)
                }
                other => {
                    anyhow::bail!("GLM layer {layer} has unsupported attention type {other:?}")
                }
            };
            layers.push(Box::new(Glm5Layer::new(
                layer,
                dense(store, &format!("{root}.input_layernorm.weight"))?,
                dense(store, &format!("{root}.post_attention_layernorm.weight"))?,
                attention,
                load_ffn(store, &format!("{root}.mlp"), layer, config, gpu)?,
                hc(store, &root, "attn", config, gpu)?,
                hc(store, &root, "ffn", config, gpu)?,
                Arc::clone(&shared_expert_schedule),
                gpu,
            )?));
        }
        tracing::info!(
            "loaded GLM-5.3-Flash EXL3: {} layers, TP rank {}/{}, EP rank {}/{}, experts {:?}",
            layers.len(),
            config.tp_rank,
            config.tp_world_size,
            config.ep_rank,
            config.ep_world_size,
            config.local_expert_range(),
        );
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, &format!("{ROOT}.embed_tokens.weight"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, &format!("{ROOT}.norm.weight"))
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // Physical layer 45 shares sparse-index state with the main model.
        // Keep it disabled until that state ownership is implemented.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exl3_marker_matches_checkpoint_contract() {
        assert_eq!((-877_912_083_i32) as u32, 0xcbac_1fed);
        assert!(Glm5NextWeightLoader.supports_tp());
    }

    #[test]
    fn exl3_trellis_geometry_preserves_matrix_orientation() {
        assert_eq!(exl3::trellis_shape(4096, 2048), [256, 128, 64]);
        assert_eq!(exl3::trellis_shape(2048, 4096), [128, 256, 64]);
    }

    #[test]
    fn bf16_hc_weights_widen_exactly_to_fp32() {
        let bf16 = [0x80, 0x3f, 0x00, 0xc0, 0x40, 0x3e];
        let widened = widen_bf16_bytes(&bf16).unwrap();
        let values = widened
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, [1.0, -2.0, 0.1875]);
    }

    #[test]
    fn kda_loader_consumes_the_rank_local_head_count_once() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.tp_world_size = 2;
        config.linear_num_key_heads = 32;
        config.linear_key_head_dim = 128;
        assert_eq!(kda_local_geometry(&config).unwrap(), (32, 4096));
        config.linear_num_key_heads = 16;
        assert!(kda_local_geometry(&config).is_err());
    }
}
