// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights};
use crate::layers::{DenseFfnLayer, FfnComponent, Glm5KdaWeights, Glm5Projection, MoeLayer};
use crate::tp_shard::{TpGdnDims, TpShardKind, shard_dense_bf16, shard_gdn_value_vector};
use crate::weight_map::{
    DenseWeight, ExpertWeight, MoeWeights, Nvfp4Variant, QuantizeCtx, dense_auto, dense_keep_f32,
    quantize_to_nvfp4, quantized_any,
};

pub(super) fn load_hc(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<HcWeights> {
    let load_site = |site: &str| -> Result<HcSiteWeights> {
        let hc = config.hc_mult;
        let mix = (2 + hc) * hc;
        let hc_dim = hc * config.hidden_size;
        let load = |suffix: &str, n: usize| {
            super::super::deepseek_v4::assemble::load_hc_f32(
                store,
                &[format!("{lp}.hc_{site}_{suffix}")],
                n,
                gpu,
            )
        };
        Ok(HcSiteWeights {
            hc_fn: load("fn", mix * hc_dim)?,
            hc_base: load("base", mix)?,
            hc_scale: load("scale", 3)?,
        })
    };
    Ok(HcWeights {
        attn: load_site("attn")?,
        ffn: load_site("ffn")?,
        head: None,
        hc_mult: config.hc_mult,
        sinkhorn_iters: config.hc_sinkhorn_iters,
        hc_eps: config.hc_eps,
    })
}

pub(super) fn load_ffn(
    store: &WeightStore,
    lp: &str,
    layer_idx: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    allow_prefill_layout: bool,
) -> Result<FfnComponent> {
    if config.mlp_only_layers.contains(&layer_idx) {
        return load_dense_ffn(store, lp, config, gpu, qctx);
    }
    load_moe(
        store,
        lp,
        layer_idx,
        config,
        gpu,
        variant,
        qctx,
        allow_prefill_layout,
    )
}

fn load_dense_ffn(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    qctx: QuantizeCtx,
) -> Result<FfnComponent> {
    let p = format!("{lp}.mlp");
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let gate = dense_auto(store, &format!("{p}.gate_proj.weight"), gpu)?;
    let up = dense_auto(store, &format!("{p}.up_proj.weight"), gpu)?;
    let down = dense_auto(store, &format!("{p}.down_proj.weight"), gpu)?;
    let weights = DenseFfnWeights {
        gate_proj: quantize_to_nvfp4(
            &gate,
            inter,
            h,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        )?,
        up_proj: quantize_to_nvfp4(
            &up,
            inter,
            h,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        )?,
        down_proj: quantize_to_nvfp4(
            &down,
            h,
            inter,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        )?,
        // Keep the memory-tight fallback for the first safe deployment.
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    };
    Ok(FfnComponent::Dense(DenseFfnLayer::new(weights, gpu)?))
}

fn load_moe(
    store: &WeightStore,
    lp: &str,
    layer_idx: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    allow_prefill_layout: bool,
) -> Result<FfnComponent> {
    let p = format!("{lp}.mlp");
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let mut experts = Vec::with_capacity(config.num_experts);
    for expert in 0..config.num_experts {
        if config.is_local_expert(expert) {
            let ep = format!("{p}.experts.{expert}");
            experts.push(ExpertWeight {
                gate_proj: quantized_any(
                    store,
                    &format!("{ep}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )
                .with_context(|| format!("GLM-5 expert {expert} gate_proj"))?,
                up_proj: quantized_any(
                    store,
                    &format!("{ep}.up_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )
                .with_context(|| format!("GLM-5 expert {expert} up_proj"))?,
                down_proj: quantized_any(
                    store,
                    &format!("{ep}.down_proj"),
                    h,
                    inter,
                    gpu,
                    variant,
                    qctx,
                )
                .with_context(|| format!("GLM-5 expert {expert} down_proj"))?,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }
    let shared = format!("{p}.shared_experts");
    let shared_expert = ExpertWeight {
        gate_proj: quantized_any(
            store,
            &format!("{shared}.gate_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        up_proj: quantized_any(
            store,
            &format!("{shared}.up_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        down_proj: quantized_any(
            store,
            &format!("{shared}.down_proj"),
            h,
            inter,
            gpu,
            variant,
            qctx,
        )?,
    };
    let bias = store.get(&format!("{p}.gate.e_score_correction_bias"))?;
    anyhow::ensure!(
        bias.num_elements() == config.num_experts,
        "GLM-5 routing bias length mismatch"
    );
    let weights = MoeWeights {
        gate: dense_auto(store, &format!("{p}.gate.weight"), gpu)?,
        shared_expert,
        // GLM-5 shared expert is always active and has no learned gate.
        shared_expert_gate: DenseWeight {
            weight: DevicePtr::NULL,
        },
        experts,
        router_pre_norm: None,
        correction_bias: Some(DenseWeight { weight: bias.ptr }),
    };
    let mut layer = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    let mmq_moe = env_flag("ATLAS_NVFP4_MMQ_MOE");
    let cutlass_moe = env_flag("ATLAS_MOE_GROUPED_CUTLASS");
    anyhow::ensure!(
        !(mmq_moe && cutlass_moe),
        "ATLAS_NVFP4_MMQ_MOE and ATLAS_MOE_GROUPED_CUTLASS are mutually exclusive"
    );
    if mmq_moe {
        layer.repack_nvfp4_mmq_unified(gpu, config)?;
    } else if cutlass_moe {
        // CUTLASS consumes the checkpoint-native packed weights and adds only
        // its block-scale swizzle. Avoid materializing the much larger Atlas
        // transposed expert twins in this mode.
        layer.build_cutlass_grouped_sfb(gpu, config, gpu.default_stream())?;
    } else if allow_prefill_layout
        && unified_moe_layout_enabled(std::env::var("ATLAS_UNIFIED_MOE_LAYOUT").ok().as_deref())
    {
        // Replace the decode-native expert storage with Atlas' transposed
        // layout one layer at a time. The transpose helper frees each source
        // phase before advancing, and EP slabs contain only locally owned
        // experts, so peak memory stays bounded on dual GB10.
        // K=4 verification batches GLM's always-on shared expert through the
        // exact-M=4 GEMV, which needs its decode-native layout. Routed experts
        // remain transposed-only to avoid hybrid layout's memory cost.
        layer.transpose_for_prefill_unified_keep_shared(gpu, config)?;
    }
    layer.maybe_cache_glm_target_shared_fp8(
        store,
        &shared,
        config,
        layer_idx,
        allow_prefill_layout,
        variant,
        gpu,
        qctx.stream,
    )?;
    Ok(FfnComponent::Moe(layer))
}

fn unified_moe_layout_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

pub(super) fn load_kda_weights(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    qctx: QuantizeCtx,
) -> Result<Glm5KdaWeights> {
    let p = format!("{lp}.self_attn");
    let load = |name: &str| dense_auto(store, &format!("{p}.{name}.weight"), gpu);
    let dims = TpGdnDims::from_config(config);
    let p_dim = dims.local_key_dim();
    let load_tp_dense =
        |name: &str, n: usize, k: usize, kind: TpShardKind| -> Result<DenseWeight> {
            let source = load(name)?;
            let (local, _, _) =
                shard_dense_bf16(source.weight, n, k, kind, dims.tp_rank, dims.tp_size, gpu)?;
            if local != source.weight {
                gpu.free(source.weight)?;
            }
            Ok(DenseWeight { weight: local })
        };
    let load_hot = |name: &str,
                    n: usize,
                    k: usize,
                    kind: TpShardKind,
                    transpose_prefill: bool|
     -> Result<Glm5Projection> {
        let dense = load_tp_dense(name, n, k, kind)?;
        let local_n = if kind == TpShardKind::ColumnParallel {
            n / dims.tp_size
        } else {
            n
        };
        let local_k = if kind == TpShardKind::RowParallel {
            k / dims.tp_size
        } else {
            k
        };
        let decode_nvfp4 = quantize_to_nvfp4(
            &dense,
            local_n,
            local_k,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        )?;
        // Quantization synchronizes internally, so the checkpoint-native BF16
        // source is no longer needed. Releasing it keeps the safe 92% memory
        // budget viable on GB10 instead of retaining two projection copies.
        gpu.free(dense.weight)?;
        let prefill_nvfp4_t = transpose_prefill
            .then(|| decode_nvfp4.transpose_for_gemm(gpu, local_n, local_k))
            .transpose()?;
        Ok(Glm5Projection {
            nvfp4: decode_nvfp4,
            prefill_nvfp4_t,
        })
    };
    let width = config.linear_conv_kernel_dim;
    let local_conv_bytes = p_dim * width * 2;
    let conv_ptr = gpu.alloc(3 * local_conv_bytes)?;
    for (slot, name) in ["q_conv1d", "k_conv1d", "v_conv1d"].iter().enumerate() {
        let weight = load_tp_dense(
            name,
            dims.full_key_dim(),
            width,
            TpShardKind::ColumnParallel,
        )?;
        gpu.copy_d2d(
            weight.weight,
            conv_ptr.offset(slot * local_conv_bytes),
            local_conv_bytes,
        )?;
        gpu.free(weight.weight)?;
    }
    let a_log_full = dense_keep_f32(store, &format!("{p}.A_log"), gpu)?;
    let (a_log, _) = shard_gdn_value_vector(a_log_full.weight, &dims, 1, 4, gpu)?;
    if a_log != a_log_full.weight {
        gpu.free(a_log_full.weight)?;
    }
    let dt_bias_full = dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?;
    let (dt_bias, _) = shard_gdn_value_vector(dt_bias_full.weight, &dims, dims.kd, 4, gpu)?;
    if dt_bias != dt_bias_full.weight {
        gpu.free(dt_bias_full.weight)?;
    }
    Ok(Glm5KdaWeights {
        q_proj: load_hot(
            "q_proj",
            dims.full_key_dim(),
            dims.h,
            TpShardKind::ColumnParallel,
            true,
        )?,
        k_proj: load_hot(
            "k_proj",
            dims.full_key_dim(),
            dims.h,
            TpShardKind::ColumnParallel,
            true,
        )?,
        v_proj: load_hot(
            "v_proj",
            dims.full_value_dim(),
            dims.h,
            TpShardKind::ColumnParallel,
            true,
        )?,
        b_proj: load_tp_dense("b_proj", dims.full_nk, dims.h, TpShardKind::ColumnParallel)?,
        f_a_proj: load("f_a_proj")?,
        f_b_proj: load_tp_dense(
            "f_b_proj",
            dims.full_key_dim(),
            dims.kd,
            TpShardKind::ColumnParallel,
        )?,
        g_a_proj: load("g_a_proj")?,
        g_b_proj: load_tp_dense(
            "g_b_proj",
            dims.full_key_dim(),
            dims.kd,
            TpShardKind::ColumnParallel,
        )?,
        conv: DenseWeight { weight: conv_ptr },
        a_log: DenseWeight { weight: a_log },
        dt_bias: DenseWeight { weight: dt_bias },
        o_norm: load("o_norm")?,
        o_proj: load_hot(
            "o_proj",
            dims.h,
            dims.full_value_dim(),
            TpShardKind::RowParallel,
            true,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::unified_moe_layout_enabled;

    #[test]
    fn unified_moe_layout_is_explicitly_opt_in() {
        assert!(unified_moe_layout_enabled(Some("1")));
        assert!(unified_moe_layout_enabled(Some("TRUE")));
        assert!(!unified_moe_layout_enabled(None));
        assert!(!unified_moe_layout_enabled(Some("0")));
        assert!(!unified_moe_layout_enabled(Some("full")));
    }
}
