// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights};
use crate::layers::{DenseFfnLayer, FfnComponent, Glm5KdaWeights, Glm5Projection, MoeLayer};
use crate::tp_shard::{TpGdnDims, TpShardKind, shard_dense_bf16, shard_gdn_value_vector};
use crate::weight_map::{
    DenseWeight, ExpertWeight, MoeWeights, Nvfp4Variant, QuantizeCtx, QuantizedWeight, dense_auto,
    quantize_to_nvfp4_scaled_by, quantized_any,
};

pub(super) fn load_hc(
    store: &WeightStore,
    lp: &str,
    layer_idx: usize,
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
        // The checkpoint's BF16 `fn` itself (`load_hc_f32` widens it exactly
        // into `hc_fn`): a store pointer, nothing allocated.
        let fn_bf16 = store
            .get(&format!("{lp}.hc_{site}_fn"))
            .ok()
            .filter(|t| t.dtype == WeightDtype::BF16 && t.num_elements() == mix * hc_dim)
            .map_or(DevicePtr::NULL, |t| t.ptr);
        Ok(HcSiteWeights {
            hc_fn: load("fn", mix * hc_dim)?,
            hc_fn_bf16: fn_bf16,
            hc_base: load("base", mix)?,
            hc_scale: load("scale", 3)?,
            lowrank: None,
        })
    };
    Ok(HcWeights {
        attn: load_site("attn")?,
        ffn: load_site("ffn")?,
        head: None,
        hc_mult: config.hc_mult,
        sinkhorn_iters: config.hc_sinkhorn_iters,
        hc_eps: config.hc_eps,
        is_first_model_layer: layer_idx == 0,
        is_last_model_layer: layer_idx + 1 == config.num_hidden_layers,
    })
}

/// The layer's FFN; `host_experts`: its expert-TP slices when already read
/// ([`super::expert_tp::read_layer_experts`]), else they are read here.
#[allow(clippy::too_many_arguments)]
#[path = "components_kda.rs"]
mod kda;
pub(super) use kda::load_kda_weights;

pub(super) fn load_ffn(
    store: &WeightStore,
    lp: &str,
    layer_idx: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    allow_prefill_layout: bool,
    host_experts: Option<super::expert_tp::HostExperts>,
) -> Result<FfnComponent> {
    if config.mlp_only_layers.contains(&layer_idx) {
        return load_dense_ffn(store, lp, config, gpu, variant, qctx);
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
        host_experts,
    )
}

fn load_dense_ffn(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<FfnComponent> {
    let p = format!("{lp}.mlp");
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    // ATLAS_GLM_FP4L_DENSE_MXFP8: the checkpoint's MXFP8 projections and no
    // NVFP4 copy at all (null placeholders).
    if let Some(native) = super::dense::native_dense(
        crate::layers::glm_native::NativePolicy::from_env(),
        store,
        &p,
        [h, inter],
        gpu,
    )? {
        let weights = DenseFfnWeights {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        };
        let installed = DenseFfnLayer::new(weights, gpu).and_then(|mut layer| {
            layer.install_native(native)?;
            Ok(layer)
        });
        return match installed {
            Ok(layer) => Ok(FfnComponent::Dense(layer)),
            Err(e) => {
                native.free(gpu)?;
                Err(e)
            }
        };
    }
    let weights = DenseFfnWeights {
        gate_proj: super::dense::load_projection(
            store,
            &format!("{p}.gate_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        up_proj: super::dense::load_projection(
            store,
            &format!("{p}.up_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        down_proj: super::dense::load_projection(
            store,
            &format!("{p}.down_proj"),
            h,
            inter,
            gpu,
            variant,
            qctx,
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
    _layer_idx: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    allow_prefill_layout: bool,
    host_experts: Option<super::expert_tp::HostExperts>,
) -> Result<FfnComponent> {
    let p = format!("{lp}.mlp");
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    anyhow::ensure!(
        !config.expert_tp || matches!(variant, Nvfp4Variant::Standard),
        "expert TP slices ModelOpt NVFP4 experts only, got {variant:?}"
    );
    let experts = if config.expert_tp {
        let host = match host_experts {
            Some(host) => host,
            None => super::expert_tp::read_layer_experts(store, &p, config)?,
        };
        super::expert_tp::upload_experts(store, &p, host, gpu)?
    } else {
        (0..config.num_experts)
            .map(|expert| -> Result<ExpertWeight> {
                if !config.is_local_expert(expert) {
                    return Ok(ExpertWeight::null());
                }
                let ep = format!("{p}.experts.{expert}");
                Ok(ExpertWeight {
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
                })
            })
            .collect::<Result<Vec<_>>>()?
    };
    let shared = format!("{p}.shared_experts");
    let (shared_gate, gate_origin) = crate::layers::moe::load_glm_shared_fp8_weight(
        store,
        &format!("{shared}.gate_proj"),
        config.shared_expert_intermediate_size,
        h,
        gpu,
        variant,
        qctx,
        allow_prefill_layout,
    )?;
    let (shared_up, up_origin) = crate::layers::moe::load_glm_shared_fp8_weight(
        store,
        &format!("{shared}.up_proj"),
        config.shared_expert_intermediate_size,
        h,
        gpu,
        variant,
        qctx,
        allow_prefill_layout,
    )?;
    let (shared_down, down_origin) = crate::layers::moe::load_glm_shared_fp8_weight(
        store,
        &format!("{shared}.down_proj"),
        h,
        config.shared_expert_intermediate_size,
        gpu,
        variant,
        qctx,
        allow_prefill_layout,
    )?;
    let shared_expert = ExpertWeight {
        gate_proj: shared_gate,
        up_proj: shared_up,
        down_proj: shared_down,
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
    layer.set_shared_fp8_origins([gate_origin, up_origin, down_origin])?;
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
        layer.transpose_shared_only(gpu, config)?;
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
    // ATLAS_GLM_FP4L_SHARED_FP8: the checkpoint's FP8 block shared expert
    // replaces the NVFP4 copies just built (and the deferred FP8 cache).
    if let Some(native) = super::shared_native::shared_native_twins(
        crate::layers::glm_native::NativePolicy::from_env(),
        store,
        &shared,
        config,
        gpu,
    )? && let Err(e) = prepare_shared_prefill(&native, config, gpu)
        .and_then(|()| layer.install_shared_native(gpu, native))
    {
        native.free(gpu)?;
        return Err(e);
    }
    // The factory installs optional shared FP8 caches only after all target,
    // MTP and head loading has released replaced checkpoint allocations.
    Ok(FfnComponent::Moe(layer))
}

/// Pick the native shared expert's W8A8 prefill GEMMs at load (cuBLASLt block
/// FP8, else the in-tree kernel): fail here, never mid-prefill.
fn prepare_shared_prefill(
    native: &crate::layers::moe::shared_native::SharedNative,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    if native.prefill != crate::layers::ops::Fp8bPrefill::W8a8 {
        return Ok(());
    }
    let (h, inter) = (config.hidden_size, config.shared_expert_intermediate_size);
    for (w, n, k) in [
        (&native.gate, inter, h),
        (&native.up, inter, h),
        (&native.down, h, inter),
    ] {
        crate::layers::ops::fp8b_w8a8_prepare(gpu, w, n, k)?;
    }
    Ok(())
}

fn unified_moe_layout_enabled(value: Option<&str>) -> bool {
    value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

pub(super) fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

#[cfg(test)]
#[path = "components_local_tests.rs"]
mod tests;
