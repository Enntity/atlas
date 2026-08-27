// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights};
use crate::layers::{DenseFfnLayer, FfnComponent, Glm5KdaWeights, MoeLayer};
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
) -> Result<FfnComponent> {
    if config.mlp_only_layers.contains(&layer_idx) {
        return load_dense_ffn(store, lp, config, gpu, qctx);
    }
    load_moe(store, lp, config, gpu, variant, qctx)
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
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
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
    Ok(FfnComponent::Moe(MoeLayer::new(
        weights,
        config.num_experts,
        None,
        gpu,
        config,
    )?))
}

pub(super) fn load_kda_weights(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Glm5KdaWeights> {
    let p = format!("{lp}.self_attn");
    let load = |name: &str| dense_auto(store, &format!("{p}.{name}.weight"), gpu);
    let p_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let width = config.linear_conv_kernel_dim;
    let conv_bytes = p_dim * width * 2;
    let conv_ptr = gpu.alloc(3 * conv_bytes)?;
    for (slot, name) in ["q_conv1d", "k_conv1d", "v_conv1d"].iter().enumerate() {
        let weight = load(name)?;
        gpu.copy_d2d(
            weight.weight,
            conv_ptr.offset(slot * conv_bytes),
            conv_bytes,
        )?;
    }
    Ok(Glm5KdaWeights {
        q_proj: load("q_proj")?,
        k_proj: load("k_proj")?,
        v_proj: load("v_proj")?,
        b_proj: load("b_proj")?,
        f_a_proj: load("f_a_proj")?,
        f_b_proj: load("f_b_proj")?,
        g_a_proj: load("g_a_proj")?,
        g_b_proj: load("g_b_proj")?,
        conv: DenseWeight { weight: conv_ptr },
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        o_norm: load("o_norm")?,
        o_proj: load("o_proj")?,
    })
}
