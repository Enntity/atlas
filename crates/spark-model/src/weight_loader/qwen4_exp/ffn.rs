// SPDX-License-Identifier: AGPL-3.0-only

//! The per-layer MoE block: 512 routed experts, top-10, plus a shared expert
//! and its sigmoid gate.
//!
//! This is `load_moe_qwen35` verbatim. Both the naming
//! (`mlp.gate`, `mlp.shared_expert.{gate,up,down}_proj`,
//! `mlp.shared_expert_gate`, `mlp.experts.{e}.{gate,up,down}_proj`) and the
//! on-disk quantization (standard ModelOpt NVFP4: packed E2M1 `weight`,
//! per-16 E4M3 `weight_scale`, per-tensor F32 `weight_scale_2`) are identical
//! to Qwen3.5/3.6 MoE. The expert COUNT and widths differ — 512 x 640 here
//! against 256 x 512 there — but those come from `config`, not from the
//! loader, so nothing needs re-deriving.
//!
//! The router (`mlp.gate`) stays the checkpoint's BF16 (`gate_nvfp4: None`),
//! as every release ships it and NVIDIA's reference engine runs it. Its
//! precision matters more than its size: at 512 experts the top-10 weights
//! cluster tightly, and an NVFP4 copy (8.5-9.3% weight error on FP4-Local)
//! changes the top-10 expert set of most tokens. Decode and verify read it
//! through `dense_gemv_bf16` and its row-identical batched tiers, prefill
//! through the q38 tile GEMM (`try_q38_router`).
//!
//! FP4-Local (`mxfp8_backbone`) ships the shared expert MXFP8: it loads as
//! the checkpoint's own bytes (`layers::moe::Mxfp8SharedExpert`) and the
//! NVFP4 shared expert the BF16 releases get is never built.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::moe::Mxfp8SharedExpert;
use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{Nvfp4Variant, load_moe_qwen35};

pub(super) fn build_moe(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    mxfp8_backbone: bool,
) -> Result<FfnComponent> {
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let inter = config.shared_expert_intermediate_size;
    let native_shared = mxfp8_shared(store, lp, inter, h)?;

    let weights = load_moe_qwen35(
        store,
        lp,
        config.num_experts,
        gpu,
        config,
        variant,
        absmax_k,
        quantize_k,
        stream,
        false,
        !mxfp8_backbone,
    )
    .with_context(|| format!("qwen4_exp: MoE block at {lp}"))?;

    let mut moe = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    // The routed prefill's W4A4 activation scales (`moe::w4a4_scales`).
    moe.load_w4a4_scales(store, &format!("{lp}.mlp"), config, gpu)?;
    if mxfp8_backbone {
        let shared = Mxfp8SharedExpert::load(store, lp, config, gpu)
            .with_context(|| format!("qwen4_exp: MXFP8 shared expert at {lp}"))?;
        moe.set_mxfp8_shared_expert(shared)?;
    }
    Ok(FfnComponent::Moe(moe))
}
