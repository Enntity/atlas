// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill mHC pre-mix dispatch for full-attention GLM-5 layers.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{HcSiteWeights, HcWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;

fn fast_prefill(tokens: u32) -> bool {
    tokens >= 128
        && matches!(
            std::env::var("ATLAS_HC_CUBLAS_PREFILL").ok().as_deref(),
            Some("1" | "true" | "yes")
        )
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_pre_prefill(
        &self,
        site: &HcSiteWeights,
        hc: &HcWeights,
        hidden: DevicePtr,
        tokens: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let streams = ctx.buffers.hc_streams();
        let h = ctx.config.hidden_size as u32;
        let hc_mult = hc.hc_mult as u32;
        if fast_prefill(tokens) {
            let mix = (2 + hc_mult) * hc_mult;
            let raw_mix = ctx.buffers.gate_logits_f32();
            ensure!(
                ctx.buffers.sizes().gate_logits_f32 >= tokens as usize * mix as usize * 4,
                "mHC TF32 pre-mix scratch is too small"
            );
            spark_runtime::cublaslt::tf32_gemm_act_weight_t(
                streams.0,
                site.hc_fn.0,
                raw_mix.0,
                tokens,
                mix,
                hc_mult * h,
                stream,
            )?;
            return ops::hc_pre_from_raw_mix(
                ctx.gpu,
                self.hc_pre_from_raw_mix_k,
                streams,
                raw_mix,
                site.hc_scale,
                site.hc_base,
                hidden,
                ctx.buffers.hc_post(),
                ctx.buffers.hc_comb(),
                tokens,
                h,
                hc_mult,
                hc.sinkhorn_iters as u32,
                ctx.config.rms_norm_eps as f32,
                hc.hc_eps,
                stream,
            );
        }
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            hidden,
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            tokens,
            h,
            hc_mult,
            hc.sinkhorn_iters as u32,
            ctx.config.rms_norm_eps as f32,
            hc.hc_eps,
            stream,
        )
    }
}
