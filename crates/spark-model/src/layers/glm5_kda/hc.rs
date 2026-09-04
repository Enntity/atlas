// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 mHC dispatch, including the prefill-only batched pre-mix GEMM.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::Glm5KdaLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::HcSiteWeights;

fn parse_fast_prefill(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true" | "yes"))
}

fn fast_prefill(tokens: u32) -> bool {
    (tokens >= 128 && parse_fast_prefill(std::env::var("ATLAS_HC_CUBLAS_PREFILL").ok().as_deref()))
        || (tokens == 5
            && parse_fast_prefill(std::env::var("ATLAS_GLM_K5_HC_CUBLAS").ok().as_deref()))
}

impl Glm5KdaLayer {
    pub(super) fn hc_pre(
        &self,
        site: &HcSiteWeights,
        hidden: DevicePtr,
        tokens: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if fast_prefill(tokens) {
            let hc = self.hc.hc_mult as u32;
            let mix = (2 + hc) * hc;
            let raw_mix = ctx.buffers.gate_logits_f32();
            ensure!(
                ctx.buffers.sizes().gate_logits_f32 >= tokens as usize * mix as usize * 4,
                "mHC TF32 pre-mix scratch is too small"
            );
            spark_runtime::cublaslt::tf32_gemm_act_weight_t(
                ctx.buffers.hc_streams().0,
                site.hc_fn.0,
                raw_mix.0,
                tokens,
                mix,
                hc * self.hidden_size as u32,
                stream,
            )?;
            return ops::hc_pre_from_raw_mix(
                ctx.gpu,
                self.hc_pre_from_raw_mix_k,
                ctx.buffers.hc_streams(),
                raw_mix,
                site.hc_scale,
                site.hc_base,
                hidden,
                ctx.buffers.hc_post(),
                ctx.buffers.hc_comb(),
                tokens,
                self.hidden_size as u32,
                hc,
                self.hc.sinkhorn_iters as u32,
                ctx.config.rms_norm_eps as f32,
                self.hc.hc_eps,
                stream,
            );
        }
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            ctx.buffers.hc_streams(),
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            hidden,
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            tokens,
            self.hidden_size as u32,
            self.hc.hc_mult as u32,
            self.hc.sinkhorn_iters as u32,
            ctx.config.rms_norm_eps as f32,
            self.hc.hc_eps,
            stream,
        )
    }

    pub(super) fn hc_post(
        &self,
        block_out: DevicePtr,
        tokens: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            block_out,
            ctx.buffers.hc_streams(),
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            ctx.buffers.hc_streams(),
            tokens,
            self.hidden_size as u32,
            self.hc.hc_mult as u32,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::parse_fast_prefill;

    #[test]
    fn fast_prefill_is_explicitly_opt_in() {
        assert!(parse_fast_prefill(Some("1")));
        assert!(parse_fast_prefill(Some("true")));
        assert!(!parse_fast_prefill(None));
        assert!(!parse_fast_prefill(Some("0")));
    }
}
