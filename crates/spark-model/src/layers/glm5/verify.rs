// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-width GLM target verification through one layer weight sweep.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::decode::dsa_verify_pool_bucket;
use super::types::{Glm5Layer, GlmAttentionWeights};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

impl Glm5Layer {
    pub(super) fn forward_verify_rows(
        &self,
        hidden: DevicePtr,
        rows: usize,
        seq_len: usize,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(rows > 0, "GLM verification requires target rows");
        let h = ctx.config.hidden_size;
        if ctx.profile {
            ctx.gpu.synchronize(stream)?;
        }
        let attention_started = ctx.profile.then(std::time::Instant::now);
        if self.layer_idx == 0 {
            ops::hc_expand(
                ctx.gpu,
                self.kernels.hc_expand,
                hidden,
                ctx.buffers.hc_streams(),
                rows as u32,
                h as u32,
                ctx.config.hc_mult as u32,
                stream,
            )?;
        }
        let normed = ctx.buffers.norm_output();
        self.hc_pre_norm(
            &self.hc_attention,
            &self.input_norm,
            hidden,
            normed,
            rows,
            ctx,
            stream,
        )?;
        let attention_output = match &self.attention {
            GlmAttentionWeights::Kda(weights) => {
                self.kda_forward_verify(weights, normed, rows, state, ctx, stream)?
            }
            GlmAttentionWeights::Dsa(weights) => {
                let pool_limit = dsa_verify_pool_bucket(
                    seq_len.saturating_add(rows),
                    ctx.config.index_kpool,
                    ctx.buffers.glm_layout().dsa_max_pools,
                );
                self.dsa_forward_verify(weights, normed, rows, pool_limit, state, ctx, stream)?
            }
        };
        self.tp_sum(attention_output, rows, ctx, stream)?;
        self.hc_post(attention_output, rows, ctx, stream)?;

        let attention_us = if let Some(started) = attention_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_micros() as u64
        } else {
            0
        };
        let ffn_started = ctx.profile.then(std::time::Instant::now);

        self.hc_pre_norm(
            &self.hc_ffn,
            &self.post_attention_norm,
            hidden,
            normed,
            rows,
            ctx,
            stream,
        )?;
        let ffn_output = self.ffn_forward(normed, rows, ctx, stream)?;
        self.hc_post(ffn_output, rows, ctx, stream)?;
        self.contract_hc_for_output_or_dflash(hidden, rows, ctx, stream)?;
        if let Some(started) = ffn_started {
            ctx.gpu.synchronize(stream)?;
            let attention_kind = match &self.attention {
                GlmAttentionWeights::Kda(_) => "kda",
                GlmAttentionWeights::Dsa(_) => "dsa",
            };
            let ffn_kind = match &self.ffn {
                super::types::GlmFfn::Dense(_) => "dense",
                super::types::GlmFfn::Exl3(_) => "exl3",
            };
            tracing::info!(
                "GLM_VERIFY_PROFILE layer={} rows={} attention={} attention_ms={:.3} ffn={} ffn_ms={:.3}",
                self.layer_idx,
                rows,
                attention_kind,
                attention_us as f64 / 1000.0,
                ffn_kind,
                started.elapsed().as_micros() as f64 / 1000.0,
            );
        }
        Ok(())
    }
}
