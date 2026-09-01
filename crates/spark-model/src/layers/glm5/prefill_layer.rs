// SPDX-License-Identifier: AGPL-3.0-only

//! Shared prefill metadata and layer assembly.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::decode::dsa_pool_limit;
use super::types::{Glm5Layer, GlmAttentionWeights};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

impl Glm5Layer {
    pub(super) fn upload_prefill_lengths(
        &self,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let base = ctx.buffers.glm_workspace();
        let layout = ctx.buffers.glm_layout();
        let i32_ptr = base.offset(layout.cu_seqlens_i32);
        let i64_ptr = base.offset(layout.cu_seqlens_i64);
        let i32_bytes = [0_i32, rows as i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let i64_bytes = [0_i64, rows as i64]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>();
        ctx.gpu.copy_h2d_async(&i32_bytes, i32_ptr, stream)?;
        ctx.gpu.copy_h2d_async(&i64_bytes, i64_ptr, stream)?;
        Ok((i32_ptr, i64_ptr))
    }

    pub(super) fn prefill_rows(
        &self,
        hidden: DevicePtr,
        rows: usize,
        seq_len_start: usize,
        state: &dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= ctx.buffers.glm_layout().max_batch_tokens,
            "GLM prefill row count is outside the workspace envelope"
        );
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
        let attention = match &self.attention {
            GlmAttentionWeights::Kda(weights) => {
                let (_, cu_i64) = self.upload_prefill_lengths(rows, ctx, stream)?;
                self.kda_prefill(weights, normed, rows, &[state], cu_i64, ctx, stream)?
            }
            GlmAttentionWeights::Dsa(weights) => {
                let layout = ctx.buffers.glm_layout();
                let pool_limit = dsa_pool_limit(
                    seq_len_start + rows,
                    ctx.config.index_kpool,
                    layout.dsa_max_pools,
                );
                let metadata = ctx
                    .attn_metadata
                    .context("GLM DSA prefill requires positions")?;
                let (cu_i32, _) = self.upload_prefill_lengths(rows, ctx, stream)?;
                self.dsa_prefill(
                    weights,
                    normed,
                    rows,
                    pool_limit,
                    &[state],
                    cu_i32,
                    metadata.positions,
                    ctx,
                    stream,
                )?
            }
        };
        self.tp_sum(attention, rows, ctx, stream)?;
        self.hc_post(attention, rows, ctx, stream)?;
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
        let ffn = self.ffn_forward(normed, rows, ctx, stream)?;
        self.hc_post(ffn, rows, ctx, stream)?;
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
                "GLM_PREFILL_PROFILE layer={} rows={} sequences=1 attention={} attention_ms={:.3} ffn={} ffn_ms={:.3}",
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
