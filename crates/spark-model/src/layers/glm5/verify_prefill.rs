// SPDX-License-Identifier: AGPL-3.0-only

//! Heterogeneous GLM target-verification + prompt-prefill layer pass.
//!
//! Attention is stateful and therefore remains on each lane's native causal
//! kernel.  The FFN is stateless across rows, so both lanes share one routed
//! expert sweep after their attention results have been folded back into the
//! appropriate mHC highway rows.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::decode::{dsa_pool_limit, dsa_verify_pool_bucket};
use super::types::{Glm5Layer, GlmAttentionWeights};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

impl Glm5Layer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_verify_with_prefill(
        &self,
        hidden: DevicePtr,
        verify_rows: usize,
        verify_seq_len: usize,
        verify_state: &mut (dyn LayerState + 'static),
        prefill_rows: usize,
        prefill_seq_len: usize,
        prefill_state: &mut (dyn LayerState + 'static),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rows = verify_rows.saturating_add(prefill_rows);
        ensure!(
            (2..=spark_runtime::buffers::GLM53_VERIFY_MAX_ROWS).contains(&verify_rows),
            "GLM heterogeneous verify width {verify_rows} is unsupported"
        );
        ensure!(
            prefill_rows > 0 && rows <= ctx.buffers.max_batch_tokens(),
            "GLM heterogeneous shape verify={verify_rows} prefill={prefill_rows} exceeds the arena"
        );
        let h = ctx.config.hidden_size;
        let row_bytes = h * size_of::<u16>();
        let normed = ctx.buffers.norm_output();

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

        match &self.attention {
            GlmAttentionWeights::Kda(weights) => {
                // KDA's projections and output collective are stateless over
                // rows. Share them across both lanes; only the recurrence and
                // rollback kernels remain lane-specific.
                self.hc_pre_norm(
                    &self.hc_attention,
                    &self.input_norm,
                    hidden,
                    normed,
                    rows,
                    ctx,
                    stream,
                )?;
                let output = self.kda_forward_verify_with_prefill(
                    weights,
                    normed,
                    verify_rows,
                    verify_state,
                    prefill_rows,
                    prefill_state,
                    ctx,
                    stream,
                )?;
                self.tp_sum(output, rows, ctx, stream)?;
                self.hc_post(output, rows, ctx, stream)?;
            }
            GlmAttentionWeights::Dsa(weights) => {
                // DSA retains its independently stateful verifier/prefill
                // kernels for now. Their projections are shared in the next
                // optimization slice; this branch preserves the proven path.
                self.hc_pre_norm_at(
                    &self.hc_attention,
                    &self.input_norm,
                    hidden,
                    normed,
                    0,
                    verify_rows,
                    ctx,
                    stream,
                )?;
                let pool_limit = dsa_verify_pool_bucket(
                    verify_seq_len.saturating_add(verify_rows),
                    ctx.config.index_kpool,
                    ctx.buffers.glm_layout().dsa_max_pools,
                );
                let verify_output = self.dsa_forward_verify(
                    weights,
                    normed,
                    verify_rows,
                    pool_limit,
                    verify_state,
                    ctx,
                    stream,
                )?;
                self.tp_sum(verify_output, verify_rows, ctx, stream)?;
                self.hc_post_at(verify_output, 0, verify_rows, ctx, stream)?;

                let prefill_hidden = hidden.offset(verify_rows * row_bytes);
                self.hc_pre_norm_at(
                    &self.hc_attention,
                    &self.input_norm,
                    prefill_hidden,
                    normed,
                    verify_rows,
                    prefill_rows,
                    ctx,
                    stream,
                )?;
                let metadata = ctx
                    .attn_metadata
                    .context("GLM heterogeneous prefill requires positions")?;
                let (cu_i32, _) = self.upload_prefill_lengths(prefill_rows, ctx, stream)?;
                let pool_limit = dsa_pool_limit(
                    prefill_seq_len.saturating_add(prefill_rows),
                    ctx.config.index_kpool,
                    ctx.buffers.glm_layout().dsa_max_pools,
                );
                let states = [&*prefill_state as &(dyn LayerState + '_)];
                let prefill_output = self.dsa_prefill(
                    weights,
                    normed,
                    prefill_rows,
                    pool_limit,
                    &states,
                    cu_i32,
                    metadata.positions.offset(verify_rows * size_of::<u32>()),
                    ctx,
                    stream,
                )?;
                self.tp_sum(prefill_output, prefill_rows, ctx, stream)?;
                self.hc_post_at(prefill_output, verify_rows, prefill_rows, ctx, stream)?;
            }
        }

        let attention_ms = if let Some(started) = attention_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let ffn_started = ctx.profile.then(std::time::Instant::now);

        // Stateless boundary: all rows now share one normalization, router,
        // expert-weight sweep and TP reduction.
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
            tracing::info!(
                "GLM_VERIFY_PREFILL_PROFILE layer={} verify_rows={} prefill_rows={} attention_ms={:.3} ffn_ms={:.3}",
                self.layer_idx,
                verify_rows,
                prefill_rows,
                attention_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(())
    }
}
