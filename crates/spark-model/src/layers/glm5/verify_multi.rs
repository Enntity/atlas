// SPDX-License-Identifier: AGPL-3.0-only

//! One GLM target weight sweep across multiple DFlash blocks.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::decode::dsa_verify_pool_bucket;
use super::types::{Glm5Layer, GlmAttentionWeights};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

impl Glm5Layer {
    pub(super) fn forward_verify_rows_multi(
        &self,
        hidden: DevicePtr,
        rows_per_seq: usize,
        seq_lens: &[usize],
        states: &mut [&mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let num_sequences = states.len();
        let rows = num_sequences.saturating_mul(rows_per_seq);
        ensure!(
            num_sequences > 0 && seq_lens.len() == num_sequences,
            "GLM multi-verify state/length mismatch"
        );
        ensure!(
            rows_per_seq > 0 && rows <= spark_runtime::buffers::GLM53_VERIFY_MAX_BATCH_ROWS,
            "GLM multi-verify shape {num_sequences}x{rows_per_seq} is unsupported"
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
        let attention_output = match &self.attention {
            GlmAttentionWeights::Kda(weights) => {
                self.kda_forward_verify_multi(weights, normed, rows_per_seq, states, ctx, stream)?
            }
            GlmAttentionWeights::Dsa(weights) => {
                let pool_limits = seq_lens
                    .iter()
                    .map(|&seq_len| {
                        dsa_verify_pool_bucket(
                            seq_len.saturating_add(rows_per_seq),
                            ctx.config.index_kpool,
                            ctx.buffers.glm_layout().dsa_max_pools,
                        )
                    })
                    .collect::<Vec<_>>();
                self.dsa_forward_verify_multi(
                    weights,
                    normed,
                    rows_per_seq,
                    &pool_limits,
                    states,
                    ctx,
                    stream,
                )?
            }
        };
        self.tp_sum(attention_output, rows, ctx, stream)?;
        self.hc_post(attention_output, rows, ctx, stream)?;

        let attention_ms = if let Some(started) = attention_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
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
                "GLM_VERIFY_MULTI_PROFILE layer={} sequences={} rows_per_seq={} attention={} attention_ms={:.3} ffn={} ffn_ms={:.3}",
                self.layer_idx,
                num_sequences,
                rows_per_seq,
                attention_kind,
                attention_ms,
                ffn_kind,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(())
    }
}
