// SPDX-License-Identifier: AGPL-3.0-only

//! Native ragged multi-sequence GLM prefill.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::decode::dsa_pool_limit;
use super::types::{Glm5Layer, GlmAttentionWeights};
use crate::layer::{BatchedAttnMetadata, ForwardContext, LayerState};
use crate::layers::ops;

impl Glm5Layer {
    pub(super) fn prefill_rows_batched(
        &self,
        hidden: DevicePtr,
        states: &[&(dyn LayerState + '_)],
        meta: &BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rows = meta.total_tokens as usize;
        ensure!(
            rows > 0 && rows <= ctx.buffers.glm_layout().max_batch_tokens,
            "GLM batched prefill rows exceed the workspace envelope"
        );
        ensure!(
            states.len() == meta.batch_size as usize
                && meta.cu_seqlens_host.len() == states.len() + 1,
            "GLM batched prefill state/metadata count mismatch"
        );

        let workspace = ctx.buffers.glm_workspace();
        let layout = ctx.buffers.glm_layout();
        let cu_i64 = workspace.offset(layout.cu_seqlens_i64);
        // All 45 layers reuse this immutable table. Upload it once before
        // layer 0's KDA kernel and retain stream ordering for later layers.
        if self.layer_idx == 0 {
            let bytes = meta
                .cu_seqlens_host
                .iter()
                .flat_map(|value| i64::from(*value).to_le_bytes())
                .collect::<Vec<_>>();
            ctx.gpu.copy_h2d_async(&bytes, cu_i64, stream)?;
        }

        let hidden_size = ctx.config.hidden_size;
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
                hidden_size as u32,
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
                self.kda_prefill(weights, normed, rows, states, cu_i64, ctx, stream)?
            }
            GlmAttentionWeights::Dsa(weights) => {
                let max_visible = meta
                    .kv_lens_host
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(rows as i32)
                    .max(1) as usize;
                let pool_limit =
                    dsa_pool_limit(max_visible, ctx.config.index_kpool, layout.dsa_max_pools);
                self.dsa_prefill(
                    weights,
                    normed,
                    rows,
                    pool_limit,
                    states,
                    meta.cu_seqlens,
                    meta.positions_stacked,
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
                "GLM_PREFILL_PROFILE layer={} rows={} sequences={} attention={} attention_ms={:.3} ffn={} ffn_ms={:.3}",
                self.layer_idx,
                rows,
                states.len(),
                attention_kind,
                attention_us as f64 / 1000.0,
                ffn_kind,
                started.elapsed().as_micros() as f64 / 1000.0,
            );
        }
        Ok(())
    }
}
