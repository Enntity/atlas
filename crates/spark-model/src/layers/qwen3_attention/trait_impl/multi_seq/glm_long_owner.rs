// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched repaired long-context K3 verify for an MLA layer.
//!
//! Attention runs per owner, unchanged: each owner's three rows are placed at
//! arena rows [0, 3) and go through the same validated eager K3 path (causal
//! semantic index, sparse attention, cache writes) with that owner's own
//! metadata and state. Only the FFN is joint, over every owner's rows.

use super::*;
use crate::layer::glm_long_owner::{GlmLongOwner, GlmLongStage, ROWS};
use anyhow::{Context, ensure};

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention) fn decode_glm_long_owners_mla(
        &self,
        owners: &mut [GlmLongOwner<'_>],
        kv_cache: &mut PagedKvCache,
        stage: &GlmLongStage,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.mla.is_some() && self.hc.is_some() && !self.ffn.is_none(),
            "GLM long owner batch requires an mHC MLA layer with FFN"
        );
        ensure!(
            self.block_idx != 0 && self.block_idx + 1 != ctx.config.num_hidden_layers,
            "GLM long owner batch MLA cannot be the first or last layer"
        );
        let rows = owners.len() * ROWS;
        let b = ctx.buffers;
        let r = stage.rows;
        let ffn_spans = [
            (b.hc_streams(), stage.highway, r.highway),
            (b.hc_post(), stage.post, r.post),
            (b.hc_comb(), stage.comb, r.comb),
            (b.norm_output(), stage.norm, r.hidden),
        ];
        // Every owner's incoming highway, before owner 0's attention reuses rows [0, 3).
        stage.copy(ctx.gpu, &ffn_spans[..1], 0, 0, rows, true, stream)?;
        let row_owner = [0usize; ROWS];
        let bs = kv_cache.block_size() as u32;
        for (owner, input) in owners.iter_mut().enumerate() {
            let first = owner * ROWS;
            stage.copy(ctx.gpu, &ffn_spans[..1], 0, first, ROWS, false, stream)?;
            let owner_ctx = ForwardContext {
                attn_metadata: Some(input.meta),
                midchunk_capture: None,
                ..*ctx
            };
            let mut c = ctx::MultiSeqCtx::new(
                self,
                &owner_ctx,
                b.hidden_states(),
                b.residual(),
                ROWS,
                &input.positions,
                bs,
                stream,
            );
            c.seq_slot = input.meta.seq_slot;
            let mut states: [&mut (dyn LayerState + 'static); 1] = [&mut *input.state];
            self.ms_hc_attention_norm_impl(
                &c,
                kv_cache,
                &owner_ctx,
                stream,
                Some(&mut states),
                Some(&row_owner),
                &input.positions,
            )?
            .context("GLM long owner MLA produced no FFN phase")?;
            stage.copy(ctx.gpu, &ffn_spans, 0, first, ROWS, true, stream)?;
        }
        // Every owner's post-attention highway, mHC coefficients and FFN input.
        stage.copy(ctx.gpu, &ffn_spans, 0, 0, rows, false, stream)?;
        self.ffn
            .forward_prefill(b.norm_output(), rows, ctx, stream)?;
        let positions: Vec<usize> = owners.iter().flat_map(|o| o.positions).collect();
        let joint = ctx::MultiSeqCtx::new(
            self,
            ctx,
            b.hidden_states(),
            b.residual(),
            rows,
            &positions,
            bs,
            stream,
        );
        self.ms_hc_supplied_post(
            &joint,
            hc_ffn::HcFfnPhase {
                hc_streams: b.hc_streams(),
                post: b.hc_post(),
                comb: b.hc_comb(),
                diag_this: false,
            },
            b.moe_output(),
            ctx,
            stream,
        )
    }
}
