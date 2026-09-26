// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched long-context verify for an MLA layer.
//!
//! Attention runs per owner, unchanged: each owner's `rows` rows are placed at
//! arena rows [0, rows) and go through the same validated eager multi-row
//! verify path (causal semantic index, sparse attention, cache writes) with
//! that owner's own metadata and state; `validate_glm_long_verify` admits the
//! width for the active lane (K3 on repaired MTP, 2..=8 on DFlash). The FFN
//! runs through `ffn_per_owner`; the mHC post is joint.

use super::*;
use crate::layer::glm_long_owner::{self as long_owner, GlmLongOwner, GlmLongStage};
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
        let rows = long_owner::owner_rows(owners)?;
        let total = owners.len() * rows;
        let b = ctx.buffers;
        let r = stage.rows;
        let ffn_spans = [
            (b.hc_streams(), stage.highway, r.highway),
            (b.hc_post(), stage.post, r.post),
            (b.hc_comb(), stage.comb, r.comb),
            (b.norm_output(), stage.norm, r.hidden),
        ];
        // Every owner's incoming highway, before owner 0's attention reuses rows [0, rows).
        stage.copy(ctx.gpu, &ffn_spans[..1], 0, 0, total, true, stream)?;
        let row_owner = &[0usize; long_owner::MAX_OWNER_ROWS][..rows];
        let bs = kv_cache.block_size() as u32;
        for (owner, input) in owners.iter_mut().enumerate() {
            let first = owner * rows;
            stage.copy(ctx.gpu, &ffn_spans[..1], 0, first, rows, false, stream)?;
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
                rows,
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
                Some(row_owner),
                &input.positions,
            )?
            .context("GLM long owner MLA produced no FFN phase")?;
            stage.copy(ctx.gpu, &ffn_spans, 0, first, rows, true, stream)?;
        }
        // Every owner's post-attention highway, mHC coefficients and FFN input.
        stage.copy(ctx.gpu, &ffn_spans, 0, 0, total, false, stream)?;
        let ffn_out = long_owner::ffn_per_owner(&self.ffn, owners.len(), rows, stage, ctx, stream)?;
        let positions: Vec<usize> = owners
            .iter()
            .flat_map(|o| o.positions.iter().copied())
            .collect();
        let joint = ctx::MultiSeqCtx::new(
            self,
            ctx,
            b.hidden_states(),
            b.residual(),
            total,
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
            ffn_out,
            ctx,
            stream,
        )
    }
}
