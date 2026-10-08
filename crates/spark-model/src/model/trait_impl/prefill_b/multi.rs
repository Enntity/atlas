// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_MULTI=1` (default off; needs
//! `ATLAS_QWEN4EXP_PREFILL_ROWINV=1`): several short prompts prefilled in
//! ONE forward. Their rows lie back to back; the row-wise work (projections,
//! mHC, MoE: row-invariant under ROWINV) runs once over every row, the
//! per-sequence work per sequence (`TransformerLayer::prefill_multi`). So
//! each prompt's logits are its single-request ROWINV prefill's, while a burst
//! of short prompts streams the weights once instead of once a prompt.
//!
//! Every rank runs [`TransformerModel::prefill_multi_pass`] on the same
//! prompts in the same order (the head sends them, `EP_CMD_PREFILL_MULTI`),
//! so the per-sequence collectives inside (the prefix-match agreement, the
//! block admission) pair up as for single prefills. Per sequence, in order:
//! the prefix lookup (a sequence that restores a snapshot leaves the pass and
//! prefills alone afterwards), the block reservation, the metadata (each
//! sequence its own region of the scratch arena); then the pass; then, per
//! sequence, the drafter capture, the finish (logits, prefix-cache insert)
//! and the eager drafter prefill, exactly as a single prefill ends.
//!
//! Scope (`prefill_multi_eligible`): qwen4_exp, every prompt one chunk of at
//! most [`MULTI_MAX_PROMPT`] tokens (under the QSA inert bound), text only,
//! no prompt logprobs, no LoRA. A sequence takes no in-pass tail checkpoint.

use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext, MultiSeg};
use crate::layers::ops::qwen4exp_rowinv::MULTI_MAX_PROMPT;
use crate::model::impl_a2_ep_worker::EP_CMD_PREFILL_MULTI;
use crate::traits::PrefillSlice;

/// `ATLAS_QWEN4EXP_PREFILL_MULTI=1` (with ROWINV). Read once.
pub fn multi_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let set = matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_MULTI").as_deref(),
            Ok("1") | Ok("true")
        );
        let rowinv = crate::layers::ops::qwen4exp_rowinv::on();
        if set && !rowinv {
            tracing::warn!(
                "ATLAS_QWEN4EXP_PREFILL_MULTI=1 ignored: it needs ATLAS_QWEN4EXP_PREFILL_ROWINV=1"
            );
        }
        set && rowinv
    })
}

impl TransformerModel {
    /// Whether this model serves `prefill_multi` at all.
    pub(in crate::model) fn prefill_multi_supported(&self) -> bool {
        multi_requested()
            && self.config.model_type == "qwen4_exp"
            && self.config.hc_mult > 0
            && !self.use_fp32_logits
            && self.lora.is_none()
    }

    /// Head side of `Model::prefill_multi`: send the prompts to the other
    /// ranks, then run the pass.
    pub(in crate::model) fn prefill_multi_head(
        &self,
        items: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        ensure!(
            self.prefill_multi_supported(),
            "prefill_multi is not enabled for this model"
        );
        for it in items.iter() {
            ensure!(
                it.chunk_start == 0
                    && it.is_last_chunk
                    && it.chunk_len == it.prompt_tokens.len()
                    && (1..=MULTI_MAX_PROMPT).contains(&it.chunk_len),
                "prefill_multi takes whole prompts of 1..={MULTI_MAX_PROMPT} tokens"
            );
        }
        if self.multi_rank_protocol_active() {
            let slots: Vec<u32> = items.iter().map(|i| i.seq.slot_idx as u32).collect();
            let lens: Vec<u32> = items.iter().map(|i| i.chunk_len as u32).collect();
            let all: Vec<u32> = items
                .iter()
                .flat_map(|i| i.prompt_tokens.iter().copied())
                .collect();
            self.ep_broadcast_seq_and_cmd(0, EP_CMD_PREFILL_MULTI, true)?;
            self.ep_broadcast_u32(items.len() as u32)?;
            self.ep_broadcast_tokens(&slots)?;
            self.ep_broadcast_tokens(&lens)?;
            self.ep_broadcast_tokens(&all)?;
        }
        let mut seqs: Vec<(&[u32], &mut crate::traits::SequenceState)> = items
            .iter_mut()
            .map(|i| (i.prompt_tokens, &mut *i.seq))
            .collect();
        self.prefill_multi_pass(&mut seqs, stream)
    }

    /// Worker side of `EP_CMD_PREFILL_MULTI`.
    pub(in crate::model) fn ep_worker_prefill_multi(
        &self,
        slots: &mut [Option<crate::traits::SequenceState>],
    ) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        ensure!((1..=64).contains(&n), "EP prefill_multi of {n} sequences");
        let ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let lens = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        let all = self.ep_broadcast_tokens(&vec![0u32; total])?;
        let mut refs = self.ep_worker_slot_refs(&ids, slots)?;
        let mut at = 0usize;
        let mut seqs: Vec<(&[u32], &mut crate::traits::SequenceState)> = Vec::with_capacity(n);
        for (seq, &len) in refs.iter_mut().zip(&lens) {
            seqs.push((&all[at..at + len as usize], &mut **seq));
            at += len as usize;
        }
        let stream = self.gpu.default_stream();
        self.prefill_multi_pass(&mut seqs, stream)?;
        // As after the worker's single prefill chunk (`0xFFFFFFF0`).
        for (_, seq) in seqs.iter() {
            if let Err(e) = self.normalize_ssm_states_dispatch(seq, stream) {
                tracing::warn!("Worker SSM state normalization failed: {e:#}");
            }
        }
        Ok(true)
    }

    /// The pass, on every rank. Returns each sequence's logits row.
    pub(in crate::model) fn prefill_multi_pass(
        &self,
        seqs: &mut [(&[u32], &mut crate::traits::SequenceState)],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };
        let began = Instant::now();
        let n = seqs.len();
        for (tokens, seq) in seqs.iter() {
            ensure!(
                (1..=MULTI_MAX_PROMPT).contains(&tokens.len())
                    && !self.tokens_have_vision_pad(tokens)
                    && seq.collect_prompt_logprobs.is_none()
                    && seq.tokens.is_empty(),
                "prefill_multi: a prompt it does not serve"
            );
        }
        // Per sequence: the prefix lookup. A sequence restoring a snapshot
        // replays from it alone after the pass (`multi[k] == false`).
        let mut multi = vec![true; n];
        {
            let mut kv = self.kv_cache.lock();
            let bs = kv.block_size();
            for (k, (tokens, seq)) in seqs.iter_mut().enumerate() {
                let (_, skip) = self.prefill_b_prefix_lookup(
                    tokens,
                    seq,
                    0,
                    tokens.len(),
                    &mut kv,
                    stream,
                    None,
                )?;
                self.pc_apply_plant(tokens, seq, 0, bs);
                multi[k] = !skip;
            }
        }
        // Logits rows: the pass's sequences first, then the restored ones.
        let rows_buf = self.prefill_multi_rows(n, stream)?;
        let mut logits = vec![DevicePtr::NULL; n];
        let rows: Vec<usize> = seqs.iter().map(|s| s.0.len()).collect();
        if multi.iter().any(|&m| m) {
            self.prefill_multi_forward(seqs, &multi, &rows, rows_buf, began, &mut logits, stream)?;
        }
        // The sequences that restored: their own single prefills.
        let mut at = multi.iter().filter(|&&m| m).count();
        for (k, (tokens, seq)) in seqs.iter_mut().enumerate().filter(|(k, _)| !multi[*k]) {
            let out = self.prefill_chunk_entry(tokens, seq, 0, tokens.len(), true, stream)?;
            logits[k] = self.prefill_multi_logits_row(rows_buf.0, at);
            self.gpu
                .copy_d2d_async(out, logits[k], self.config.vocab_size * 2, stream)?;
            at += 1;
        }
        Ok(logits)
    }

    /// Embed, admit and describe the `multi` sequences, run the layers over
    /// all their rows, then finish each one.
    #[allow(clippy::too_many_arguments)]
    fn prefill_multi_forward(
        &self,
        seqs: &mut [(&[u32], &mut crate::traits::SequenceState)],
        multi: &[bool],
        rows: &[usize],
        (rows_base, rows_cap): (DevicePtr, usize),
        began: Instant,
        logits: &mut [DevicePtr],
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let hidden = self.buffers.hidden_states();
        let picked: Vec<usize> = (0..seqs.len()).filter(|&k| multi[k]).collect();
        let mut row0 = vec![0usize; seqs.len()];
        let mut total = 0usize;
        for &k in &picked {
            row0[k] = total;
            total += rows[k];
        }
        ensure!(
            total <= self.buffers.max_batch_tokens(),
            "prefill_multi: {total} rows exceed the {}-row arena",
            self.buffers.max_batch_tokens()
        );
        // Zero the arena once; each prompt's embedding at its rows.
        let first = picked[0];
        let pre = self.prefill_b_zero_and_embed(true, seqs[first].0, 0, rows[first], stream)?;
        for &k in &picked[1..] {
            let dst = hidden.offset(row0[k] * h * 2);
            self.prefill_b_embed_chunk_at(seqs[k].0, 0, rows[k], dst, stream)?;
        }
        let _ = super::embed_chunk::take_staged_ids();
        self.buffers.note_rows(total);

        // Admission and metadata, a region of the scratch arena each (after
        // the MoE top-k staging the pass's `total` rows need).
        let topk = self.config.num_experts_per_tok;
        let mut at = (total * topk * 8 + 255) & !255;
        let scratch = self.buffers.scratch();
        let mut metas: Vec<AttnMetadataDev> = Vec::with_capacity(picked.len());
        {
            let mut kv = self.kv_cache.lock();
            for &k in &picked {
                let (tokens, seq) = &mut seqs[k];
                let len = rows[k];
                self.reserve_prefill_blocks(seq, len, &mut kv, stream)?;
                match self.prefill_b_proc_range(
                    tokens,
                    seq,
                    0,
                    len,
                    true,
                    0,
                    false,
                    hidden.offset(row0[k] * h * 2),
                    stream,
                )? {
                    super::proc_range::ProcRange::Compute {
                        proc_start: 0,
                        proc_count,
                        effective_seq_len_start: 0,
                    } if proc_count == len => {}
                    _ => anyhow::bail!("prefill_multi: a sequence that does not compute from 0"),
                }
                self.ple_prefill_warm(tokens, 0, seq)?;
                let region = self.buffers.scratch_bytes().saturating_sub(at);
                let m = self.prefill_b_upload_meta_at(
                    tokens,
                    seq,
                    0,
                    len,
                    0,
                    len,
                    0,
                    &kv,
                    scratch.offset(at),
                    region,
                    stream,
                )?;
                ensure!(m.needs_paged, "prefill_multi attends paged");
                // The metadata is packed in ONE pinned staging buffer that the
                // next sequence repacks: drain this upload first.
                self.gpu.synchronize(stream)?;
                self.prefill_b_upload_paged(
                    seq,
                    len,
                    0,
                    len,
                    m.meta_base,
                    m.slot_offset,
                    &kv,
                    stream,
                )?;
                let page = seq.chunked_prefill_meta.as_ref().expect("uploaded above");
                let (ph, pw) = if m.use_mrope {
                    (
                        m.meta_base.offset(m.pos_stream_bytes),
                        m.meta_base.offset(m.pos_stream_bytes * 2),
                    )
                } else {
                    (m.meta_base, m.meta_base)
                };
                metas.push(AttnMetadataDev {
                    positions: m.meta_base,
                    positions_h: ph,
                    positions_w: pw,
                    slot: m.meta_base.offset(m.slot_offset),
                    seq_len: page.seq_len,
                    block_table: page.block_table,
                    max_blocks_per_seq: seq.block_table.len() as u32,
                    num_seqs: 1,
                    seq_slot: DevicePtr::NULL,
                    moe_row_adapter: DevicePtr::NULL,
                });
                at = (at + m.slot_offset + len * 8 + 255) & !255;
                ensure!(
                    at <= self.buffers.scratch_bytes(),
                    "prefill_multi: scratch arena full"
                );
            }
        }
        // The pass's token ids, host and device, back to back.
        let ids: Vec<u32> = picked
            .iter()
            .flat_map(|&k| seqs[k].0.iter().copied())
            .collect();
        // SAFETY: a `&[u32]` viewed as its `len * 4` bytes.
        let id_bytes =
            unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4) };
        self.gpu
            .copy_h2d_async(id_bytes, self.buffers.token_ids(), stream)?;
        self.gpu.synchronize(stream)?;
        let t_meta = began.elapsed();

        let ctx_for = |meta: AttnMetadataDev, at: usize, len: usize| ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(meta),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: Some(self.buffers.token_ids().offset(at * 4)),
            host_token_ids: Some(&ids[at..at + len]),
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.moe_lora_route(-1),
        };
        let pass_ctx = ctx_for(metas[0], 0, total);
        let seg_ctx: Vec<ForwardContext> = picked
            .iter()
            .enumerate()
            .map(|(i, &k)| ctx_for(metas[i], row0[k], rows[k]))
            .collect();
        {
            let _rowinv = crate::layers::ops::qwen4exp_rowinv::enter(
                self.gpu.as_ref(),
                self.buffers.sizes().hc_lowrank_scratch,
                self.config.hc_mult * h,
                self.config.hc_lowrank,
                true,
            );
            let mut kv = self.kv_cache.lock();
            // ATLAS_GLM_DET_TRACE: pass rows as positions of the first sequence's
            // request (the `prefill_multi` line maps them back).
            let _det = crate::det_trace::enter(self.config.ep_rank, seqs[first].1.slot_idx, 0);
            let det = crate::det_trace::on_stream(self.gpu.as_ref(), stream);
            det.tap("emb", hidden, (0, total), h * 2);
            let hc_row = self.config.hc_mult
                * h
                * crate::layers::ops::hc_elem_bytes(&self.config.model_type);
            for (i, layer) in self.layers.iter().enumerate() {
                crate::det_trace::set_layer(i);
                let mut segs: Vec<MultiSeg<'_, '_>> = Vec::with_capacity(picked.len());
                let mut c = 0usize;
                for (k, (_, seq)) in seqs.iter_mut().enumerate() {
                    if !multi[k] {
                        continue;
                    }
                    let seq = &mut **seq;
                    segs.push(MultiSeg {
                        row0: row0[k],
                        rows: rows[k],
                        start: 0,
                        kv_write_start: 0,
                        state: seq.layer_states[i].as_mut(),
                        block_table: &mut seq.block_table,
                        disk_block_ids: &mut seq.disk_block_ids,
                        disk_last_offloaded_per_layer: &mut seq.disk_last_offloaded_per_layer,
                        ctx: &seg_ctx[c],
                    });
                    c += 1;
                }
                layer
                    .prefill_multi(hidden, total, &mut segs, &mut kv, &pass_ctx, stream)
                    .map_err(|e| anyhow::anyhow!("multi-sequence prefill layer {i}: {e}"))?;
                det.tap("out", self.buffers.hc_streams(), (0, total), hc_row);
            }
        }
        let t_fwd = began.elapsed();
        // Every sequence's logits in one pass over the head, when it can.
        let last: Vec<usize> = picked.iter().map(|&k| row0[k] + rows[k] - 1).collect();
        let batched = self.prefill_multi_lm_head(rows_base, rows_cap, &last, stream)?;

        // Finish each sequence as its single prefill does.
        for (i, &k) in picked.iter().enumerate() {
            let (tokens, seq) = &mut seqs[k];
            let len = rows[k];
            self.try_mtp_prefill_capture_from(seq, 0, len, hidden.offset(row0[k] * h * 2), stream)?;
            let mut kv = self.kv_cache.lock();
            let bs = kv.block_size();
            seq.tokens.extend_from_slice(tokens);
            seq.seq_len = len;
            seq.last_decode_ckpt_block = len / bs;
            let row = self.prefill_multi_logits_row(rows_base, i);
            let out = self.prefill_b_finalize_last_at(
                tokens,
                seq,
                &mut kv,
                0,
                len,
                len,
                row0[k],
                0,
                batched.then_some(row),
                stream,
            )?;
            drop(kv);
            let marks = [
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                t_meta,
                t_fwd,
            ];
            self.warm_trace_chunk(seq, len, began, (0, len), pre, marks, Some(out), stream)?;
            if !batched {
                self.gpu
                    .copy_d2d_async(out, row, self.config.vocab_size * 2, stream)?;
            }
            logits[k] = row;
            self.try_eager_drafter_prefill(seq, true, stream)?;
        }
        let layout: Vec<(usize, usize, usize)> = picked
            .iter()
            .map(|&k| (seqs[k].1.slot_idx, row0[k], rows[k]))
            .collect();
        tracing::info!(
            "prefill_multi: {} sequences, {total} rows in one pass ({:.1} ms to the forward's end); \
             (slot, row0, rows) {layout:?}",
            picked.len(),
            t_fwd.as_secs_f64() * 1e3
        );
        Ok(())
    }
}
