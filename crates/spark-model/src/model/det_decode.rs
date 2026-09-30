// SPDX-License-Identifier: AGPL-3.0-only

//! Model-side taps of the decode determinism tracer
//! (`ATLAS_GLM_DET_TRACE_DECODE`, see [`crate::det_trace::decode`]): the
//! state decode starts from, the drafter's inputs and drafts, the verify
//! forward's inputs and head, and the committed state. Every tap only reads.

use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::det_trace::{self, At, Scope, decode};
use crate::layer::SsmLayerState;
use crate::layers::DflashProposerState;
use crate::speculative::ProposerState;
use crate::traits::SequenceState;

fn dflash_state(state: Option<&dyn ProposerState>) -> Option<&DflashProposerState> {
    state?.as_any().downcast_ref::<DflashProposerState>()
}

impl TransformerModel {
    /// `at` moved past the last layer, where the head and the state log.
    fn det_after_layers(&self, at: At) -> At {
        At {
            layer: self.layers.len(),
            ..at
        }
    }

    /// Every KDA layer's recurrent and conv state of `seq`, as left by
    /// whatever ran on `stream` and the secondary stream, as `stages`.
    fn det_kda_state(&self, at: At, seq: &SequenceState, stream: u64, stages: [&str; 2]) {
        if !stages.into_iter().any(decode::wanted) {
            return;
        }
        let states = || {
            seq.layer_states
                .iter()
                .filter_map(|state| state.as_any().downcast_ref::<SsmLayerState>())
        };
        let pool = &self.ssm_pool;
        let h: Vec<_> = states().map(|s| (s.h_state, pool.h_stored_bytes)).collect();
        let conv: Vec<_> = states().map(|s| (s.conv_state, pool.conv_bytes)).collect();
        // A verify commit writes the state on the secondary stream.
        let gpu = self.gpu.as_ref();
        let _ = gpu.synchronize(self.secondary_stream);
        decode::spans(at, gpu, stream, stages[0], h.len(), &h, "");
        decode::spans(at, gpu, stream, stages[1], conv.len(), &conv, "");
    }

    /// The request's last prefill chunk is done: log what decode starts from.
    pub(in crate::model) fn det_decode_pre(&self, seq: &SequenceState, prompt: usize, stream: u64) {
        let Some(at) = decode::pre_at(self.config.ep_rank, seq.slot_idx, prompt) else {
            return;
        };
        let at = self.det_after_layers(at);
        let ctx_rows = dflash_state(seq.proposer_state.as_deref()).map_or(0, |d| d.ctx_len);
        let values = [
            seq.slot_idx,
            prompt,
            seq.cached_prefix_tokens,
            seq.marconi_skip_to,
            seq.kv_valid_tokens,
            ctx_rows,
        ];
        decode::values(at, "pre", 0, &values.map(|v| v as u32));
        if decode::traced(&at) {
            self.det_kda_state(at, seq, stream, ["kda_h", "kda_conv"]);
        }
    }

    /// Before a propose for `seq`: `d_in` and `d_hid` (the stack row at
    /// `hidden`). Returns where the propose's lines sit, if it is traced.
    pub(in crate::model) fn det_propose_enter(
        &self,
        seq: &SequenceState,
        token: u32,
        position: usize,
        num_drafts: usize,
        hidden: Option<DevicePtr>,
    ) -> Option<At> {
        let at = decode::step_at(self.config.ep_rank, seq.slot_idx, position)?;
        let d = dflash_state(seq.proposer_state.as_deref())?;
        let values = [
            token as usize,
            position,
            num_drafts,
            d.ctx_len,
            d.ctx_committed,
            d.ctx_count_drafter,
            d.seq_len,
            d.last_num_accepted,
            d.skip_next_decode_append as usize,
            d.lane_id,
            d.end_floor,
        ];
        decode::values(at, "d_in", 0, &values.map(|v| v as u32));
        let (gpu, stream) = (self.gpu.as_ref(), self.gpu.default_stream());
        if let Some(hidden) = hidden {
            let stack = [(hidden, d.ctx_slot_bytes)];
            decode::spans(at, gpu, stream, "d_hid", 1, &stack, "");
        }
        Some(at)
    }

    /// After the propose at `at`: the context it attended and its drafts.
    pub(in crate::model) fn det_propose_done(
        &self,
        at: At,
        slot: usize,
        state: &dyn ProposerState,
        drafts: &[u32],
    ) {
        let (gpu, stream) = (self.gpu.as_ref(), self.gpu.default_stream());
        if let Some(d) = dflash_state(Some(state)) {
            let ctx = [(d.ctx_hidden_acc, d.ctx_len * d.ctx_slot_bytes)];
            let rows = d.ctx_len.to_string();
            if decode::first_propose(slot) {
                decode::spans(at, gpu, stream, "d_ctx0", d.ctx_len, &ctx, &rows);
            }
            decode::spans(at, gpu, stream, "x_d_ctx", d.ctx_len, &ctx, &rows);
            if decode::wanted("d_pos") {
                let positions: Vec<u8> = d
                    .ctx_positions
                    .iter()
                    .flat_map(|p| p.to_le_bytes())
                    .collect();
                let (first, last) = (d.ctx_positions.first(), d.ctx_positions.last());
                let v = format!(
                    "{},{},{}",
                    d.ctx_positions.len(),
                    first.copied().unwrap_or(-1),
                    last.copied().unwrap_or(-1)
                );
                decode::hashed(at, "d_pos", (0, d.ctx_positions.len()), &positions, &v);
            }
        }
        decode::values(at, "d_out", 0, drafts);
    }

    /// [`Self::det_propose_enter`] for every sequence of a batched propose.
    pub(in crate::model) fn det_propose_batch_enter(
        &self,
        seqs: &[&mut SequenceState],
        tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        hiddens: &[DevicePtr],
    ) -> Vec<(usize, Option<At>)> {
        if !decode::on() {
            return Vec::new();
        }
        (0..seqs.len())
            .map(|i| {
                let hidden = hiddens.get(i).copied();
                let seq = &*seqs[i];
                let at = self.det_propose_enter(seq, tokens[i], positions[i], num_drafts, hidden);
                (seq.slot_idx, at)
            })
            .collect()
    }

    /// [`Self::det_propose_done`] for every sequence of a batched propose.
    pub(in crate::model) fn det_propose_batch_done(
        &self,
        entered: &[(usize, Option<At>)],
        states: &[&mut dyn ProposerState],
        drafts: &[Vec<u32>],
    ) {
        for ((&(slot, at), state), drafts) in entered.iter().zip(states).zip(drafts) {
            if let Some(at) = at {
                self.det_propose_done(at, slot, &**state, drafts);
            }
        }
    }

    /// A verify of `tokens` starts on `seq`, its rows embedded: `tok` and
    /// `emb`. The layers' taps log to the returned scope.
    pub(in crate::model) fn det_verify_enter(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        stream: u64,
    ) -> Option<Scope> {
        let at = decode::step_at(self.config.ep_rank, seq.slot_idx, seq.seq_len)?;
        decode::values(at, "tok", 0, tokens);
        let scope = decode::enter(at);
        let (hidden, row) = (self.buffers.hidden_states(), self.config.hidden_size * 2);
        det_trace::on_stream(self.gpu.as_ref(), stream).tap("emb", hidden, (0, tokens.len()), row);
        Some(scope)
    }

    /// `out`: the mHC highway rows after the verify layer just run.
    pub(in crate::model) fn det_verify_layer_out(&self, rows: usize, stream: u64) {
        if decode::scope_at().is_none() {
            return;
        }
        let elem = crate::layers::ops::hc_elem_bytes(&self.config.model_type);
        let row = self.config.hc_mult * self.config.hidden_size * elem;
        let highway = self.buffers.hc_streams();
        det_trace::on_stream(self.gpu.as_ref(), stream).tap("out", highway, (0, rows), row);
    }

    /// The verify forward of `top1.len()` rows is done and read back:
    /// `final`, `logits` and `top1`.
    pub(in crate::model) fn det_verify_done(&self, top1: &[u32], stream: u64) {
        let Some(at) = decode::scope_at() else {
            return;
        };
        let at = self.det_after_layers(at);
        det_trace::set_layer(at.layer);
        let (gpu, k) = (self.gpu.as_ref(), top1.len());
        let (h, vocab) = (self.config.hidden_size, self.config.vocab_size);
        det_trace::on_stream(gpu, stream).tap("final", self.buffers.hidden_states(), (0, k), h * 2);
        // The vocabulary split projects this rank's half only.
        let (first, columns) = if self.glm_verify_logits_argmax_only() {
            let rank = self.comm.as_ref().map_or(0, |comm| comm.rank());
            (rank * (vocab / 2), vocab / 2)
        } else {
            (0, vocab)
        };
        let base = self.buffers.logits();
        let rows: Vec<_> = (0..k)
            .map(|row| (base.offset((row * vocab + first) * 2), columns * 2))
            .collect();
        decode::logits(at, gpu, stream, &rows);
        decode::values(at, "top1", 0, top1);
    }

    /// `committed` of the `k` verified rows of `seq` were committed: `acc`
    /// and the state (`st_h`, `st_conv`), then the request's next step starts.
    pub(in crate::model) fn det_committed(&self, seq: &SequenceState, committed: usize, k: usize) {
        if !decode::on() {
            return;
        }
        if let Some(at) = decode::step_at(self.config.ep_rank, seq.slot_idx, seq.seq_len) {
            let at = self.det_after_layers(at);
            decode::values(at, "acc", 0, &[committed as u32, k as u32]);
            self.det_kda_state(at, seq, self.gpu.default_stream(), ["st_h", "st_conv"]);
        }
        decode::end_step(seq.slot_idx);
    }
}

#[cfg(test)]
#[path = "det_decode_tests.rs"]
mod tests;
