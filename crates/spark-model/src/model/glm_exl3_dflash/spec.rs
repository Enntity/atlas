// SPDX-License-Identifier: AGPL-3.0-only

//! F4 phase 2: the native EXL3 DFlash2 speculative lane
//! (`ATLAS_GLM_EXL3_DFLASH_SPEC=1`).
//!
//! One step for the pending token `x` at position `P` (not yet processed by
//! the target):
//! 1. the drafter appends the previous context row(s) and proposes
//!    `d1..dk` after `x` (same as the draft-only observer);
//! 2. one `k + 1`-row verify forward of `[x, d1..dk]` at `P..P+k`
//!    (`decode_verify_graphed_kgamma`, EXL3 arms) yields the per-row argmax
//!    `v0..vk` and captures every row's drafter context;
//! 3. accepted = longest prefix with `d_j == v_{j-1}`; the step emits
//!    `d1..d_a` and the bonus `v_a`;
//! 4. rollback: `seq_len = P + a + 1`, the KDA/conv state takes the snapshot
//!    after row `a` (`commit_accepted_prefix`, which also rewinds the
//!    per-layer aux state), MLA/indexer rows past `seq_len` are
//!    position-addressed and rewritten by the next verify;
//! 5. the drafter context gains rows `0..=a` (positions `P..=P+a`) only.
//!
//! Every input is identical on both TP ranks (drafts are replicated, the
//! verify argmax comes from the gathered logits), so the head only tells the
//! worker which slot and token; each rank runs the whole step itself.

use anyhow::{Context, Result, ensure};

use super::super::types::TransformerModel;
use super::Exl3DflashObserver;
use crate::traits::{Model, SequenceState};

/// Worker command: one speculative step for the addressed slot; one u32
/// payload word (the pending token) follows.
pub(crate) const EP_CMD_EXL3_DFLASH_SPEC: u32 = 0xFFFF_FFE4;

impl TransformerModel {
    /// Head side of [`Model::exl3_dflash_spec_step`]: `Ok(None)` (serial
    /// decode this step) unless the lane is on and the slot's drafter state
    /// is healthy and the verify rows fit the sequence's block capacity.
    pub(crate) fn exl3_dflash_spec_step_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
    ) -> Result<Option<Vec<u32>>> {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return Ok(None);
        };
        if !obs.spec {
            return Ok(None);
        }
        let ready = obs
            .seqs
            .lock()
            .get(&seq.slot_idx)
            .is_some_and(|so| so.broken.is_none() && so.chunk.is_none());
        let block = self.kv_cache.lock().block_size();
        let fits = (seq.seq_len + obs.k + 1).div_ceil(block) <= self.max_blocks_per_seq as usize;
        if !(ready && fits) {
            return Ok(None);
        }
        if self.multi_rank_protocol_active() {
            self.ep_broadcast_seq_and_cmd(
                seq.slot_idx as u32,
                EP_CMD_EXL3_DFLASH_SPEC,
                self.ep_protocol_v2,
            )?;
            self.ep_broadcast_u32(token)?;
        }
        self.exl3_dflash_spec_local(token, seq).map(Some)
    }

    /// The speculative step itself, run identically on every rank.
    pub(crate) fn exl3_dflash_spec_local(
        &self,
        token: u32,
        seq: &mut SequenceState,
    ) -> Result<Vec<u32>> {
        let obs: &Exl3DflashObserver = self
            .exl3_dflash
            .as_deref()
            .context("EXL3 DFlash speculative step without the observer")?;
        ensure!(
            !obs.verify_save.is_null(),
            "EXL3 DFlash speculative step without verify capture rows"
        );
        let mut seqs = obs.seqs.lock();
        let so = seqs
            .get_mut(&seq.slot_idx)
            .context("EXL3 DFlash speculative step for a slot without drafter state")?;
        ensure!(
            so.broken.is_none(),
            "EXL3 DFlash speculative step on a stopped drafter: {:?}",
            so.broken
        );
        let pos = seq.seq_len;
        so.observe(pos, token);
        // ATLAS_GLM_EXL3_DFLASH_PROFILE=1: synchronized phase timing of every
        // step (draft / verify / commit), summarized in the final line.
        let timed = obs.profile;
        let sync_now = |on: bool| -> Result<std::time::Instant> {
            if on {
                self.gpu.synchronize(self.gpu.default_stream())?;
            }
            Ok(std::time::Instant::now())
        };
        let t0 = sync_now(timed)?;
        let drafts = self.exl3_dflash_step(obs, so, token, pos, seq)?;
        let t1 = sync_now(timed)?;
        ensure!(
            drafts.len() == obs.k,
            "EXL3 DFlash drafter returned {} drafts, expected {}",
            drafts.len(),
            obs.k
        );
        let mut rows = Vec::with_capacity(obs.k + 1);
        rows.push(token);
        rows.extend_from_slice(&drafts);
        let width = rows.len();
        let base_tokens = seq.tokens.len();
        let check = obs.rollback_check_every > 0
            && (so.spec_steps + 1).is_multiple_of(obs.rollback_check_every);
        if check {
            self.checkpoint_ssm_states(seq)?;
        }
        let argmax = self.decode_verify_graphed_kgamma(&rows, seq, 0)?;
        let t2 = sync_now(timed)?;
        ensure!(
            argmax.len() == width,
            "EXL3 DFlash verify returned {} rows for {width}",
            argmax.len()
        );
        let accepted = accepted_prefix(&drafts, &argmax).min(obs.force_accept.unwrap_or(obs.k));
        let bonus = argmax[accepted];
        // Keep `x, d1..d_a`; the bonus is the next step's pending token.
        seq.seq_len = pos + accepted + 1;
        seq.tokens.truncate(base_tokens + accepted + 1);
        self.commit_accepted_prefix(seq, accepted + 1, width)?;
        self.sync_secondary()?;
        if check && accepted + 1 < width {
            let (state_same, argmax_same) =
                self.exl3_rollback_check(seq, &rows, &argmax, accepted, pos, base_tokens)?;
            so.rollback_checks.0 += 1;
            so.rollback_checks.1 += u64::from(state_same);
            so.rollback_checks.2 += u64::from(argmax_same);
            if !(state_same && argmax_same) {
                tracing::warn!(
                    "DFLASH_ROLLBACK_CHECK rank={} slot={} pos={pos} accepted={accepted}: \
                     state_identical={state_same} argmax_identical={argmax_same}",
                    obs.rank,
                    seq.slot_idx
                );
            }
        }

        let gpu = self.gpu.as_ref();
        let stream = gpu.default_stream();
        let d = so.dstate();
        obs.make_room(gpu, d, accepted + 1, stream)?;
        gpu.copy_d2d_async(
            obs.verify_save,
            d.ctx_hidden_acc.offset(d.ctx_len * obs.slot_bytes),
            (accepted + 1) * obs.slot_bytes,
            stream,
        )?;
        d.ctx_positions
            .extend((pos..=pos + accepted).map(|p| p as i32));
        d.ctx_len += accepted + 1;
        so.pending_capture = None;

        let mut emitted = drafts[..accepted].to_vec();
        emitted.push(bonus);
        for (i, &t) in emitted.iter().enumerate() {
            so.observe(pos + 1 + i, t);
        }
        so.spec_steps += 1;
        so.spec_emitted += emitted.len() as u64;
        if timed {
            let t3 = sync_now(true)?;
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
            so.spec_ms[0] += ms(t0, t1);
            so.spec_ms[1] += ms(t1, t2);
            so.spec_ms[2] += ms(t2, t3);
        }
        Ok(emitted)
    }

    /// Exact rollback check (see `ATLAS_GLM_EXL3_DFLASH_ROLLBACK_CHECK`).
    /// Precondition: the pre-verify SSM checkpoint was taken, the first
    /// verify committed `accepted + 1` rows. Leaves the sequence committed
    /// from the replay (identical when the check passes).
    fn exl3_rollback_check(
        &self,
        seq: &mut SequenceState,
        rows: &[u32],
        argmax: &[u32],
        accepted: usize,
        pos: usize,
        base_tokens: usize,
    ) -> Result<(bool, bool)> {
        let width = rows.len();
        let first = self.exl3_ssm_fingerprint(seq)?;
        // Back to the pre-verify state: SSM checkpoint, aux rewound by the
        // committed rows, seq_len/tokens.
        let stream = self.gpu.default_stream();
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            self.layers[i].rollback_aux_verify(
                state.as_mut(),
                0,
                accepted + 1,
                self.gpu.as_ref(),
                stream,
            )?;
        }
        seq.seq_len = pos;
        seq.tokens.truncate(base_tokens);
        self.rollback_ssm_states(seq, 0)?;
        // Same accepted rows, different rejected rows.
        let vocab = self.config.vocab_size as u32;
        let mut replay = rows.to_vec();
        for t in replay.iter_mut().skip(accepted + 1) {
            *t = (*t + 1) % vocab;
        }
        let argmax2 = self.decode_verify_graphed_kgamma(&replay, seq, 0)?;
        seq.seq_len = pos + accepted + 1;
        seq.tokens.truncate(base_tokens + accepted + 1);
        self.commit_accepted_prefix(seq, accepted + 1, width)?;
        self.sync_secondary()?;
        let second = self.exl3_ssm_fingerprint(seq)?;
        Ok((first == second, argmax2[..=accepted] == argmax[..=accepted]))
    }

    /// FNV-1a over every KDA layer's live h and conv state (D2H; slow).
    fn exl3_ssm_fingerprint(&self, seq: &SequenceState) -> Result<u64> {
        use crate::layer::SsmLayerState;
        let gpu = self.gpu.as_ref();
        gpu.synchronize(gpu.default_stream())?;
        let h_bytes = self.ssm_pool.h_stored_bytes;
        let conv_bytes = self.config.ssm_conv_state_bytes();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut buf = Vec::new();
        for (i, state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != atlas_core::config::LayerType::LinearAttention {
                continue;
            }
            let ssm = state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .context("EXL3 rollback check: expected SsmLayerState")?;
            for (ptr, bytes) in [(ssm.h_state, h_bytes), (ssm.conv_state, conv_bytes)] {
                buf.resize(bytes, 0);
                gpu.copy_d2h(ptr, &mut buf)?;
                for b in &buf {
                    hash ^= u64::from(*b);
                    hash = hash.wrapping_mul(0x1000_0000_01b3);
                }
            }
        }
        Ok(hash)
    }

    /// Verify capture after layer `layer_idx`: rows `0..rows` of the highway
    /// into the observer's verify rows (no-op without the probe/lane).
    pub(crate) fn exl3_dflash_verify_capture(
        &self,
        layer_idx: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return Ok(());
        };
        if obs.verify_save.is_null() {
            return Ok(());
        }
        let Some(slot) = obs.capture_layers.iter().position(|&l| l == layer_idx) else {
            return Ok(());
        };
        ensure!(
            rows <= obs.k + 1,
            "EXL3 DFlash verify capture of {rows} rows exceeds {}",
            obs.k + 1
        );
        obs.contract_rows(
            self.gpu.as_ref(),
            self.buffers.hc_streams(),
            0,
            rows,
            obs.verify_save.offset(slot * obs.hidden * 2),
            obs.slot_bytes,
            stream,
        )
    }
}

/// Longest prefix of `drafts` equal to the verify argmax chain: draft `j`
/// (0-based) is checked against verify row `j`, which predicts the token
/// after row `j`'s input.
pub(crate) fn accepted_prefix(drafts: &[u32], argmax: &[u32]) -> usize {
    drafts
        .iter()
        .zip(argmax)
        .take_while(|(d, v)| d == v)
        .count()
}

#[cfg(test)]
mod tests {
    use super::accepted_prefix;

    #[test]
    fn accepted_prefix_stops_at_first_mismatch() {
        assert_eq!(accepted_prefix(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(accepted_prefix(&[1, 2, 3], &[1, 9, 3, 4]), 1);
        assert_eq!(accepted_prefix(&[1, 2, 3], &[0, 2, 3, 4]), 0);
        assert_eq!(accepted_prefix(&[], &[7]), 0);
    }
}
