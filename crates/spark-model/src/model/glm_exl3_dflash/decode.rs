// SPDX-License-Identifier: AGPL-3.0-only
//! Decode hooks: score earlier drafts, draft after each serial token and,
//! with the speculative lane, verify and commit.

use super::*;

impl TransformerModel {
    /// Before the serial decode of `token` at `seq.seq_len`: score earlier
    /// drafts against `token`, append the previous position's context row and
    /// propose `k` drafts after `token`. Never fails the decode: a drafter
    /// error stops drafting for this sequence.
    ///
    /// Only a failed verify-probe rollback is returned as an error: the
    /// target state could then differ from the serial one.
    pub(crate) fn exl3_dflash_before_decode(
        &self,
        token: u32,
        seq: &mut SequenceState,
    ) -> Result<()> {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return Ok(());
        };
        let mut seqs = obs.seqs.lock();
        let Some(so) = seqs.get_mut(&seq.slot_idx) else {
            return Ok(());
        };
        if so.broken.is_some() {
            return Ok(());
        }
        let pos = seq.seq_len;
        so.observe(pos, token);
        let drafts = match self.exl3_dflash_step(obs, so, token, pos, seq) {
            Ok(drafts) => drafts,
            Err(error) => {
                tracing::warn!(
                    "EXL3 DFlash rank {} slot {}: drafting stopped at position {pos}: {error:#}",
                    obs.rank,
                    seq.slot_idx
                );
                so.broken = Some(format!("{error:#}"));
                return Ok(());
            }
        };
        if obs.probe_every > 0 && so.steps.is_multiple_of(obs.probe_every) && drafts.len() == obs.k
        {
            let mut rows = Vec::with_capacity(drafts.len() + 1);
            rows.push(token);
            rows.extend_from_slice(&drafts);
            match self.exl3_dflash_verify_probe(&rows, seq) {
                Ok(argmax) => so.parity.record(pos, &drafts, &argmax),
                Err(error) => {
                    // The probe restored the target state before returning a
                    // probe error; a restore failure is the `?` inside.
                    tracing::warn!(
                        "EXL3 DFlash rank {} slot {}: verify probe disabled at {pos}: {error:#}",
                        obs.rank,
                        seq.slot_idx
                    );
                    so.broken = Some(format!("verify probe: {error:#}"));
                }
            }
            if obs.log_every > 0 && so.steps.is_multiple_of(obs.log_every) {
                obs.parity_line("periodic", &request_label(seq), seq.slot_idx, &so.parity);
            }
        }
        Ok(())
    }

    /// One `rows.len()`-row verify forward of `rows` at `seq.seq_len`
    /// (the production K=gamma verify, EXL3 arms), then a full restore:
    /// SSM state from the pre-verify checkpoint, per-layer aux state, and
    /// `seq_len`/`tokens`. MLA K/V and index rows past `seq_len` are
    /// position-addressed and rewritten by the serial decode. Returns the
    /// per-row argmax. Restore errors propagate (outer `Result`); a verify
    /// error is returned only after the restore succeeded.
    fn exl3_dflash_verify_probe(&self, rows: &[u32], seq: &mut SequenceState) -> Result<Vec<u32>> {
        let len = seq.seq_len;
        let tokens = seq.tokens.len();
        use crate::traits::Model;
        self.checkpoint_ssm_states(seq)?;
        let verified = self.decode_verify_graphed_kgamma(rows, seq, 0);
        seq.seq_len = len;
        seq.tokens.truncate(tokens);
        let stream = self.gpu.default_stream();
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            self.layers[i].rollback_aux_verify(
                state.as_mut(),
                0,
                rows.len(),
                self.gpu.as_ref(),
                stream,
            )?;
        }
        self.rollback_ssm_states(seq, 0)?;
        verified
    }

    pub(super) fn exl3_dflash_step(
        &self,
        obs: &Exl3DflashObserver,
        so: &mut SeqObs,
        token: u32,
        pos: usize,
        seq: &SequenceState,
    ) -> Result<Vec<u32>> {
        let gpu = self.gpu.as_ref();
        let stream = gpu.default_stream();
        let profile_now =
            obs.profile && obs.log_every > 0 && (so.steps + 1).is_multiple_of(obs.log_every);
        let clock = |sync: bool| -> Result<std::time::Instant> {
            if sync {
                gpu.synchronize(stream)?;
            }
            Ok(std::time::Instant::now())
        };
        let t_start = clock(profile_now)?;
        match so.pending_capture.take() {
            Some(prev) => {
                ensure!(
                    prev + 1 == pos,
                    "captured position {prev} does not precede decode position {pos}"
                );
                let d = so.dstate();
                obs.make_room(gpu, d, 1, stream)?;
                gpu.copy_d2d_async(
                    obs.save,
                    d.ctx_hidden_acc.offset(d.ctx_len * obs.slot_bytes),
                    obs.slot_bytes,
                    stream,
                )?;
                d.ctx_positions.push(prev as i32);
                d.ctx_len += 1;
            }
            None => {
                let d = so.dstate();
                ensure!(
                    pos > 0 && d.ctx_positions.last().copied() == Some((pos - 1) as i32),
                    "context ends at {:?}, decode position {pos} (prefill capture missing)",
                    d.ctx_positions.last()
                );
            }
        }
        // propose() must not append again: every row is ours.
        so.dstate().skip_next_decode_append = false;
        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
            request_id: seq.request_id.clone(),
        };
        let owner = so.owner;
        let t_appended = clock(profile_now)?;
        let propose = |state: &mut dyn ProposerState| {
            obs.head.propose(
                token,
                DevicePtr::NULL,
                pos,
                obs.k,
                state,
                Some(owner),
                &ctx,
                stream,
                None,
                None,
                None,
            )
        };
        let drafts = propose(so.state.as_mut())?;
        if profile_now {
            let t_proposed = clock(true)?;
            // Same block again: context precompute is already committed, so
            // this times fc-free block forward + head + selector + readback.
            let again = propose(so.state.as_mut())?;
            let t_repeat = clock(true)?;
            let (head_ms, selector_ms) = obs.head.profile_tail_ms(gpu, 5)?;
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
            let propose_ms = ms(t_appended, t_proposed);
            let block_ms = ms(t_proposed, t_repeat);
            tracing::info!(
                "DFLASH_PROFILE rank={} step={} pos={pos} ctx={} append_ms={:.3} propose_ms={:.3} \
                 block_repeat_ms={:.3} precompute_est_ms={:.3} head_ms={:.3} selector_ms={:.3} \
                 layers_embed_readback_est_ms={:.3} repeat_identical={}",
                obs.rank,
                so.steps + 1,
                so.dstate().ctx_len,
                ms(t_start, t_appended),
                propose_ms,
                block_ms,
                propose_ms - block_ms,
                head_ms,
                selector_ms,
                block_ms - head_ms - selector_ms,
                again == drafts
            );
        }
        so.scorer.record(pos, &drafts);
        so.steps += 1;
        if obs.trace {
            tracing::info!(
                "DFLASH_DRAFT rank={} req={} slot={} pos={pos} bonus={token} ctx={} drafts={:?}",
                obs.rank,
                request_label(seq),
                seq.slot_idx,
                so.dstate().ctx_len,
                drafts
            );
        }
        if obs.log_every > 0 && so.steps.is_multiple_of(obs.log_every) {
            obs.log_line(
                "periodic",
                &request_label(seq),
                seq.slot_idx,
                so.scorer.stats(),
            );
        }
        Ok(drafts)
    }

    /// After a successful serial decode of position `pos`: its context row
    /// sits in the save buffer for the next step.
    pub(crate) fn exl3_dflash_after_decode(&self, slot: usize, pos: usize) {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return;
        };
        if let Some(so) = obs.seqs.lock().get_mut(&slot)
            && so.broken.is_none()
        {
            so.pending_capture = Some(pos);
        }
    }

    /// Batched (n > 1) decode does not capture: stop drafting for those
    /// sequences (phase 1 measures C=1).
    pub(crate) fn exl3_dflash_batched_decode(&self, slots: &[usize]) {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return;
        };
        let mut seqs = obs.seqs.lock();
        for slot in slots {
            if let Some(so) = seqs.get_mut(slot)
                && so.broken.is_none()
            {
                so.broken = Some("batched decode (phase 1 drafts at C=1 only)".to_string());
            }
        }
    }
}
