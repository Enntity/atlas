// SPDX-License-Identifier: AGPL-3.0-only
//! Model hooks around the serial path: install, sequence lifetime and the
//! prefill/decode context capture.

use super::*;

impl TransformerModel {
    /// Factory: install the observer built by [`load`].
    pub fn attach_exl3_dflash(&mut self, parts: Exl3DflashParts) -> Result<()> {
        ensure!(
            self.exl3_dflash.is_none(),
            "EXL3 DFlash observer already attached"
        );
        ensure!(
            self.proposer.is_none(),
            "EXL3 DFlash observer cannot coexist with a speculative proposer"
        );
        let observer = Exl3DflashObserver::build(parts, &self.config, self.gpu.as_ref())?;
        tracing::info!(
            "EXL3 DFlash draft-only observer ACTIVE on rank {} (k={}, capture after layers {:?}, \
             DFLASH_ACCEPT every {} steps)",
            observer.rank,
            observer.k,
            observer.capture_layers,
            observer.log_every
        );
        self.exl3_dflash = Some(Box::new(observer));
        Ok(())
    }

    /// New sequence in `seq.slot_idx`: fresh drafter state (any previous
    /// occupant is retired first). Never fails the allocation: on error the
    /// sequence simply runs without drafts.
    pub(crate) fn exl3_dflash_on_alloc(&self, seq: &SequenceState) {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return;
        };
        if let Err(error) = self.exl3_dflash_on_alloc_inner(obs, seq) {
            tracing::warn!(
                "EXL3 DFlash rank {} slot {}: no drafter state for this sequence: {error:#}",
                obs.rank,
                seq.slot_idx
            );
        }
    }

    fn exl3_dflash_on_alloc_inner(
        &self,
        obs: &Exl3DflashObserver,
        seq: &SequenceState,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();
        if let Some(old) = obs.seqs.lock().remove(&seq.slot_idx) {
            obs.retire(gpu, seq.slot_idx, old, "-");
        }
        let generation = obs.generation.fetch_add(1, Ordering::Relaxed);
        let owner = SequenceGeneration::new(seq.slot_idx, generation)?;
        let mut state = obs.head.alloc_state(gpu)?;
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .context("EXL3 DFlash: drafter state type")?;
        dstate.lifecycle = Some(CaptureDescriptor::bind(
            owner,
            0,
            0,
            obs.head.gamma,
            dstate.ctx_slot_bytes,
        )?);
        obs.seqs.lock().insert(
            seq.slot_idx,
            SeqObs {
                owner,
                state,
                scorer: DraftScorer::new(obs.k),
                pending_capture: None,
                chunk: None,
                broken: None,
                steps: 0,
                parity: VerifyParity::default(),
                observed_through: None,
                spec_steps: 0,
                spec_emitted: 0,
                rollback_checks: (0, 0, 0),
                think_budget_fired: 0,
                think_budget_kept: 0,
                think_budget_dropped: 0,
                spec_ms: [0.0; 3],
            },
        );
        Ok(())
    }

    /// Sequence released: final `DFLASH_ACCEPT` line and drafter cleanup.
    pub(crate) fn exl3_dflash_on_free(&self, seq: &SequenceState) {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return;
        };
        if obs.token_log && obs.rank == 0 && seq.tokens.len() > seq.prompt_len {
            tracing::info!(
                "DFLASH_TOKENS rank=0 req={} n={} tokens={:?}",
                request_label(seq),
                seq.tokens.len() - seq.prompt_len,
                &seq.tokens[seq.prompt_len..]
            );
        }
        if let Some(old) = obs.seqs.lock().remove(&seq.slot_idx) {
            obs.retire(self.gpu.as_ref(), seq.slot_idx, old, &request_label(seq));
        }
    }

    /// Prefill capture after layer `layer_idx` over `rows` highway rows at
    /// absolute positions `pos0..pos0 + rows`.
    /// Never fails the prefill: an error stops drafting for this sequence.
    pub(crate) fn exl3_dflash_prefill_layer(
        &self,
        seq: &SequenceState,
        layer_idx: usize,
        pos0: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return Ok(());
        };
        if let Err(error) =
            self.exl3_dflash_prefill_layer_inner(obs, seq, layer_idx, pos0, rows, stream)
        {
            tracing::warn!(
                "EXL3 DFlash rank {} slot {}: prefill capture failed at layer {layer_idx}: {error:#}",
                obs.rank,
                seq.slot_idx
            );
            if let Some(so) = obs.seqs.lock().get_mut(&seq.slot_idx) {
                so.broken = Some(format!("prefill capture: {error:#}"));
            }
        }
        Ok(())
    }

    fn exl3_dflash_prefill_layer_inner(
        &self,
        obs: &Exl3DflashObserver,
        seq: &SequenceState,
        layer_idx: usize,
        pos0: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(slot) = obs.capture_layers.iter().position(|&l| l == layer_idx) else {
            return Ok(());
        };
        let gpu = self.gpu.as_ref();
        let mut seqs = obs.seqs.lock();
        let Some(so) = seqs.get_mut(&seq.slot_idx) else {
            return Ok(());
        };
        if so.broken.is_some() {
            return Ok(());
        }
        if slot == 0 {
            so.chunk = None;
            let d = so.dstate();
            let end = pos0 + rows;
            let window_start = seq.tokens.len().max(end).saturating_sub(d.max_ctx_len);
            let first = pos0.max(window_start);
            if first >= end {
                return Ok(());
            }
            if let Some(&last) = d.ctx_positions.last()
                && last as usize >= first
            {
                so.broken = Some(format!(
                    "prefill row {first} does not follow context position {last}"
                ));
                return Ok(());
            }
            obs.make_room(gpu, d, end - first, stream)?;
            let acc_row = d.ctx_len;
            so.chunk = Some(ChunkRows {
                first,
                row0: first - pos0,
                rows: end - first,
                acc_row,
            });
        }
        let Some(chunk) = so.chunk else {
            return Ok(());
        };
        let acc = so.dstate().ctx_hidden_acc;
        obs.contract_rows(
            gpu,
            self.buffers.hc_streams(),
            chunk.row0,
            chunk.rows,
            acc.offset(chunk.acc_row * obs.slot_bytes + slot * obs.hidden * 2),
            obs.slot_bytes,
            stream,
        )?;
        if slot + 1 == obs.capture_layers.len() {
            let d = so.dstate();
            d.ctx_positions
                .extend((chunk.first..chunk.first + chunk.rows).map(|p| p as i32));
            d.ctx_len += chunk.rows;
            so.chunk = None;
            so.pending_capture = None;
        }
        Ok(())
    }

    /// Decode capture after layer `layer_idx` (row 0 of the highway) into the
    /// observer's save row. Stream-ordered and graph-capturable.
    pub(crate) fn exl3_dflash_decode_capture(&self, layer_idx: usize, stream: u64) -> Result<()> {
        let Some(obs) = self.exl3_dflash.as_deref() else {
            return Ok(());
        };
        let Some(slot) = obs.capture_layers.iter().position(|&l| l == layer_idx) else {
            return Ok(());
        };
        obs.contract_rows(
            self.gpu.as_ref(),
            self.buffers.hc_streams(),
            0,
            1,
            obs.save.offset(slot * obs.hidden * 2),
            obs.slot_bytes,
            stream,
        )
    }
}
