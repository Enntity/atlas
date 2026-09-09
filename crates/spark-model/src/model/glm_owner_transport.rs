// SPDX-License-Identifier: AGPL-3.0-only
//! Selected E7 exchange; the first header attempt begins terminal ownership.
use super::{
    TransformerModel,
    glm_owner_wire::{self as wire, Bounds, OwnerRecord, PAYLOAD_WORDS, Packet},
};
use crate::{
    layer::glm_owner_verify::GlmOwnerBatchShape,
    traits::{Model, SequenceState},
};
use anyhow::{Context, Result, ensure};

impl TransformerModel {
    pub(in crate::model) fn owner_packet(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &[&SequenceState],
        tokens: &[[u32; 5]],
    ) -> Result<[u32; PAYLOAD_WORDS]> {
        self.owner_compute_preflight(shape, seqs, tokens)?;
        match shape.owners() {
            3 => self.owner_packet_fixed::<3>(shape, seqs.try_into()?, tokens),
            4 => self.owner_packet_fixed::<4>(shape, seqs.try_into()?, tokens),
            _ => anyhow::bail!("E7 owner shape changed"),
        }
    }

    fn owner_packet_fixed<const N: usize>(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &[&SequenceState; N],
        tokens: &[[u32; 5]],
    ) -> Result<[u32; PAYLOAD_WORDS]> {
        let mode = self
            .glm_owner_verify_mode
            .context("E7 cold owner mode missing")?;
        let mut inputs = std::array::from_fn::<_, N, _>(|_| None);
        let mut states = std::array::from_fn::<_, N, _>(|_| None);
        for (i, seq) in seqs.iter().enumerate() {
            inputs[i] = Some(self.paired_input(seq)?);
            states[i] = Some(
                seq.proposer_state
                    .as_deref()
                    .context("E7 proposer missing")?,
            );
        }
        let inputs = inputs.map(|v| v.expect("all actual inputs validated"));
        let states = states.map(|v| v.expect("all actual states validated"));
        let facts = self
            .paired_handoff()
            .context("E7 handoff missing")?
            .validate_verify_owners(shape, &inputs, tokens, &states, &self.glm_repair_context())?;
        let mut owners = [None; 4];
        for i in 0..N {
            owners[i] = Some(OwnerRecord {
                slot: u32::try_from(seqs[i].slot_idx)?,
                generation: facts[i].0,
                attempt: facts[i].1,
                base: u32::try_from(seqs[i].seq_len)?,
                tokens: tokens[i],
            });
        }
        Packet {
            shape,
            mode,
            owners,
        }
        .encode(self.owner_wire_bounds()?)
    }

    fn owner_wire_bounds(&self) -> Result<Bounds> {
        Ok(Bounds {
            capacity: self.paired_owner_capacity()?,
            vocab_size: self.config.vocab_size,
            // Explicit K5 target envelope; actual private/context limits are
            // additionally checked by owner_compute_preflight on both ranks.
            context_tokens: 2048,
        })
    }

    /// No host/device allocation and no silent local fallback. Only E7's two
    /// fixed extents are admitted; stack byte encoding avoids pointer casts.
    fn owner_exchange_words<const N: usize>(&self, words: &mut [u32; N]) -> Result<()> {
        ensure!(
            N == PAYLOAD_WORDS || N == wire::VERDICT_WORDS,
            "E7 exchange extent"
        );
        let comm = self.comm.as_ref().context("E7 communicator missing")?;
        self.paired_wire_profile(comm.rank())?;
        let count = N * 4;
        let scratch = self.buffers.scratch();
        ensure!(
            !scratch.is_null() && self.buffers.sizes().scratch >= count,
            "E7 staging extent"
        );
        let mut bytes = [0u8; PAYLOAD_WORDS * 4];
        if comm.rank() == 0 {
            for (word, dst) in words.iter().zip(bytes[..count].chunks_exact_mut(4)) {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            self.gpu.copy_h2d(&bytes[..count], scratch)?;
        }
        comm.broadcast(scratch.0, count, 0)?;
        if comm.rank() == 1 {
            self.gpu.synchronize(self.gpu.default_stream())?;
            self.gpu.copy_d2h(scratch, &mut bytes[..count])?;
            for (word, src) in words.iter_mut().zip(bytes[..count].chunks_exact(4)) {
                *word = u32::from_le_bytes(src.try_into()?);
            }
        }
        Ok(())
    }

    pub(in crate::model) fn owner_send_verify(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &mut [&mut SequenceState],
        tokens: &[[u32; 5]],
    ) -> Result<[[u32; 5]; 4]> {
        self.paired_wire_profile(0)?;
        ensure!(seqs.len() == shape.owners(), "E7 head owner count");
        // Only the live prefix is passed, never the duplicate inactive borrow.
        let mut borrowed = [&*seqs[0]; 4];
        for (i, seq) in seqs.iter().enumerate() {
            borrowed[i] = seq;
        }
        let mut packet = self.owner_packet(shape, &borrowed[..shape.owners()], tokens)?;
        (|| {
            self.ep_broadcast_seq_and_cmd(0, wire::EP_GLM_OWNER_VERIFY, true)?;
            self.owner_exchange_words(&mut packet)?;
            self.sync_secondary()?;
            self.owner_compute_verify(shape, seqs, tokens)
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    pub(in crate::model) fn owner_send_verdict(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &mut [&mut SequenceState],
        tokens: &[[u32; 5]],
        accepted: &[usize],
    ) -> Result<()> {
        (|| {
            self.paired_wire_profile(0)?;
            ensure!(seqs.len() == shape.owners(), "E7 verdict owner count");
            self.glm_owner_verify_mode
                .context("E7 cold owner mode missing")?;
            let mut borrowed = [&*seqs[0]; 4];
            for (i, seq) in seqs.iter().enumerate() {
                borrowed[i] = seq;
            }
            self.owner_verdict_preflight(shape, &borrowed[..shape.owners()], tokens, accepted)?;
            let mut counts = [0; 4];
            counts[..shape.owners()].copy_from_slice(accepted);
            let mut packet = wire::encode_verdict(shape, counts)?;
            self.owner_exchange_words(&mut packet)?;
            self.owner_finish_verify(shape, seqs, tokens, accepted)?;
            self.owner_log_commit(shape);
            Ok(())
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    pub(in crate::model) fn owner_receive_verify(
        &self,
        preamble: u32,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        (|| {
            self.paired_wire_profile(1)?;
            ensure!(
                preamble == 0 && slots.len() == self.paired_owner_capacity()?,
                "E7 worker preamble/registry mismatch"
            );
            let mode = self
                .glm_owner_verify_mode
                .context("E7 cold owner mode missing")?;
            let mut payload = [0; PAYLOAD_WORDS];
            self.owner_exchange_words(&mut payload)?;
            let packet = Packet::decode(&payload, self.owner_wire_bounds()?)?;
            ensure!(packet.mode == mode, "E7 worker compute mode mismatch");
            let count = packet.shape.owners();
            let mut tokens = [[0; 5]; 4];
            let mut selected: [Option<&mut SequenceState>; 4] = std::array::from_fn(|_| None);
            let mut ordinal = 0;
            for (physical, entry) in slots.iter_mut().enumerate() {
                if ordinal == count {
                    break;
                }
                let record = packet.owners[ordinal].context("E7 live packet owner missing")?;
                if physical != record.slot as usize {
                    continue;
                }
                let seq = entry.as_mut().context("E7 worker actual owner missing")?;
                ensure!(
                    seq.slot_idx == physical,
                    "E7 worker physical registry identity mismatch"
                );
                tokens[ordinal] = record.tokens;
                selected[ordinal] = Some(seq);
                ordinal += 1;
            }
            ensure!(ordinal == count, "E7 worker owner coverage mismatch");
            let [a, b, c, d] = selected;
            let (a, b, c) = (
                a.context("E7 owner0")?,
                b.context("E7 owner1")?,
                c.context("E7 owner2")?,
            );
            match d {
                Some(d) => self.owner_receive_selected(
                    packet.shape,
                    &mut [a, b, c, d],
                    &tokens[..count],
                    &payload,
                ),
                None => self.owner_receive_selected(
                    packet.shape,
                    &mut [a, b, c],
                    &tokens[..count],
                    &payload,
                ),
            }
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    fn owner_receive_selected(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &mut [&mut SequenceState],
        tokens: &[[u32; 5]],
        payload: &[u32; PAYLOAD_WORDS],
    ) -> Result<bool> {
        let mut borrowed = [&*seqs[0]; 4];
        for (i, seq) in seqs.iter().enumerate() {
            borrowed[i] = seq;
        }
        let expected = self.owner_packet(shape, &borrowed[..shape.owners()], tokens)?;
        ensure!(
            expected == *payload,
            "E7 actual mode/owner/generation/attempt/token mismatch"
        );
        self.sync_secondary()?;
        self.owner_compute_verify(shape, seqs, tokens)?;
        let mut verdict = [0; wire::VERDICT_WORDS];
        self.owner_exchange_words(&mut verdict)?;
        let accepted = wire::decode_verdict(&verdict, shape)?;
        self.owner_finish_verify(shape, seqs, tokens, &accepted[..shape.owners()])?;
        self.owner_log_commit(shape);
        Ok(true)
    }

    fn owner_log_commit(&self, shape: GlmOwnerBatchShape) {
        let key = if shape.owners() == 3 {
            "log:glm_e7_c3_committed"
        } else {
            "log:glm_e7_c4_committed"
        };
        if self.stats.once(key) {
            tracing::info!(
                rank = self.config.ep_rank,
                mode = ?self.glm_owner_verify_mode,
                owners = shape.owners(),
                rows = shape.rows(),
                "GLM E7 local owner verification committed"
            );
        }
    }
}
