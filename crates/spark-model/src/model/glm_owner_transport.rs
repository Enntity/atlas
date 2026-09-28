// SPDX-License-Identifier: AGPL-3.0-only
//! Selected E7/E8 exchange; the first header attempt begins terminal ownership.
use super::{
    TransformerModel,
    glm_owner_wire::{self as wire, Bounds, OwnerRecord},
};
use crate::{
    layer::glm_owner_verify::GlmOwnerBatchShape,
    traits::{Model, SequenceState},
};
use anyhow::{Context, Result, ensure};

#[path = "glm_owner_transport_words.rs"]
mod words;
use words::Payload;

impl TransformerModel {
    pub(in crate::model) fn owner_packet(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &[&SequenceState],
        tokens: &[[u32; 5]],
    ) -> Result<Payload> {
        self.owner_compute_preflight(shape, seqs, tokens)?;
        match shape.owners() {
            3 => self.owner_packet_fixed::<3>(shape, seqs.try_into()?, tokens),
            4 => self.owner_packet_fixed::<4>(shape, seqs.try_into()?, tokens),
            5 => self.owner_packet_fixed::<5>(shape, seqs.try_into()?, tokens),
            6 => self.owner_packet_fixed::<6>(shape, seqs.try_into()?, tokens),
            7 => self.owner_packet_fixed::<7>(shape, seqs.try_into()?, tokens),
            8 => self.owner_packet_fixed::<8>(shape, seqs.try_into()?, tokens),
            _ => anyhow::bail!("owner transport shape changed"),
        }
    }

    fn owner_packet_fixed<const N: usize>(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &[&SequenceState; N],
        tokens: &[[u32; 5]],
    ) -> Result<Payload> {
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
        let mut owners = [None; 8];
        for i in 0..N {
            owners[i] = Some(OwnerRecord {
                slot: u32::try_from(seqs[i].slot_idx)?,
                generation: facts[i].0,
                attempt: facts[i].1,
                base: u32::try_from(seqs[i].seq_len)?,
                tokens: tokens[i],
            });
        }
        Payload::encode(shape, mode, owners, self.owner_wire_bounds()?)
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
    /// and E8's fixed extents are admitted; stack encoding avoids pointer casts.
    fn owner_exchange_words<const N: usize>(&self, words: &mut [u32; N]) -> Result<()> {
        ensure!(
            N == wire::PAYLOAD_WORDS
                || N == wire::VERDICT_WORDS
                || N == super::glm_owner8_wire::PAYLOAD_WORDS
                || N == super::glm_owner8_wire::VERDICT_WORDS,
            "owner exchange extent"
        );
        let comm = self.comm.as_ref().context("E7 communicator missing")?;
        self.paired_wire_profile(comm.rank())?;
        let count = N * 4;
        let scratch = self.buffers.scratch();
        ensure!(
            !scratch.is_null() && self.buffers.sizes().scratch >= count,
            "E7 staging extent"
        );
        let mut bytes = [0u8; super::glm_owner8_wire::PAYLOAD_WORDS * 4];
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
    ) -> Result<[[u32; 5]; 8]> {
        self.paired_wire_profile(0)?;
        ensure!(seqs.len() == shape.owners(), "E7 head owner count");
        // Only the live prefix is passed, never the duplicate inactive borrow.
        let mut borrowed = [&*seqs[0]; 8];
        for (i, seq) in seqs.iter().enumerate() {
            borrowed[i] = seq;
        }
        let mut packet = self.owner_packet(shape, &borrowed[..shape.owners()], tokens)?;
        (|| {
            self.ep_broadcast_seq_and_cmd(0, packet.command(), true)?;
            packet.exchange(self)?;
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
            let mut borrowed = [&*seqs[0]; 8];
            for (i, seq) in seqs.iter().enumerate() {
                borrowed[i] = seq;
            }
            self.owner_verdict_preflight(shape, &borrowed[..shape.owners()], tokens, accepted)?;
            words::exchange_verdict(self, shape, Some(accepted))?;
            self.owner_finish_verify(shape, seqs, tokens, accepted)?;
            self.owner_log_commit(shape);
            Ok(())
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    pub(in crate::model) fn owner_receive_verify(
        &self,
        preamble: u32,
        command: u32,
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
            let mut payload = Payload::empty(command)?;
            payload.exchange(self)?;
            let packet = payload.decode(self.owner_wire_bounds()?)?;
            ensure!(packet.mode == mode, "E7 worker compute mode mismatch");
            let count = packet.shape.owners();
            let mut tokens = [[0; 5]; 8];
            let mut selected: [Option<&mut SequenceState>; 8] = std::array::from_fn(|_| None);
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
            match count {
                3 => self.owner_receive_fixed::<3>(
                    packet.shape,
                    selected,
                    &tokens[..count],
                    &payload,
                ),
                4 => self.owner_receive_fixed::<4>(
                    packet.shape,
                    selected,
                    &tokens[..count],
                    &payload,
                ),
                5 => self.owner_receive_fixed::<5>(
                    packet.shape,
                    selected,
                    &tokens[..count],
                    &payload,
                ),
                6 => self.owner_receive_fixed::<6>(
                    packet.shape,
                    selected,
                    &tokens[..count],
                    &payload,
                ),
                7 => self.owner_receive_fixed::<7>(
                    packet.shape,
                    selected,
                    &tokens[..count],
                    &payload,
                ),
                8 => self.owner_receive_fixed::<8>(
                    packet.shape,
                    selected,
                    &tokens[..count],
                    &payload,
                ),
                _ => anyhow::bail!("owner worker shape changed"),
            }
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    fn owner_receive_fixed<const N: usize>(
        &self,
        shape: GlmOwnerBatchShape,
        mut selected: [Option<&mut SequenceState>; 8],
        tokens: &[[u32; 5]],
        payload: &Payload,
    ) -> Result<bool> {
        ensure!(
            N == shape.owners()
                && selected[..N].iter().all(Option::is_some)
                && selected[N..].iter().all(Option::is_none),
            "owner worker fixed coverage changed"
        );
        let mut seqs: [&mut SequenceState; N] = std::array::from_fn(|i| {
            selected[i]
                .take()
                .expect("complete distinct owner coverage")
        });
        self.owner_receive_selected(shape, &mut seqs, tokens, payload)
    }

    fn owner_receive_selected(
        &self,
        shape: GlmOwnerBatchShape,
        seqs: &mut [&mut SequenceState],
        tokens: &[[u32; 5]],
        payload: &Payload,
    ) -> Result<bool> {
        let mut borrowed = [&*seqs[0]; 8];
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
        let accepted = words::exchange_verdict(self, shape, None)?;
        self.owner_finish_verify(shape, seqs, tokens, &accepted[..shape.owners()])?;
        self.owner_log_commit(shape);
        Ok(true)
    }

    fn owner_log_commit(&self, shape: GlmOwnerBatchShape) {
        let key = match shape.owners() {
            3 => "log:glm_e7_c3_committed",
            4 => "log:glm_e7_c4_committed",
            5 => "log:glm_e8_c5_committed",
            6 => "log:glm_e8_c6_committed",
            7 => "log:glm_e8_c7_committed",
            8 => "log:glm_e8_c8_committed",
            _ => unreachable!("validated owner shape"),
        };
        if self.stats.once(key) {
            tracing::info!(
                rank = self.config.ep_rank,
                mode = ?self.glm_owner_verify_mode,
                owners = shape.owners(),
                rows = shape.rows(),
                "{}",
                if shape.owners() <= 4 {
                    "GLM E7 local owner verification committed"
                } else {
                    "GLM E8 local owner verification committed"
                }
            );
        }
    }
}
