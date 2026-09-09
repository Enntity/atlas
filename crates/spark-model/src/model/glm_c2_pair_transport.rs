// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed temporal pair command; every post-header error remains session-terminal.
use super::TransformerModel;
use crate::layer::glm_pair_verify::{GlmPairFfn, GlmPairWorkspace};
use crate::traits::{Model, SequenceState};
use anyhow::{Context, Result, ensure};

pub(in crate::model) const EP_GLM_PAIR_VERIFY: u32 = 0xffff_ffe6;
const VERSION: u32 = 1;
const WORDS: usize = 26; // version/owners/rows/compute mode + two 11-word owners.

impl TransformerModel {
    pub(in crate::model) fn paired_pair_packet(
        &self,
        seqs: [&SequenceState; 2],
        tokens: &[[u32; 5]; 2],
    ) -> Result<[u32; WORDS]> {
        let mode = self
            .glm_pair_verify_mode
            .context("joint verification disabled")?;
        crate::layers::glm5_mtp::Glm5MtpHead::validate_fixed_pair_slots(
            [seqs[0].slot_idx, seqs[1].slot_idx],
            self.paired_owner_capacity()?,
        )?;
        GlmPairWorkspace::new(&self.glm_repair_context(), mode)?;
        for seq in seqs {
            self.paired_target_preflight(seq, 5)?;
        }
        {
            let cache = self.kv_cache.lock();
            let mut additional = 0usize;
            for seq in seqs {
                let (_, needed) = self.paired_target_budget(seq, &cache)?;
                additional = additional
                    .checked_add(needed.saturating_sub(seq.block_table.len()))
                    .context("paired aggregate target budget overflow")?;
            }
            ensure!(
                additional <= cache.num_free_blocks(),
                "paired aggregate target budget exhausted"
            );
            ensure!(
                seqs[0]
                    .block_table
                    .iter()
                    .all(|b| !seqs[1].block_table.contains(b)),
                "paired historical target maps alias"
            );
        }
        let inputs = [self.paired_input(seqs[0])?, self.paired_input(seqs[1])?];
        let states = [
            seqs[0]
                .proposer_state
                .as_deref()
                .context("paired proposer0 missing")?,
            seqs[1]
                .proposer_state
                .as_deref()
                .context("paired proposer1 missing")?,
        ];
        let facts = self
            .paired_handoff()
            .context("paired head missing")?
            .validate_verify_pair(&inputs, tokens, states, &self.glm_repair_context())?;
        let mut packet = [0; WORDS];
        packet[..4].copy_from_slice(&[
            VERSION,
            2,
            10,
            match mode {
                GlmPairFfn::TwoK5 => 1,
                GlmPairFfn::Joint => 2,
                GlmPairFfn::JointSharedM10 => 3,
            },
        ]);
        for owner in 0..2 {
            let start = 4 + owner * 11;
            let (generation, attempt) = facts[owner];
            packet[start..start + 6].copy_from_slice(&[
                u32::try_from(seqs[owner].slot_idx)?,
                generation as u32,
                (generation >> 32) as u32,
                attempt as u32,
                (attempt >> 32) as u32,
                u32::try_from(seqs[owner].seq_len)?,
            ]);
            packet[start + 6..start + 11].copy_from_slice(&tokens[owner]);
        }
        Ok(packet)
    }

    pub(in crate::model) fn paired_send_verify_pair(
        &self,
        seqs: [&mut SequenceState; 2],
        tokens: &[[u32; 5]; 2],
    ) -> Result<[[u32; 5]; 2]> {
        self.paired_wire_profile(0)?;
        let packet = self.paired_pair_packet([&*seqs[0], &*seqs[1]], tokens)?;
        let group_base = u32::try_from(seqs[0].slot_idx)?;
        (|| {
            self.ep_broadcast_seq_and_cmd(group_base, EP_GLM_PAIR_VERIFY, true)?;
            self.ep_broadcast_tokens(&packet)?;
            self.sync_secondary()?;
            self.paired_compute_verify(seqs, tokens)
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    pub(in crate::model) fn paired_receive_verify_pair(
        &self,
        preamble_slot: u32,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        (|| {
            self.paired_wire_profile(1)?;
            let capacity = self.paired_owner_capacity()?;
            ensure!(
                slots.len() == capacity,
                "paired worker registry/capacity mismatch"
            );
            let base = usize::try_from(preamble_slot)?;
            let second = base
                .checked_add(1)
                .context("paired worker group overflow")?;
            crate::layers::glm5_mtp::Glm5MtpHead::validate_fixed_pair_slots(
                [base, second],
                capacity,
            )?;
            let (left, right) = slots[base..=second].split_at_mut(1);
            let s0 = left[0].as_mut().context("paired worker owner0 missing")?;
            let s1 = right[0].as_mut().context("paired worker owner1 missing")?;
            ensure!(
                s0.slot_idx == base && s1.slot_idx == second,
                "paired worker physical registry differs from preamble"
            );
            let payload = self.ep_broadcast_tokens(&[0; WORDS])?;
            ensure!(payload.len() == WORDS, "paired worker payload extent");
            let tokens: [[u32; 5]; 2] = std::array::from_fn(|owner| {
                payload[10 + owner * 11..15 + owner * 11]
                    .try_into()
                    .expect("fixed payload slice")
            });
            let expected = self.paired_pair_packet([s0, s1], &tokens)?;
            ensure!(
                payload == expected,
                "paired worker mode/owner/generation/attempt/token mismatch"
            );
            self.sync_secondary()?;
            self.paired_compute_verify([s0, s1], &tokens)?;
            let acknowledgement = self.ep_broadcast_tokens(&[0; 4])?;
            ensure!(
                acknowledgement.len() == 4
                    && acknowledgement[..2] == [VERSION, 2]
                    && acknowledgement[2] <= 4
                    && acknowledgement[3] <= 4,
                "paired worker verdict header/count mismatch"
            );
            self.paired_finish_local(
                [s0, s1],
                &tokens,
                [acknowledgement[2] as usize, acknowledgement[3] as usize],
            )?;
            Ok(true)
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    pub(in crate::model) fn paired_send_pair_verdict(
        &self,
        seqs: [&mut SequenceState; 2],
        tokens: &[[u32; 5]; 2],
        accepted: [usize; 2],
    ) -> Result<()> {
        // This call follows an issued pair; even a local malformed verdict is
        // terminal, not a retryable preflight failure.
        (|| {
            self.paired_wire_profile(0)?;
            self.paired_verdict_bases([&*seqs[0], &*seqs[1]], tokens, accepted)?;
            self.ep_broadcast_tokens(&[VERSION, 2, accepted[0] as u32, accepted[1] as u32])?;
            self.paired_finish_local(seqs, tokens, accepted)
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    fn paired_verdict_bases(
        &self,
        seqs: [&SequenceState; 2],
        tokens: &[[u32; 5]; 2],
        accepted: [usize; 2],
    ) -> Result<[usize; 2]> {
        crate::layers::glm5_mtp::Glm5MtpHead::validate_fixed_pair_slots(
            [seqs[0].slot_idx, seqs[1].slot_idx],
            self.paired_owner_capacity()?,
        )?;
        ensure!(
            accepted.iter().all(|&a| a <= 4),
            "paired accepted count exceeds four"
        );
        let mut bases = [0; 2];
        for owner in 0..2 {
            let seq = seqs[owner];
            bases[owner] = seq
                .seq_len
                .checked_sub(5)
                .context("paired verdict lacks five rows")?;
            ensure!(
                seq.tokens.len() == seq.seq_len
                    && seq.tokens.get(bases[owner]..) == Some(tokens[owner].as_slice()),
                "paired verdict canonical issued append changed"
            );
            self.paired_ssm_bindings(seq)?;
        }
        Ok(bases)
    }

    fn paired_finish_local(
        &self,
        mut seqs: [&mut SequenceState; 2],
        tokens: &[[u32; 5]; 2],
        accepted: [usize; 2],
    ) -> Result<()> {
        let bases = self.paired_verdict_bases([&*seqs[0], &*seqs[1]], tokens, accepted)?;
        for owner in 0..2 {
            seqs[owner].seq_len = bases[owner] + accepted[owner] + 1;
            let end = seqs[owner].seq_len;
            seqs[owner].tokens.truncate(end);
        }
        let capability = self
            .paired_handoff()
            .context("paired verdict head missing")?;
        self.paired_with_states(&mut seqs, |inputs, states, ctx| {
            capability.record_verify_pair(inputs, bases, tokens, accepted, states, ctx)
        })?;
        for owner in 0..2 {
            self.trim_proposer_state(seqs[owner], accepted[owner], 0)?;
            self.commit_accepted_prefix(seqs[owner], accepted[owner] + 1, 5)?;
        }
        self.sync_secondary()?;
        crate::speculative::glm_paired_execution::GlmPairedExecution::check_communication_health(
            self,
        )?;
        if self.stats.once("log:glm_e6_committed") {
            tracing::info!(
                rank = self.config.ep_rank,
                mode = ?self.glm_pair_verify_mode,
                owners = 2,
                rows = 10,
                "GLM E6 local paired verification committed"
            );
        }
        Ok(())
    }
}
