// SPDX-License-Identifier: AGPL-3.0-only
//! Selected commands own preflight-to-wire ordering; not native fault containment.
use super::*;
use crate::traits::Model;

const SELECTED_E1_VERSION: u32 = 1;

impl TransformerModel {
    pub(in crate::model) fn paired_wire_profile(&self, rank: usize) -> Result<()> {
        let comm = self
            .comm
            .as_ref()
            .context("paired wire communicator missing")?;
        ensure!(
            self.paired_handoff().is_some()
                && self.ep_protocol_v2
                && self.levers.max_decode_seqs == 2
                && crate::layers::glm5_mtp::distributed_enabled()
                && comm.world_size() == 2
                && comm.rank() == rank
                && !self.ep_cmd_buf.is_null(),
            "paired transport requires actual selected EP-v2 two-slot rank"
        );
        Ok(())
    }

    /// First-header failure can precede any Verification. Do not merely poison
    /// the addressed request: the peer may already be waiting inside a command.
    pub(in crate::model) fn paired_transport_error(&self, error: anyhow::Error) -> anyhow::Error {
        let latched = self
            .paired_handoff()
            .context("paired transport capability missing")
            .and_then(|capability| capability.fail_session(self.gpu.as_ref()));
        error.context(format!(
            "paired issued command is terminal; latch={latched:?}"
        ))
    }

    fn paired_proposal_packet(
        &self,
        seq: &SequenceState,
        seed: u32,
        position: usize,
        drafts: usize,
        grammar: bool,
    ) -> Result<[u32; 8]> {
        let (generation, attempt) =
            self.paired_validate_propose(seq, seed, position, drafts, grammar)?;
        Ok([
            SELECTED_E1_VERSION,
            generation as u32,
            (generation >> 32) as u32,
            attempt as u32,
            (attempt >> 32) as u32,
            u32::try_from(position)?,
            4,
            seed,
        ])
    }

    pub(in crate::model) fn paired_send_propose(
        &self,
        seq: &mut SequenceState,
        seed: u32,
        position: usize,
        drafts: usize,
        grammar: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.paired_wire_profile(0)?;
        let payload =
            self.paired_proposal_packet(seq, seed, position, drafts, grammar.is_some())?;
        let slot = u32::try_from(seq.slot_idx)?;
        // No fallible preflight below this line is considered safely retryable.
        // B2 must arm its process-fatal guard before entering this first header.
        (|| {
            self.ep_broadcast_seq_and_cmd(slot, 0xffffffe1, true)?;
            self.ep_broadcast_tokens(&payload)?;
            let result = self.run_mtp_propose_inner(seed, position, 4, seq, None)?;
            ensure!(
                result.len() == 4,
                "paired E1 returned a non-four draft count"
            );
            Ok(result)
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    pub(in crate::model) fn paired_send_verify(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
    ) -> Result<Vec<u32>> {
        self.paired_wire_profile(0)?;
        self.paired_validate_verify(seq, tokens)?;
        let slot = u32::try_from(seq.slot_idx)?;
        (|| {
            self.ep_broadcast_seq_and_cmd(slot, 0xfffffff5, true)?;
            self.ep_broadcast_u32(5)?;
            self.ep_broadcast_tokens(tokens)?;
            self.sync_secondary()?;
            self.decode_verify_graphed_kgamma(tokens, seq, self.gpu.default_stream())
        })()
        .map_err(|error| self.paired_transport_error(error))
    }

    pub(in crate::model) fn paired_receive_propose(&self, seq: &mut SequenceState) -> Result<bool> {
        // The worker has already received the preamble. Every failure here is
        // terminal, including an invalid format before local proposal claim.
        (|| {
            self.paired_wire_profile(1)?;
            let payload = self.ep_broadcast_tokens(&[0; 8])?;
            ensure!(
                payload.len() == 8 && payload[0] == SELECTED_E1_VERSION && payload[6] == 4,
                "paired E1 version or fixed width mismatch"
            );
            let seed = payload[7];
            let position = payload[5] as usize;
            let expected = self.paired_proposal_packet(seq, seed, position, 4, false)?;
            ensure!(
                payload == expected,
                "paired E1 owner generation/attempt/position mismatch"
            );
            let drafts = self.run_mtp_propose_inner(seed, position, 4, seq, None)?;
            ensure!(
                drafts.len() == 4,
                "paired worker E1 returned a non-four draft count"
            );
            Ok(true)
        })()
        .map_err(|error| self.paired_transport_error(error))
    }
}
