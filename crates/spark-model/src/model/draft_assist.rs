// SPDX-License-Identifier: AGPL-3.0-only

//! Rank-split propose (`ATLAS_GLM_DRAFT_TP`), model side.
//!
//! The head announces a single-sequence propose that will swap halves with
//! the worker (`layers::dflash_head::rank_split`); the worker serves them
//! from its own copy of the drafter, which proposes nothing.
//!
//! Wire protocol, v2-addressed:
//!
//! ```text
//! EC  (seq_id 0)  -> the worker enqueues its walk of the propose's swaps
//! ```
//!
//! The command carries no payload: both ranks plan the swaps from the same
//! drafter shapes and the same agreed switch (`startup_parity`). The head
//! sends it only outside any other command, so the worker is at its command
//! loop; the worker enqueues and returns there, its next command ordered
//! behind the walk on the same stream.

use std::sync::Arc;

use anyhow::{Context, Result};
use spark_comm::CommBackend;

use super::types::TransformerModel;
use crate::layers::BlockDiffusionDraftHead;
use crate::speculative::DraftProposer;

pub(super) const EP_CMD_DRAFT_ASSIST: u32 = 0xFFFF_FFEC;

impl TransformerModel {
    /// Install the worker rank's copy of the drafter: it serves split
    /// proposes and is never the proposer.
    pub fn set_draft_assist(&mut self, head: Arc<BlockDiffusionDraftHead>) {
        self.draft_assist = Some(head);
    }

    /// Head: the communicator a propose swaps over, when it will split.
    pub(super) fn draft_split_comm(
        &self,
        proposer: &dyn DraftProposer,
        grammar: bool,
    ) -> Option<&dyn CommBackend> {
        let comm = self.comm_ref()?;
        (self.multi_rank_protocol_active() && proposer.rank_split_ready(comm, grammar))
            .then_some(comm)
    }

    /// Head: announce the split propose about to run.
    pub(super) fn announce_draft_split(&self) -> Result<()> {
        self.ep_broadcast_seq_and_cmd(0, EP_CMD_DRAFT_ASSIST, self.ep_protocol_v2)
    }

    /// Worker side of EC.
    pub(super) fn draft_assist_serve(&self) -> Result<bool> {
        let head = self
            .draft_assist
            .as_ref()
            .context("rank-split propose announced, but this rank holds no drafter")?;
        let comm = self
            .comm_ref()
            .context("rank-split propose without a communicator")?;
        head.rank_split_serve(self.gpu.as_ref(), comm, self.gpu.default_stream())?;
        Ok(true)
    }
}
