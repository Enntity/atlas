// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed four-stream DFlash2 proposal.

use anyhow::Result;
use std::mem::size_of;

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::layer::ForwardContext;
use crate::speculative::ProposerState;

pub(super) fn requested_block_rows(num_drafts: usize, configured_gamma: usize) -> usize {
    num_drafts.min(configured_gamma.saturating_sub(1)) + 1
}

impl BlockDiffusionDraftHead {
    pub(super) fn propose_drafts_batch(
        &self,
        last_tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        anyhow::ensure!(
            last_tokens.len() == states.len() && positions.len() == states.len(),
            "DFlash batch proposal inputs are not aligned"
        );
        let mut typed = states
            .iter_mut()
            .map(|state| {
                state
                    .as_any_mut()
                    .downcast_mut::<DflashProposerState>()
                    .ok_or_else(|| anyhow::anyhow!("invalid DFlash proposer state"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut paged = Vec::with_capacity(typed.len());
        for (sequence, (state, &position)) in typed.iter_mut().zip(positions).enumerate() {
            let (_, context_count) =
                self.prepare_paged_state(position, state, ctx, stream, None)?;
            anyhow::ensure!(
                state.block_table.len() <= self.scratch.option_b_max_blocks,
                "DFlash block table exceeds graph-stable capacity"
            );
            let bytes = state
                .block_table
                .iter()
                .flat_map(|block| block.to_le_bytes())
                .collect::<Vec<_>>();
            let stable = self
                .scratch
                .option_b_block_tables_dev
                .offset(sequence * self.scratch.option_b_max_blocks * size_of::<u32>());
            ctx.gpu.copy_h2d_async(&bytes, stable, stream)?;
            paged.push((stable, context_count));
        }
        let block_rows = requested_block_rows(num_drafts, self.gamma);
        let cap = block_rows - 1;
        let mut drafts =
            self.forward_block_batch(last_tokens, positions, &paged, block_rows, ctx, stream)?;
        for (state, sequence_drafts) in typed.iter_mut().zip(drafts.iter_mut()) {
            sequence_drafts.truncate(cap);
            state.last_num_drafted = sequence_drafts.len();
            // Greedy rejection compares exact token ids; sparse proposal
            // distributions are only required for stochastic sampling. Never
            // let a distribution from a prior serial proposal leak here.
            state.last_candidate_ids.clear();
            state.last_candidate_scores.clear();
        }
        Ok(drafts)
    }
}

#[cfg(test)]
mod tests {
    use super::requested_block_rows;

    #[test]
    fn concurrent_depth_uses_only_requested_drafter_rows() {
        assert_eq!(requested_block_rows(7, 17), 8);
        assert_eq!(requested_block_rows(16, 17), 17);
        assert_eq!(requested_block_rows(32, 17), 17);
    }
}
