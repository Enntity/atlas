// SPDX-License-Identifier: AGPL-3.0-only

//! Request-owned phase. A verified verdict is not inferred from legacy trim.

use crate::speculative::glm_pair_plan::{Finish, FinishPlan, ProposalPlan, VerifiedCommit};
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, Default)]
pub(super) enum RepairPhase {
    #[default]
    Capture,
    Proposed(ProposalPlan),
    Pending(PendingRepair),
    Failed,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PendingRepair {
    pub plan: FinishPlan,
    pub tokens: [u32; 5],
    pub drafts: usize,
    pub cached_rows: usize,
    trim_seen: bool,
}

impl RepairPhase {
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        generation: u64,
        capture_generation: u64,
        base: usize,
        tokens: &[u32],
        accepted: usize,
        position: usize,
        cached_rows: usize,
        hidden_rows: usize,
    ) -> Result<()> {
        let Self::Proposed(proposal) = *self else {
            anyhow::bail!("GLM verified event requires an outstanding proposal");
        };
        let plan = proposal.finish(Finish::Verified(VerifiedCommit {
            generation,
            capture_generation,
            accepted,
            verify_token_rows: tokens.len(),
            normalized_hidden_rows: hidden_rows,
            target_position: position,
            observed_cache_rows: cached_rows,
            hidden_base_position: base,
        }))?;
        // The checked proposal fixes the valid prefix; paired storage stays K5.
        let mut owned_tokens = [0; 5];
        owned_tokens[..tokens.len()].copy_from_slice(tokens);
        *self = Self::Pending(PendingRepair {
            plan,
            tokens: owned_tokens,
            drafts: proposal.drafts(),
            cached_rows,
            trim_seen: false,
        });
        Ok(())
    }

    pub fn acknowledge(&mut self, accepted: usize) -> Result<()> {
        let Self::Pending(pending) = self else {
            anyhow::bail!("GLM trim has no explicit verified verdict");
        };
        ensure!(
            !pending.trim_seen && pending.plan.bonus_hidden_row() == Some(accepted),
            "GLM trim mismatches or duplicates verified verdict"
        );
        pending.trim_seen = true;
        Ok(())
    }

    pub fn pending(
        self,
        generation: u64,
        position: usize,
        hidden_row: usize,
    ) -> Result<PendingRepair> {
        let Self::Pending(pending) = self else {
            anyhow::bail!("GLM E1 requires pending verified state");
        };
        ensure!(
            pending.trim_seen,
            "GLM E1 precedes verified trim acknowledgement"
        );
        ensure!(
            pending.plan.state().generation() == generation,
            "GLM E1 generation is stale"
        );
        ensure!(
            pending.plan.state().target_position() == position,
            "GLM E1 position is stale"
        );
        ensure!(
            pending.plan.bonus_hidden_row() == Some(hidden_row),
            "GLM E1 bonus hidden row is stale"
        );
        Ok(pending)
    }
}

#[cfg(test)]
#[path = "repair_state_tests.rs"]
mod tests;
