// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix-cache snapshot policy for hybrid SSM models: the restore-depth rank
//! agreement and branch-point checkpoints.
//!
//! # Branch-point checkpoints (`ATLAS_GLM_PC_BRANCH=1`, default off)
//!
//! A prefill saves one SSM checkpoint per turn: the tail split, one block
//! below the last block boundary under the prompt end. That serves the next
//! turn of the SAME conversation. A NEW conversation that shares a long
//! prefix with an earlier one (agentic clients: the same 20-60K-token system
//! prompt and tool list in every session, subagent and side request) matches
//! that prefix in the KV radix but finds no snapshot at or below the match,
//! because every existing snapshot sits at some other conversation's tail. It
//! then recomputes all of it: "Prefix cache hit: N tokens but no SSM snapshot
//! — recomputing all KV".
//!
//! With the flag, such a prefill (radix match of at least
//! `ATLAS_GLM_PC_BRANCH_MIN` tokens, default 2048, with no restorable
//! snapshot at all) splits the chunk that spans the match point and saves a
//! checkpoint exactly there. The match point is where the radix branches, so
//! at least two requests share it. The next request with that prefix restores
//! it and replays only its own suffix. This is Marconi's branch-point
//! admission (MLSys'25, arXiv:2411.19379). The checkpoint is an ordinary
//! block-aligned prefill checkpoint, so it stays valid under
//! `ATLAS_MARCONI_PREFILL_ONLY`. The cost is one extra prefill pass the first
//! time a prefix is shared. A conversation's own next turn restores its
//! previous tail, so it never pays that pass.
//!
//! # Restore-depth rank agreement
//!
//! TP2 ranks keep their own snapshot pools and indexes. Only the radix match
//! is min-reduced across ranks (F83), and the Marconi restore depth sets each
//! rank's processed row range, so ranks that restore at different depths run
//! mismatched collectives. With either flag the ranks also agree on the
//! restore depth: the minimum depth any rank can restore, taken only if every
//! rank holds an exact-prefix snapshot at that depth, and otherwise a full
//! recompute everywhere. That costs one 4-byte min-reduction per prefill, plus
//! a second one when there is something to restore.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::prefix_cache::PrefixMatch;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// `ATLAS_GLM_PC_BRANCH=1`: branch-point checkpoints. Read once.
pub(in crate::model) fn glm_pc_branch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PC_BRANCH").as_deref() == Ok("1"))
}

/// Smallest shared prefix worth a branch checkpoint (`ATLAS_GLM_PC_BRANCH_MIN`,
/// default 2048 tokens). The checkpoint costs one extra prefill pass, roughly
/// one sweep of the active weights (~0.1-0.2 s on GB10 TP2), once. Each later
/// request that shares the prefix then skips at least this many tokens.
pub(in crate::model) fn glm_pc_branch_min_tokens() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("ATLAS_GLM_PC_BRANCH_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048)
    })
}

/// Whether ranks must agree on the Marconi restore depth: on with either
/// prefix-cache policy flag, because both make later decisions depend on it.
pub(in crate::model) fn pc_rank_agree_enabled() -> bool {
    spark_runtime::radix_tree::glm_pc_evict_enabled() || glm_pc_branch_enabled()
}

/// The tail-split cut for a `total`-token prompt: one block below the last
/// block boundary strictly under `total` (`prefill_chunk_dispatch_with`).
pub(super) fn tail_cut(total: usize, bs: usize) -> usize {
    ((total.saturating_sub(1) / bs) * bs).saturating_sub(bs)
}

/// Where to save a branch checkpoint, if anywhere: at the radix match
/// `matched` when nothing was restored (`skip_to == 0`), the shared prefix is
/// at least `min` tokens long, and it lies below the tail cut (which already
/// gets a checkpoint).
pub(super) fn branch_checkpoint_at(
    matched: usize,
    skip_to: usize,
    total: usize,
    bs: usize,
    min: usize,
) -> Option<usize> {
    (skip_to == 0 && matched > 0 && matched >= min && matched < tail_cut(total, bs))
        .then_some(matched)
}

/// The split point for a chunk `[start, start + len)`: the planned branch
/// checkpoint when it lies strictly inside the chunk. A checkpoint on a chunk
/// boundary needs no split. Chunks carrying DFlash verify owners
/// (`passengers`) never split.
pub(super) fn branch_split_at(
    at: Option<usize>,
    (start, len): (usize, usize),
    passengers: bool,
) -> Option<usize> {
    at.filter(|&a| !passengers && a > start && a < start + len)
}

impl TransformerModel {
    /// Whether `snap_id` at `snap_tok` may be restored for this lookup. This
    /// is the base restore gate, unchanged; see `prefill_b_prefix_lookup` for
    /// why each term exists.
    pub(super) fn marconi_restorable(
        &self,
        snap_id: usize,
        snap_tok: usize,
        is_tail: bool,
        matched: usize,
        total: usize,
        seq: &SequenceState,
    ) -> bool {
        let exact_without_hidden =
            snap_tok == matched && matched == total && !self.ssm_snapshots.has_hidden(snap_id);
        let bypass_exact = snap_tok == matched
            && matched == total
            && std::env::var("ATLAS_MARCONI_EXACT").as_deref() != Ok("1");
        snap_tok >= crate::model::mtp_carry::marconi_min_tokens()
            && snap_tok > 0
            && matched <= total
            && !exact_without_hidden
            && !bypass_exact
            && (!is_tail
                || self
                    .ssm_snapshots
                    .session_matches(snap_id, seq.session_hash))
            && (!self.requires_aux_state() || self.ssm_snapshots.has_aux(snap_id))
    }

    /// Agree on one restore `(snapshot, depth, is_tail)` across ranks (see
    /// the module docs). Returns the local choice unchanged when agreement is
    /// off or this is a single-rank world; `(None, 0, false)` means no rank
    /// restores. Every rank issues the same collectives: one min-reduction,
    /// plus a second exactly when the agreed depth is non-zero.
    pub(super) fn pc_agree_restore(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        prefix_match: &PrefixMatch,
        matched: usize,
        total: usize,
        local: (Option<usize>, usize),
    ) -> Result<(Option<usize>, usize, bool)> {
        let is_tail = prefix_match.ssm_snapshot_is_tail;
        if !pc_rank_agree_enabled() || !self.multi_rank_protocol_active() {
            return Ok((local.0, local.1, is_tail));
        }
        let restorable = |id: usize, tok: usize, tail: bool| {
            self.marconi_restorable(id, tok, tail, matched, total, seq)
        };
        let depth = match local {
            (Some(id), tok) if restorable(id, tok, is_tail) => tok,
            _ => 0,
        };
        let agreed = self.ep_min_u32(depth as u32)? as usize;
        if agreed == 0 {
            return Ok((None, 0, false));
        }
        let mine = if agreed == depth {
            local.0.map(|id| (id, is_tail))
        } else {
            self.prefix_cache
                .snapshot_at(tokens, agreed, seq.adapter_id)
                .filter(|&id| restorable(id, agreed, false))
                .map(|id| (id, false))
        };
        let all = self.ep_min_u32(u32::from(mine.is_some()))? == 1;
        if agreed != depth || !all {
            tracing::info!(
                "pc rank-agree: local restore depth {depth}, agreed {agreed}, \
                 every rank holds it: {all}"
            );
        }
        Ok(match mine {
            Some((id, tail)) if all => (Some(id), agreed, tail),
            _ => (None, 0, false),
        })
    }

    /// Plan this prefill's branch checkpoint (chunk 0, after the restore
    /// decision). Inputs are rank-agreed, so every rank plans the same split.
    pub(super) fn pc_plan_branch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        matched: usize,
        skip_to: usize,
        bs: usize,
    ) {
        seq.pc_branch_at = None;
        if !glm_pc_branch_enabled()
            || self.config.num_ssm_layers() == 0
            || !self.ssm_snapshots.is_enabled()
            || !self.prefix_cache.is_active()
            || self.tokens_have_vision_pad(tokens)
        {
            return;
        }
        let min = glm_pc_branch_min_tokens();
        seq.pc_branch_at = branch_checkpoint_at(matched, skip_to, tokens.len(), bs, min);
        if let Some(at) = seq.pc_branch_at {
            tracing::info!(
                "pc branch checkpoint planned at token {at} ({matched}-token shared prefix, \
                 no restorable snapshot)"
            );
        }
    }

    /// Run `[start, start + len)` as two chunks split at the planned branch
    /// checkpoint `at`; the first ends at `at`, where
    /// `prefill_b_save_checkpoint` saves it.
    pub(super) fn pc_branch_split(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        (start, len): (usize, usize),
        at: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_chunk_dispatch(tokens, seq, start, at - start, false, stream)?;
        self.prefill_chunk_dispatch(tokens, seq, at, start + len - at, is_last_chunk, stream)
    }
}

#[cfg(test)]
#[path = "pc_policy_tests.rs"]
mod tests;
