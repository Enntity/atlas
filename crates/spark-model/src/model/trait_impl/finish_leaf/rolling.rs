// SPDX-License-Identifier: AGPL-3.0-only

//! The rolling leaf of one sequence: where a step saves it, what the save
//! copies, who owns the snapshot slot, and when the leaf is stale.

use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::radix_tree::{prefix_hash_push, prefix_hash_seed};

use super::super::super::ssm_batched_copy::StateCopy;
use super::super::super::ssm_pool::SsmStatePool;
use super::super::super::ssm_snapshot::SsmSnapshotPool;

/// How many of the `rows` rows a step adds on top of `pre` tokens reach the
/// last `span`-token boundary inside the step: `Some(r)`, `1 <= r <= rows`,
/// when `pre + r` is that boundary, `None` when the step crosses none.
pub(in crate::model) fn boundary_row(pre: usize, rows: usize, span: usize) -> Option<usize> {
    let at = (pre + rows).checked_div(span)? * span;
    (at > pre).then(|| at - pre)
}

/// The rolling-leaf save of one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) struct LeafSave {
    /// Rows of the step folded into the state before the save.
    pub rows: usize,
    /// The token position the saved state is at.
    pub at: usize,
    /// The verify's conv snapshot (after this row) that holds the boundary's
    /// conv state. `None`: the live conv state does.
    pub conv_row: Option<usize>,
}

/// The save a step makes when it leaves a sequence at `end` tokens after
/// committing the first `rows` of `k` verified rows (a plain decode step: 1
/// of 1). `end` is the length AFTER the rejected rows were rolled back: a
/// caller that commits first labels the leaf too deep.
///
/// The live conv state holds all `k` rows until the commit's rewind, so the
/// boundary's is the verify's snapshot after its row, unless that row is the
/// verify's last. The verify writes those snapshots for rows `0..k - 1`.
pub(in crate::model) fn leaf_save(
    end: usize,
    rows: usize,
    k: usize,
    span: usize,
) -> Option<LeafSave> {
    let pre = end.checked_sub(rows)?;
    boundary_row(pre, rows, span).map(|r| LeafSave {
        rows: r,
        at: pre + r,
        conv_row: (r < k).then(|| r - 1),
    })
}

/// The prefix hash of `tokens` continuing from `seed` (the FNV-1a fold the
/// snapshot index keys on, with no adapter).
pub(in crate::model) fn hash_tokens(seed: u64, tokens: &[u32]) -> u64 {
    tokens.iter().fold(seed, |h, &t| prefix_hash_push(h, t))
}

/// A sequence's rolling leaf: snapshot slot `snap` holds its exact SSM state
/// after its first `tokens` tokens, which hashed to `hash` when it was saved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FinishLeaf {
    pub snap: usize,
    pub tokens: usize,
    pub hash: u64,
}

impl FinishLeaf {
    /// The leaf in slot `snap` for the first `at` tokens of `tokens`. `prev`
    /// is the leaf the slot held before: its hash is extended when it lies
    /// under `at`, so a rewind below it and a different regeneration show up
    /// as a mismatch against the full hash taken at finish.
    pub(in crate::model) fn advance(
        prev: Option<Self>,
        snap: usize,
        tokens: &[u32],
        at: usize,
    ) -> Self {
        let hash = match prev.filter(|l| l.tokens <= at) {
            Some(l) => hash_tokens(l.hash, &tokens[l.tokens..at]),
            None => hash_tokens(prefix_hash_seed(0), &tokens[..at]),
        };
        Self {
            snap,
            tokens: at,
            hash,
        }
    }

    /// Whether the leaf still describes a prefix of `tokens` covered by
    /// `cached_blocks` cached blocks of `bs` tokens.
    pub(in crate::model) fn valid(&self, tokens: &[u32], cached_blocks: usize, bs: usize) -> bool {
        self.tokens <= tokens.len()
            && self.tokens.is_multiple_of(bs)
            && self.tokens / bs <= cached_blocks
            && hash_tokens(prefix_hash_seed(0), &tokens[..self.tokens]) == self.hash
    }
}

/// Holder of a sequence's [`FinishLeaf`]. The sequence owns the snapshot slot
/// until `cache_sequence` registers it or `free_sequence` returns it; a
/// sequence dropped without either (abort, unwind) hands the slot to the
/// pool's orphan list instead of leaking it.
#[derive(Default)]
pub(crate) struct LeafCell {
    leaf: Mutex<Option<FinishLeaf>>,
    orphans: OnceLock<Arc<Mutex<Vec<usize>>>>,
}

impl LeafCell {
    pub(in crate::model) fn get(&self) -> Option<FinishLeaf> {
        *self.leaf.lock()
    }
    pub(in crate::model) fn take(&self) -> Option<FinishLeaf> {
        self.leaf.lock().take()
    }
    pub(in crate::model) fn set(&self, leaf: FinishLeaf, orphans: &Arc<Mutex<Vec<usize>>>) {
        self.orphans.get_or_init(|| orphans.clone());
        *self.leaf.lock() = Some(leaf);
    }
}

impl Drop for LeafCell {
    fn drop(&mut self) {
        if let (Some(leaf), Some(orphans)) = (self.leaf.get_mut().take(), self.orphans.get()) {
            orphans.lock().push(leaf.snap);
        }
    }
}

/// The copies that save SSM-pool slot `ssm_slot` into snapshot slot `snap`:
/// the live h state, and the conv state from the verify snapshot after row
/// `conv_row` (`None`: the live conv state).
pub(in crate::model) fn leaf_copies(
    pool: &SsmStatePool,
    snaps: &SsmSnapshotPool,
    ssm_slot: usize,
    snap: usize,
    conv_row: Option<usize>,
) -> (Vec<StateCopy>, Vec<StateCopy>) {
    let plan = |src: &dyn Fn(usize) -> DevicePtr, dst: &dyn Fn(usize) -> DevicePtr, bytes| {
        (0..snaps.num_ssm_layers())
            .map(|i| StateCopy {
                src: src(i),
                dst: dst(i),
                bytes,
            })
            .collect::<Vec<_>>()
    };
    let conv_src = |i| match conv_row {
        Some(row) => pool.conv_intermediate(i, ssm_slot, row),
        None => pool.conv_state(i, ssm_slot),
    };
    (
        plan(
            &|i| pool.h_state(i, ssm_slot),
            &|i| snaps.tail_h_dst(i, snap),
            snaps.h_bytes(),
        ),
        plan(
            &conv_src,
            &|i| snaps.tail_conv_dst(i, snap),
            snaps.conv_bytes(),
        ),
    )
}

/// The `matched_tokens` a finishing sequence passes to the radix insert: the
/// tokens whose nodes its prefill already inserted and ref-bumped.
///
/// That is the prompt's WHOLE blocks. Base passes `prompt_len` itself, which
/// also counts the block the prompt end falls inside: that node is created
/// here, so it gets the cache's reference only, the sequence's `release`
/// takes it to zero, and a zero-ref node stops every later walk. With a
/// prompt that does not end on a block boundary, no block of the generated
/// output was ever matched by the next turn. Left as is with the flag off:
/// there the worker rank caches nothing, so matching those blocks on the head
/// alone would only keep them recent in its LRU at the expense of useful ones.
pub(in crate::model) fn owned_from(prompt_len: usize, bs: usize) -> usize {
    prompt_len / bs * bs
}
