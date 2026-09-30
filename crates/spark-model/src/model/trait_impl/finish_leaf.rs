// SPDX-License-Identifier: AGPL-3.0-only

//! End-of-turn SSM snapshot on a block boundary
//! (`ATLAS_GLM_PC_FINISH_LEAF=1`, default off).
//!
//! # What it is for
//!
//! A prefill saves one SSM checkpoint, a block or two below the prompt end.
//! The next turn of the conversation restores it and re-prefills everything
//! after it: the gap to the prompt end, all of the previous turn's output and
//! the new message. With this flag the sequence also keeps the state it had
//! at the last block boundary its decode crossed, registers it when the turn
//! finishes, and the next turn restores there instead.
//!
//! # Why the older leaves were wrong
//!
//! The finish leaf (`finish_leaf_snapshot`) copies the LIVE state at finish
//! and labels it `seq.tokens.len()`. Production turns it off, along with the
//! prompt-end and decode-cadence snapshots (`ATLAS_MARCONI_PREFILL_ONLY`),
//! because restores gave wrong answers. What the code shows:
//!
//! 1. *Position.* A DFlash step that finished the sequence returned before
//!    its commit (fixed separately), so the head's state was not the state
//!    at `seq.tokens.len()`: with KDA records the h state had none of the
//!    step's rows, the conv state had all `k` rows including the rejected
//!    drafts, and the aux rewind had not run.
//! 2. *Rank symmetry.* `cache_sequence` runs on the head only, so only the
//!    head held the leaf. It came into reach two turns later (its blocks
//!    are then prompt blocks), and when the message in between was shorter
//!    than the 17-32 token tail gap it was the head's deepest snapshot while
//!    the worker's was the prefill checkpoint: two restore depths, so every
//!    TP collective paired one rank's row `i` with the other rank's row
//!    `i + d`. The extra head-only snapshots also made the two 16-slot pools
//!    evict different victims, which desynchronizes restores of any kind.
//! 3. *Alignment.* `seq.tokens.len()` is rarely a block boundary and a lookup
//!    only returns snapshots at or below its block-aligned match, so the
//!    leaf could not serve the turn it was saved for. With sub-block matching
//!    it would pair with a partial block whose tail holds rejected rows.
//! 4. *Reachability.* The radix insert at finish left the block under the
//!    prompt end without a sequence reference (see [`owned_from`]), so no
//!    block of the generated output was ever matched by the next turn.
//!
//! The prompt-end leaf is exact as saved (both ranks, end of prefill), but it
//! sits mid-block 17-32 tokens above the tail checkpoint and buys at most
//! that, so it stays off.
//!
//! # What this does instead
//!
//! The state is saved at the moment it is exactly on a boundary, by model
//! code that both ranks run from rank-identical inputs:
//!
//! * A records commit folds the accepted rows of a verify step into the h
//!   state one row at a time. When the step crosses a boundary the fold is
//!   issued in two launches, to the boundary and from it, with the save
//!   between. `kda_commit_records` applies the same per-row fold from and to
//!   FP32 memory, so the split is bit-identical to the single launch. The
//!   conv state at the boundary is the verify's own snapshot after that row.
//! * A plain decode step advances the state in place by one token, so the
//!   live state is the boundary state whenever the new length is a boundary.
//!
//! The save reuses one Marconi snapshot slot per sequence (a rolling slot),
//! so a turn costs one slot however long it is. At finish the head tells the
//! worker to cache its mirror of the sequence too ([`EP_CMD_CACHE_SEQUENCE`]),
//! and both register the slot at the boundary. A restore goes through the
//! usual gate plus the restore-depth rank agreement (`pc_policy`), so a rank
//! that lost its copy makes both ranks fall back together. The leaf only
//! pairs with a whole-block KV match: the flag is ignored unless sub-block
//! matching is off (`ATLAS_PREFIX_SUBBLOCK=0`).
//!
//! # Numerics
//!
//! Exact, but not the bits a cold prefill of the same prompt produces. The
//! restored state and the KV rows of the previous turn's output come from the
//! decode kernels that generated that output (the state the model really
//! had), where the flag-off path recomputes both with the prefill kernels.
//! The next prefill also starts at a different row, so its pass shapes differ
//! (see the accumulation-order note in `pc_policy`).
//!
//! # Cost
//!
//! One D2D copy of the sequence's SSM state per rank (74 MiB; about 3 ms in
//! `scripts/finish-leaf-bench`, a preliminary standalone reading on a shared
//! GB10) each time decode crosses a save boundary
//! (`ATLAS_GLM_PC_FINISH_LEAF_BLOCKS` blocks apart, default 1), on the stream
//! that advanced the state: the secondary stream for a verify commit, where
//! it overlaps the head's next propose. The DFlash drafter's context on the
//! next turn shrinks with the re-prefill it replaces (it is filled from the
//! rows a prefill processes); a wider span keeps more of it.
//!
//! # Rank env parity
//!
//! Both ranks must run with the same `ATLAS_GLM_PC_FINISH_LEAF`,
//! `ATLAS_GLM_PC_FINISH_LEAF_BLOCKS` and `ATLAS_PREFIX_SUBBLOCK`: the flag
//! turns on the restore-depth agreement collectives and the cache command
//! (see "Rank env parity" in `pc_policy`).

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::DevicePtr;

use super::super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// Head to worker: cache your mirror of this slot's sequence (then the usual
/// free follows). Sent only with the flag on.
pub(in crate::model) const EP_CMD_CACHE_SEQUENCE: u32 = 0xFFFF_FFF8;

/// The flag needs whole-block prefix matching, so that a restore of the leaf
/// never pairs with a partial block.
fn resolve(flag: Option<&str>, subblock: Option<&str>) -> bool {
    let on = flag == Some("1");
    if on && subblock != Some("0") {
        tracing::warn!(
            "ATLAS_GLM_PC_FINISH_LEAF=1 ignored: it needs whole-block prefix matching \
             (ATLAS_PREFIX_SUBBLOCK=0)"
        );
    }
    on && subblock == Some("0")
}

/// `ATLAS_GLM_PC_FINISH_LEAF=1` with `ATLAS_PREFIX_SUBBLOCK=0`. Read once.
pub(in crate::model) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let var = |k| std::env::var(k).ok();
        resolve(
            var("ATLAS_GLM_PC_FINISH_LEAF").as_deref(),
            var("ATLAS_PREFIX_SUBBLOCK").as_deref(),
        )
    })
}

/// Blocks between rolling saves (`ATLAS_GLM_PC_FINISH_LEAF_BLOCKS`, default 1:
/// every block boundary). `n` makes the end-of-turn leaf land on the last
/// boundary that is a multiple of `n` blocks, for `1/n` of the copies.
fn span_blocks() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_GLM_PC_FINISH_LEAF_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1)
    })
}

/// How many of the `rows` rows a step adds on top of `pre` tokens reach the
/// last `span`-token boundary inside the step: `Some(r)`, `1 <= r <= rows`,
/// when `pre + r` is that boundary, `None` when the step crosses none.
pub(super) fn boundary_row(pre: usize, rows: usize, span: usize) -> Option<usize> {
    let at = (pre + rows).checked_div(span)? * span;
    (at > pre).then(|| at - pre)
}

/// FNV-1a over `tokens`, continuing from `seed`.
fn hash_tokens(seed: u64, tokens: &[u32]) -> u64 {
    tokens
        .iter()
        .fold(seed, |h, &t| (h ^ u64::from(t)).wrapping_mul(0x100000001b3))
}
const HASH_SEED: u64 = 0xcbf29ce484222325;

/// A sequence's rolling leaf: snapshot slot `snap` holds its exact SSM state
/// after its first `tokens` tokens, which hashed to `hash` when it was saved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FinishLeaf {
    pub snap: usize,
    pub tokens: usize,
    pub hash: u64,
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
    pub(super) fn get(&self) -> Option<FinishLeaf> {
        *self.leaf.lock()
    }
    pub(super) fn take(&self) -> Option<FinishLeaf> {
        self.leaf.lock().take()
    }
    pub(super) fn set(&self, leaf: FinishLeaf, orphans: &Arc<Mutex<Vec<usize>>>) {
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
pub(super) fn leaf_copies(
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

/// Whether `leaf` still describes a prefix of `tokens` covered by
/// `cached_blocks` cached blocks of `bs` tokens.
pub(super) fn leaf_valid(
    leaf: FinishLeaf,
    tokens: &[u32],
    cached_blocks: usize,
    bs: usize,
) -> bool {
    leaf.tokens <= tokens.len()
        && leaf.tokens.is_multiple_of(bs)
        && leaf.tokens / bs <= cached_blocks
        && hash_tokens(HASH_SEED, &tokens[..leaf.tokens]) == leaf.hash
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
pub(super) fn owned_from(prompt_len: usize, bs: usize) -> usize {
    prompt_len / bs * bs
}

impl TransformerModel {
    /// The save span in tokens when `seq` can carry a rolling leaf now.
    fn finish_leaf_span(&self, seq: &SequenceState) -> Option<usize> {
        let ok = enabled()
            && self.ssm_snapshots.is_enabled()
            && self.prefix_cache.is_active()
            && self.config.num_ssm_layers() > 0
            && self.ssm_pool.h_stored_bytes == self.ssm_snapshots.h_bytes()
            && !self.requires_aux_state()
            && seq.slot_idx < self.ssm_pool.max_slots
            && seq.hss_window_start() == 0
            && seq.seq_len == seq.tokens.len()
            && !self.seq_ssm_h_is_f16(seq);
        ok.then(|| self.kv_cache.lock().block_size() * span_blocks())
    }

    /// A free snapshot slot for a rolling leaf, evicting one if needed.
    fn finish_leaf_reserve(&self, session_hash: u64) -> Option<usize> {
        let snaps = &self.ssm_snapshots;
        for orphan in std::mem::take(&mut *snaps.orphans.lock()) {
            snaps.free(orphan);
        }
        snaps.reserve_tail_slot(session_hash).or_else(|| {
            snaps
                .reclaim_from_cache(
                    self.prefix_cache.as_ref(),
                    &mut self.kv_cache.lock(),
                    self.ssm_tier_store.as_deref(),
                    self.gpu.as_ref(),
                )
                .then(|| snaps.reserve_tail_slot(session_hash))
                .flatten()
        })
    }

    /// Save the state `seq` has on `stream` as its rolling leaf at `at`
    /// tokens (see [`leaf_copies`] for `conv_row`). A failed save drops the
    /// leaf: the turn then registers none.
    fn finish_leaf_save(
        &self,
        seq: &SequenceState,
        at: usize,
        conv_row: Option<usize>,
        stream: u64,
    ) {
        // Below the restore floor a leaf could never be restored.
        if at < crate::model::mtp_carry::marconi_min_tokens() {
            return;
        }
        let held = seq.finish_leaf.take();
        let Some(snap) = held
            .map(|l| l.snap)
            .or_else(|| self.finish_leaf_reserve(seq.session_hash))
        else {
            return;
        };
        let (h, conv) = leaf_copies(
            &self.ssm_pool,
            &self.ssm_snapshots,
            seq.slot_idx,
            snap,
            conv_row,
        );
        let saved = run_ssm_state_copies(self.gpu.as_ref(), &h, &conv, stream)
            .and_then(|()| self.record_snapshot_save_dispatch(stream));
        if let Err(e) = saved {
            tracing::warn!("finish-leaf save at token {at}: {e:#}");
            return self.finish_leaf_return(snap);
        }
        // Extend the previous leaf's hash: a rewind below it shows up as a
        // mismatch against the full hash taken at finish.
        let hash = match held.filter(|l| l.tokens <= at) {
            Some(l) => hash_tokens(l.hash, &seq.tokens[l.tokens..at]),
            None => hash_tokens(HASH_SEED, &seq.tokens[..at]),
        };
        let leaf = FinishLeaf {
            snap,
            tokens: at,
            hash,
        };
        seq.finish_leaf.set(leaf, &self.ssm_snapshots.orphans);
        tracing::debug!(
            "finish-leaf save: slot {} token {at} snapshot {snap}",
            seq.slot_idx
        );
    }

    /// KDA records commit of the first `rows` rows of a `k`-row verify
    /// (`seq.tokens` already rolled back to the accepted prefix), saving the
    /// rolling leaf where the step crosses a save boundary.
    pub(super) fn commit_kda_records_with_leaf(
        &self,
        seq: &mut SequenceState,
        rows: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let rewind = rows < k;
        let end = seq.tokens.len();
        let split = self
            .finish_leaf_span(seq)
            .and_then(|span| boundary_row(end.checked_sub(rows)?, rows, span));
        match split {
            None => self.commit_kda_records(seq, 0..rows, rewind, stream),
            Some(r) if r == rows => {
                self.commit_kda_records(seq, 0..rows, rewind, stream)?;
                self.finish_leaf_save(seq, end, None, stream);
                Ok(())
            }
            Some(r) => {
                // Fold to the boundary, save (conv from the verify's snapshot
                // after row r - 1; the live conv holds all k rows), fold on.
                self.commit_kda_records(seq, 0..r, false, stream)?;
                self.finish_leaf_save(seq, end - (rows - r), Some(r - 1), stream);
                self.commit_kda_records(seq, r..rows, rewind, stream)
            }
        }
    }

    /// A plain decode step advanced `seq` in place: save the rolling leaf
    /// when it now sits on a save boundary.
    pub(in crate::model) fn finish_leaf_after_decode(&self, seq: &SequenceState) {
        let at = seq.tokens.len();
        if !enabled() || at == 0 || seq.finish_leaf.get().is_some_and(|l| l.tokens == at) {
            return;
        }
        if let Some(span) = self.finish_leaf_span(seq)
            && boundary_row(at - 1, 1, span).is_some()
        {
            self.finish_leaf_save(seq, at, None, self.gpu.default_stream());
        }
    }

    /// Head: have the worker cache its mirror of `seq` (it holds the same
    /// blocks and rolling leaf), so both ranks match and restore alike.
    pub(super) fn finish_leaf_mirror_cache(&self, seq: &SequenceState) {
        let head = self.comm.as_ref().is_some_and(|c| c.rank() == 0);
        if enabled() && head && seq.slot_idx < self.ssm_pool.max_slots {
            let sent = self.ep_broadcast_seq_and_cmd(
                seq.slot_idx as u32,
                EP_CMD_CACHE_SEQUENCE,
                self.ep_protocol_v2,
            );
            if let Err(e) = sent {
                tracing::error!("finish-leaf: EP broadcast cache-sequence: {e:#}");
            }
        }
    }

    /// `cache_sequence` with the flag on, on either rank: cache the whole
    /// blocks of `seq.tokens`, then register the rolling leaf at its boundary
    /// or drop it if it no longer describes a cached prefix of the sequence.
    pub(super) fn finish_leaf_cache(&self, seq: &SequenceState, bs: usize) {
        let acquired = self.prefix_cache.insert(
            &seq.tokens,
            &seq.block_table,
            &seq.disk_block_ids,
            bs,
            owned_from(seq.prompt_len, bs),
            seq.adapter_id,
        );
        super::super::block_mgmt::cache_acquires_refs(&acquired, &mut self.kv_cache.lock());
        let Some(leaf) = seq.finish_leaf.take() else {
            return;
        };
        let blocks = leaf.tokens / bs;
        if !leaf_valid(leaf, &seq.tokens, seq.block_table.len(), bs) {
            tracing::info!("finish-leaf: stale leaf at token {} dropped", leaf.tokens);
            return self.finish_leaf_return(leaf.snap);
        }
        let displaced = self.prefix_cache.insert_intermediate_snapshot(
            &seq.tokens[..leaf.tokens],
            &seq.block_table[..blocks],
            seq.disk_block_ids.get(..blocks).unwrap_or(&[]),
            bs,
            leaf.snap,
            seq.session_hash,
            leaf.tokens,
            seq.adapter_id,
        );
        if let Some(old) = displaced {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "finish-leaf: snapshot {} registered at token {} of {}",
            leaf.snap,
            leaf.tokens,
            seq.tokens.len()
        );
    }

    /// Return a rolling leaf that was never registered.
    pub(super) fn finish_leaf_release(&self, seq: &SequenceState) {
        if let Some(leaf) = seq.finish_leaf.take() {
            self.finish_leaf_return(leaf.snap);
        }
    }

    /// Hand an unregistered rolling slot back to the pool. Its last save may
    /// still be in flight on the secondary stream, and the next owner of the
    /// slot saves on another stream, so drain first (off the hot path: only
    /// an aborted, failed or stale leaf comes here).
    fn finish_leaf_return(&self, snap: usize) {
        let drained = self
            .sync_secondary_dispatch()
            .and_then(|()| self.gpu.synchronize(self.gpu.default_stream()));
        if let Err(e) = drained {
            tracing::warn!("finish-leaf: drain before returning snapshot {snap}: {e:#}");
        }
        self.ssm_snapshots.free(snap);
    }
}

#[cfg(test)]
#[path = "finish_leaf_tests.rs"]
mod tests;
