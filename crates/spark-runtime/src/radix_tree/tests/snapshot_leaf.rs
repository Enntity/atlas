// SPDX-License-Identifier: AGPL-3.0-only

//! Finish leaves in the snapshot index: they never supersede or displace a
//! conversation's frontier checkpoint. Like the chain tests, these drive the
//! index directly (the flags are process-wide `OnceLock`s).

use super::*;
use crate::radix_tree::snapshot::SnapshotEntry;

const POOL: usize = 16;

/// One conversation's token path: a shared system prompt, then its own turns.
fn convo(session: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| if i < 2048 { i } else { session * 1_000_000 + i })
        .collect()
}

fn hash(tokens: &[u32]) -> u64 {
    hash_token_prefix(tokens, tokens.len(), 0)
}

/// A prefill checkpoint for `tokens`, as `RadixTree` registers one.
fn checkpoint(idx: &mut SsmSnapshotIndex, tokens: &[u32], id: usize) -> Option<usize> {
    let old = idx.insert(hash(tokens), id, 0, tokens.len());
    idx.link_chain(tokens, 0, hash(tokens));
    old
}

fn leaf(idx: &mut SsmSnapshotIndex, tokens: &[u32], id: usize) -> Option<usize> {
    idx.insert_leaf(hash(tokens), id, 0, tokens.len())
}

fn entry(idx: &SsmSnapshotIndex, id: usize) -> &SnapshotEntry {
    idx.entries.iter().find(|e| e.snapshot_id == id).unwrap()
}

fn resident(idx: &SsmSnapshotIndex, id: usize) -> bool {
    idx.entries.iter().any(|e| e.snapshot_id == id)
}

fn evict(idx: &mut SsmSnapshotIndex) -> usize {
    let v = idx.chain_victim(true).unwrap();
    idx.entries.swap_remove(v).snapshot_id
}

#[test]
fn a_leaf_does_not_supersede_the_checkpoint_under_it() {
    let mut idx = SsmSnapshotIndex::new();
    let t = convo(1, 31_000);
    checkpoint(&mut idx, &t[..30_000], 1);
    assert_eq!(leaf(&mut idx, &t[..30_400], 2), None);
    let (cut, l) = (entry(&idx, 1).chain, entry(&idx, 2).chain);
    assert!(!cut.superseded && !cut.branch, "still the frontier");
    assert!(l.leaf && l.id == 0 && !l.superseded);
    // The next turn's checkpoint links past the leaf, to the checkpoint.
    checkpoint(&mut idx, &t, 3);
    let (cut, l, next) = (
        entry(&idx, 1).chain,
        entry(&idx, 2).chain,
        entry(&idx, 3).chain,
    );
    assert!(cut.superseded && !cut.branch, "no fork at the checkpoint");
    assert_eq!(cut.id, next.id, "one conversation, one chain");
    assert!(l.leaf && l.superseded, "the leaf has served its turn");
}

#[test]
fn victims_are_dead_history_then_leaves_then_frontiers() {
    let mut idx = SsmSnapshotIndex::new();
    let (a, b) = (convo(1, 32_000), convo(2, 40_000));
    checkpoint(&mut idx, &b, 100); // B's frontier: the oldest entry of all
    checkpoint(&mut idx, &a[..30_000], 1);
    checkpoint(&mut idx, &a[..31_000], 2); // supersedes 1
    leaf(&mut idx, &a[..31_400], 3);
    let order: Vec<usize> = (0..4).map(|_| evict(&mut idx)).collect();
    assert_eq!(order, [1, 3, 100, 2]);
}

#[test]
fn a_leaf_never_takes_a_frontier_slot() {
    let mut idx = SsmSnapshotIndex::new();
    let a = convo(1, 32_000);
    checkpoint(&mut idx, &convo(2, 40_000), 100);
    checkpoint(&mut idx, &a[..30_000], 1);
    assert_eq!(idx.evict_for_leaf(), None, "two frontiers, nothing to take");
    checkpoint(&mut idx, &a[..31_000], 2);
    leaf(&mut idx, &a[..31_400], 3);
    assert_eq!(idx.evict_for_leaf(), Some(1), "dead history first");
    assert_eq!(idx.evict_for_leaf(), Some(3), "then another leaf");
    assert_eq!(idx.evict_for_leaf(), None);
    assert!(resident(&idx, 100) && resident(&idx, 2));
    // A spilled entry holds no slot to take.
    leaf(&mut idx, &a[..31_400], 4);
    idx.entries.iter_mut().for_each(|e| e.tiered = e.chain.leaf);
    assert_eq!(idx.evict_for_leaf(), None);
}

#[test]
fn a_checkpoint_and_a_leaf_at_one_prefix() {
    let mut idx = SsmSnapshotIndex::new();
    let t = convo(1, 30_000);
    // A checkpoint already there wins: the leaf's slot comes straight back.
    checkpoint(&mut idx, &t, 1);
    assert_eq!(leaf(&mut idx, &t, 2), Some(2));
    assert!(!entry(&idx, 1).chain.leaf && idx.entries.len() == 1);
    // A newer leaf replaces an older one and stays a leaf.
    let u = convo(3, 30_000);
    assert_eq!(leaf(&mut idx, &u, 5), None);
    assert_eq!(leaf(&mut idx, &u, 6), Some(5));
    assert!(entry(&idx, 6).chain.leaf);
    // A checkpoint saved over a leaf takes its place and joins a chain.
    assert_eq!(checkpoint(&mut idx, &u, 7), Some(6));
    let c = entry(&idx, 7).chain;
    assert!(!c.leaf && c.id != 0);
}

#[test]
fn the_restoring_turn_settles_its_leaf() {
    let mut idx = SsmSnapshotIndex::new();
    let t = convo(1, 31_000);
    checkpoint(&mut idx, &t[..30_000], 1);
    leaf(&mut idx, &t[..30_400], 2);
    // Not a leaf, or nothing there: no effect either way.
    for keep in [true, false] {
        idx.settle_leaf(&t[..30_000], 0, keep);
        idx.settle_leaf(&t[..30_416], 0, keep);
    }
    let (cut, l) = (entry(&idx, 1).chain, entry(&idx, 2).chain);
    assert!(!cut.superseded && l.leaf && !l.superseded);
    // The turn saves its own checkpoint: the leaf is the first victim, and
    // still restorable until it goes.
    idx.settle_leaf(&t[..30_400], 0, false);
    assert!(entry(&idx, 2).chain.superseded && !entry(&idx, 1).chain.superseded);
    assert_eq!(idx.resident_at(&t, 30_400, 0), Some(2));
    assert_eq!(idx.evict_for_leaf(), Some(2));
    // The turn saves none: the leaf becomes the frontier.
    leaf(&mut idx, &t[..30_400], 3);
    idx.settle_leaf(&t[..30_400], 0, true);
    let (cut, l) = (entry(&idx, 1).chain, entry(&idx, 3).chain);
    assert!(cut.superseded && !l.leaf && !l.superseded);
    assert_eq!(cut.id, l.id);
    assert_eq!(evict(&mut idx), 1, "the old checkpoint is dead history now");
}

/// `turns` rounds of `convs` conversations taking turns on one 16-slot pool,
/// each turn saving its prefill checkpoint and, with `leaves`, a finish leaf
/// through the rolling-slot rules. Returns, per turn after the first round,
/// whether the conversation still held (its checkpoint, its leaf).
fn serve(convs: u32, turns: usize, leaves: bool) -> Vec<(bool, bool)> {
    let mut idx = SsmSnapshotIndex::new();
    let (mut next_id, mut held) = (0usize, Vec::new());
    let mut last = vec![(usize::MAX, usize::MAX); convs as usize];
    for turn in 0..turns {
        for c in 0..convs {
            let prompt = convo(c + 1, 30_000 + turn * 800);
            let (cut, lf) = last[c as usize];
            if turn > 0 {
                held.push((resident(&idx, cut), resident(&idx, lf)));
                // The restore bumps the winner, as a lookup does, and settles
                // the leaf it took (this turn saves its own checkpoint).
                let hit = idx.lookup_tiered(&prompt, prompt.len(), 0, 0).unwrap();
                idx.settle_leaf(&prompt[..hit.token_count], 0, false);
            }
            // The prefill checkpoint: a free slot or the chain victim.
            if idx.entries.len() == POOL {
                evict(&mut idx);
            }
            next_id += 1;
            checkpoint(&mut idx, &prompt[..prompt.len() - 32], next_id);
            last[c as usize] = (next_id, usize::MAX);
            // The rolling slot (held outside the index while the turn
            // decodes), then the leaf 400 tokens into the output.
            if leaves && (idx.entries.len() < POOL || idx.evict_for_leaf().is_some()) {
                next_id += 1;
                let out = convo(c + 1, 30_000 + turn * 800 + 400);
                assert_eq!(leaf(&mut idx, &out, next_id), None);
                last[c as usize].1 = next_id;
            }
            assert!(idx.entries.len() <= POOL);
        }
    }
    held
}

/// The pool-pressure contract: with leaves on, every conversation keeps the
/// checkpoint it keeps with leaves off, whatever the number of conversations.
/// Leaves live in the slack. A checkpoint and a leaf per conversation fit the
/// pool up to 8 conversations, and every leaf is then there for its turn.
/// Beyond that a strict rotation is the worst case for the leaves (the least
/// recently used one belongs to the conversation whose turn is next), and
/// none survives; no checkpoint is lost for it.
#[test]
fn leaves_never_cost_a_conversation_its_checkpoint() {
    for convs in [3, 8, 9, 12, 15] {
        let (on, off) = (serve(convs, 4, true), serve(convs, 4, false));
        assert!(off.iter().all(|&(cut, _)| cut), "convs={convs}: base");
        assert!(on.iter().all(|&(cut, _)| cut), "convs={convs}: with leaves");
        let kept = on.iter().filter(|&&(_, lf)| lf).count();
        let want = if convs <= 8 { on.len() } else { 0 };
        assert_eq!(kept, want, "convs={convs}: leaves in reach of their turn");
    }
}
