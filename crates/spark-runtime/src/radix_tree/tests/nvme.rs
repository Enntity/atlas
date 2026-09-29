// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe spill-tier bookkeeping: spill-on-evict, restore planning/promotion,
//! disk budget LRU, pins, failure handling and `insert` re-homing.

use crate::prefix_cache::{NvmePrefixTier, PrefixCache, SpillOrder};
use crate::radix_tree::RadixTree;

const BS: usize = 16;

fn toks(base: u32, blocks: usize) -> Vec<u32> {
    (base..base + (blocks * BS) as u32).collect()
}

/// Cache a finished request: insert as a cache miss, then the sequence exits.
fn cache(tree: &RadixTree, tokens: &[u32], blocks: &[u32]) {
    tree.insert(tokens, blocks, &[], BS, 0, 0);
    tree.release(tokens, BS, 0);
}

fn spilled_tree(max_slots: u32) -> RadixTree {
    let tree = RadixTree::new();
    assert!(tree.enable(max_slots));
    tree
}

#[test]
fn disabled_tier_deletes_and_plans_nothing() {
    let tree = RadixTree::new();
    let t = toks(0, 2);
    cache(&tree, &t, &[10, 20]);
    let ev = tree.evict(2);
    assert_eq!(ev.physical, vec![20, 10]);
    assert!(ev.spill.is_empty());
    assert!(!tree.is_enabled());
    assert!(tree.plan_restore(&t, BS, 0).disk.is_empty());
}

#[test]
fn enable_is_once_and_needs_budget() {
    let tree = RadixTree::new();
    assert!(!tree.enable(0));
    assert!(tree.enable(4));
    assert!(!tree.enable(8));
}

#[test]
fn evict_spills_leaf_first_and_keeps_nodes() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);

    let ev = tree.evict(1);
    assert_eq!(ev.physical, vec![30]);
    assert_eq!(ev.spill.len(), 1);
    assert_eq!(ev.spill[0].block, 30);

    // Resident-only lookup now stops above the spilled block.
    let m = tree.lookup(&t, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&t, BS, 0);

    let ev = tree.evict(2);
    assert_eq!(ev.physical, vec![20, 10]);
    assert_eq!(ev.spill.len(), 2);
    assert!(tree.lookup(&t, BS, 0, 0).is_empty());
    assert_eq!(tree.stats().0, 0, "no resident entries left");
    assert_eq!(tree.nvme_stats().slots_used, 3);
    assert_eq!(tree.nvme_stats().spills, 3);
}

#[test]
fn restore_round_trip_promotes_in_path_order() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);
    let ev = tree.evict(3);
    let slot_of = |b: u32| ev.spill.iter().find(|o| o.block == b).unwrap().slot;

    let plan = tree.plan_restore(&t, BS, 0);
    assert_eq!(plan.resident_tokens, 0);
    assert_eq!(plan.pinned_tokens, 3 * BS);
    let slots: Vec<u32> = plan.disk.iter().map(|d| d.slot).collect();
    assert_eq!(slots, vec![slot_of(10), slot_of(20), slot_of(30)]);
    for (d, b) in plan.disk.iter().zip([10, 20, 30]) {
        let o = ev.spill.iter().find(|o| o.block == b).unwrap();
        assert_eq!(d.tag, o.tag, "plan carries the spill's record tag");
    }

    let give_back = tree.complete_restore(&t, BS, 0, &plan, &[100, 101, 102], false);
    assert!(give_back.is_empty());
    let m = tree.lookup(&t, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![100, 101, 102]);
    tree.release(&t, BS, 0);
    let s = tree.nvme_stats();
    assert_eq!((s.restores, s.slots_used), (3, 0));
    // Pin released: the restored chain is evictable (spillable) again.
    assert_eq!(tree.evict(3).spill.len(), 3);
}

#[test]
fn restore_continues_below_resident_prefix() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);
    tree.evict(1); // only the leaf goes to disk
    let plan = tree.plan_restore(&t, BS, 0);
    assert_eq!(plan.resident_tokens, 2 * BS);
    assert_eq!(plan.disk.len(), 1);
    // A longer prompt plans the same run (only full, cached blocks).
    let mut longer = t.clone();
    longer.extend(1000..1010);
    let plan2 = tree.plan_restore(&longer, BS, 0);
    assert_eq!(plan2.disk, plan.disk);
    tree.complete_restore(&longer, BS, 0, &plan2, &[], false);
    tree.complete_restore(&t, BS, 0, &plan, &[77], false);
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![10, 20, 77]);
}

#[test]
fn pinned_path_survives_eviction_during_restore() {
    let tree = spilled_tree(8);
    let a = toks(0, 2);
    cache(&tree, &a, &[10, 20]);
    tree.evict(1); // 20 → disk; 10 stays resident
    let plan = tree.plan_restore(&a, BS, 0);
    assert_eq!(plan.disk.len(), 1);
    // The restore's own allocation evicts: the pinned resident parent must not
    // be picked (it has no resident child, so it WOULD be without the pin).
    assert!(tree.evict(1).is_empty());
    tree.complete_restore(&a, BS, 0, &plan, &[21], false);
    assert_eq!(tree.lookup(&a, BS, 0, 0).matched_blocks, vec![10, 21]);
}

#[test]
fn failed_read_drops_node_and_disk_subtree() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);
    tree.evict(3);
    let plan = tree.plan_restore(&t, BS, 0);
    // First record restored, second read failed.
    let give_back = tree.complete_restore(&t, BS, 0, &plan, &[100], true);
    assert!(give_back.is_empty());
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![100]);
    tree.release(&t, BS, 0);
    assert!(
        tree.plan_restore(&t, BS, 0).disk.is_empty(),
        "never restore past a bad record"
    );
    let s = tree.nvme_stats();
    assert_eq!((s.restore_failures, s.slots_used), (1, 0));
}

#[test]
fn unadopted_blocks_are_handed_back() {
    let tree = spilled_tree(8);
    let t = toks(0, 1);
    cache(&tree, &t, &[10]);
    tree.evict(1);
    let plan = tree.plan_restore(&t, BS, 0);
    // More blocks than the plan has nodes: the extra one goes back.
    let give_back = tree.complete_restore(&t, BS, 0, &plan, &[5, 6], false);
    assert_eq!(give_back, vec![6]);
}

#[test]
fn spill_failure_degrades_to_plain_eviction() {
    let tree = spilled_tree(8);
    let t = toks(0, 2);
    cache(&tree, &t, &[10, 20]);
    let ev = tree.evict(2);
    // The parent's write failed: it and the (on-disk) child are dropped.
    let parent: Vec<SpillOrder> = ev.spill.iter().copied().filter(|o| o.block == 10).collect();
    assert!(tree.spill_failed(&parent).is_empty());
    assert!(tree.plan_restore(&t, BS, 0).disk.is_empty());
    let s = tree.nvme_stats();
    assert_eq!((s.spill_failures, s.slots_used), (1, 0));
}

#[test]
fn budget_drops_coldest_disk_leaves_first() {
    let tree = spilled_tree(2);
    let a = toks(0, 2); // older conversation
    let b = toks(5000, 2); // newer conversation
    cache(&tree, &a, &[10, 20]);
    cache(&tree, &b, &[30, 40]);
    assert_eq!(tree.evict(2).spill.len(), 2); // a fills the budget
    // b spills by dropping a's coldest leaf (20), then a's now-leaf root (10).
    assert_eq!(tree.evict(2).spill.len(), 2);
    let s = tree.nvme_stats();
    assert_eq!((s.disk_drops, s.slots_used), (2, 2));
    assert!(tree.plan_restore(&a, BS, 0).disk.is_empty());
    let plan = tree.plan_restore(&b, BS, 0);
    assert_eq!(plan.disk.len(), 2);
    tree.complete_restore(&b, BS, 0, &plan, &[], false);

    // A newer block displaces b's deepest record; b stays restorable to depth 1.
    let c = toks(9000, 1);
    cache(&tree, &c, &[50]);
    let ev = tree.evict(1);
    assert_eq!(ev.spill.len(), 1);
    assert_eq!(ev.spill[0].block, 50);
    let plan = tree.plan_restore(&b, BS, 0);
    assert_eq!(plan.disk.len(), 1);
    tree.complete_restore(&b, BS, 0, &plan, &[], false);
}

#[test]
fn colder_candidate_is_deleted_not_spilled() {
    let tree = spilled_tree(1);
    let old = toks(0, 1);
    let new = toks(7000, 1);
    cache(&tree, &old, &[10]);
    cache(&tree, &new, &[20]);
    // Touch `new` so it is hotter, spill it, then evict `old`.
    let _ = tree.lookup(&new, BS, 0, 0);
    tree.release(&new, BS, 0);
    // LRU picks `old` first: spilled into the only slot.
    assert_eq!(tree.evict(1).spill[0].block, 10);
    // `new` is hotter than `old`'s record: displaces it.
    assert_eq!(tree.evict(1).spill[0].block, 20);
    let s = tree.nvme_stats();
    assert_eq!((s.disk_drops, s.cold_drops), (1, 0));
    // A block colder than the on-disk record: deleted outright.
    let older = toks(8000, 1);
    cache(&tree, &older, &[30]);
    // Re-touch `new`'s disk node via a plan so its record is the hottest.
    let plan = tree.plan_restore(&new, BS, 0);
    tree.complete_restore(&new, BS, 0, &plan, &[], false);
    let ev = tree.evict(1);
    assert_eq!(ev.physical, vec![30]);
    assert!(ev.spill.is_empty());
    assert_eq!(tree.nvme_stats().cold_drops, 1);
}

#[test]
fn insert_rehomes_disk_nodes_onto_recomputed_blocks() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);
    tree.evict(2); // 30, 20 → disk
    // A sequence that did not restore recomputes all 3 blocks and inserts.
    let acq = tree.insert(&t, &[11, 21, 31], &[], BS, 0, 0);
    assert_eq!(acq.blocks, vec![21, 31], "re-homed blocks take a cache ref");
    tree.release(&t, BS, 0);
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![10, 21, 31]);
    assert_eq!(tree.nvme_stats().slots_used, 0);
}

#[test]
fn snapshot_anchor_depth_is_read_only() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    tree.insert_with_snapshot(&t[..2 * BS], &[10, 20], &[], BS, 7, 0, 0, 0);
    tree.release(&t[..2 * BS], BS, 0);
    assert_eq!(tree.snapshot_anchor_depth(&t, 3 * BS, 0, 0), 2 * BS);
    assert_eq!(tree.snapshot_anchor_depth(&t, BS, 0, 0), 0);
    assert_eq!(tree.snapshot_anchor_depth(&toks(900, 3), 3 * BS, 0, 0), 0);
    assert_eq!(tree.snapshot_count(), 1);
}
