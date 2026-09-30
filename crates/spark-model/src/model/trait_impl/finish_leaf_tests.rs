// SPDX-License-Identifier: AGPL-3.0-only

//! The rolling finish leaf: where it lands, what it copies, when it is
//! dropped, and that two ranks place and restore it alike.

use super::super::super::ssm_batched_copy::run_ssm_state_copies;
use super::super::super::ssm_pool::SsmStatePool;
use super::super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::prefill_b::pc_policy::{Agreed, agree_restore, tail_cut};
use super::*;
use crate::ssm_reserve::SsmRollbackMode;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::radix_tree::RadixTree;

const BS: usize = 16;

#[test]
fn the_flag_needs_whole_block_matching() {
    assert!(resolve(Some("1"), Some("0")));
    assert!(!resolve(Some("1"), None));
    assert!(!resolve(Some("1"), Some("1")));
    assert!(!resolve(None, Some("0")));
    assert!(!resolve(Some("0"), Some("0")));
}

#[test]
fn boundary_row_is_the_last_boundary_the_step_reaches() {
    // A 5-row step from 45 commits 46..=50: row 3 lands on 48.
    assert_eq!(boundary_row(45, 5, BS), Some(3));
    // Ending exactly on the boundary: the whole step.
    assert_eq!(boundary_row(45, 3, BS), Some(3));
    assert_eq!(boundary_row(47, 1, BS), Some(1));
    // Starting on a boundary does not count: that state is already behind.
    assert_eq!(boundary_row(48, 5, BS), None);
    assert_eq!(boundary_row(48, 16, BS), Some(16));
    assert_eq!(boundary_row(45, 2, BS), None);
    // A step wider than a block keeps the deepest boundary.
    assert_eq!(boundary_row(45, 40, BS), Some(35));
    // A coarser span skips the block boundaries in between.
    assert_eq!(boundary_row(45, 5, 4 * BS), None);
    assert_eq!(boundary_row(60, 9, 4 * BS), Some(4));
    assert_eq!(boundary_row(45, 5, 0), None);
}

/// Steps of a decode: `(rows, plain)`, a verify commit of 1..=9 rows or one
/// plain decode token, from a small LCG so the mix is fixed.
fn steps(seed: u64, n: usize) -> Vec<(usize, bool)> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let plain = (x >> 40).is_multiple_of(5);
            (if plain { 1 } else { 1 + (x >> 33) as usize % 9 }, plain)
        })
        .collect()
}

/// The leaf position a sequence of `len` tokens holds after `steps`, saving
/// wherever [`boundary_row`] says (both the commit and the plain-decode hook
/// reduce to it).
fn run(mut len: usize, steps: &[(usize, bool)], span: usize) -> (usize, Option<usize>) {
    let mut leaf = None;
    for &(rows, _) in steps {
        if let Some(r) = boundary_row(len, rows, span) {
            leaf = Some(len + r);
        }
        len += rows;
    }
    (len, leaf)
}

/// At finish the leaf is the last save boundary at or below the final
/// committed token, whatever the step widths were.
#[test]
fn the_leaf_ends_on_the_last_boundary_at_or_below_the_end() {
    for seed in 0..200u64 {
        for span in [BS, 4 * BS] {
            let start = 300 + seed as usize % 37;
            let (end, leaf) = run(start, &steps(seed, 3 + seed as usize % 90), span);
            let want = end / span * span;
            assert_eq!(
                leaf,
                (want > start).then_some(want),
                "seed={seed} span={span}"
            );
        }
    }
}

#[test]
fn the_hash_chain_equals_the_full_hash() {
    let tokens: Vec<u32> = (0..200).map(|i| i * 7 + 3).collect();
    let full = hash_tokens(HASH_SEED, &tokens[..160]);
    let chained = hash_tokens(hash_tokens(HASH_SEED, &tokens[..48]), &tokens[48..160]);
    assert_eq!(full, chained);
    assert_ne!(full, hash_tokens(HASH_SEED, &tokens[..144]));
}

#[test]
fn a_leaf_is_registered_only_for_a_cached_prefix_it_still_describes() {
    let tokens: Vec<u32> = (0..100).collect();
    let leaf = |at: usize| FinishLeaf {
        snap: 3,
        tokens: at,
        hash: hash_tokens(HASH_SEED, &tokens[..at]),
    };
    assert!(leaf_valid(leaf(96), &tokens, 6, BS));
    // The turn was rewound below the leaf and ended there.
    assert!(!leaf_valid(leaf(96), &tokens[..90], 5, BS));
    // Its blocks are not all in the block table.
    assert!(!leaf_valid(leaf(96), &tokens, 5, BS));
    // The sequence was rewound and regenerated differently under the leaf.
    let mut other = tokens.clone();
    other[70] = 9_999;
    assert!(!leaf_valid(leaf(96), &other, 6, BS));
    // Never off a block boundary.
    assert!(!leaf_valid(leaf(90), &tokens, 6, BS));
}

#[test]
fn a_dropped_sequence_hands_its_slot_to_the_orphan_list() {
    let orphans = Arc::new(Mutex::new(Vec::new()));
    let leaf = FinishLeaf {
        snap: 5,
        tokens: 96,
        hash: 1,
    };
    // Registered or released (taken): nothing to reclaim.
    let cell = LeafCell::default();
    cell.set(leaf, &orphans);
    assert_eq!(cell.get(), Some(leaf));
    assert_eq!(cell.take(), Some(leaf));
    drop(cell);
    assert!(orphans.lock().is_empty());
    // Dropped while still holding the slot.
    let cell = LeafCell::default();
    cell.set(leaf, &orphans);
    drop(cell);
    assert_eq!(*orphans.lock(), vec![5]);
    // A sequence that never saved a leaf has no list and nothing to hand over.
    drop(LeafCell::default());
}

fn tiny_config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.linear_num_key_heads = 2;
    c.linear_key_head_dim = 4;
    c.linear_num_value_heads = 2;
    c.linear_value_head_dim = 4;
    c.linear_conv_kernel_dim = 4;
    c
}

/// The save takes h from the live state and conv from the verify snapshot
/// after the boundary row (the live conv then holds all `k` rows), or the
/// live conv when the step ends on the boundary. A restore returns both.
#[test]
fn the_save_pairs_live_h_with_the_boundary_conv() {
    const SEQ: usize = 1;
    const SNAP: usize = 2;
    let gpu = MockGpuBackend::new();
    let pool = SsmStatePool::new(
        &tiny_config(),
        4,
        true,
        4,
        3,
        false,
        SsmRollbackMode::Snapshot,
        &gpu,
    )
    .unwrap();
    let layers = pool.num_ssm_layers;
    let snaps =
        SsmSnapshotPool::new(3, pool.h_bytes, pool.conv_bytes, layers, 0, 0, 8, &gpu).unwrap();
    let (h_live, conv_live, conv_row) = (
        |l: usize| 10 + l as u8,
        |l: usize| 80 + l as u8,
        |l, r| (150 + 8 * l + r) as u8,
    );
    for l in 0..layers {
        let put = |v: u8, n: usize, p| gpu.copy_h2d(&vec![v; n], p).unwrap();
        put(h_live(l), pool.h_stored_bytes, pool.h_state(l, SEQ));
        put(conv_live(l), pool.conv_bytes, pool.conv_state(l, SEQ));
        for r in 0..pool.num_intermediates {
            put(
                conv_row(l, r),
                pool.conv_bytes,
                pool.conv_intermediate(l, SEQ, r),
            );
        }
    }
    let saved = |row: Option<usize>| {
        let (h, conv) = leaf_copies(&pool, &snaps, SEQ, SNAP, row);
        run_ssm_state_copies(&gpu, &h, &conv, 0).unwrap();
        // Restore into another slot and read that slot's live state back.
        snaps.restore(SNAP, 3, &pool, &gpu, 0).unwrap();
        (0..layers)
            .map(|l| {
                let (mut h, mut c) = (vec![0u8; pool.h_stored_bytes], vec![0u8; pool.conv_bytes]);
                gpu.copy_d2h(pool.h_state(l, 3), &mut h).unwrap();
                gpu.copy_d2h(pool.conv_state(l, 3), &mut c).unwrap();
                assert!(h.iter().all(|&b| b == h[0]) && c.iter().all(|&b| b == c[0]));
                (h[0], c[0])
            })
            .collect::<Vec<_>>()
    };
    let want = |conv: &dyn Fn(usize) -> u8| {
        (0..layers)
            .map(|l| (h_live(l), conv(l)))
            .collect::<Vec<_>>()
    };
    assert_eq!(saved(Some(1)), want(&|l| conv_row(l, 1)));
    assert_eq!(saved(None), want(&conv_live));
}

/// One rank's prefix cache through a turn: prefill (prompt blocks and the
/// tail checkpoint), then finish (all whole blocks, and the leaf when the
/// rank kept one), as `cache_sequence` does on both ranks with the flag on.
struct Rank {
    cache: RadixTree,
    rank: usize,
}

impl Rank {
    fn turn(rank: usize, tokens: &[u32], prompt: usize, leaf: Option<usize>) -> Self {
        let cache = RadixTree::new();
        let blocks: Vec<u32> = (0..(tokens.len() / BS + 1) as u32).collect();
        let cut = tail_cut(prompt, BS);
        let acquired = cache.insert(&tokens[..prompt], &blocks[..prompt / BS], &[], BS, 0, 0);
        assert_eq!(acquired.blocks.len(), prompt / BS);
        cache.insert_intermediate_snapshot(
            &tokens[..cut],
            &blocks[..cut / BS],
            &[],
            BS,
            cut + rank,
            0,
            cut,
            0,
        );
        cache.insert(
            tokens,
            &blocks[..tokens.len() / BS],
            &[],
            BS,
            owned_from(prompt, BS),
            0,
        );
        if let Some(at) = leaf {
            let none = cache.insert_intermediate_snapshot(
                &tokens[..at],
                &blocks[..at / BS],
                &[],
                BS,
                at + rank,
                0,
                at,
                0,
            );
            assert_eq!(none, None);
        }
        cache.release(tokens, BS, 0);
        Self { cache, rank }
    }

    /// `(matched, restorable depth)` for the next turn's prompt.
    fn lookup(&self, next: &[u32]) -> (usize, usize) {
        let m = self.cache.lookup(next, BS, 0, 0);
        assert_eq!(
            m.ssm_snapshot,
            (m.ssm_snapshot_tokens > 0).then(|| m.ssm_snapshot_tokens + self.rank)
        );
        (m.matched_tokens, m.ssm_snapshot_tokens)
    }
}

/// The block the prompt end falls inside is created at finish, so the
/// finishing sequence must own it: passed `prompt_len` itself (as base does),
/// the sequence's release takes that node to zero references and the next
/// turn's walk stops there, before every block of the generated output.
#[test]
fn the_block_under_the_prompt_end_is_owned_by_the_finishing_sequence() {
    let (prompt, end) = (1_000usize, 1_504usize);
    let tokens: Vec<u32> = (0..end as u32).collect();
    let blocks: Vec<u32> = (0..(end / BS) as u32).collect();
    let matched = |owned: usize| {
        let cache = RadixTree::new();
        cache.insert(&tokens[..prompt], &blocks[..prompt / BS], &[], BS, 0, 0);
        cache.insert(&tokens, &blocks, &[], BS, owned, 0);
        cache.release(&tokens, BS, 0);
        cache.peek_matched_tokens(&tokens, BS, 0)
    };
    assert_eq!(owned_from(prompt, BS), 992);
    assert_eq!(matched(owned_from(prompt, BS)), end);
    assert_eq!(
        matched(prompt),
        992,
        "base: the generated blocks are unreachable"
    );
    // A prompt that ends on a boundary has no such block either way.
    assert_eq!(owned_from(1_008, BS), 1_008);
}

/// Both ranks' [`agree_restore`] results for the next turn, over the
/// min-reductions their own lookups produce.
fn agreed(ranks: &[Rank; 2], next: &[u32]) -> [(usize, Agreed); 2] {
    let looked = [ranks[0].lookup(next), ranks[1].lookup(next)];
    assert_eq!(looked[0].0, looked[1].0, "ranks must match the same blocks");
    let depth = looked[0].1.min(looked[1].1);
    let holds =
        |i: usize| looked[i].1 == depth || ranks[i].cache.snapshot_at(next, depth, 0).is_some();
    let all = u32::from(depth > 0 && holds(0) && holds(1));
    [0, 1].map(|i| {
        let mut replies = [depth as u32, all].into_iter();
        agree_restore(
            looked[i].1,
            |_| Ok(replies.next().expect("at most two reductions")),
            |at| ranks[i].cache.snapshot_at(next, at, 0),
        )
        .unwrap()
    })
}

/// A turn decoded from the same verdicts leaves the same leaf on both ranks;
/// the next turn then matches the same blocks and restores at the leaf.
#[test]
fn both_ranks_place_and_restore_the_same_leaf() {
    for seed in 0..40u64 {
        let prompt = 1_000 + seed as usize * 13;
        let verdicts = steps(seed, 60 + seed as usize);
        // Head and worker apply the same verdict stream to mirrored sequences.
        let (end, leaf) = run(prompt, &verdicts, BS);
        assert_eq!((end, leaf), run(prompt, &verdicts, BS));
        let at = leaf.expect("a 60-step turn crosses a block boundary");
        assert_eq!(at, end / BS * BS);
        let tokens: Vec<u32> = (0..end as u32).map(|t| t * 3 + seed as u32).collect();
        let next: Vec<u32> = tokens.iter().copied().chain(900_000..900_040).collect();
        let ranks = [0, 1].map(|r| Rank::turn(r, &tokens, prompt, leaf));
        assert_eq!(ranks[0].lookup(&next), (at, at));
        assert_eq!(
            agreed(&ranks, &next),
            [(at, Agreed::Local); 2],
            "seed={seed}"
        );
    }
}

/// A rank that could not keep its leaf (snapshot pool exhausted) restores
/// from the prefill checkpoint; the agreement brings the other rank down to
/// it, so both resume at the same row.
#[test]
fn a_rank_without_the_leaf_pulls_both_to_the_tail_checkpoint() {
    let (prompt, end) = (1_000, 1_500);
    let (at, cut) = (end / BS * BS, tail_cut(prompt, BS));
    let tokens: Vec<u32> = (0..end as u32).collect();
    let next: Vec<u32> = tokens.iter().copied().chain(900_000..900_040).collect();
    let ranks = [
        Rank::turn(0, &tokens, prompt, Some(at)),
        Rank::turn(1, &tokens, prompt, None),
    ];
    assert_eq!(ranks[0].lookup(&next), (at, at));
    assert_eq!(ranks[1].lookup(&next), (at, cut));
    assert_eq!(
        agreed(&ranks, &next),
        [(cut, Agreed::At(cut)), (cut, Agreed::Local)]
    );
}

/// A next turn that diverges inside the previous output (a re-rendered
/// history) matches fewer blocks than the leaf covers and falls back to the
/// prefill checkpoint on both ranks: the leaf is never paired with a shorter
/// match.
#[test]
fn a_match_below_the_leaf_does_not_restore_it() {
    let (prompt, end) = (1_000, 1_500);
    let (at, cut) = (end / BS * BS, tail_cut(prompt, BS));
    let tokens: Vec<u32> = (0..end as u32).collect();
    let mut next: Vec<u32> = tokens.iter().copied().chain(900_000..900_040).collect();
    next[1_300] = 7;
    let ranks = [0, 1].map(|r| Rank::turn(r, &tokens, prompt, Some(at)));
    assert_eq!(ranks[0].lookup(&next), (1_296, cut));
    assert_eq!(agreed(&ranks, &next), [(cut, Agreed::Local); 2]);
}
