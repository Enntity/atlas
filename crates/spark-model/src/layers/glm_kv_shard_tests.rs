// SPDX-License-Identifier: AGPL-3.0-only

//! Scratch layout, policy and LSE-merge math of the GLM KV shard. The merge
//! tests run a CPU mirror of the split kernel's partial semantics and of
//! `glm_sparse_decode_split_merge` against plain softmax attention.

use super::*;

#[test]
fn flag_is_explicit() {
    assert!(!parse("F", None).unwrap());
    assert!(!parse("F", Some("0")).unwrap());
    assert!(parse("F", Some("1")).unwrap());
    for bad in ["", "true", "2", " 1"] {
        assert!(parse("F", Some(bad)).is_err());
    }
}

fn disjoint(regions: &[(usize, usize)]) -> bool {
    let mut sorted = regions.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0)
}

#[test]
fn merge_layout_regions_are_aligned_and_disjoint() {
    for rows in 1..=MERGE_MAX_ROWS as u32 {
        let splits = merge_splits(rows);
        assert!((1..=MAX_SPLITS).contains(&splits), "rows {rows}");
        let m = MergeLayout::new(rows, splits);
        let (r, s) = (rows as usize, splits as usize);
        let part = r * 32 * 512 * 4;
        let lse = r * 32 * 4;
        let regions = [
            (m.q_peer, r * 32 * 512 * 2),
            (m.ids, r * 2051 * 4),
            (m.own_o, (s + 1) * part),
            (m.own_lse, (s + 1) * lse),
            (m.out_lse, lse),
            (m.peer_o, s * part),
            (m.peer_lse, s * lse),
            (m.send, MergeLayout::partial_bytes(rows)),
            (m.recv, MergeLayout::partial_bytes(rows)),
        ];
        assert!(disjoint(&regions), "rows {rows}");
        assert!(regions.iter().all(|&(o, _)| o % 256 == 0));
        assert!(regions.iter().all(|&(o, n)| o + n <= m.total));
    }
}

#[test]
fn scratch_layout_holds_views_pieces_and_the_widest_owner() {
    let bytes = 16 * 528;
    let l = ScratchLayout::new(32772, 8256, bytes);
    let widest = (1..=MERGE_MAX_ROWS as u32)
        .map(|r| MergeLayout::new(r, merge_splits(r)).total)
        .max()
        .unwrap();
    let work = (2 * (PIECE_BLOCKS * bytes).next_multiple_of(256)).max(widest);
    let regions = [
        (l.view, 32772 * bytes),
        (l.mine_slot, 32772 * 4),
        (l.mine_dst, 32772 * 4),
        (l.peer_dst, 32772 * 4),
        (l.check, 16),
        (l.slots, 8256 * 8),
        (l.work, work),
    ];
    assert!(disjoint(&regions));
    assert_eq!(l.total, l.work + work.next_multiple_of(256));
    // 512K tokens of fp8_g128 latents: a 264 MiB view + ~38 MiB of work.
    assert!(l.total < 310 << 20, "{}", l.total);
    assert!(widest < 40 << 20, "{widest}");
}

#[test]
fn table_hash_distinguishes_order_and_length() {
    assert_ne!(table_hash(&[1, 2]), table_hash(&[2, 1]));
    assert_ne!(table_hash(&[0]), table_hash(&[0, 0]));
    assert_eq!(table_hash(&[5, 9, 3]), table_hash(&[5, 9, 3]));
}

/// One partition's normalized output and natural LSE (`-inf`, zeros when
/// it selects nothing): the `*_split` kernel's contract.
fn partial(q: &[f32], keys: &[Vec<f32>], ids: &[Option<usize>], scale: f32) -> (Vec<f32>, f32) {
    let scores: Vec<(usize, f32)> = ids
        .iter()
        .flatten()
        .map(|&t| {
            (
                t,
                scale * q.iter().zip(&keys[t]).map(|(a, b)| a * b).sum::<f32>(),
            )
        })
        .collect();
    let Some(m) = scores.iter().map(|s| s.1).reduce(f32::max) else {
        return (vec![0.0; q.len()], f32::NEG_INFINITY);
    };
    let mut out = vec![0.0; q.len()];
    let mut l = 0.0f32;
    for &(t, s) in &scores {
        let p = (s - m).exp();
        l += p;
        out.iter_mut().zip(&keys[t]).for_each(|(o, v)| *o += p * v);
    }
    out.iter_mut().for_each(|o| *o /= l);
    (out, m + l.ln())
}

/// CPU mirror of `glm_sparse_decode_split_merge` for one (row, head).
fn merge(parts: &[(Vec<f32>, f32)]) -> (Vec<f32>, f32) {
    let dim = parts[0].0.len();
    let live: Vec<&(Vec<f32>, f32)> = parts.iter().filter(|p| p.1 != f32::NEG_INFINITY).collect();
    let Some(mx) = live.iter().map(|p| p.1).reduce(f32::max) else {
        return (vec![0.0; dim], f32::NEG_INFINITY);
    };
    let weights: Vec<f32> = live.iter().map(|p| (p.1 - mx).exp()).collect();
    let sum: f32 = weights.iter().sum();
    let mut out = vec![0.0; dim];
    for (p, w) in live.iter().zip(&weights) {
        out.iter_mut()
            .zip(&p.0)
            .for_each(|(o, v)| *o += w / sum * v);
    }
    (out, mx + sum.ln())
}

/// Deterministic pseudo-random values in [-1, 1).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// A row's sharded attention: each rank's `splits` partitions over the
/// selected IDs it owns; the peer's heads get one FP32-merged partial, which
/// joins the own partitions in the final merge.
fn sharded(
    q: &[f32],
    keys: &[Vec<f32>],
    selected: &[i32],
    owner: impl Fn(usize) -> usize,
    rank: usize,
    splits: usize,
    scale: f32,
) -> Vec<f32> {
    let local = |r: usize| -> Vec<Option<usize>> {
        selected
            .iter()
            .map(|&t| (t >= 0 && owner(t as usize) == r).then_some(t as usize))
            .collect()
    };
    let parts_of = |r: usize| -> Vec<(Vec<f32>, f32)> {
        let ids = local(r);
        let per = ids.len().div_ceil(splits);
        ids.chunks(per.max(1))
            .map(|c| partial(q, keys, c, scale))
            .collect()
    };
    let mut own = parts_of(rank);
    own.push(merge(&parts_of(1 - rank)));
    merge(&own).0
}

#[test]
fn owner_split_lse_merge_equals_full_softmax_attention() {
    let (dim, tokens) = (64usize, 300usize);
    let keys: Vec<Vec<f32>> = (0..tokens).map(|t| values(t as u64 + 7, dim)).collect();
    let q = values(99, dim);
    let scale = 0.0625;
    // A sparse selection with padding, blocks of 16 owned by parity.
    let mut selected: Vec<i32> = (0..tokens as i32).filter(|t| t % 3 != 1).collect();
    selected.extend([-1, -1, -1]);
    let all: Vec<Option<usize>> = selected
        .iter()
        .map(|&t| (t >= 0).then_some(t as usize))
        .collect();
    let (want, _) = partial(&q, &keys, &all, scale);
    let owners: [fn(usize) -> usize; 2] = [|t| (t / 16) % 2, |t| usize::from(t >= 250)];
    for block_owner in owners {
        for rank in 0..2 {
            for splits in [1, 4, 15] {
                let got = sharded(&q, &keys, &selected, block_owner, rank, splits, scale);
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "{g} vs {w}");
                }
            }
        }
    }
}

#[test]
fn a_rank_owning_nothing_contributes_nothing() {
    let keys: Vec<Vec<f32>> = (0..20).map(|t| values(t + 1, 8)).collect();
    let q = values(5, 8);
    let selected: Vec<i32> = (0..20).collect();
    let all: Vec<Option<usize>> = (0..20).map(Some).collect();
    let (want, _) = partial(&q, &keys, &all, 0.5);
    for rank in 0..2 {
        let got = sharded(&q, &keys, &selected, |_| 1, rank, 3, 0.5);
        assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-6));
    }
    // Nothing selected at all: zeros, as the unsharded kernel writes.
    let none = sharded(&q, &keys, &[-1, -1], |_| 0, 0, 2, 0.5);
    assert!(none.iter().all(|&v| v == 0.0));
}
