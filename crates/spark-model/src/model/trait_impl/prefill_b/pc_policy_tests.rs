// SPDX-License-Identifier: AGPL-3.0-only

//! Placement arithmetic for the prefix-cache policy (`pc_policy`).

use super::{branch_checkpoint_at, branch_split_at, tail_cut};

const BS: usize = 16;

/// The warm-turn tail checkpoint sits one block below the last block
/// boundary strictly under the prompt end: for a prompt of `n` tokens,
/// `floor((n-1)/16)*16 - 16`. The next turn's block-floored match lands on
/// `floor((n-1)/16)*16` or one block below it, so the checkpoint is always
/// eligible and the anchor-to-match gap is 0-31 tokens.
#[test]
fn tail_cut_is_one_block_below_the_last_boundary() {
    assert_eq!(tail_cut(40_000, BS), 39_968);
    assert_eq!(tail_cut(40_001, BS), 39_984);
    assert_eq!(tail_cut(40_016, BS), 39_984);
    assert_eq!(tail_cut(40_017, BS), 40_000);
    assert_eq!(tail_cut(17, BS), 0);
    assert_eq!(tail_cut(0, BS), 0);
    for n in 33..2_000 {
        let floor = (n - 1) / BS * BS;
        assert_eq!(tail_cut(n, BS), floor - BS, "n={n}");
        assert!(n - tail_cut(n, BS) > BS && n - tail_cut(n, BS) <= 2 * BS);
    }
}

#[test]
fn branch_checkpoint_only_without_any_restore() {
    // A new session sharing a 30K system prompt, nothing restorable.
    assert_eq!(
        branch_checkpoint_at(30_000, 0, 32_000, BS, 2048),
        Some(30_000)
    );
    // A restore happened (the conversation's own tail, or an earlier
    // branch point): the conversation continues, no extra pass.
    assert_eq!(branch_checkpoint_at(30_000, 29_968, 32_000, BS, 2048), None);
    assert_eq!(branch_checkpoint_at(30_000, 20_000, 32_000, BS, 2048), None);
}

#[test]
fn branch_checkpoint_respects_the_minimum_and_the_tail() {
    assert_eq!(branch_checkpoint_at(2_032, 0, 10_000, BS, 2048), None);
    assert_eq!(
        branch_checkpoint_at(2_048, 0, 10_000, BS, 2048),
        Some(2_048)
    );
    assert_eq!(branch_checkpoint_at(0, 0, 10_000, BS, 0), None);
    // At or past the tail cut the tail checkpoint already covers it: an
    // identical retried prompt (matched == floor) never pays a second split.
    let total = 30_000;
    let cut = tail_cut(total, BS);
    assert_eq!(branch_checkpoint_at(cut, 0, total, BS, 2048), None);
    assert_eq!(branch_checkpoint_at(cut + BS, 0, total, BS, 2048), None);
    assert_eq!(
        branch_checkpoint_at(cut - BS, 0, total, BS, 2048),
        Some(cut - BS)
    );
}

#[test]
fn split_only_strictly_inside_the_chunk() {
    let at = Some(20_000);
    assert_eq!(branch_split_at(at, (16_384, 8_192), false), Some(20_000));
    // On a chunk boundary: the chunk end saves it, no split.
    assert_eq!(branch_split_at(at, (20_000, 8_192), false), None);
    assert_eq!(branch_split_at(at, (11_808, 8_192), false), None);
    // Outside the chunk, no plan, or verify passengers aboard: no split.
    assert_eq!(branch_split_at(at, (0, 8_192), false), None);
    assert_eq!(branch_split_at(at, (24_576, 8_192), false), None);
    assert_eq!(branch_split_at(None, (16_384, 8_192), false), None);
    assert_eq!(branch_split_at(at, (16_384, 8_192), true), None);
}

/// Splitting a chunk at the planned point and re-checking each half never
/// splits again, so the recursion in `pc_branch_split` is exactly one level.
#[test]
fn a_split_chunk_does_not_split_again() {
    let at = Some(20_000);
    let (start, len) = (16_384, 8_192);
    let a = branch_split_at(at, (start, len), false).unwrap();
    assert_eq!(branch_split_at(at, (start, a - start), false), None);
    assert_eq!(branch_split_at(at, (a, start + len - a), false), None);
}
