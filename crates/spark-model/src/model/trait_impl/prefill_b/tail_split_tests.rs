// SPDX-License-Identifier: AGPL-3.0-only

//! `tail_split_cut` arithmetic (host-side only).

use super::tail_split_cut;

#[test]
fn cut_is_one_block_below_the_last_boundary_under_total() {
    // 15_999 / 64 = 249 blocks -> boundary 15_936, cut one block below.
    assert_eq!(tail_split_cut(16_000, 8_192, 64), Some(15_872));
    // A block-aligned total: the boundary strictly under it is 16_320.
    assert_eq!(tail_split_cut(16_384, 8_192, 64), Some(16_256));
}

#[test]
fn no_cut_at_or_before_the_chunk_start() {
    assert_eq!(tail_split_cut(16_000, 15_872, 64), None);
    assert_eq!(tail_split_cut(16_000, 15_900, 64), None);
    // A prompt of at most two blocks cuts at 0.
    assert_eq!(tail_split_cut(128, 0, 64), None);
    assert_eq!(tail_split_cut(0, 0, 64), None);
}

#[test]
fn the_tail_pass_holds_one_to_two_blocks_and_never_splits_again() {
    for bs in [16, 64] {
        for total in 0..2_000 {
            for start in [0, 1, 60, 64, 512] {
                let Some(cut) = tail_split_cut(total, start, bs) else {
                    continue;
                };
                assert!(start < cut && cut < total && cut % bs == 0);
                // Owners ride this pass; the fused gate needs >= 2 rows.
                assert!(total - cut > bs && total - cut <= 2 * bs);
                assert_eq!(tail_split_cut(total, cut, bs), None);
            }
        }
    }
}
