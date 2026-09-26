// SPDX-License-Identifier: AGPL-3.0-only
//! Metadata layout and E9/EA width words (host-side only).
use super::*;

const WIDTHS: [usize; 3] = [2, K3_ROWS, owner::MAX_OWNER_ROWS];
/// 262K context at 16-token blocks, and a tiny table.
const MAX_BLOCKS: [usize; 2] = [16_384, 3];

#[test]
fn k3_metadata_layout_matches_the_fixed_width_layout() {
    for mb in MAX_BLOCKS {
        for owners in 1..=MAX_OWNERS {
            let l = MetaLayout::new(owners, K3_ROWS, mb);
            let joint = align(META_BLOCK_TABLE + owners * 3 * mb * 4);
            let own = align(META_BLOCK_TABLE + 3 * mb * 4);
            for o in 0..owners {
                assert_eq!(l.owner_offset(o), META_BASE + joint + o * own);
            }
            assert_eq!(l.end(), META_BASE + joint + owners * own);
        }
    }
}

#[test]
fn metadata_blocks_are_aligned_disjoint_and_hold_their_rows() {
    for mb in MAX_BLOCKS {
        for rows in WIDTHS {
            for owners in 1..=MAX_OWNERS {
                let l = MetaLayout::new(owners, rows, mb);
                let block = |r: usize| META_BLOCK_TABLE + r * mb * 4;
                assert!(l.joint_bytes >= block(owners * rows));
                assert!(l.owner_bytes >= block(rows));
                assert_eq!(META_BASE % 256, 0);
                let mut end = META_BASE + l.joint_bytes;
                for o in 0..owners {
                    let at = l.owner_offset(o);
                    assert_eq!(at % 256, 0, "{owners}x{rows} owner {o}");
                    assert!(at >= end, "{owners}x{rows} owner {o} overlaps");
                    end = at + block(rows);
                }
                assert!(end <= l.end());
            }
        }
    }
    // Wider blocks never shrink the layout.
    let widest = MetaLayout::new(MAX_OWNERS, owner::MAX_OWNER_ROWS, MAX_BLOCKS[0]);
    assert!(widest.end() > MetaLayout::new(MAX_OWNERS, K3_ROWS, MAX_BLOCKS[0]).end());
}

#[test]
fn width_word_keeps_the_k3_wire_and_round_trips_other_widths() {
    for word in 0..MAX_OWNERS + 1 {
        assert_eq!(encode_width(word, K3_ROWS), word as u32);
        assert_eq!(decode_width(word as u32), (word, K3_ROWS));
        for rows in WIDTHS {
            assert_eq!(decode_width(encode_width(word, rows)), (word, rows));
        }
    }
    assert_eq!(encode_width(4, 8), 0x0008_0004);
    assert_eq!(encode_width(1, 2), 0x0002_0001);
}

#[test]
fn owner_index_is_bounded_by_owners_and_width() {
    for rows in WIDTHS {
        for o in 0..MAX_OWNERS {
            assert!(owner_supported(o, rows));
        }
        assert!(!owner_supported(MAX_OWNERS, rows));
    }
    for rows in [0, 1, owner::MAX_OWNER_ROWS + 1] {
        assert!(!owner_supported(0, rows));
    }
}
