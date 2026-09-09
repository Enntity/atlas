// SPDX-License-Identifier: AGPL-3.0-only
//! Actual wider model traversal with byte-sentinel layers, not GPU numerics.
use super::*;
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;

#[test]
fn malformed_last_owner_verdict_detaches_nothing_and_fails_session() {
    if flow::isolated(
        "pair_group_tests::owner_compute::malformed_last_owner_verdict_detaches_nothing_and_fails_session",
    ) {
        return;
    }
    let (mut f, histories) = Four::prepare_fixture(Fixture::new_owner_compute(0));
    let shape = GlmOwnerBatchShape::new(4).unwrap();
    let tokens: [[u32; 5]; 4] = histories
        .each_ref()
        .map(|h| h.issued.as_slice().try_into().unwrap());
    f.f.model
        .owner_compute_verify(shape, &mut f.states.each_mut(), &tokens)
        .unwrap();
    let saved = f.states.each_ref().map(|s| (s.seq_len, s.tokens.clone()));
    let slab = f.f.gpu.slab_for_owners(4);
    let bytes = f.f.gpu.read_span(slab, 4 * 6 * ROW_BYTES);
    f.f.gpu.clear();
    assert!(
        f.f.model
            .owner_finish_verify(shape, &mut f.states.each_mut(), &tokens, &[0, 1, 4, 5])
            .is_err()
    );
    assert!(
        f.f.gpu.trace().is_empty(),
        "last invalid verdict must not detach an earlier owner"
    );
    assert_eq!(
        f.states.each_ref().map(|s| (s.seq_len, s.tokens.clone())),
        saved
    );
    assert_eq!(f.f.gpu.read_span(slab, bytes.len()), bytes);
    for seq in &mut f.states {
        assert!(
            f.f.model
                .run_mtp_propose_inner(7, seq.seq_len, 4, seq, None)
                .is_err()
        );
    }
    assert!(
        f.f.gpu.trace().is_empty(),
        "failed cohort cannot start another proposal"
    );
}

#[test]
fn actual_wider_model_produces_each_physical_owner_rows() {
    if flow::isolated(
        "pair_group_tests::owner_compute::actual_wider_model_produces_each_physical_owner_rows",
    ) {
        return;
    }
    for rank in 0..2 {
        for physical in [&[0usize, 2, 3][..], &[0, 1, 2, 3][..]] {
            let (mut f, histories) = Four::prepare_fixture(Fixture::new_owner_compute(rank));
            let shape = GlmOwnerBatchShape::new(physical.len()).unwrap();
            let tokens: Vec<[u32; 5]> = physical
                .iter()
                .map(|&slot| histories[slot].issued.as_slice().try_into().unwrap())
                .collect();
            let untouched = (physical.len() == 3).then(|| peer_snapshot(&f, 1));
            let capacity = f
                .states
                .each_ref()
                .map(|s| (s.tokens.capacity(), s.block_table.capacity()));
            f.f.gpu.clear();
            let mut selected: Vec<_> = f
                .states
                .iter_mut()
                .filter(|seq| physical.contains(&seq.slot_idx))
                .collect();
            let predictions =
                f.f.model
                    .owner_compute_verify(shape, &mut selected, &tokens)
                    .expect("actual3/4-owner model must traverse every selected target");
            for (ordinal, &slot) in physical.iter().enumerate() {
                let expected = flow::normalized(&histories[slot]);
                assert_eq!(
                    predictions[ordinal],
                    std::array::from_fn(|row| u32::from(expected[row][0] % 8))
                );
                assert_eq!(f.states[slot].seq_len, histories[slot].base + 5);
                assert_eq!(
                    &f.states[slot].tokens[histories[slot].base..],
                    &tokens[ordinal]
                );
                for (row, bytes) in expected.iter().enumerate() {
                    assert_eq!(
                        f.f.gpu.read_span(
                            f.f.model
                                .buffers
                                .norm_output()
                                .offset((ordinal * 5 + row) * ROW_BYTES),
                            ROW_BYTES
                        ),
                        *bytes
                    );
                }
            }
            assert!(
                predictions[physical.len()..]
                    .iter()
                    .all(|row| *row == [0; 5])
            );
            if let Some(before) = untouched {
                assert_eq!(peer_snapshot(&f, 1), before);
            }
            assert!(
                !f.f.gpu
                    .trace()
                    .iter()
                    .any(|event| matches!(event, Event::Alloc(_, _) | Event::Free(_))),
                "wider target traversal must not allocate/free device storage"
            );
            assert_eq!(
                f.states
                    .each_ref()
                    .map(|s| (s.tokens.capacity(), s.block_table.capacity())),
                capacity
            );
            let accepted = &([0usize, 1, 4, 2])[..physical.len()];
            let mut selected: Vec<_> = f
                .states
                .iter_mut()
                .filter(|seq| physical.contains(&seq.slot_idx))
                .collect();
            f.f.model
                .owner_finish_verify(shape, &mut selected, &tokens, accepted)
                .expect("actual wider verdict must commit every selected owner");
            for (ordinal, &slot) in physical.iter().enumerate().rev() {
                let end = histories[slot].base + accepted[ordinal] + 1;
                let expected = flow::normalized(&histories[slot]);
                assert_eq!(f.states[slot].seq_len, end);
                assert_eq!(
                    &f.states[slot].tokens[histories[slot].base..],
                    &tokens[ordinal][..accepted[ordinal] + 1]
                );
                assert_eq!(f.slab_row(slot, 5), expected[accepted[ordinal]]);
                for (row, bytes) in expected.iter().enumerate().take(accepted[ordinal]) {
                    assert_eq!(f.slab_row(slot, row + 1), *bytes);
                }
                let seed = predictions[ordinal][accepted[ordinal]];
                assert_eq!(
                    f.f.model
                        .run_mtp_propose_inner(seed, end, 4, &mut f.states[slot], None)
                        .unwrap()
                        .len(),
                    4
                );
            }
        }
    }
}
