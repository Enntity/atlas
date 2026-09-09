// SPDX-License-Identifier: AGPL-3.0-only
//! Actual C5–C8 target/verdict ownership; byte sentinels, not GPU numerics.
use super::*;
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;

#[test]
fn actual_five_through_eight_compute_and_high_slot_drains_preserve_owners() {
    if flow::isolated(
        "pair_group_tests::owner8_compute::actual_five_through_eight_compute_and_high_slot_drains_preserve_owners",
    ) {
        return;
    }
    for rank in 0..2 {
        for physical in [
            &[0usize, 2, 4, 6, 7][..],
            &[0, 1, 3, 4, 6, 7][..],
            &[1, 2, 3, 4, 5, 6, 7][..],
            &[0, 1, 2, 3, 4, 5, 6, 7][..],
            &[1, 5, 7][..],
            &[0, 2, 5, 7][..],
        ] {
            exercise_cohort(rank, physical);
        }
    }
}

fn exercise_cohort(rank: usize, physical: &[usize]) {
    let (mut f, histories) = Eight::prepared(rank);
    let shape = GlmOwnerBatchShape::new(physical.len()).unwrap();
    let tokens: Vec<[u32; 5]> = physical
        .iter()
        .map(|&slot| histories[slot].issued.as_slice().try_into().unwrap())
        .collect();
    let untouched: Vec<_> = (0..8)
        .filter(|slot| !physical.contains(slot))
        .map(|slot| (slot, peer_snapshot(&f, slot)))
        .collect();
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
            .expect("actual selected owner count must traverse every target");
    assert!(predictions.len() >= physical.len());
    for (ordinal, &slot) in physical.iter().enumerate() {
        let expected = flow::normalized(&histories[slot]);
        assert_eq!(
            predictions[ordinal],
            std::array::from_fn(|row| u32::from(expected[row][0] % 8)),
            "rank {rank}, physical {slot}, packed ordinal {ordinal}"
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
                *bytes,
                "rank {rank}, physical {slot}, target row {row}"
            );
        }
    }
    assert!(
        predictions[physical.len()..]
            .iter()
            .all(|row| *row == [0; 5])
    );
    for (slot, before) in &untouched {
        assert_eq!(&peer_snapshot(&f, *slot), before);
    }
    assert!(
        !f.f.gpu
            .trace()
            .iter()
            .any(|event| matches!(event, Event::Alloc(_, _) | Event::Free(_))),
        "owner target traversal must use existing device storage"
    );
    assert_eq!(
        f.states
            .each_ref()
            .map(|s| (s.tokens.capacity(), s.block_table.capacity())),
        capacity
    );
    let accepted = &([0usize, 4, 1, 3, 2, 0, 4, 2])[..physical.len()];
    let mut selected: Vec<_> = f
        .states
        .iter_mut()
        .filter(|seq| physical.contains(&seq.slot_idx))
        .collect();
    f.f.model
        .owner_finish_verify(shape, &mut selected, &tokens, accepted)
        .expect("actual verdict must commit each selected physical owner");
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
    for (slot, before) in &untouched {
        assert_eq!(&peer_snapshot(&f, *slot), before);
    }
}

#[test]
fn malformed_eighth_verdict_preserves_all_owners_before_terminal_refusal() {
    if flow::isolated(
        "pair_group_tests::owner8_compute::malformed_eighth_verdict_preserves_all_owners_before_terminal_refusal",
    ) {
        return;
    }
    for rank in 0..2 {
        let (mut f, histories) = Eight::prepared(rank);
        let shape = GlmOwnerBatchShape::new(8).unwrap();
        let mut tokens: [[u32; 5]; 8] = histories
            .each_ref()
            .map(|h| h.issued.as_slice().try_into().unwrap());
        f.f.model
            .owner_compute_verify(shape, &mut f.states.each_mut(), &tokens)
            .expect("actual eight-owner producer must precede malformed verdict");
        let before: Vec<_> = (0..8).map(|slot| peer_snapshot(&f, slot)).collect();
        let proposer_addresses = f
            .states
            .each_ref()
            .map(|s| flow::private(s) as *const _ as usize);
        let target_free = f.f.model.kv_cache.lock().num_free_blocks();
        // All counts are valid. Only the final owner's canonical issued token
        // is wrong, so validation must inspect the whole cohort before detach.
        tokens[7][4] ^= 1;
        f.f.gpu.clear();
        let error =
            f.f.model
                .owner_finish_verify(
                    shape,
                    &mut f.states.each_mut(),
                    &tokens,
                    &[0, 4, 1, 3, 2, 0, 4, 2],
                )
                .unwrap_err();
        assert!(
            format!("{error:#}").contains("owner verdict canonical issued append changed"),
            "must reach the last owner's actual canonical verdict check: {error:#}"
        );
        assert!(f.f.gpu.trace().is_empty());
        assert_eq!(f.f.model.kv_cache.lock().num_free_blocks(), target_free);
        assert_eq!(
            f.states
                .each_ref()
                .map(|s| flow::private(s) as *const _ as usize),
            proposer_addresses
        );
        // Read retained raw spans, not a pool API that the terminal latch now
        // correctly refuses. Both selected target and private owners stay live.
        for (slot, saved) in before.iter().enumerate() {
            let seq = &f.states[slot];
            assert_eq!(seq.slot_idx, saved.slot);
            assert_eq!(seq.seq_len, saved.target_len);
            assert_eq!(seq.tokens, saved.target_tokens);
            assert_eq!(seq.block_table, saved.target_blocks);
            assert_eq!(seq.mtp_capture_gen, saved.capture_generation);
            assert_eq!(flow::private(seq).seq_len, saved.private_len);
            assert_eq!(flow::private(seq).block_table, saved.private_blocks);
            for (pointer, bytes) in &saved.spans {
                assert_eq!(f.f.gpu.read_span(*pointer, bytes.len()), *bytes);
            }
        }
        for seq in &mut f.states {
            assert!(
                f.f.model
                    .run_mtp_propose_inner(7, seq.seq_len, 4, seq, None)
                    .is_err()
            );
        }
        assert!(
            f.f.gpu.trace().is_empty(),
            "last-owner verdict error must stop every later owner before work"
        );
    }
}
