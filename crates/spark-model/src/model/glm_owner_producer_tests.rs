// SPDX-License-Identifier: AGPL-3.0-only
//! Real issued proposals reach immutable owner-batch producer validation.
use super::*;
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;
use crate::speculative::glm_repair::GlmPairedHandoff;

fn validate_actual_owners(physical: &[usize]) {
    for rank in 0..2 {
        let (f, history) = Four::prepared(rank);
        let shape = GlmOwnerBatchShape::new(physical.len()).unwrap();
        let inputs: Vec<_> = physical
            .iter()
            .map(|&slot| f.f.model.paired_input(&f.states[slot]).unwrap())
            .collect();
        let tokens: Vec<[u32; 5]> = physical
            .iter()
            .map(|&slot| history[slot].issued.as_slice().try_into().unwrap())
            .collect();
        let states: Vec<_> = physical
            .iter()
            .map(|&slot| f.states[slot].proposer_state.as_deref().unwrap())
            .collect();
        let ctx = f.f.model.glm_repair_context();
        let cap: &dyn GlmPairedHandoff = f.f.head.as_ref();
        // Existing actual Single validation establishes valid issued owners.
        for ordinal in 0..physical.len() {
            cap.validate_verify(&inputs[ordinal], &tokens[ordinal], states[ordinal], &ctx)
                .unwrap();
        }
        let before: Vec<_> = (0..4).map(|slot| peer_snapshot(&f, slot)).collect();
        f.f.gpu.clear();
        let result = cap.validate_verify_owners(shape, &inputs, &tokens, &states, &ctx);
        assert!(
            f.f.gpu.trace().is_empty(),
            "immutable validation must issue no GPU work"
        );
        assert_eq!(
            (0..4)
                .map(|slot| peer_snapshot(&f, slot))
                .collect::<Vec<_>>(),
            before
        );
        let facts = result.expect("actual3/4 issued owners must reach owner-batch producer");
        assert!(
            facts[..physical.len()]
                .iter()
                .all(|&(generation, attempt)| generation > 0 && attempt > 0)
        );
        assert!(facts[physical.len()..].iter().all(|&fact| fact == (0, 0)));
        // Validation is not a producer claim: repeat succeeds without state change.
        assert_eq!(
            cap.validate_verify_owners(shape, &inputs, &tokens, &states, &ctx)
                .unwrap(),
            facts
        );
    }
}

#[test]
fn actual_three_owner_producer_accepts_noncontiguous_physical_slots() {
    if flow::isolated(
        "pair_group_tests::owner_producer::actual_three_owner_producer_accepts_noncontiguous_physical_slots",
    ) {
        return;
    }
    validate_actual_owners(&[0, 2, 3]);
    validate_actual_owners(&[1, 2, 3]);
}

#[test]
fn actual_four_owner_producer_accepts_all_issued_owners() {
    if flow::isolated(
        "pair_group_tests::owner_producer::actual_four_owner_producer_accepts_all_issued_owners",
    ) {
        return;
    }
    validate_actual_owners(&[0, 1, 2, 3]);
}

#[test]
fn actual_five_through_eight_issued_owner_validation_preserves_every_owner() {
    if flow::isolated(
        "pair_group_tests::owner_producer::actual_five_through_eight_issued_owner_validation_preserves_every_owner",
    ) {
        return;
    }
    for rank in 0..2 {
        let (f, history) = Eight::prepared(rank);
        let cap: &dyn GlmPairedHandoff = f.f.head.as_ref();
        let ctx = f.f.model.glm_repair_context();
        let before: Vec<_> = (0..8).map(|slot| peer_snapshot(&f, slot)).collect();
        for physical in [
            &[0usize, 2, 4, 6, 7][..],
            &[0, 1, 3, 4, 6, 7][..],
            &[1, 2, 3, 4, 5, 6, 7][..],
            &[0, 1, 2, 3, 4, 5, 6, 7][..],
        ] {
            let shape = GlmOwnerBatchShape::new(physical.len()).unwrap();
            let inputs: Vec<_> = physical
                .iter()
                .map(|&slot| f.f.model.paired_input(&f.states[slot]).unwrap())
                .collect();
            let tokens: Vec<[u32; 5]> = physical
                .iter()
                .map(|&slot| history[slot].issued.as_slice().try_into().unwrap())
                .collect();
            let states: Vec<_> = physical
                .iter()
                .map(|&slot| f.states[slot].proposer_state.as_deref().unwrap())
                .collect();
            for ordinal in 0..physical.len() {
                cap.validate_verify(&inputs[ordinal], &tokens[ordinal], states[ordinal], &ctx)
                    .unwrap();
            }
            f.f.gpu.clear();
            let facts = cap
                .validate_verify_owners(shape, &inputs, &tokens, &states, &ctx)
                .expect(
                    "actual five-through-eight issued owners need immutable producer validation",
                );
            assert!(
                facts.len() >= physical.len(),
                "every live owner needs its actual issued facts"
            );
            assert!(
                facts[..physical.len()]
                    .iter()
                    .all(|&(generation, attempt)| generation > 0 && attempt > 0)
            );
            assert!(facts[physical.len()..].iter().all(|&fact| fact == (0, 0)));
            assert_eq!(
                cap.validate_verify_owners(shape, &inputs, &tokens, &states, &ctx)
                    .unwrap(),
                facts
            );
            let mut bad_tokens = tokens.clone();
            bad_tokens.last_mut().unwrap()[4] ^= 1;
            assert!(
                cap.validate_verify_owners(shape, &inputs, &bad_tokens, &states, &ctx)
                    .is_err()
            );
            assert_eq!(
                cap.validate_verify_owners(shape, &inputs, &tokens, &states, &ctx)
                    .unwrap(),
                facts
            );
            assert!(
                f.f.gpu.trace().is_empty(),
                "immutable validation/refusal must not write"
            );
            assert_eq!(
                (0..8)
                    .map(|slot| peer_snapshot(&f, slot))
                    .collect::<Vec<_>>(),
                before
            );
        }
    }
}
