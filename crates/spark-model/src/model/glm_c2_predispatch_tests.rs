// SPDX-License-Identifier: AGPL-3.0-only
//! Actual constructed Model capability: repeatable checks, no receipt fabrication.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::traits::Model;

fn snapshot(
    f: &Fixture,
) -> (
    Vec<u8>,
    Vec<Vec<Vec<u8>>>,
    Vec<(usize, Vec<u32>, Vec<u32>)>,
    usize,
) {
    (
        f.gpu.read_span(f.gpu.slab(), SLAB_BYTES),
        (0..2)
            .map(|i| flow::bytes(f, i, flow::private(&f.seqs[i]).seq_len))
            .collect(),
        f.seqs
            .iter()
            .map(|seq| (seq.seq_len, seq.tokens.clone(), seq.block_table.clone()))
            .collect(),
        f.model.kv_cache.lock().num_free_blocks(),
    )
}

#[test]
fn actual_selected_capability_repeated_verify_checks_do_not_claim_or_mutate() {
    if flow::isolated(
        "predispatch_tests::actual_selected_capability_repeated_verify_checks_do_not_claim_or_mutate",
    ) {
        return;
    }
    for rank in 0..2 {
        let (mut f, history) = flow::prepare(rank, [0, 1]);
        let before = snapshot(&f);
        f.gpu.clear();
        let model: &dyn Model = &f.model;
        let execution = model
            .glm_paired_execution()
            .expect("actual paired head must expose checked execution");
        for _ in 0..3 {
            for owner in 0..2 {
                execution
                    .validate_verify(&f.seqs[owner], &history[owner].issued)
                    .unwrap();
                assert!(
                    execution
                        .validate_verify(&f.seqs[owner], &history[1 - owner].issued)
                        .is_err()
                );
            }
        }
        assert!(f.gpu.trace().is_empty());
        assert_eq!(snapshot(&f), before);
        f.model
            .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
            .unwrap();
        f.gpu.clear();
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .validate_verify(&f.seqs[1], &history[1].issued)
                .is_err()
        );
        assert!(f.gpu.trace().is_empty());
    }
}

#[test]
fn actual_bootstrap_proposal_preflight_does_not_consume_missing_pair_or_bonus() {
    if flow::isolated(
        "predispatch_tests::actual_bootstrap_proposal_preflight_does_not_consume_missing_pair_or_bonus",
    ) {
        return;
    }
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        for (owner, prompt) in [[1, 2, 3, 4], [4, 3, 2, 1]].iter().enumerate() {
            f.model.prefill(prompt, &mut f.seqs[owner], CALLER).unwrap();
            f.gpu.clear();
            assert!(
                f.model
                    .glm_paired_execution()
                    .unwrap()
                    .validate_propose(&f.seqs[owner], 7, 5, 4, None)
                    .is_err()
            );
            assert!(f.gpu.trace().is_empty());
            f.model
                .decode(5 + owner as u32, &mut f.seqs[owner], CALLER)
                .unwrap();
        }
        let before = snapshot(&f);
        f.gpu.clear();
        for _ in 0..3 {
            for owner in 0..2 {
                let cap = f.model.glm_paired_execution().unwrap();
                cap.validate_propose(&f.seqs[owner], 7 - owner as u32, 5, 4, None)
                    .unwrap();
                assert!(cap.validate_propose(&f.seqs[owner], 7, 4, 4, None).is_err());
                assert!(cap.validate_propose(&f.seqs[owner], 7, 5, 3, None).is_err());
                assert!(cap.validate_propose(&f.seqs[owner], 8, 5, 4, None).is_err());
                assert!(
                    cap.validate_propose(&f.seqs[owner], 7, 5, 4, Some(&[0]))
                        .is_err()
                );
            }
        }
        assert!(f.gpu.trace().is_empty());
        assert_eq!(snapshot(&f), before);
        for owner in 0..2 {
            let drafts = f
                .model
                .run_mtp_propose_inner(7 - owner as u32, 5, 4, &mut f.seqs[owner], None)
                .unwrap();
            let mut issued = vec![7 - owner as u32];
            issued.extend(drafts);
            f.model
                .glm_paired_execution()
                .unwrap()
                .validate_verify(&f.seqs[owner], &issued)
                .unwrap();
        }
    }
}

#[test]
fn actual_pending_all_acceptances_preflight_preserves_acknowledgements_and_rows() {
    if flow::isolated(
        "predispatch_tests::actual_pending_all_acceptances_preflight_preserves_acknowledgements_and_rows",
    ) {
        return;
    }
    for rank in 0..2 {
        for accepted in 0..5 {
            let (mut f, mut history) = flow::prepare(rank, [0, 1]);
            flow::head_verdict(&mut f, 0, &history[0], accepted);
            let position = f.seqs[0].seq_len;
            f.gpu.clear();
            assert!(
                f.model
                    .glm_paired_execution()
                    .unwrap()
                    .validate_propose(&f.seqs[0], 7, position, 4, None)
                    .is_err()
            );
            assert!(f.gpu.trace().is_empty());
            flow::acknowledge(&mut f, 0, accepted, accepted % 2 == 0);
            let before = snapshot(&f);
            f.gpu.clear();
            for _ in 0..3 {
                f.model
                    .glm_paired_execution()
                    .unwrap()
                    .validate_propose(&f.seqs[0], 7, position, 4, None)
                    .unwrap();
            }
            assert!(f.gpu.trace().is_empty());
            assert_eq!(snapshot(&f), before);
            flow::continue_owner(&mut f, 0, &mut history[0], accepted);
        }
    }
}

fn drain_target(f: &Fixture) -> Vec<u32> {
    let mut cache = f.model.kv_cache.lock();
    let mut blocks = Vec::new();
    while let Some(block) = cache.try_alloc_block() {
        blocks.push(block);
    }
    blocks
}

#[test]
fn invalid_selected_profile_and_map_never_turn_capability_into_legacy_fallback() {
    if flow::isolated(
        "predispatch_tests::invalid_selected_profile_and_map_never_turn_capability_into_legacy_fallback",
    ) {
        return;
    }
    for rank in 0..2 {
        let (mut f, history) = flow::prepare(rank, [0, 1]);
        let before = snapshot(&f);
        f.gpu.clear();
        let original = f.model.config.model_type.clone();
        f.model.config.model_type = "foreign_model".into();
        let cap = f
            .model
            .glm_paired_execution()
            .expect("invalid selected model is not legacy");
        assert!(cap.validate_verify(&f.seqs[0], &history[0].issued).is_err());
        f.model.config.model_type = original;
        let original = f.seqs[0].block_table[0];
        f.seqs[0].block_table[0] = 256;
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .validate_verify(&f.seqs[0], &history[0].issued)
                .is_err()
        );
        f.seqs[0].block_table[0] = original;
        let original = f.seqs[0].tokens[0];
        f.seqs[0].tokens[0] = (original + 1) % 8;
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .validate_verify(&f.seqs[0], &history[0].issued)
                .is_err()
        );
        f.seqs[0].tokens[0] = original;
        assert!(f.gpu.trace().is_empty());
        assert_eq!(snapshot(&f), before);
        f.model
            .glm_paired_execution()
            .unwrap()
            .validate_verify(&f.seqs[0], &history[0].issued)
            .unwrap();
        f.model
            .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
            .unwrap();
    }
}

#[test]
fn actual_k5_exhaustion_refuses_before_claim_and_capacity_restoration_can_execute() {
    if flow::isolated(
        "predispatch_tests::actual_k5_exhaustion_refuses_before_claim_and_capacity_restoration_can_execute",
    ) {
        return;
    }
    for rank in 0..2 {
        let (mut f, mut history) = flow::prepare(rank, [0, 1]);
        for _ in 0..2 {
            flow::head_verdict(&mut f, 0, &history[0], 4);
            flow::acknowledge(&mut f, 0, 4, false);
            flow::continue_owner(&mut f, 0, &mut history[0], 4);
        }
        assert_eq!(history[0].base, 15);
        assert_eq!(f.seqs[0].block_table.len(), 1);
        let held = drain_target(&f);
        let before = snapshot(&f);
        f.gpu.clear();
        assert!(
            f.model
                .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
                .is_err()
        );
        assert!(f.gpu.trace().is_empty());
        assert_eq!(
            snapshot(&f),
            before,
            "capacity failure must not consume the live owner"
        );
        f.model.kv_cache.lock().free_blocks(&held);
        f.model
            .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
            .unwrap();
    }
}

#[test]
fn actual_pending_proposal_checks_following_target_budget_without_claiming_blocks() {
    if flow::isolated(
        "predispatch_tests::actual_pending_proposal_checks_following_target_budget_without_claiming_blocks",
    ) {
        return;
    }
    for rank in 0..2 {
        let (mut f, mut history) = flow::prepare(rank, [0, 1]);
        flow::head_verdict(&mut f, 0, &history[0], 4);
        flow::acknowledge(&mut f, 0, 4, false);
        flow::continue_owner(&mut f, 0, &mut history[0], 4);
        flow::head_verdict(&mut f, 0, &history[0], 4);
        flow::acknowledge(&mut f, 0, 4, false);
        assert_eq!(f.seqs[0].seq_len, 15);
        let held = drain_target(&f);
        let before = snapshot(&f);
        f.gpu.clear();
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .validate_propose(&f.seqs[0], 7, 15, 4, None)
                .is_err()
        );
        assert!(
            f.model
                .run_mtp_propose_inner(7, 15, 4, &mut f.seqs[0], None)
                .is_err()
        );
        assert!(f.gpu.trace().is_empty());
        assert_eq!(snapshot(&f), before);
        f.model.kv_cache.lock().free_blocks(&held);
        flow::continue_owner(&mut f, 0, &mut history[0], 4);
    }
}
