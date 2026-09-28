// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::speculative::glm_pair_plan::{BootstrapInput, Limits, Profile};

fn limits(drafts: usize) -> Limits {
    Limits::new(
        Profile {
            sequences: 1,
            drafts,
            continuous: true,
            grammar: false,
            adaptive_depth: false,
            catchup: false,
            carry: false,
            prefix_reuse: false,
        },
        2044,
        2048,
        4,
    )
    .unwrap()
}

fn first_at_depth(drafts: usize) -> ProposalPlan {
    let l = limits(drafts);
    let f = l
        .bootstrap(BootstrapInput {
            generation: 9,
            capture_generation: 9,
            prompt_tokens: 2,
            target_position: 3,
            token_rows: 3,
            normalized_hidden_rows: 2,
            cached_rows: 1,
        })
        .unwrap();
    l.propose(f.state(), 9, 3, 2, drafts).unwrap()
}

fn first() -> ProposalPlan {
    first_at_depth(4)
}

#[test]
fn verified_state_keeps_zero_seed_and_full_accept_fifth_row() {
    for accepted in 0..=4 {
        let mut phase = RepairPhase::Proposed(first());
        phase
            .record(9, 9, 3, &[7, 1, 2, 3, 4], accepted, 4 + accepted, 6, 5)
            .unwrap();
        assert!(
            phase.pending(9, 4 + accepted, accepted).is_err(),
            "trim required"
        );
        phase.acknowledge(accepted).unwrap();
        let p = phase.pending(9, 4 + accepted, accepted).unwrap();
        assert_eq!(p.plan.state().cache_rows(), 3 + accepted);
        assert_eq!(&p.tokens[1..1 + accepted], &[1, 2, 3, 4][..accepted]);
        assert!(
            phase.acknowledge(accepted).is_err(),
            "duplicate trim rejected"
        );
    }
}

#[test]
fn record_and_e1_reject_stale_metadata_without_changing_phase() {
    let mut phase = RepairPhase::Proposed(first());
    assert!(phase.record(9, 9, 2, &[7, 1, 2, 3, 4], 2, 6, 6, 5).is_err());
    assert!(matches!(phase, RepairPhase::Proposed(_)));
    phase.record(9, 9, 3, &[7, 1, 2, 3, 4], 2, 6, 6, 5).unwrap();
    assert!(
        phase.acknowledge(0).is_err(),
        "zero discard cannot mimic verdict"
    );
    phase.acknowledge(2).unwrap();
    assert!(phase.pending(10, 6, 2).is_err());
    assert!(phase.pending(9, 7, 2).is_err());
    assert!(phase.pending(9, 6, 1).is_err());
    assert!(phase.record(9, 9, 3, &[7, 1, 2, 3, 4], 2, 6, 6, 5).is_err());
    assert!(phase.pending(9, 6, 2).is_ok());
}

#[test]
fn fresh_or_failed_phase_cannot_be_treated_as_verified_zero() {
    let mut phase = RepairPhase::Capture;
    assert!(phase.acknowledge(0).is_err());
    assert!(phase.pending(9, 3, 0).is_err());
    phase = RepairPhase::Failed;
    assert!(phase.pending(9, 3, 0).is_err());
    assert!(phase.acknowledge(0).is_err());
}

#[test]
fn single_draft_verdict_owns_width_and_requires_record_before_trim() {
    for accepted in 0..=1 {
        let mut phase = RepairPhase::Proposed(first_at_depth(1));
        assert!(phase.acknowledge(accepted).is_err());
        assert!(
            phase
                .record(9, 9, 3, &[7, 1, 2, 3, 4], accepted, 4 + accepted, 3, 5)
                .is_err()
        );
        assert!(phase.record(9, 9, 3, &[7, 1], 2, 6, 3, 2).is_err());
        phase
            .record(9, 9, 3, &[7, 1], accepted, 4 + accepted, 3, 2)
            .unwrap();
        phase.acknowledge(accepted).unwrap();
        let p = phase.pending(9, 4 + accepted, accepted).unwrap();
        assert_eq!(p.drafts, 1);
        assert_eq!(p.plan.state().cache_rows(), 3 + accepted);
        assert!(phase.pending(10, 4 + accepted, accepted).is_err());
    }
}

#[test]
fn two_draft_verdict_reject_partial_full_and_next_cycle_keep_owned_width() {
    for accepted in 0..=2 {
        let mut proposal = first_at_depth(2);
        for next_accepted in [accepted, 2, 0] {
            let base = proposal.position();
            let cache = proposal.speculative_cache_end();
            let position = base + next_accepted + 1;
            let mut phase = RepairPhase::Proposed(proposal);
            assert!(phase.acknowledge(next_accepted).is_err());
            for tokens in [&[7, 1][..], &[7, 1, 2, 3, 4][..]] {
                assert!(
                    phase
                        .record(9, 9, base, tokens, next_accepted, position, cache, 3)
                        .is_err()
                );
            }
            assert!(
                phase
                    .record(9, 9, base, &[7, 1, 2], 3, base + 4, cache, 3)
                    .is_err()
            );
            assert!(
                phase
                    .record(9, 9, base, &[7, 1, 2], next_accepted, position, cache, 2)
                    .is_err()
            );
            phase
                .record(9, 9, base, &[7, 1, 2], next_accepted, position, cache, 3)
                .unwrap();
            phase.acknowledge(next_accepted).unwrap();
            let pending = phase.pending(9, position, next_accepted).unwrap();
            assert_eq!(pending.drafts, 2);
            assert_eq!(pending.plan.state().cache_rows(), cache - 1 + next_accepted);
            assert_eq!(pending.plan.write().map_or(0, |w| w.rows()), next_accepted);
            assert!(pending.plan.keep_seed());
            assert!(phase.pending(10, position, next_accepted).is_err());
            let state = pending.plan.state();
            assert!(
                limits(2)
                    .propose(state, 9, position, state.cache_rows(), 1)
                    .is_err()
            );
            proposal = limits(2)
                .propose(state, 9, position, state.cache_rows(), 2)
                .unwrap();
        }
    }
}
