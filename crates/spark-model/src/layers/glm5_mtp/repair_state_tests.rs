// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::speculative::glm_pair_plan::{BootstrapInput, Limits, Profile};

fn limits() -> Limits {
    Limits::new(
        Profile {
            sequences: 1,
            drafts: 4,
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

fn first() -> ProposalPlan {
    let l = limits();
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
    l.propose(f.state(), 9, 3, 2, 4).unwrap()
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
