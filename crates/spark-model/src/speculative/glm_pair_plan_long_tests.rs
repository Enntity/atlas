// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn served_36k_profile_is_bounded_before_mtp2_limits() {
    let served_context = 36_864;
    let repair_context = crate::speculative::glm_repair_policy::repair_context(served_context);
    assert_eq!(repair_context, 32_768);
    assert!(
        Limits::new(
            Profile {
                drafts: 2,
                ..profile()
            },
            repair_context,
            32_784,
            2,
        )
        .is_ok()
    );
    assert!(
        Limits::new(
            Profile {
                drafts: 2,
                ..profile()
            },
            served_context,
            32_784,
            2,
        )
        .is_err()
    );
}

#[test]
fn two_draft_limits_cover_all_verdicts_and_keep_fixed_width() {
    let limits = Limits::new(
        Profile {
            drafts: 2,
            ..profile()
        },
        128,
        144,
        2,
    )
    .unwrap();
    let state = limits.bootstrap(bootstrap(15, true)).unwrap().state();
    for accepted in 0..=2 {
        let proposal = limits.propose(state, 7, 16, 15, 2).unwrap();
        let plan = proposal
            .finish(Finish::Verified(VerifiedCommit {
                generation: 7,
                capture_generation: 7,
                accepted,
                verify_token_rows: 3,
                normalized_hidden_rows: 3,
                target_position: 17 + accepted,
                observed_cache_rows: 17,
                hidden_base_position: 16,
            }))
            .unwrap();
        assert_eq!(plan.state().cache_rows(), 16 + accepted);
        assert_eq!(plan.state().target_position(), 17 + accepted);
        assert_eq!(
            plan.write().map(|write| write.rows()),
            (accepted > 0).then_some(accepted)
        );
        assert_eq!(plan.bonus_hidden_row(), Some(accepted));
    }
    assert!(limits.propose(state, 7, 16, 15, 1).is_err());
    assert!(limits.propose(state, 7, 16, 15, 4).is_err());
}

#[test]
fn indexed_context_checks_every_verdict_across_old_and_physical_boundaries() {
    let limits = Limits::new(
        Profile {
            drafts: 2,
            ..profile()
        },
        32768,
        32784,
        2,
    )
    .unwrap();
    for prompt in [2047, 2048, 2049, 4095, 15807, 32766] {
        let state = limits.bootstrap(bootstrap(prompt, true)).unwrap().state();
        for accepted in 0..=2 {
            let proposal = limits.propose(state, 7, prompt + 1, prompt, 2).unwrap();
            let plan = proposal
                .finish(Finish::Verified(VerifiedCommit {
                    generation: 7,
                    capture_generation: 7,
                    accepted,
                    verify_token_rows: 3,
                    normalized_hidden_rows: 3,
                    target_position: prompt + 2 + accepted,
                    observed_cache_rows: prompt + 2,
                    hidden_base_position: prompt + 1,
                }))
                .unwrap();
            assert_eq!(plan.state().cache_rows(), prompt + 1 + accepted);
            assert_eq!(plan.state().target_position(), prompt + 2 + accepted);
        }
        let short = Limits::new(
            Profile {
                drafts: 2,
                ..profile()
            },
            32768,
            prompt + 2,
            2,
        )
        .unwrap();
        assert!(short.propose(state, 7, prompt + 1, prompt, 2).is_err());
    }
    assert!(
        Limits::new(
            Profile {
                drafts: 2,
                ..profile()
            },
            32769,
            32784,
            2
        )
        .is_err()
    );
    assert!(
        Limits::new(
            Profile {
                drafts: 4,
                ..profile()
            },
            32768,
            32784,
            4
        )
        .is_err()
    );
}
