// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn profile() -> Profile {
    Profile {
        sequences: 1,
        drafts: 4,
        continuous: true,
        grammar: false,
        adaptive_depth: false,
        catchup: false,
        carry: false,
        prefix_reuse: false,
    }
}

fn limits() -> Limits {
    Limits::new(profile(), 128, 144, 4).unwrap()
}

fn bootstrap(prompt: usize, eager: bool) -> BootstrapInput {
    BootstrapInput {
        generation: 7,
        capture_generation: 7,
        prompt_tokens: prompt,
        target_position: prompt + 1,
        token_rows: prompt + 1,
        normalized_hidden_rows: prompt,
        cached_rows: if eager { prompt - 1 } else { 0 },
    }
}

#[test]
fn eager_owned_tail_has_exact_local_row_and_absolute_position_bounds() {
    for prompt in [1, 2, 15, 16, 127] {
        let input = EagerTailInput {
            generation: 9,
            prompt_tokens: prompt,
            target_position: prompt + 1,
            token_rows: prompt + 1,
            cached_rows: prompt - 1,
            tail_position: prompt - 1,
        };
        let plan = limits().bootstrap_eager_tail(input).unwrap();
        assert_eq!(
            plan.write().unwrap(),
            PairWrite {
                cache_start: prompt - 1,
                token_start: prompt,
                hidden_start: 0,
                rows: 1
            }
        );
        assert_eq!(plan.state().cache_rows(), prompt);
        assert_eq!(plan.state().generation(), 9);
        for case in 0..6 {
            let mut bad = input;
            match case {
                0 => bad.generation = 0,
                1 => bad.prompt_tokens = 0,
                2 => bad.target_position += 1,
                3 => bad.token_rows = prompt,
                4 => bad.cached_rows += 1,
                _ => bad.tail_position += 1,
            }
            assert!(limits().bootstrap_eager_tail(bad).is_err());
        }
    }
    for prompt in [128, usize::MAX] {
        assert!(
            limits()
                .bootstrap_eager_tail(EagerTailInput {
                    generation: 1,
                    prompt_tokens: prompt,
                    target_position: prompt,
                    token_rows: prompt,
                    cached_rows: prompt - 1,
                    tail_position: prompt - 1
                })
                .is_err()
        );
    }
}

#[test]
fn bootstrap_covers_shifted_prompt_and_first_generated_token_exactly_once() {
    for prompt in [1, 2, 15, 16] {
        for eager in [false, true] {
            let input = bootstrap(prompt, eager);
            let plan = limits().bootstrap(input).unwrap();
            let span = plan.write().unwrap();
            let start = input.cached_rows;
            assert_eq!(
                span,
                PairWrite {
                    cache_start: start,
                    token_start: start + 1,
                    hidden_start: start,
                    rows: prompt - start
                }
            );
            assert_eq!(plan.state().cache_rows(), prompt);
            assert_eq!(plan.state().target_position(), prompt + 1);
            let mut rebuilt: Vec<(usize, usize)> = (0..start).map(|j| (j + 1, j)).collect();
            rebuilt.extend((0..span.rows).map(|i| (span.token_start + i, span.hidden_start + i)));
            assert_eq!(rebuilt, (0..prompt).map(|j| (j + 1, j)).collect::<Vec<_>>());
        }
    }
}

fn proposal(state: PairState, bounds: Limits) -> ProposalPlan {
    bounds
        .propose(state, 7, state.target_position(), state.cache_rows(), 4)
        .unwrap()
}

fn verified(p: &ProposalPlan, accepted: usize) -> VerifiedCommit {
    VerifiedCommit {
        generation: 7,
        capture_generation: 7,
        accepted,
        verify_token_rows: 5,
        normalized_hidden_rows: 5,
        hidden_base_position: p.position(),
        target_position: p.position() + accepted + 1,
        observed_cache_rows: p.speculative_cache_end(),
    }
}

#[test]
fn verified_counts_zero_through_four_preserve_seed_and_align_true_hidden() {
    for accepted in 0..=4 {
        let state = limits().bootstrap(bootstrap(3, true)).unwrap().state();
        let p = proposal(state, limits());
        let result = p.finish(Finish::Verified(verified(&p, accepted))).unwrap();
        assert_eq!(
            result.state().cache_rows(),
            state.cache_rows() + accepted + 1
        );
        assert_eq!(
            result.state().target_position(),
            state.target_position() + accepted + 1
        );
        assert_eq!(result.bonus_hidden_row(), Some(accepted));
        assert_eq!(result.keep_seed(), true);
        if accepted == 0 {
            assert_eq!(result.write(), None);
        } else {
            let w = result.write().unwrap();
            assert_eq!(
                w,
                PairWrite {
                    cache_start: state.cache_rows() + 1,
                    token_start: 1,
                    hidden_start: 0,
                    rows: accepted
                }
            );
            for i in 0..w.rows {
                assert_eq!(w.token_start + i, w.hidden_start + i + 1);
            }
        }
        if accepted == 4 {
            assert_eq!(
                result.state().cache_rows(),
                p.speculative_cache_end() + 1,
                "full acceptance appends the final draft's never-generated KV input"
            );
        }
    }
}

#[test]
fn repeated_partial_full_zero_commits_have_no_compacted_gaps() {
    let mut state = limits().bootstrap(bootstrap(14, true)).unwrap().state();
    for accepted in [0, 4, 1, 3, 2, 4, 0, 0] {
        let p = proposal(state, limits());
        let next = p.finish(Finish::Verified(verified(&p, accepted))).unwrap();
        state = next.state();
        assert_eq!(state.cache_rows() + 1, state.target_position());
        assert_eq!(state.generation(), 7);
    }
    assert!(
        state.cache_rows() > 16,
        "sequence crosses a paged-block boundary"
    );
}

#[test]
fn discard_is_not_zero_acceptance_or_an_invented_target_advance() {
    let state = limits().bootstrap(bootstrap(2, true)).unwrap().state();
    let p = proposal(state, limits());
    let discarded = p
        .finish(Finish::DiscardUnverified {
            generation: 7,
            observed_cache_rows: p.speculative_cache_end(),
        })
        .unwrap();
    assert_eq!(discarded.state(), state);
    assert_eq!(discarded.write(), None);
    assert_eq!(discarded.bonus_hidden_row(), None);
    assert!(!discarded.keep_seed());
    let committed = p.finish(Finish::Verified(verified(&p, 0))).unwrap();
    assert_eq!(committed.state().cache_rows(), state.cache_rows() + 1);
    // A later serial step creates an uncovered gap; it cannot silently resume.
    assert!(
        limits()
            .propose(state, 7, state.target_position() + 1, state.cache_rows(), 4)
            .unwrap_err()
            .to_string()
            .contains("position")
    );
}

#[test]
fn unsupported_profiles_fail_at_construction() {
    let changes: [fn(&mut Profile); 8] = [
        |p| p.sequences = 2,
        |p| p.drafts = 3,
        |p| p.continuous = false,
        |p| p.grammar = true,
        |p| p.adaptive_depth = true,
        |p| p.catchup = true,
        |p| p.carry = true,
        |p| p.prefix_reuse = true,
    ];
    for change in changes {
        let mut p = profile();
        change(&mut p);
        assert!(
            Limits::new(p, 128, 144, 4)
                .unwrap_err()
                .to_string()
                .contains("fixed continuous")
        );
    }
    assert!(
        Limits::new(profile(), usize::MAX, 144, 4)
            .unwrap_err()
            .to_string()
            .contains("overflow")
    );
    assert!(
        Limits::new(profile(), 2045, 2064, 4)
            .unwrap_err()
            .to_string()
            .contains("2048")
    );
    assert!(Limits::new(profile(), 2044, 2064, 4).is_ok());
}

#[test]
fn bootstrap_rejects_stale_short_missing_and_inconsistent_coverage() {
    let cases: [(fn(&mut BootstrapInput), &str); 8] = [
        (|i| i.generation = 0, "generation"),
        (|i| i.capture_generation = 8, "generation"),
        (|i| i.normalized_hidden_rows = 1, "hidden"),
        (|i| i.token_rows = 2, "token"),
        (|i| i.target_position = 2, "position"),
        (|i| i.cached_rows = 2, "cache"),
        (|i| i.prompt_tokens = 0, "prompt"),
        (|i| i.prompt_tokens = usize::MAX, "overflow"),
    ];
    for (change, message) in cases {
        let mut input = bootstrap(2, false);
        change(&mut input);
        assert!(
            limits()
                .bootstrap(input)
                .unwrap_err()
                .to_string()
                .contains(message),
            "{message}"
        );
    }
}

#[test]
fn verify_generation_row_counts_and_cache_ownership_fail_closed() {
    let p = proposal(
        limits().bootstrap(bootstrap(2, true)).unwrap().state(),
        limits(),
    );
    let cases: [(fn(&mut VerifiedCommit), &str); 8] = [
        (|v| v.generation = 8, "generation"),
        (|v| v.capture_generation = 8, "generation"),
        (|v| v.accepted = 5, "accepted"),
        (|v| v.verify_token_rows = 4, "token"),
        (|v| v.normalized_hidden_rows = 4, "hidden"),
        (|v| v.hidden_base_position -= 1, "hidden base"),
        (|v| v.target_position += 1, "position"),
        (|v| v.observed_cache_rows -= 1, "cache"),
    ];
    for (change, message) in cases {
        let mut v = verified(&p, 4);
        change(&mut v);
        assert!(
            p.finish(Finish::Verified(v))
                .unwrap_err()
                .to_string()
                .contains(message),
            "{message}"
        );
    }
    assert!(
        p.finish(Finish::DiscardUnverified {
            generation: 8,
            observed_cache_rows: p.speculative_cache_end()
        })
        .is_err()
    );
}

#[test]
fn actual_cache_and_staging_capacities_cover_full_accept_extra_row() {
    let bounds = Limits::new(profile(), 128, 6, 4).unwrap();
    let state = bounds.bootstrap(bootstrap(2, true)).unwrap().state();
    // Four transient writes fit, but the possible full-accept fifth row
    // must be reserved before executing even the first proposal write.
    assert!(
        bounds
            .propose(state, 7, state.target_position(), state.cache_rows(), 4)
            .unwrap_err()
            .to_string()
            .contains("cache capacity")
    );
    let bounds = Limits::new(profile(), 128, 144, 2).unwrap();
    assert!(
        bounds
            .propose(state, 7, state.target_position(), state.cache_rows(), 4)
            .unwrap_err()
            .to_string()
            .contains("staging")
    );
    let bounds = Limits::new(profile(), 128, 7, 4).unwrap();
    let p = proposal(state, bounds);
    assert!(p.finish(Finish::Verified(verified(&p, 4))).is_ok());
    let bounds = Limits::new(profile(), 128, 1, 4).unwrap();
    assert!(
        bounds
            .bootstrap(bootstrap(2, false))
            .unwrap_err()
            .to_string()
            .contains("cache capacity")
    );
}

#[test]
fn finished_request_can_release_without_fabricated_next_proposal() {
    let p = proposal(
        limits().bootstrap(bootstrap(1, false)).unwrap().state(),
        limits(),
    );
    let final_plan = p.finish(Finish::Verified(verified(&p, 4))).unwrap();
    assert_eq!(final_plan.bonus_hidden_row(), Some(4));
    // Pure plans have no destructor/device side effects. A terminal owner may
    // discard the plan and free its request rather than execute repair.
    let mut input = bootstrap(1, false);
    input.generation = 8;
    input.capture_generation = 8;
    let fresh = limits().bootstrap(input).unwrap();
    assert_eq!(fresh.state().generation(), 8);
    assert!(limits().propose(fresh.state(), 7, 2, 1, 4).is_err());
}

#[test]
fn single_draft_history_keeps_seed_and_full_accept_extra_pair() {
    let bounds = Limits::new(
        Profile {
            drafts: 1,
            ..profile()
        },
        128,
        144,
        1,
    )
    .unwrap();
    for eager in [false, true] {
        let mut state = bounds.bootstrap(bootstrap(3, eager)).unwrap().state();
        for accepted in [1, 0, 1, 1, 0] {
            let p = bounds
                .propose(state, 7, state.target_position(), state.cache_rows(), 1)
                .unwrap();
            assert!(
                bounds
                    .propose(state, 7, state.target_position(), state.cache_rows(), 4)
                    .is_err()
            );
            let input = VerifiedCommit {
                verify_token_rows: 2,
                normalized_hidden_rows: 2,
                ..verified(&p, accepted)
            };
            for bad in [
                VerifiedCommit {
                    accepted: 2,
                    ..input
                },
                VerifiedCommit {
                    verify_token_rows: 5,
                    ..input
                },
                VerifiedCommit {
                    normalized_hidden_rows: 1,
                    ..input
                },
                VerifiedCommit {
                    capture_generation: 8,
                    ..input
                },
            ] {
                assert!(p.finish(Finish::Verified(bad)).is_err());
            }
            let next = p.finish(Finish::Verified(input)).unwrap();
            assert!(next.keep_seed());
            assert_eq!(next.state().cache_rows(), state.cache_rows() + accepted + 1);
            assert_eq!(next.bonus_hidden_row(), Some(accepted));
            assert_eq!(
                next.write().map(|w| (
                    w.cache_start(),
                    w.token_start(),
                    w.hidden_start(),
                    w.rows()
                )),
                (accepted > 0).then_some((state.cache_rows() + 1, 1, 0, accepted))
            );
            state = next.state();
        }
    }
}

#[test]
fn single_draft_capacity_and_unqualified_depths_fail_before_proposal() {
    for drafts in [0, 3, 5] {
        assert!(
            Limits::new(
                Profile {
                    drafts,
                    ..profile()
                },
                128,
                144,
                4
            )
            .is_err()
        );
    }
    for (cache, staging) in [(4, 1), (5, 0)] {
        let bounds = Limits::new(
            Profile {
                drafts: 1,
                ..profile()
            },
            128,
            cache,
            staging,
        )
        .unwrap();
        let state = bounds.bootstrap(bootstrap(3, true)).unwrap().state();
        assert!(bounds.propose(state, 7, 4, 3, 1).is_err());
    }
}

#[path = "glm_pair_plan_long_tests.rs"]
mod long_tests;
