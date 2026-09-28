// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn limits() -> PlanLimits {
    PlanLimits {
        max_requests: 4,
        max_decode_requests: 4,
        max_prefill_requests: 2,
        max_context_tokens: 64,
        max_scheduled_tokens: 16,
        max_payload_tokens: 32,
        max_arena_tokens: 8,
        max_prefill_chunk_tokens: 8,
        vocab_size: 256,
        wire_slot_capacity: 8,
    }
}

fn profile() -> PlanProfile {
    PlanProfile {
        execution: ExecutionMode::LegacySerial,
        features: vec![],
        prefill_state: PrefillStatePolicy::NormalizeAfterChunk,
    }
}

fn mixed() -> ScheduledIntent {
    ScheduledIntent {
        step: StepId {
            session: 11,
            sequence: 42,
        },
        tokens: vec![7, 8, 9, 10, 11, 12],
        work: vec![
            RequestWork {
                key: RequestKey {
                    wire_slot: 3,
                    generation: 7,
                },
                expected_position: 19,
                kind: WorkItem::DecodeOne {
                    token: TokenSpan {
                        offset: 0,
                        count: 1,
                    },
                },
            },
            RequestWork {
                key: RequestKey {
                    wire_slot: 1,
                    generation: 9,
                },
                expected_position: 2,
                kind: WorkItem::PrefillChunk {
                    prompt: TokenSpan {
                        offset: 1,
                        count: 4,
                    },
                    prompt_revision: 3,
                    chunk_start: 2,
                    chunk_len: 2,
                },
            },
            RequestWork {
                key: RequestKey {
                    wire_slot: 2,
                    generation: 0,
                },
                expected_position: 63,
                kind: WorkItem::DecodeOne {
                    token: TokenSpan {
                        offset: 5,
                        count: 1,
                    },
                },
            },
        ],
    }
}

#[test]
fn preserves_interleaved_work_order_and_consumes_results_before_arena_reuse() {
    let plan = ValidatedStepPlan::new(mixed(), profile(), limits()).unwrap();
    assert_eq!(
        plan.step(),
        StepId {
            session: 11,
            sequence: 42
        }
    );
    assert_eq!(plan.work(), mixed().work.as_slice());
    assert_eq!(plan.tokens(), &[7, 8, 9, 10, 11, 12]);
    assert_eq!(plan.scheduled_tokens(), 4); // full prompt payload contains six tokens
    assert_eq!(
        plan.substeps(),
        &[
            SequentialSubstep::DecodeOne {
                work_index: 0,
                position_after: 20
            },
            SequentialSubstep::ConsumeLogits { work_index: 0 },
            SequentialSubstep::PrefillChunk {
                work_index: 1,
                position_after: 4,
                is_last: true
            },
            SequentialSubstep::NormalizeSsm { work_index: 1 },
            SequentialSubstep::ConsumeLogits { work_index: 1 },
            SequentialSubstep::DecodeOne {
                work_index: 2,
                position_after: 64
            },
            SequentialSubstep::ConsumeLogits { work_index: 2 },
        ]
    );
}

#[test]
fn intermediate_prefill_has_no_logits_and_normalization_is_explicit() {
    let mut intent = mixed();
    if let WorkItem::PrefillChunk { chunk_len, .. } = &mut intent.work[1].kind {
        *chunk_len = 1;
    }
    let plan = ValidatedStepPlan::new(intent.clone(), profile(), limits()).unwrap();
    assert!(plan.substeps().contains(&SequentialSubstep::PrefillChunk {
        work_index: 1,
        position_after: 3,
        is_last: false
    }));
    assert!(
        !plan
            .substeps()
            .contains(&SequentialSubstep::ConsumeLogits { work_index: 1 })
    );
    assert!(
        plan.substeps()
            .contains(&SequentialSubstep::NormalizeSsm { work_index: 1 })
    );
    let mut p = profile();
    p.prefill_state = PrefillStatePolicy::Preserve;
    let plan = ValidatedStepPlan::new(intent, p, limits()).unwrap();
    assert!(
        !plan
            .substeps()
            .iter()
            .any(|s| matches!(s, SequentialSubstep::NormalizeSsm { .. }))
    );
}

#[test]
fn budgets_distinguish_scheduled_tokens_full_payload_and_serial_arena() {
    let mut l = limits();
    l.max_scheduled_tokens = 4;
    l.max_payload_tokens = 6;
    l.max_arena_tokens = 2;
    assert!(ValidatedStepPlan::new(mixed(), profile(), l).is_ok());
    for which in 0..7 {
        let mut l = limits();
        match which {
            0 => l.max_requests = 2,
            1 => l.max_decode_requests = 1,
            2 => l.max_prefill_requests = 0,
            3 => l.max_scheduled_tokens = 3,
            4 => l.max_payload_tokens = 5,
            5 => l.max_arena_tokens = 1,
            _ => l.max_prefill_chunk_tokens = 1,
        }
        assert!(ValidatedStepPlan::new(mixed(), profile(), l).is_err());
    }
}

#[test]
fn zero_configuration_and_unsupported_profiles_are_not_implicit_defaults() {
    for which in 0..7 {
        let mut l = limits();
        match which {
            0 => l.max_requests = 0,
            1 => l.max_context_tokens = 0,
            2 => l.max_scheduled_tokens = 0,
            3 => l.max_payload_tokens = 0,
            4 => l.max_arena_tokens = 0,
            5 => l.vocab_size = 0,
            _ => l.wire_slot_capacity = 0,
        }
        assert!(ValidatedStepPlan::new(mixed(), profile(), l).is_err());
    }
    for execution in [ExecutionMode::Speculative, ExecutionMode::PackedMixed] {
        let mut p = profile();
        p.execution = execution;
        assert!(ValidatedStepPlan::new(mixed(), p, limits()).is_err());
    }
    for feature in [
        ExecutionFeature::Multimodal,
        ExecutionFeature::Adapters,
        ExecutionFeature::PromptLogprobs,
        ExecutionFeature::PrefixReuse,
        ExecutionFeature::Swapping,
    ] {
        let mut p = profile();
        p.features.push(feature);
        assert!(ValidatedStepPlan::new(mixed(), p, limits()).is_err());
    }
}

#[test]
fn duplicate_or_reused_wire_slots_and_invalid_tokens_are_rejected() {
    for generation in [7, 8] {
        let mut intent = mixed();
        intent.work[2].key = RequestKey {
            wire_slot: 3,
            generation,
        };
        assert!(ValidatedStepPlan::new(intent, profile(), limits()).is_err());
    }
    let mut intent = mixed();
    intent.work[0].key.wire_slot = 8;
    assert!(ValidatedStepPlan::new(intent, profile(), limits()).is_err());
    let mut intent = mixed();
    intent.tokens[1] = 256; // even uncomputed prompt prefix must be valid
    assert!(ValidatedStepPlan::new(intent, profile(), limits()).is_err());
    let mut intent = mixed();
    intent.work.clear();
    intent.tokens.clear();
    assert!(ValidatedStepPlan::new(intent, profile(), limits()).is_err());
}

#[test]
fn malformed_spans_counts_positions_and_overflows_fail_closed() {
    for (offset, count) in [(1, 1), (0, 0), (0, 2), (usize::MAX, 1), (0, usize::MAX)] {
        let mut intent = mixed();
        intent.work[0].kind = WorkItem::DecodeOne {
            token: TokenSpan { offset, count },
        };
        assert!(ValidatedStepPlan::new(intent, profile(), limits()).is_err());
    }
    for (offset, count, start, chunk) in [
        (0, 4, 2, 2),
        (2, 4, 2, 2),
        (1, 5, 2, 2),
        (1, 0, 0, 1),
        (1, 4, 1, 2),
        (1, 4, 2, 0),
        (1, 4, 2, 3),
        (1, 4, usize::MAX, 1),
        (1, 4, 2, usize::MAX),
    ] {
        let mut intent = mixed();
        intent.work[1].kind = WorkItem::PrefillChunk {
            prompt: TokenSpan { offset, count },
            prompt_revision: 3,
            chunk_start: start,
            chunk_len: chunk,
        };
        assert!(ValidatedStepPlan::new(intent, profile(), limits()).is_err());
    }
    let mut intent = mixed();
    intent.tokens.push(1);
    assert!(ValidatedStepPlan::new(intent, profile(), limits()).is_err());
    for position in [64, usize::MAX] {
        let mut intent = mixed();
        intent.work[2].expected_position = position;
        assert!(ValidatedStepPlan::new(intent, profile(), limits()).is_err());
    }
    let mut intent = mixed();
    intent.work[0].expected_position = 0;
    intent.work[2].expected_position = 0;
    let mut l = limits();
    l.max_context_tokens = 3;
    let error = ValidatedStepPlan::new(intent, profile(), l).err().unwrap();
    assert!(error.to_string().contains("prompt exceeds context"));
}

#[test]
fn arithmetic_failures_reach_the_specific_checked_guards() {
    let mut l = limits();
    l.max_payload_tokens = usize::MAX;
    let error = ValidatedStepPlan::new(mixed(), profile(), l).err().unwrap();
    assert!(error.to_string().contains("payload byte limit overflow"));

    let mut intent = mixed();
    intent.work[0].kind = WorkItem::DecodeOne {
        token: TokenSpan {
            offset: usize::MAX,
            count: 1,
        },
    };
    let error = ValidatedStepPlan::new(intent, profile(), limits())
        .err()
        .unwrap();
    assert!(error.to_string().contains("token span overflow"));

    let mut intent = mixed();
    intent.work[1].expected_position = usize::MAX;
    if let WorkItem::PrefillChunk {
        chunk_start,
        chunk_len,
        ..
    } = &mut intent.work[1].kind
    {
        *chunk_start = usize::MAX;
        *chunk_len = 1;
    }
    let error = ValidatedStepPlan::new(intent, profile(), limits())
        .err()
        .unwrap();
    assert!(error.to_string().contains("chunk end overflow"));

    let mut intent = mixed();
    intent.work[0].expected_position = usize::MAX;
    let error = ValidatedStepPlan::new(intent, profile(), limits())
        .err()
        .unwrap();
    assert!(error.to_string().contains("position overflow"));
}

#[test]
fn generation_and_step_are_preserved_not_claimed_as_live_or_wire_validation() {
    let mut intent = mixed();
    intent.work[0].key.generation = u64::MAX;
    let a = ValidatedStepPlan::new(intent.clone(), profile(), limits()).unwrap();
    let b = ValidatedStepPlan::new(intent, profile(), limits()).unwrap();
    assert_eq!(a.work()[0].key.generation, u64::MAX);
    assert_eq!(a.step(), b.step()); // no session registry or replay prevention in this pure slice
}

#[test]
fn disabled_prefill_allows_zero_chunk_budget_but_enabled_prefill_does_not() {
    let mut intent = mixed();
    intent.work.truncate(1);
    intent.tokens.truncate(1);
    let mut l = limits();
    l.max_prefill_requests = 0;
    l.max_prefill_chunk_tokens = 0;
    assert!(ValidatedStepPlan::new(intent.clone(), profile(), l).is_ok());
    l.max_prefill_requests = 1;
    let error = ValidatedStepPlan::new(intent, profile(), l).err().unwrap();
    assert!(error.to_string().contains("positive chunk budget"));
}
