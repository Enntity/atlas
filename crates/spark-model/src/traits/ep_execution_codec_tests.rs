// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::traits::execution_plan::*;

pub(super) fn limits() -> LegacyWireLimits {
    LegacyWireLimits {
        slot_capacity: 8,
        vocab_size: 256,
        max_prompt_tokens: 2048,
        max_chunk_tokens: 64,
        max_decode_rows: 4,
        max_control_bytes: 65536,
        staging_bytes: 8192,
    }
}

pub(super) fn bytes(call: WireCall<'_>) -> Vec<u8> {
    let mut out = vec![0; call.byte_len().unwrap()];
    call.write_le(&mut out).unwrap();
    out
}

fn plan(normalize: bool) -> ValidatedStepPlan {
    let p = PlanProfile {
        execution: ExecutionMode::LegacySerial,
        features: vec![],
        prefill_state: if normalize {
            PrefillStatePolicy::NormalizeAfterChunk
        } else {
            PrefillStatePolicy::Preserve
        },
    };
    let l = PlanLimits {
        max_requests: 4,
        max_decode_requests: 3,
        max_prefill_requests: 1,
        max_context_tokens: 2048,
        max_scheduled_tokens: 67,
        max_payload_tokens: 8192,
        max_arena_tokens: 64,
        max_prefill_chunk_tokens: 64,
        vocab_size: 256,
        wire_slot_capacity: 8,
    };
    ValidatedStepPlan::new(
        ScheduledIntent {
            step: StepId {
                session: 9,
                sequence: 7,
            },
            tokens: vec![21, 30, 31, 32, 33, 22],
            work: vec![
                RequestWork {
                    key: RequestKey {
                        wire_slot: 3,
                        generation: 2,
                    },
                    expected_position: 10,
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
                        generation: 5,
                    },
                    expected_position: 2,
                    kind: WorkItem::PrefillChunk {
                        prompt: TokenSpan {
                            offset: 1,
                            count: 4,
                        },
                        prompt_revision: 9,
                        chunk_start: 2,
                        chunk_len: 2,
                    },
                },
                RequestWork {
                    key: RequestKey {
                        wire_slot: 2,
                        generation: 1,
                    },
                    expected_position: 20,
                    kind: WorkItem::DecodeOne {
                        token: TokenSpan {
                            offset: 5,
                            count: 1,
                        },
                    },
                },
            ],
        },
        p,
        l,
    )
    .unwrap()
}

#[test]
fn singleton_decode_and_prefill_pin_little_endian_broadcast_boundaries() {
    let prompt = [10, 11, 12, 13];
    for dialect in [LegacyDialect::V1, LegacyDialect::V2] {
        let decode = encode_command(
            LegacyCommand::Decode { slot: 0, token: 42 },
            dialect,
            limits(),
        )
        .unwrap();
        let mut expected = vec![vec![42, 0, 0, 0]];
        if dialect == LegacyDialect::V2 {
            expected.insert(0, vec![0, 0, 0, 0]);
        }
        assert_eq!(
            decode
                .calls()
                .iter()
                .copied()
                .map(bytes)
                .collect::<Vec<_>>(),
            expected
        );
        let prefill = encode_command(
            LegacyCommand::Prefill {
                slot: 0,
                chunk_start: 2,
                chunk_len: 2,
                prompt: &prompt[..],
            },
            dialect,
            limits(),
        )
        .unwrap();
        let mut expected = vec![
            vec![0xf0, 0xff, 0xff, 0xff],
            vec![2, 0, 0, 0],
            vec![2, 0, 0, 0],
            vec![4, 0, 0, 0],
            vec![10, 0, 0, 0, 11, 0, 0, 0, 12, 0, 0, 0, 13, 0, 0, 0],
        ];
        if dialect == LegacyDialect::V2 {
            expected.insert(0, vec![0, 0, 0, 0]);
        }
        assert_eq!(
            prefill
                .calls()
                .iter()
                .copied()
                .map(bytes)
                .collect::<Vec<_>>(),
            expected
        );
        let WireCall::Words(borrowed) = prefill.calls().last().unwrap() else {
            panic!("one full-prompt bulk")
        };
        assert_eq!(
            borrowed.as_ptr(),
            prompt.as_ptr(),
            "encode must borrow the full prompt"
        );
    }
}

#[test]
fn plan_lowering_preserves_interleave_and_local_obligations_without_opcodes() {
    let p = plan(true);
    let t = encode_plan(&p, LegacyDialect::V2, limits()).unwrap();
    let locals: Vec<_> = t
        .ops()
        .iter()
        .filter_map(|op| match op {
            TranscriptOp::Local(s) => Some(*s),
            _ => None,
        })
        .collect();
    assert_eq!(locals, p.substeps());
    let wire: Vec<_> = t
        .ops()
        .iter()
        .filter_map(|op| match op {
            TranscriptOp::Wire(w) => Some(*w),
            _ => None,
        })
        .collect();
    assert_eq!(
        wire,
        vec![
            WireCall::U32(3),
            WireCall::U32(21),
            WireCall::U32(1),
            WireCall::U32(0xfffffff0),
            WireCall::U32(2),
            WireCall::U32(2),
            WireCall::U32(4),
            WireCall::Words(&p.tokens()[1..5]),
            WireCall::U32(2),
            WireCall::U32(22)
        ]
    );
    assert_eq!(t.wire_bytes(), 52);
    let normalized = t
        .ops()
        .iter()
        .position(|op| {
            matches!(
                op,
                TranscriptOp::Local(SequentialSubstep::NormalizeSsm { work_index: 1 })
            )
        })
        .unwrap();
    assert!(matches!(
        t.ops()[normalized + 1],
        TranscriptOp::Local(SequentialSubstep::ConsumeLogits { work_index: 1 })
    ));
    assert_eq!(
        t.ops()[normalized + 2],
        TranscriptOp::Wire(WireCall::U32(2))
    );
    assert!(encode_plan(&p, LegacyDialect::V1, limits()).is_err());
    assert!(
        encode_plan(&plan(false), LegacyDialect::V2, limits()).is_err(),
        "legacy worker always normalizes after prefill"
    );
}

#[test]
fn batched_decode_and_controls_have_separate_canonical_fixtures() {
    let slots = [7, 1, 5];
    let tokens = [10, 11, 12];
    let t = encode_command(
        LegacyCommand::DecodeBatch {
            slots: &slots[..],
            tokens: &tokens[..],
        },
        LegacyDialect::V2,
        limits(),
    )
    .unwrap();
    assert_eq!(
        t.calls(),
        &[
            WireCall::U32(0),
            WireCall::U32(0xffffffe0),
            WireCall::U32(3),
            WireCall::Words(&slots),
            WireCall::Words(&tokens)
        ]
    );
    assert!(
        encode_command(
            LegacyCommand::DecodeBatch {
                slots: &slots[..],
                tokens: &tokens[..]
            },
            LegacyDialect::V1,
            limits()
        )
        .is_err()
    );
    let replace = encode_command(
        LegacyCommand::ReplaceSlot { slot: 5 },
        LegacyDialect::V2,
        limits(),
    )
    .unwrap();
    assert_eq!(
        replace.calls(),
        &[WireCall::U32(5), WireCall::U32(0xfffffff1)]
    );
    let shutdown = encode_command(LegacyCommand::Shutdown, LegacyDialect::V2, limits()).unwrap();
    assert_eq!(
        shutdown.calls(),
        &[WireCall::U32(0), WireCall::U32(u32::MAX)]
    );
}

pub(super) fn header(
    words: &[u32],
    dialect: LegacyDialect,
    l: LegacyWireLimits,
) -> anyhow::Result<ValidatedHeader> {
    let bytes: Vec<_> = words.iter().map(|x| x.to_le_bytes()).collect();
    let calls: Vec<_> = bytes.iter().map(|b| &b[..]).collect();
    parse_header(dialect, &calls, l)
}

#[test]
fn staged_parser_borrows_unaligned_little_endian_payload_and_preserves_order() {
    let h = header(&[3, 0xfffffff0, 2, 2, 4], LegacyDialect::V2, limits()).unwrap();
    assert_eq!(h.bulk_byte_lengths(), &[16]);
    let mut payload = vec![0xaa];
    payload.extend([10u32, 11, 12, 13].iter().flat_map(|x| x.to_le_bytes()));
    let LegacyCommand::Prefill {
        slot,
        chunk_start,
        chunk_len,
        prompt,
    } = h.parse_payloads(&[&payload[1..]]).unwrap()
    else {
        panic!("prefill")
    };
    assert_eq!((slot, chunk_start, chunk_len), (3, 2, 2));
    assert_eq!(prompt.iter().collect::<Vec<_>>(), vec![10, 11, 12, 13]);
    assert_eq!(prompt.as_bytes().as_ptr(), payload[1..].as_ptr());
    let h = header(&[0, 0xffffffe0, 3], LegacyDialect::V2, limits()).unwrap();
    assert_eq!(h.bulk_byte_lengths(), &[12, 12]);
    let slots = bytes(WireCall::Words(&[7, 1, 5]));
    let tokens = bytes(WireCall::Words(&[10, 11, 12]));
    let LegacyCommand::DecodeBatch { slots, tokens } =
        h.parse_payloads(&[&slots, &tokens]).unwrap()
    else {
        panic!("batch")
    };
    assert_eq!(slots.iter().collect::<Vec<_>>(), vec![7, 1, 5]);
    assert_eq!(tokens.iter().collect::<Vec<_>>(), vec![10, 11, 12]);
}

#[test]
fn header_rejections_precede_any_bulk_access() {
    let cases = [
        vec![0, 0xffffffe0, 0],
        vec![0, 0xffffffe0, 5],
        vec![1, 0xffffffe0, 2],
        vec![0, 0xfffffff0, 0, 0, 4],
        vec![0, 0xfffffff0, 65, 0, 100],
        vec![0, 0xfffffff0, 1, u32::MAX, 4],
        vec![0, 0xfffffff0, 2, 3, 4],
        vec![0, 0xfffffff0, 1, 0, 2049],
        vec![8, 1],
        vec![0, 256],
        vec![0],
        vec![0, 1, 2],
        vec![0, 0xfffffff0, 2, 0],
    ];
    for words in cases {
        assert!(
            header(&words, LegacyDialect::V2, limits()).is_err(),
            "{words:?}"
        );
    }
    let mut l = limits();
    l.staging_bytes = 15;
    assert!(header(&[0, 0xfffffff0, 2, 2, 4], LegacyDialect::V2, l).is_err());
    l.staging_bytes = 16;
    l.max_control_bytes = 36;
    assert!(header(&[0, 0xfffffff0, 2, 2, 4], LegacyDialect::V2, l).is_ok());
    l.max_control_bytes = 35;
    assert!(header(&[0, 0xfffffff0, 2, 2, 4], LegacyDialect::V2, l).is_err());
    assert!(parse_header(LegacyDialect::V1, &[&[1, 0, 0]], limits()).is_err());
}

#[test]
fn reserved_tokens_and_unsupported_commands_fail_in_all_payload_positions() {
    let mut l = limits();
    l.vocab_size = u32::MAX;
    for token in 0xffffffe0..=u32::MAX {
        assert!(
            encode_command(
                LegacyCommand::Decode { slot: 0, token },
                LegacyDialect::V2,
                l
            )
            .is_err()
        );
        assert!(
            encode_command(
                LegacyCommand::Prefill {
                    slot: 0,
                    chunk_start: 0,
                    chunk_len: 1,
                    prompt: &[token][..]
                },
                LegacyDialect::V2,
                l
            )
            .is_err()
        );
        let h = header(&[0, 0xfffffff0, 1, 0, 1], LegacyDialect::V2, l).unwrap();
        assert!(h.parse_payloads(&[&token.to_le_bytes()]).is_err());
    }
    for cmd in [
        0xffffffe1, 0xffffffe2, 0xfffffff2, 0xfffffff3, 0xfffffff4, 0xfffffff5,
    ] {
        assert!(header(&[0, cmd], LegacyDialect::V2, l).is_err());
    }
    // Existing worker ignores shutdown's slot preamble, even when out of range.
    assert!(matches!(
        header(&[u32::MAX, u32::MAX], LegacyDialect::V2, limits())
            .unwrap()
            .parse_payloads(&[])
            .unwrap(),
        LegacyCommand::Shutdown
    ));
}

#[test]
fn malformed_bulks_and_serialization_destinations_are_rejected() {
    let h = header(&[0, 0xffffffe0, 2], LegacyDialect::V2, limits()).unwrap();
    let slots = bytes(WireCall::Words(&[1, 1]));
    let tokens = bytes(WireCall::Words(&[2, 3]));
    assert!(h.parse_payloads(&[&slots, &tokens]).is_err());
    let slots = bytes(WireCall::Words(&[1, 7]));
    assert!(h.parse_payloads(&[&slots, &tokens]).is_ok());
    for calls in [
        vec![],
        vec![&slots[..]],
        vec![&slots[..], &tokens[..7]],
        vec![&slots[..], &tokens[..], &tokens[..]],
    ] {
        assert!(h.parse_payloads(&calls).is_err());
    }
    assert!(WireCall::Words(&[1, 2]).write_le(&mut [0; 7]).is_err());
    assert!(WireCall::U32(1).write_le(&mut [0; 8]).is_err());
}

#[test]
fn invalid_limits_and_total_plan_budget_are_explicit() {
    for which in 0..8 {
        let mut l = limits();
        match which {
            0 => l.slot_capacity = 0,
            1 => l.vocab_size = 0,
            2 => l.max_prompt_tokens = 0,
            3 => l.max_chunk_tokens = 0,
            4 => l.max_decode_rows = 0,
            5 => l.max_control_bytes = 0,
            6 => l.staging_bytes = 0,
            _ => l.max_control_bytes = usize::MAX,
        }
        assert!(
            header(&[0, 1], LegacyDialect::V2, l).is_err(),
            "case {which}"
        );
    }
    let mut l = limits();
    l.max_control_bytes = 51;
    assert!(encode_plan(&plan(true), LegacyDialect::V2, l).is_err());
}
