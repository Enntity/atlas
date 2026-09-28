// SPDX-License-Identifier: AGPL-3.0-only

use super::tests::{bytes, header, limits};
use super::*;

fn map_payload<P, Q>(command: LegacyCommand<P>, mut map: impl FnMut(P) -> Q) -> LegacyCommand<Q> {
    match command {
        LegacyCommand::Decode { slot, token } => LegacyCommand::Decode { slot, token },
        LegacyCommand::Prefill {
            slot,
            chunk_start,
            chunk_len,
            prompt,
        } => LegacyCommand::Prefill {
            slot,
            chunk_start,
            chunk_len,
            prompt: map(prompt),
        },
        LegacyCommand::DecodeBatch { slots, tokens } => LegacyCommand::DecodeBatch {
            slots: map(slots),
            tokens: map(tokens),
        },
        LegacyCommand::ReplaceSlot { slot } => LegacyCommand::ReplaceSlot { slot },
        LegacyCommand::Shutdown => LegacyCommand::Shutdown,
    }
}

fn roundtrip(command: LegacyCommand<&[u32]>, dialect: LegacyDialect) {
    let wire = encode_command(command, dialect, limits()).unwrap();
    let calls: Vec<_> = wire.calls().iter().copied().map(bytes).collect();
    let scalar_count = wire
        .calls()
        .iter()
        .take_while(|c| matches!(c, WireCall::U32(_)))
        .count();
    let scalar_refs: Vec<_> = calls[..scalar_count].iter().map(Vec::as_slice).collect();
    let bulk_refs: Vec<_> = calls[scalar_count..].iter().map(Vec::as_slice).collect();
    let header = parse_header(dialect, &scalar_refs, limits()).unwrap();
    assert_eq!(header.wire_bytes(), wire.wire_bytes());
    let decoded = header.parse_payloads(&bulk_refs).unwrap();
    assert_eq!(
        map_payload(decoded, |p| p.iter().collect::<Vec<_>>()),
        map_payload(command, |p| p.to_vec())
    );
}

#[test]
fn canonical_encode_parse_roundtrips_cover_both_dialects_and_prefill_seams() {
    let prompt = [0, 1, 254, 255, 7];
    for dialect in [LegacyDialect::V1, LegacyDialect::V2] {
        for token in [0, 1, 254, 255] {
            roundtrip(LegacyCommand::Decode { slot: 0, token }, dialect);
        }
        for (chunk_start, chunk_len) in [(0, 1), (1, 3), (4, 1), (0, 5)] {
            roundtrip(
                LegacyCommand::Prefill {
                    slot: 0,
                    chunk_start,
                    chunk_len,
                    prompt: &prompt,
                },
                dialect,
            );
        }
        roundtrip(LegacyCommand::ReplaceSlot { slot: 0 }, dialect);
        roundtrip(LegacyCommand::Shutdown, dialect);
    }
    for slots in [[7, 1, 5], [5, 7, 1], [1, 5, 7]] {
        roundtrip(
            LegacyCommand::DecodeBatch {
                slots: &slots,
                tokens: &[0, 255, 1],
            },
            LegacyDialect::V2,
        );
    }
}

#[test]
fn batch_payload_validation_covers_slot_bounds_lengths_and_reserved_tokens() {
    let shape = header(&[0, BATCH, 2], LegacyDialect::V2, limits()).unwrap();
    for (ids, toks, reason) in [
        ([1, 8], [1, 2], "slot outside"),
        ([1, 1], [1, 2], "duplicate"),
        ([1, 7], [1, 256], "token outside"),
        ([1, 7], [1, BATCH], "token outside"),
    ] {
        let slots = bytes(WireCall::Words(&ids));
        let tokens = bytes(WireCall::Words(&toks));
        let error = shape.parse_payloads(&[&slots, &tokens]).err().unwrap();
        assert!(error.to_string().contains(reason), "{error}");
        assert!(
            encode_command(
                LegacyCommand::DecodeBatch {
                    slots: &ids,
                    tokens: &toks
                },
                LegacyDialect::V2,
                limits()
            )
            .is_err()
        );
    }
    assert!(
        encode_command(
            LegacyCommand::DecodeBatch {
                slots: &[1, 2],
                tokens: &[1]
            },
            LegacyDialect::V2,
            limits()
        )
        .is_err()
    );
    let combined = bytes(WireCall::Words(&[1, 7, 1, 2]));
    assert!(
        shape.parse_payloads(&[&combined]).is_err(),
        "two bulks cannot become one"
    );
    let full = bytes(WireCall::Words(&[1, 7]));
    assert!(
        shape.parse_payloads(&[&full, &combined]).is_err(),
        "trailing bulk words reject"
    );
}

#[test]
fn checked_overflow_and_huge_headers_fail_before_payload_stage() {
    assert!(checked_bytes(usize::MAX).is_err());
    assert_eq!(checked_bytes(usize::MAX / 4).unwrap(), (usize::MAX / 4) * 4);
    let error = header(&[0, PREFILL, 1, u32::MAX, 4], LegacyDialect::V2, limits())
        .err()
        .unwrap();
    assert!(error.to_string().contains("prefill end overflow"));
    let mut l = limits();
    l.max_prompt_tokens = u32::MAX;
    l.staging_bytes = usize::MAX;
    let error = header(&[0, PREFILL, 1, 0, u32::MAX], LegacyDialect::V2, l)
        .err()
        .unwrap();
    assert!(error.to_string().contains("control budget") || error.to_string().contains("overflow"));
    assert!(parse_header(LegacyDialect::V2, &[&[0u8; 4][..]; 6], limits()).is_err());
    let flat = bytes(WireCall::Words(&[0, 1]));
    assert!(parse_header(LegacyDialect::V2, &[&flat], limits()).is_err());
    assert_eq!(bytes(WireCall::U32(0x01020304)), vec![4, 3, 2, 1]);
}
