// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use std::sync::atomic::Ordering;

fn state(seq: &mut SequenceState) -> &mut Glm5MtpProposerState {
    seq.proposer_state
        .as_mut()
        .unwrap()
        .as_any_mut()
        .downcast_mut()
        .unwrap()
}

#[test]
fn actual_body_and_final_copy_failures_after_input_spend_attempt() {
    for body_failure in [true, false] {
        fixture(0, |head, ctx, gpu, saved| {
            let mut seq = sequence(head, ctx, 1);
            gpu.fail_body.store(body_failure, Ordering::Relaxed);
            gpu.fail_final_read.store(!body_failure, Ordering::Relaxed);
            for attempt in 1..=8 {
                prepared(state(&mut seq), 1, 3);
                arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
                gpu.events.lock().clear();
                assert!(
                    head.forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None)
                        .is_err()
                );
                let events = gpu.events.lock();
                assert_eq!(events.first(), Some(&Event::Read(saved, ROW_BYTES, 7)));
                assert!(events.contains(&Event::Body));
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| matches!(e, Event::Read(_, ROW_BYTES, _)))
                        .count(),
                    if body_failure { 2 } else { 3 }
                );
                if !body_failure {
                    assert_eq!(
                        events.last(),
                        Some(&Event::Read(ctx.buffers.norm_output(), ROW_BYTES, 7))
                    );
                }
                assert_eq!(state(&mut seq).hidden_trace.spent, attempt);
            }
            gpu.fail_body.store(false, Ordering::Relaxed);
            prepared(state(&mut seq), 1, 3);
            let queries = gpu.capture_queries.load(Ordering::Relaxed);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
            gpu.events.lock().clear();
            head.forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None)
                .unwrap();
            assert!(
                gpu.events
                    .lock()
                    .iter()
                    .all(|e| !matches!(e, Event::Read(..)))
            );
            assert_eq!(gpu.capture_queries.load(Ordering::Relaxed), queries);
        });
    }
}

#[test]
fn exact_raw_bf16_digest_includes_last_byte_and_distinguishes_zero_from_failure() {
    fixture(0, |_, _, gpu, saved| {
        let mut row = [0u8; ROW_BYTES];
        gpu.inner.copy_h2d(&row, saved).unwrap();
        let zero = snapshot(gpu, saved, false, 13).unwrap();
        assert_eq!(zero, <[u8; 32]>::from(Sha256::digest(row)));
        assert_ne!(zero, [0; 32]);
        row[ROW_BYTES - 1] = 0x80; // BF16 negative zero: preserve representation.
        gpu.inner.copy_h2d(&row, saved).unwrap();
        let negative_zero = snapshot(gpu, saved, false, 13).unwrap();
        assert_ne!(zero, negative_zero);
        assert_eq!(negative_zero, <[u8; 32]>::from(Sha256::digest(row)));
        for (index, byte) in row.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        gpu.inner.copy_h2d(&row, saved).unwrap();
        assert_eq!(
            snapshot(gpu, saved, false, 13).unwrap(),
            <[u8; 32]>::from(Sha256::digest(row))
        );
        assert_eq!(
            *gpu.events.lock(),
            vec![Event::Read(saved, ROW_BYTES, 13); 3]
        );
        gpu.fail_read.store(true, Ordering::Relaxed);
        assert!(snapshot(gpu, saved, false, 13).is_err());
    });
}

#[test]
fn actual_hook_rechecks_capture_and_step_ownership_before_io() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, 1);
        for fault in 0..6 {
            prepared(state(&mut seq), 1, 3);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
            gpu.events.lock().clear();
            gpu.capturing.store(fault == 0, Ordering::Relaxed);
            ctx.graph_capture = fault == 1;
            let ptr = if fault == 2 {
                ctx.buffers.norm_output()
            } else {
                saved
            };
            let step = if fault == 3 { 4 } else { 0 };
            let token = if fault == 4 { 2 } else { 3 };
            let position = if fault == 5 { 4 } else { 3 };
            assert!(
                head.forward_one(token, ptr, position, step, state(&mut seq), ctx, 7, None)
                    .is_err()
            );
            assert!(gpu.events.lock().is_empty());
            gpu.capturing.store(false, Ordering::Relaxed);
            ctx.graph_capture = false;
        }
    });
}

#[test]
fn final_owner_and_cursor_reject_before_copy() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, 1);
        arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
        let state = state(&mut seq);
        let mut record = state
            .hidden_trace
            .input(3, 3, 0, saved, state.seq_len, ctx, 7)
            .unwrap()
            .unwrap();
        record.post_eh(ctx.buffers.hidden_states(), ctx, 7).unwrap();
        gpu.events.lock().clear();
        assert!(
            record
                .final_hidden(saved, state.seq_len + 1, ctx, 7)
                .is_err()
        );
        assert!(
            record
                .final_hidden(ctx.buffers.norm_output(), state.seq_len, ctx, 7)
                .is_err()
        );
        assert!(gpu.events.lock().is_empty());
    });
}

#[test]
fn strict_flag_and_row_address_validation() {
    for (raw, expected) in [(None, false), (Some("0"), false), (Some("1"), true)] {
        assert_eq!(parse(raw).unwrap(), expected);
    }
    for raw in ["", "true", "yes", "2", " 1", "1 "] {
        assert!(parse(Some(raw)).is_err());
    }
    for ptr in [DevicePtr::NULL, DevicePtr(1), DevicePtr(u64::MAX - 1)] {
        assert!(validate_row(ptr).is_err());
    }
    assert!(validate_row(DevicePtr(8192)).is_ok());
}

#[test]
fn explicit_non_unicode_flag_is_not_treated_as_absent() {
    use std::env::VarError;
    assert!(!parse_environment(Err(VarError::NotPresent)).unwrap());
    assert!(parse_environment(Ok("1".into())).unwrap());
    // Test the environment API's error value directly: no process-global mutation.
    assert!(parse_environment(Err(VarError::NotUnicode(std::ffi::OsString::new()))).is_err());
}

#[test]
fn actual_no_pool_route_from_production_resolver_is_trace_eligible() {
    for rank in 0..2 {
        fixture(rank, |head, ctx, gpu, saved| {
            let mut seq = sequence(head, ctx, 1);
            ctx.moe_lora_route = crate::lora::resolve_moe_lora_route(-1, -1, false);
            assert_eq!(ctx.moe_lora_route, crate::layer::MoeLoraRoute::Fold);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7)
                .expect("actual native no-pool Fold is inert and must be eligible");
            assert!(gpu.events.lock().is_empty());
        });
    }
}
