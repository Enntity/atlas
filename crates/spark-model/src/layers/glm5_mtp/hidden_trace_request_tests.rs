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
fn actual_request_arm_rejects_sequence_generation_not_owned_by_prepared_repair() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, gpu, 1);
        seq.mtp_capture_gen = 2;
        assert!(
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).is_err(),
            "sequence capture stamp must match actual prepared repair generation"
        );
        assert!(gpu.events.lock().is_empty());
    });
}

#[test]
fn real_sequence_arm_bounds_all_attempts_and_resets_reused_state_only_for_new_request() {
    for rank in 0..2 {
        fixture(rank, |head, ctx, gpu, saved| {
            let mut seq = sequence(head, ctx, gpu, 11);
            for attempt in 1..=10 {
                prepared_with_source(head, ctx, gpu, state(&mut seq), 11, 3);
                gpu.events.lock().clear();
                let queries = gpu.capture_queries.load(Ordering::Relaxed);
                arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
                let drafts = head
                    .propose(
                        3,
                        saved,
                        3,
                        4,
                        state(&mut seq),
                        None,
                        ctx,
                        7,
                        None,
                        None,
                        None,
                    )
                    .unwrap();
                assert_eq!(drafts, [7, 7, 7, 7]);
                let reads: Vec<_> = gpu
                    .events
                    .lock()
                    .iter()
                    .filter_map(|e| {
                        if let Event::Read(p, ROW_BYTES, s) = e {
                            Some((*p, ROW_BYTES, *s))
                        } else {
                            None
                        }
                    })
                    .collect();
                assert_eq!(reads.len(), if attempt <= 8 { 9 } else { 0 });
                if attempt <= 8 {
                    assert_eq!(reads[0], (saved, ROW_BYTES, 7));
                    assert_eq!(reads[1], (ctx.buffers.hidden_states(), ROW_BYTES, 7));
                    assert!(
                        reads[2..]
                            .iter()
                            .all(|v| *v == (ctx.buffers.norm_output(), ROW_BYTES, 7))
                    );
                } else {
                    assert_eq!(gpu.capture_queries.load(Ordering::Relaxed), queries);
                }
                assert_eq!(state(&mut seq).hidden_trace.spent, attempt.min(8));
            }
            // Keep the same boxed proposer state and slot; actual sequence capture
            // ownership changes as it does at a new request's cold chunk zero.
            seq.mtp_capture_gen = 12;
            prepared_with_source(head, ctx, gpu, state(&mut seq), 12, 3);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
            assert_eq!(state(&mut seq).hidden_trace.spent, 1);
            assert_eq!(
                state(&mut seq).hidden_trace.active.unwrap().request.rank,
                rank
            );
            seq.mtp_capture_gen = 11;
            prepared_with_source(head, ctx, gpu, state(&mut seq), 11, 3);
            assert!(arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).is_err());
            assert_eq!(state(&mut seq).hidden_trace.spent, 1);
            head.free_state(gpu, None, state(&mut seq)).unwrap();
            head.free_state(gpu, None, state(&mut seq)).unwrap();
            assert_eq!(state(&mut seq).hidden_trace.spent, 0);
            assert!(state(&mut seq).hidden_trace.enabled);
            assert!(state(&mut seq).hidden_trace.identity.is_none());
        });
    }
}

#[test]
fn actual_disabled_hook_has_no_reads_queries_or_changed_drafts_and_hidden() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, gpu, 1);
        state(&mut seq).hidden_trace.enabled = false;
        let allocations = gpu.inner.alloc_count();
        arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
        let off = head
            .propose(
                3,
                saved,
                3,
                4,
                state(&mut seq),
                None,
                ctx,
                7,
                None,
                None,
                None,
            )
            .unwrap();
        let off_events = gpu.events.lock().clone();
        let off_hidden = gpu.inner.read_alloc(ctx.buffers.norm_output()).unwrap();
        assert!(off_events.iter().all(|e| !matches!(e, Event::Read(..))));
        assert_eq!(gpu.capture_queries.load(Ordering::Relaxed), 0);
        assert_eq!(state(&mut seq).hidden_trace.spent, 0);
        assert_eq!(gpu.inner.alloc_count(), allocations);
        state(&mut seq).hidden_trace.enabled = true;
        prepared_with_source(head, ctx, gpu, state(&mut seq), 1, 3);
        let allocations = gpu.inner.alloc_count();
        gpu.events.lock().clear();
        arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
        let on = head
            .propose(
                3,
                saved,
                3,
                4,
                state(&mut seq),
                None,
                ctx,
                7,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(on, off);
        assert_eq!(
            gpu.inner.read_alloc(ctx.buffers.norm_output()).unwrap(),
            off_hidden
        );
        let on_events: Vec<_> = gpu
            .events
            .lock()
            .iter()
            .filter(|e| !matches!(e, Event::Read(..)))
            .cloned()
            .collect();
        assert_eq!(off_events, on_events);
        assert_eq!(
            gpu.inner.alloc_count(),
            allocations,
            "trace adds no GPU allocation"
        );
    });
}

#[test]
fn actual_capture_and_bad_profile_refuse_before_diagnostic_io() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, gpu, 1);
        for stream_capture in [false, true] {
            ctx.graph_capture = !stream_capture;
            gpu.capturing.store(stream_capture, Ordering::Relaxed);
            assert!(arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).is_err());
            assert!(gpu.events.lock().is_empty());
        }
        ctx.graph_capture = false;
        gpu.capturing.store(false, Ordering::Relaxed);
        for fault in 0..5 {
            seq.cached_prefix_tokens = usize::from(fault == 0);
            seq.adapter_id = u64::from(fault == 1);
            let drafts = if fault == 2 { 3 } else { 4 };
            let source = if fault == 4 { DevicePtr::NULL } else { saved };
            assert!(arm_prepared(&mut seq, 3, 3, drafts, source, 0, fault == 3, ctx, 7).is_err());
            assert!(gpu.events.lock().is_empty());
        }
    });
}

#[test]
fn actual_failed_input_copy_spends_attempt_and_does_not_execute_body() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, gpu, 1);
        gpu.fail_read.store(true, Ordering::Relaxed);
        for attempt in 1..=8 {
            prepared_with_source(head, ctx, gpu, state(&mut seq), 1, 3);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
            gpu.events.lock().clear();
            assert!(
                head.forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None)
                    .is_err()
            );
            assert_eq!(*gpu.events.lock(), [Event::Read(saved, ROW_BYTES, 7)]);
            assert_eq!(state(&mut seq).hidden_trace.spent, attempt);
        }
        prepared_with_source(head, ctx, gpu, state(&mut seq), 1, 3);
        arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
        gpu.events.lock().clear();
        assert_eq!(
            head.forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None)
                .unwrap(),
            7
        );
        assert!(
            gpu.events
                .lock()
                .iter()
                .all(|e| !matches!(e, Event::Read(..)))
        );
    });
}
