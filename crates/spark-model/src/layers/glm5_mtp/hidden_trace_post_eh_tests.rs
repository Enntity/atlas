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
fn actual_post_eh_hook_reads_projection_sentinel_before_body_overwrite() {
    for rank in 0..2 {
        fixture(rank, |head, ctx, gpu, saved| {
            let mut seq = sequence(head, ctx, gpu, 1);
            for (attempt, last) in [0x5a, 0xda].into_iter().enumerate() {
                prepared_with_source(head, ctx, gpu, state(&mut seq), 1, 3);
                arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
                gpu.read_hashes.lock().clear();
                gpu.eh_last_byte.store(last, Ordering::Relaxed);
                assert_eq!(
                    head.forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None)
                        .unwrap(),
                    7
                );
                let hashes = gpu.read_hashes.lock();
                let mut expected = [0xa5; ROW_BYTES];
                expected[ROW_BYTES - 1] = last;
                assert_eq!(hashes.len(), if attempt == 0 { 7 } else { 3 });
                assert_eq!(hashes[1], <[u8; 32]>::from(Sha256::digest(expected)));
                assert_ne!(hashes[1], hashes[0]);
                assert_ne!(hashes[1], *hashes.last().unwrap());
            }
        });
    }
}

#[test]
fn actual_post_eh_failure_spends_all_attempts_before_body_then_exhausts() {
    for rank in 0..2 {
        fixture(rank, |head, ctx, gpu, saved| {
            let mut seq = sequence(head, ctx, gpu, 1);
            gpu.fail_post_eh_read.store(true, Ordering::Relaxed);
            for attempt in 1..=9 {
                prepared_with_source(head, ctx, gpu, state(&mut seq), 1, 3);
                arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
                gpu.events.lock().clear();
                let queries = gpu.capture_queries.load(Ordering::Relaxed);
                let result = head.forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None);
                let events = gpu.events.lock();
                if attempt <= 8 {
                    assert!(result.is_err());
                    assert!(!events.contains(&Event::Body));
                    let reads: Vec<_> = events
                        .iter()
                        .filter(|e| matches!(e, Event::Read(..)))
                        .cloned()
                        .collect();
                    assert_eq!(
                        reads,
                        [
                            Event::Read(saved, ROW_BYTES, 7),
                            Event::Read(ctx.buffers.hidden_states(), ROW_BYTES, 7)
                        ]
                    );
                } else {
                    assert_eq!(result.unwrap(), 7);
                    assert!(events.iter().all(|e| !matches!(e, Event::Read(..))));
                    assert_eq!(gpu.capture_queries.load(Ordering::Relaxed), queries);
                }
                assert_eq!(state(&mut seq).hidden_trace.spent, attempt.min(8));
            }
            seq.mtp_capture_gen = 2;
            prepared_with_source(head, ctx, gpu, state(&mut seq), 2, 3);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
            assert_eq!(state(&mut seq).hidden_trace.spent, 1);
        });
    }
}

#[test]
fn post_eh_metadata_missing_duplicate_capture_and_later_step_are_fail_closed() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, gpu, 1);
        arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
        let state = state(&mut seq);
        let mut record = state
            .hidden_trace
            .input(3, 3, 0, saved, state.seq_len, ctx, 7)
            .unwrap()
            .unwrap();
        let owner = ctx.buffers.hidden_states();
        gpu.events.lock().clear();
        assert!(
            record
                .final_hidden(ctx.buffers.norm_output(), state.seq_len + 1, ctx, 7)
                .is_err()
        );
        assert!(record.post_eh(saved, ctx, 7).is_err());
        for fault in 0..3 {
            gpu.capturing.store(fault == 0, Ordering::Relaxed);
            ctx.graph_capture = fault == 1;
            if fault == 2 {
                let mut small = ctx.config.clone();
                small.hidden_size = 1;
                let buffers =
                    spark_runtime::buffers::BufferArena::new(&small, 1, 16, 16, 1, gpu).unwrap();
                let small_ctx = ForwardContext {
                    buffers: &buffers,
                    gpu: ctx.gpu,
                    config: ctx.config,
                    dispatch: ctx.dispatch,
                    derived: ctx.derived,
                    levers: ctx.levers,
                    stats: ctx.stats,
                    comm: ctx.comm,
                    ssm_batch: None,
                    host_token_ids: None,
                    attn_metadata: None,
                    profile: false,
                    graph_capture: false,
                    gdn_exact_replay: false,
                    token_ids: None,
                    routed_lora_layers: None,
                    midchunk_capture: None,
                    moe_lora_route: ctx.moe_lora_route,
                };
                assert!(
                    record
                        .post_eh(buffers.hidden_states(), &small_ctx, 7)
                        .is_err()
                );
            } else {
                assert!(record.post_eh(owner, ctx, 7).is_err());
            }
        }
        ctx.graph_capture = false;
        gpu.capturing.store(false, Ordering::Relaxed);
        assert!(gpu.events.lock().is_empty());
        record.post_eh(owner, ctx, 7).unwrap();
        assert!(record.post_eh(owner, ctx, 7).is_err());
        let hash = record.post_eh.unwrap();
        assert_eq!(
            format!("{}", OptionalHex(Some(&hash))),
            format!("{}", Hex(&hash))
        );
        assert_eq!(format!("{}", OptionalHex(None)), "None");
        // Post-EH alone no longer admits a complete first-attempt record.
        assert!(
            record
                .final_hidden(ctx.buffers.norm_output(), state.seq_len + 1, ctx, 7)
                .is_err()
        );
        record.step = 1;
        let queries = gpu.capture_queries.load(Ordering::Relaxed);
        gpu.events.lock().clear();
        gpu.capturing.store(true, Ordering::Relaxed);
        ctx.graph_capture = true;
        record.post_eh(DevicePtr::NULL, ctx, 7).unwrap();
        assert_eq!(gpu.capture_queries.load(Ordering::Relaxed), queries);
        assert!(gpu.events.lock().is_empty());
    });
}
