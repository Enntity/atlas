// SPDX-License-Identifier: AGPL-3.0-only
//! Drive actual proposer hook ordering and request-owned first-attempt budget.
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

fn prefix(head: &Glm5MtpHead, state: &mut Glm5MtpProposerState, gpu: &support::TraceGpu) {
    let mut cache = head.kv_cache.lock();
    while state.block_table.is_empty() {
        state.block_table.push(cache.alloc_block().unwrap());
    }
    let block = state.block_table[0];
    for (side, pool) in [cache.k_pool_ptr(0), cache.v_pool_ptr(0)]
        .into_iter()
        .enumerate()
    {
        let bytes: Vec<_> = (0..2048)
            .map(|i| (i ^ (i >> 8) ^ (side * 31)) as u8)
            .collect();
        gpu.inner
            .copy_h2d(&bytes, pool.offset(block as usize * 16384))
            .unwrap();
    }
}

#[test]
fn actual_first_hook_request_repair_and_missing_order_fail_without_kv_reads() {
    for fault in 0..8 {
        fixture(0, |head, ctx, gpu, saved| {
            let mut seq = sequence(head, ctx, gpu, 1);
            prefix(head, state(&mut seq), gpu);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
            let s = state(&mut seq);
            let mut trace = s
                .hidden_trace
                .input(3, 3, 0, saved, 2, ctx, 7)
                .unwrap()
                .unwrap();
            trace.post_eh(ctx.buffers.hidden_states(), ctx, 7).unwrap();
            match fault {
                0 => trace.post_eh = None,
                1 => trace.request.hidden_row = 1,
                2 => trace.request.identity.generation = 2,
                3 => trace.request.position = 4,
                4 => s.seq_len = 3,
                5 => s.repair = repair_state::RepairPhase::Capture,
                6 => prepared_with_source(head, ctx, gpu, s, 1, 4),
                7 => ctx.graph_capture = true,
                _ => unreachable!(),
            }
            let cache = head.kv_cache.lock();
            gpu.events.lock().clear();
            assert!(trace.kv_after(&cache, s, ctx, 7).is_err());
            assert!(trace.kv_before(&cache, s, ctx, 7).is_err());
            assert!(trace.kv_before(&cache, s, ctx, 7).is_err());
            assert!(gpu.events.lock().is_empty());
            assert_eq!(s.hidden_trace.spent, 1);
        });
    }
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, gpu, 1);
        prefix(head, state(&mut seq), gpu);
        arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
        let s = state(&mut seq);
        let mut trace = s
            .hidden_trace
            .input(3, 3, 0, saved, 2, ctx, 7)
            .unwrap()
            .unwrap();
        trace.post_eh(ctx.buffers.hidden_states(), ctx, 7).unwrap();
        let cache = head.kv_cache.lock();
        trace.kv_before(&cache, s, ctx, 7).unwrap();
        gpu.events.lock().clear();
        assert!(trace.kv_before(&cache, s, ctx, 7).is_err());
        assert!(gpu.events.lock().is_empty());
        trace.kv_after(&cache, s, ctx, 7).unwrap();
        gpu.events.lock().clear();
        assert!(trace.kv_after(&cache, s, ctx, 7).is_err());
        assert!(gpu.events.lock().is_empty());
    });
}

#[test]
fn actual_first_hook_reads_prefix_before_body_and_distinct_written_row_after() {
    for rank in 0..2 {
        fixture(rank, |head, ctx, gpu, saved| {
            let mut seq = sequence(head, ctx, gpu, 1);
            prefix(head, state(&mut seq), gpu);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
            let s = state(&mut seq);
            let mut trace = s
                .hidden_trace
                .input(3, 3, 0, saved, 2, ctx, 7)
                .unwrap()
                .unwrap();
            head.forward_body_one(3, saved, 3, s, ctx, 7, Some(&mut trace))
                .unwrap();
            let probe = trace.kv.as_ref().expect("actual first attempt probe");
            let mut expected = Sha256::new();
            expected.update(b"atlas/glm53/mtp-kv/appended/v1\0");
            expected.update(2u64.to_le_bytes());
            expected.update(512u32.to_le_bytes());
            expected.update(2u32.to_le_bytes());
            expected.update([0x62; 1024]);
            expected.update([0x72; 1024]);
            assert_eq!(probe.appended, Some(expected.finalize().into()));
            let events = gpu.events.lock();
            let body = events.iter().position(|e| *e == Event::Body).unwrap();
            let reads: Vec<_> = events
                .iter()
                .enumerate()
                .filter(|(_, e)| matches!(e, Event::Read(_, 1024 | 2048, 7)))
                .collect();
            assert_eq!(reads.len(), 4);
            assert!(reads[0].0 < body && reads[1].0 < body);
            assert!(reads[2].0 > body && reads[3].0 > body);
            assert_eq!(s.seq_len, 3);
        });
    }
}

#[test]
fn actual_each_kv_copy_failure_spends_first_attempt_without_replacement() {
    for fail in 1..=4 {
        for rank in 0..2 {
            fixture(rank, |head, ctx, gpu, saved| {
                let mut seq = sequence(head, ctx, gpu, 1);
                prefix(head, state(&mut seq), gpu);
                arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
                gpu.fail_kv_read_at.store(fail, Ordering::Relaxed);
                let err = head
                    .forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None)
                    .unwrap_err();
                assert!(err.to_string().contains("injected KV"));
                assert_eq!(gpu.kv_reads.load(Ordering::Relaxed), fail);
                assert_eq!(gpu.events.lock().contains(&Event::Body), fail > 2);
                assert_eq!(state(&mut seq).hidden_trace.spent, 1);
                gpu.fail_kv_read_at.store(usize::MAX, Ordering::Relaxed);
                gpu.kv_reads.store(0, Ordering::Relaxed);
                prepared_with_source(head, ctx, gpu, state(&mut seq), 1, 3);
                arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
                head.forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None)
                    .unwrap();
                assert_eq!(gpu.kv_reads.load(Ordering::Relaxed), 0);
                assert_eq!(state(&mut seq).hidden_trace.spent, 2);
            });
        }
    }
}

#[test]
fn actual_unselected_later_steps_attempts_and_disabled_trace_do_not_read_kv() {
    fixture(0, |head, ctx, gpu, saved| {
        let mut seq = sequence(head, ctx, gpu, 1);
        prefix(head, state(&mut seq), gpu);
        for attempt in 1..=9 {
            prepared_with_source(head, ctx, gpu, state(&mut seq), 1, 3);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
            gpu.kv_reads.store(0, Ordering::Relaxed);
            head.propose(
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
            assert_eq!(
                gpu.kv_reads.load(Ordering::Relaxed),
                if attempt == 1 { 4 } else { 0 }
            );
        }
        seq.mtp_capture_gen = 2;
        prepared_with_source(head, ctx, gpu, state(&mut seq), 2, 3);
        arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
        gpu.kv_reads.store(0, Ordering::Relaxed);
        head.forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None)
            .unwrap();
        assert_eq!(gpu.kv_reads.load(Ordering::Relaxed), 4);
        state(&mut seq).hidden_trace.enabled = false;
        state(&mut seq).hidden_trace.reset();
        prepared_with_source(head, ctx, gpu, state(&mut seq), 2, 3);
        gpu.kv_reads.store(0, Ordering::Relaxed);
        head.forward_one(3, saved, 3, 0, state(&mut seq), ctx, 7, None)
            .unwrap();
        assert_eq!(gpu.kv_reads.load(Ordering::Relaxed), 0);
    });
}
