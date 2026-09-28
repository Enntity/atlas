// SPDX-License-Identifier: AGPL-3.0-only
//! Actual complete-chunk F0 and request-owned primer; no native collective proof.
use super::bootstrap_transport_tests::Snapshot;
use super::{
    fixture::*, isolated, transport_test_fixture as wire, verdict_continuation_tests as flow,
};
use crate::traits::Model;
use std::sync::atomic::Ordering;

fn packets(owner: usize, tokens: &[u32]) -> Vec<Vec<u32>> {
    vec![
        vec![owner as u32],
        vec![0xfffffff0],
        vec![tokens.len() as u32],
        vec![0],
        vec![tokens.len() as u32],
        tokens.to_vec(),
        vec![0],
        vec![0],
    ]
}

#[test]
fn actual_immutable_cold_validation_then_complete_transport() {
    if isolated(
        "cold_prefill_transport_tests::actual_immutable_cold_validation_then_complete_transport",
    ) {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut head = Fixture::new(0);
        let tx = wire::Wire::install(&mut head, 0);
        tx.enable_cold_prefix();
        let prompts = [vec![1, 2, 3, 4], vec![6, 5, 4, 3, 2, 1]];
        for owner in 0..2 {
            head.seqs[owner].prompt_len = prompts[owner].len();
        }
        head.gpu.clear();
        let free = head.model.kv_cache.lock().num_free_blocks();
        for _ in 0..2 {
            for owner in order {
                head.model
                    .glm_paired_execution()
                    .unwrap()
                    .validate_cold_prefill(&head.seqs[owner], &prompts[owner])
                    .unwrap();
            }
        }
        assert!(tx.packets().is_empty());
        assert!(head.gpu.trace().is_empty());
        assert_eq!(head.model.kv_cache.lock().num_free_blocks(), free);
        for owner in order {
            tx.clear();
            head.gpu.clear();
            head.model
                .glm_paired_execution()
                .unwrap()
                .cold_prefill(&mut head.seqs[owner], &prompts[owner])
                .unwrap();
            assert_eq!(tx.packets(), packets(owner, &prompts[owner]));
            assert_eq!(tx.roots(), [0, 0, 0, 0, 0, 0, 0, 1]);
            assert_eq!(head.seqs[owner].tokens, prompts[owner]);
            assert_eq!(
                flow::private(&head.seqs[owner]).seq_len,
                prompts[owner].len() - 1
            );
            assert!(
                head.gpu
                    .trace()
                    .contains(&Event::Target(prompts[owner].len(), 0, DEFAULT))
            );
        }
    }
}

#[test]
fn actual_head_packets_replay_both_owners_and_bootstrap() {
    if isolated(
        "cold_prefill_transport_tests::actual_head_packets_replay_both_owners_and_bootstrap",
    ) {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut head = Fixture::new(0);
        let mut peer = Fixture::new(1);
        let tx = wire::Wire::install(&mut head, 0);
        let rx = wire::Wire::install(&mut peer, 1);
        tx.enable_cold_prefix();
        rx.enable_cold_prefix();
        for owner in order {
            let tokens = if owner == 0 {
                [1, 2, 3, 4]
            } else {
                [4, 3, 2, 1]
            };
            tx.clear();
            head.model
                .glm_paired_execution()
                .unwrap()
                .cold_prefill(&mut head.seqs[owner], &tokens)
                .unwrap();
            rx.queue(&tx.packets());
            assert!(wire::worker(&mut peer).unwrap());
            rx.done();
            assert_eq!(rx.roots(), tx.roots());
            wire::same_private(&head, &peer, owner);
            assert_eq!(head.seqs[owner].tokens, peer.seqs[owner].tokens);
            tx.clear();
            head.model
                .glm_paired_execution()
                .unwrap()
                .bootstrap(&mut head.seqs[owner], 5 + owner as u32)
                .unwrap();
            rx.queue(&tx.packets());
            assert!(wire::worker(&mut peer).unwrap());
            rx.done();
            wire::same_private(&head, &peer, owner);
        }
        assert_ne!(flow::slab(&head, 0, 0), flow::slab(&head, 1, 0));
    }
}

#[test]
fn cold_profile_rejections_are_zero_wire_and_non_claiming() {
    if isolated(
        "cold_prefill_transport_tests::cold_profile_rejections_are_zero_wire_and_non_claiming",
    ) {
        return;
    }
    for case in 0..12 {
        let mut f = Fixture::new(if case == 11 { 1 } else { 0 });
        let rank = if case == 11 { 1 } else { 0 };
        let tx = wire::Wire::install(&mut f, rank);
        tx.enable_cold_prefix();
        let capacity = f.model.mtp_prefill_capacity;
        let mut tokens = vec![1, 2, 3, 4];
        match case {
            0 => tokens.clear(),
            1 => tokens = vec![1],
            2 => tokens = vec![1; 1025],
            3 => tokens[0] = 8,
            4 => f.model.ep_protocol_v2 = false,
            5 => f.gpu.capturing.store(true, Ordering::Relaxed),
            6 => f.seqs[0].slot_idx = 2,
            7 => f.seqs[0].seq_len = 1,
            8 => f.seqs[0].cached_prefix_tokens = 1,
            9 => f.seqs[0].mtp_capture_gen = 1,
            10 => f.model.mtp_prefill_capacity = 3,
            11 => {}
            _ => unreachable!(),
        }
        f.gpu.clear();
        let slab = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
        let free = f.model.kv_cache.lock().num_free_blocks();
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .validate_cold_prefill(&f.seqs[0], &tokens)
                .is_err(),
            "case{case}"
        );
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .cold_prefill(&mut f.seqs[0], &tokens)
                .is_err(),
            "case{case}"
        );
        assert!(tx.packets().is_empty());
        assert!(f.gpu.trace().is_empty());
        assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), slab);
        assert_eq!(f.model.kv_cache.lock().num_free_blocks(), free);
        if rank == 0 {
            f.model.ep_protocol_v2 = true;
            f.gpu.capturing.store(false, Ordering::Relaxed);
            f.seqs[0].slot_idx = 0;
            f.seqs[0].seq_len = 0;
            f.seqs[0].cached_prefix_tokens = 0;
            f.seqs[0].mtp_capture_gen = 0;
            f.model.mtp_prefill_capacity = capacity;
            f.model
                .glm_paired_execution()
                .unwrap()
                .cold_prefill(&mut f.seqs[0], &[1, 2, 3, 4])
                .unwrap();
        }
    }
}

fn setup_fault(rank: usize, owner: usize) -> (Fixture, std::sync::Arc<wire::Wire>, Snapshot) {
    let mut f = Fixture::new(rank);
    let peer = 1 - owner;
    // Publish the peer through the existing actual producer before installing wire.
    f.model
        .prefill(&[4, 3, 2, 1], &mut f.seqs[peer], CALLER)
        .unwrap();
    let saved = Snapshot::new(&f, peer);
    let w = wire::Wire::install(&mut f, rank);
    w.enable_cold_prefix();
    if rank == 1 {
        w.queue(&packets(owner, &[1, 2, 3, 4]));
    }
    f.gpu.clear();
    (f, w, saved)
}
fn execute(f: &mut Fixture, owner: usize, rank: usize) -> anyhow::Result<()> {
    if rank == 0 {
        f.model
            .glm_paired_execution()
            .unwrap()
            .cold_prefill(&mut f.seqs[owner], &[1, 2, 3, 4])?;
    } else {
        assert!(wire::worker(f)?);
    }
    Ok(())
}
fn terminal(f: &mut Fixture, saved: &Snapshot, owner: usize) {
    saved.unchanged(f, 1 - owner);
    f.gpu.clear();
    f.model.gpu.synchronize(DEFAULT).unwrap();
    f.gpu.clear();
    for slot in [owner, 1 - owner] {
        assert!(
            f.model
                .prefill(&[1, 2, 3, 4], &mut f.seqs[slot], CALLER)
                .is_err()
        );
    }
    assert!(f.model.decode(5, &mut f.seqs[1 - owner], CALLER).is_err());
    assert!(f.gpu.trace().is_empty());
    saved.unchanged(f, 1 - owner);
}

#[test]
fn issued_f0_faults_latch_without_touching_published_peer() {
    if isolated(
        "cold_prefill_transport_tests::issued_f0_faults_latch_without_touching_published_peer",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            // Derive each operation ordinal from a fresh successful actual control.
            for boundary in 0..4 {
                let (mut f, w, saved) = setup_fault(rank, owner);
                let capture = f.model.mtp_prefill_hidden.offset(3 * ROW_BYTES);
                let tail = f.gpu.slab().offset(owner * 6 * ROW_BYTES);
                execute(&mut f, owner, rank).unwrap();
                w.done();
                saved.unchanged(&f, 1 - owner);
                let trace = f.gpu.trace();
                let tail_index = trace
                    .iter()
                    .position(|e| *e == Event::Copy(capture, tail, ROW_BYTES, DEFAULT))
                    .unwrap();
                let index = match boundary {
                    0 => trace
                        .iter()
                        .position(|e| *e == Event::Target(4, 0, DEFAULT))
                        .unwrap(),
                    1 => trace
                        .iter()
                        .position(|e| matches!(e, Event::Kv(_, DEFAULT)))
                        .unwrap(),
                    2 => tail_index,
                    3 => tail_index + 1,
                    _ => unreachable!(),
                };
                if boundary == 3 {
                    assert_eq!(trace[index], Event::Sync(DEFAULT));
                }
                let (mut f, _, saved) = setup_fault(rank, owner);
                f.gpu.fail.store(index + 1, Ordering::Relaxed);
                let err = execute(&mut f, owner, rank).unwrap_err();
                assert!(format!("{err:#}").contains("injected fixture operation failure"));
                assert_eq!(f.gpu.trace().len(), index + 1);
                terminal(&mut f, &saved, owner);
            }
            // First header (head only), token payload, and both prefix roots.
            for ordinal in if rank == 0 {
                vec![1, 6, 7, 8]
            } else {
                vec![6, 7, 8]
            } {
                let (mut f, w, saved) = setup_fault(rank, owner);
                w.fail.store(ordinal, Ordering::Relaxed);
                let err = execute(&mut f, owner, rank).unwrap_err();
                assert!(format!("{err:#}").contains("injected command transfer failure"));
                assert_eq!(w.packets().len(), ordinal);
                assert!(!f.gpu.trace().iter().any(|e| matches!(e, Event::Target(..))));
                terminal(&mut f, &saved, owner);
            }
        }
    }
}

#[test]
fn worker_bounds_metadata_before_bulk_receive() {
    if isolated("cold_prefill_transport_tests::worker_bounds_metadata_before_bulk_receive") {
        return;
    }
    for metadata in [
        [4, 1, 4],
        [3, 0, 4],
        [1, 0, 1],
        [1025, 0, 1025],
        [u32::MAX, 0, u32::MAX],
    ] {
        let (mut f, w, saved) = setup_fault(1, 0);
        // Replace queued valid input using a fresh wire; no token packet is supplied.
        let _ = w;
        let rx = wire::Wire::install(&mut f, 1);
        rx.enable_cold_prefix();
        rx.queue(&[
            vec![0],
            vec![0xfffffff0],
            vec![metadata[0]],
            vec![metadata[1]],
            vec![metadata[2]],
        ]);
        let free = f.model.kv_cache.lock().num_free_blocks();
        let error = wire::worker(&mut f).unwrap_err();
        rx.done();
        assert!(format!("{error:#}").contains("bounded complete cold chunk"));
        assert_eq!(rx.packets().len(), 5);
        assert!(
            !f.gpu
                .trace()
                .iter()
                .any(|e| matches!(e, Event::Target(..) | Event::Alloc(..)))
        );
        assert_eq!(f.model.kv_cache.lock().num_free_blocks(), free);
        terminal(&mut f, &saved, 0);
    }
}

#[test]
fn actual_worker_cold_f0_publishes_owned_tail() {
    if isolated("cold_prefill_transport_tests::actual_worker_cold_f0_publishes_owned_tail") {
        return;
    }
    let mut peer = Fixture::new(1);
    let rx = wire::Wire::install(&mut peer, 1);
    rx.enable_cold_prefix();
    let tokens = [1, 2, 3, 4];
    rx.queue(&packets(1, &tokens));
    peer.gpu.clear();
    assert!(wire::worker(&mut peer).unwrap());
    rx.done();
    assert_eq!(rx.roots(), [0, 0, 0, 0, 0, 0, 0, 1]);
    assert_eq!(peer.seqs[1].tokens, tokens);
    assert_eq!(flow::private(&peer.seqs[1]).seq_len, 3);
    assert_eq!(flow::slab(&peer, 1, 0), vec![0x27; ROW_BYTES]);
}
