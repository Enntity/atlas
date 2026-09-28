// SPDX-License-Identifier: AGPL-3.0-only
//! Predictable head refusal and received identity rejection through real APIs.
use super::{fixture::*, transport_test_fixture as wire, verdict_continuation_tests as flow};
use crate::speculative::DraftProposer;
use crate::traits::Model;
use std::sync::atomic::Ordering;

pub(super) fn terminal(f: &mut Fixture) {
    f.model.gpu.synchronize(DEFAULT).unwrap();
    f.gpu.clear();
    for owner in 0..2 {
        let position = f.seqs[owner].seq_len;
        assert!(
            f.model
                .run_mtp_propose_inner(7, position, 4, &mut f.seqs[owner], None)
                .is_err()
        );
        assert!(f.model.decode(1, &mut f.seqs[owner], CALLER).is_err());
    }
    assert!(f.head.alloc_state(f.model.gpu.as_ref()).is_err());
    assert!(
        f.gpu.trace().is_empty(),
        "terminal state must refuse before further backend work"
    );
}

#[test]
fn selected_head_bad_proposal_is_zero_wire_and_does_not_consume_owner() {
    if flow::isolated(
        "transport_boundary_tests::selected_head_bad_proposal_is_zero_wire_and_does_not_consume_owner",
    ) {
        return;
    }
    for owner in 0..2 {
        for case in 0..7 {
            let mut f = wire::bootstrapped(0, [1, 0]);
            let tx = wire::Wire::install(&mut f, 0);
            let position = f.seqs[owner].seq_len;
            let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
            let (mut seed, mut requested, mut width, mut grammar) = (7, position, 4, None);
            match case {
                0 => seed = 8,
                1 => requested -= 1,
                2 => width = 3,
                3 => grammar = Some(&[][..]),
                4 => f.model.ep_protocol_v2 = false,
                5 => f.gpu.capturing.store(true, Ordering::Relaxed),
                6 => f.model.levers.max_decode_seqs = 1,
                _ => unreachable!(),
            }
            assert!(
                f.model
                    .glm_paired_execution()
                    .unwrap()
                    .propose(&mut f.seqs[owner], seed, requested, width, grammar)
                    .is_err()
            );
            assert!(tx.packets().is_empty(), "case {case}");
            assert!(f.gpu.trace().is_empty(), "case {case}");
            assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
            f.model.ep_protocol_v2 = true;
            f.gpu.capturing.store(false, Ordering::Relaxed);
            f.model.levers.max_decode_seqs = 2;
            f.model
                .glm_paired_execution()
                .unwrap()
                .propose(&mut f.seqs[owner], 7, position, 4, None)
                .unwrap();
        }
    }
}

#[test]
fn selected_head_wrong_issued_tokens_are_zero_wire_then_actual_f5_succeeds() {
    if flow::isolated(
        "transport_boundary_tests::selected_head_wrong_issued_tokens_are_zero_wire_then_actual_f5_succeeds",
    ) {
        return;
    }
    for owner in 0..2 {
        let (mut f, histories) = flow::prepare(0, [0, 1]);
        let tx = wire::Wire::install(&mut f, 0);
        f.gpu.clear();
        let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
        assert!(
            f.model
                .glm_paired_execution()
                .unwrap()
                .verify(&mut f.seqs[owner], &histories[1 - owner].issued)
                .is_err()
        );
        assert!(tx.packets().is_empty());
        assert!(f.gpu.trace().is_empty());
        assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
        f.model
            .glm_paired_execution()
            .unwrap()
            .verify(&mut f.seqs[owner], &histories[owner].issued)
            .unwrap();
    }
}

#[test]
fn worker_rejects_each_versioned_identity_field_before_private_compute_and_latches() {
    if flow::isolated(
        "transport_boundary_tests::worker_rejects_each_versioned_identity_field_before_private_compute_and_latches",
    ) {
        return;
    }
    for owner in 0..2 {
        let mut head = wire::bootstrapped(0, [1, 0]);
        let tx = wire::Wire::install(&mut head, 0);
        let position = head.seqs[owner].seq_len;
        head.model
            .glm_paired_execution()
            .unwrap()
            .propose(&mut head.seqs[owner], 7, position, 4, None)
            .unwrap();
        for field in 0..9 {
            let mut f = wire::bootstrapped(1, [1, 0]);
            let rx = wire::Wire::install(&mut f, 1);
            let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
            let mut packets = tx.packets();
            if field == 8 {
                packets[0][0] = 1 - owner as u32;
            } else if field == 7 {
                packets[2][field] = 8; // Invalid vocabulary, not a valid different sampler seed.
            } else {
                packets[2][field] ^= 1;
            }
            rx.queue(&packets);
            let error = wire::worker(&mut f).unwrap_err();
            assert!(format!("{error:#}").contains("paired"), "{error:#}");
            rx.done();
            assert!(!f.gpu.trace().iter().any(|e| matches!(
                e,
                Event::Kernel(_, _, _)
                    | Event::Target(_, _, _)
                    | Event::Body(_, _)
                    | Event::Kv(_, _)
            )));
            assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
            terminal(&mut f);
        }
    }
}

#[test]
fn worker_consumes_valid_selected_nonraw_seed_not_global_hidden_row() {
    if flow::isolated(
        "transport_boundary_tests::worker_consumes_valid_selected_nonraw_seed_not_global_hidden_row",
    ) {
        return;
    }
    for owner in 0..2 {
        let mut head = wire::bootstrapped(0, [0, 1]);
        let tx = wire::Wire::install(&mut head, 0);
        let position = head.seqs[owner].seq_len;
        head.model
            .glm_paired_execution()
            .unwrap()
            .propose(&mut head.seqs[owner], 7, position, 4, None)
            .unwrap();
        let mut packets = tx.packets();
        packets[2][7] = 2; // A different valid caller-selected seed is supported.
        let mut worker = wire::bootstrapped(1, [0, 1]);
        let mut control = wire::bootstrapped(1, [0, 1]);
        let rx = wire::Wire::install(&mut worker, 1);
        let drafts = control
            .model
            .run_mtp_propose_inner(2, position, 4, &mut control.seqs[owner], None)
            .unwrap();
        rx.queue(&packets);
        assert!(wire::worker(&mut worker).unwrap());
        rx.done();
        wire::same_private(&worker, &control, owner);
        assert_eq!(worker.gpu.eh_pairs(), control.gpu.eh_pairs());
        let mut issued = vec![2];
        issued.extend(drafts);
        worker
            .model
            .decode_verify_graphed_kgamma(&issued, &mut worker.seqs[owner], CALLER)
            .unwrap();
    }
}

#[test]
fn actual_next_block_exhaustion_refuses_e1_and_f5_before_first_header() {
    if flow::isolated(
        "transport_boundary_tests::actual_next_block_exhaustion_refuses_e1_and_f5_before_first_header",
    ) {
        return;
    }
    for verify in [false, true] {
        let (mut f, mut history) = flow::prepare(0, [0, 1]);
        for round in 0..2 {
            flow::head_verdict(&mut f, 0, &history[0], 4);
            flow::acknowledge(&mut f, 0, 4, false);
            if round == 0 || verify {
                flow::continue_owner(&mut f, 0, &mut history[0], 4);
            }
        }
        assert_eq!(f.seqs[0].seq_len, 15);
        assert_eq!(f.seqs[0].block_table.len(), 1);
        let tx = wire::Wire::install(&mut f, 0);
        let taken = {
            let mut cache = f.model.kv_cache.lock();
            let free = cache.num_free_blocks();
            (0..free)
                .map(|_| cache.alloc_block().unwrap())
                .collect::<Vec<_>>()
        };
        let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
        f.gpu.clear();
        let cap = f.model.glm_paired_execution().unwrap();
        let result = if verify {
            cap.verify(&mut f.seqs[0], &history[0].issued)
        } else {
            cap.propose(&mut f.seqs[0], 7, 15, 4, None)
        };
        assert!(format!("{:#}", result.unwrap_err()).contains("free-block budget"));
        assert!(tx.packets().is_empty());
        assert!(f.gpu.trace().is_empty());
        assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
        f.model.kv_cache.lock().free_blocks(&taken);
        let cap = f.model.glm_paired_execution().unwrap();
        if verify {
            cap.verify(&mut f.seqs[0], &history[0].issued).unwrap();
        } else {
            cap.propose(&mut f.seqs[0], 7, 15, 4, None).unwrap();
        }
    }
}
