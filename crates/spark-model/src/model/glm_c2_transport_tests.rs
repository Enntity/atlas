// SPDX-License-Identifier: AGPL-3.0-only
//! Actual optional capability and worker transport, with no scheduler activation.
use super::{fixture::*, transport_test_fixture as wire, verdict_continuation_tests as flow};
use crate::traits::Model;

#[test]
fn actual_selected_first_e1_head_and_worker_use_owned_packet() {
    if flow::isolated("transport_tests::actual_selected_first_e1_head_and_worker_use_owned_packet")
    {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        let mut head = wire::bootstrapped(0, order);
        let mut worker = wire::bootstrapped(1, order);
        let tx = wire::Wire::install(&mut head, 0);
        let rx = wire::Wire::install(&mut worker, 1);
        for owner in order {
            tx.clear();
            let position = head.seqs[owner].seq_len;
            let seed = 7 - owner as u32;
            let drafts = head
                .model
                .glm_paired_execution()
                .unwrap()
                .propose(&mut head.seqs[owner], seed, position, 4, None)
                .expect("actual selected E1 transport must execute");
            assert_eq!(drafts.len(), 4);
            let packets = tx.packets();
            assert_eq!(
                packets,
                vec![
                    vec![owner as u32],
                    vec![0xffffffe1],
                    vec![1, 1, 0, 1, 0, position as u32, 4, seed]
                ]
            );
            rx.queue(&packets);
            assert!(wire::worker(&mut worker).unwrap());
            rx.done();
            wire::same_private(&head, &worker, owner);
        }
    }
}

#[test]
fn actual_selected_f5_head_emits_existing_payload_then_real_verify() {
    if flow::isolated(
        "transport_tests::actual_selected_f5_head_emits_existing_payload_then_real_verify",
    ) {
        return;
    }
    for owner in 0..2 {
        let (mut head, history) = flow::prepare(0, [1, 0]);
        let (mut worker, _) = flow::prepare(1, [1, 0]);
        let tx = wire::Wire::install(&mut head, 0);
        let rx = wire::Wire::install(&mut worker, 1);
        let issued = &history[owner].issued;
        let verified = head
            .model
            .glm_paired_execution()
            .unwrap()
            .verify(&mut head.seqs[owner], issued)
            .expect("actual selected F5 transport must execute");
        assert_eq!(verified.len(), 5);
        assert_eq!(
            tx.packets(),
            vec![
                vec![owner as u32],
                vec![0xfffffff5],
                vec![5],
                issued.clone()
            ]
        );
        let accepted = 4;
        head.model.ep_broadcast_u32(accepted).unwrap();
        head.model
            .record_glm_mtp_verified_impl(
                &mut head.seqs[owner],
                history[owner].base,
                issued,
                accepted as usize,
            )
            .unwrap();
        flow::acknowledge(&mut head, owner, accepted as usize, false);
        rx.queue(&tx.packets());
        assert!(wire::worker(&mut worker).unwrap());
        rx.done();
        wire::same_private(&head, &worker, owner);
    }
}

#[test]
fn actual_selected_worker_accepts_versioned_first_e1() {
    if flow::isolated("transport_tests::actual_selected_worker_accepts_versioned_first_e1") {
        return;
    }
    let mut f = wire::bootstrapped(1, [1, 0]);
    let rx = wire::Wire::install(&mut f, 1);
    let position = f.seqs[0].seq_len;
    rx.queue(&[
        vec![0],
        vec![0xffffffe1],
        vec![1, 1, 0, 1, 0, position as u32, 4, 7],
    ]);
    assert!(wire::worker(&mut f).expect("actual selected worker E1 must execute"));
    rx.done();
    assert_eq!(flow::private(&f.seqs[0]).seq_len, position + 3);
    assert!(f.gpu.trace().iter().any(|e| matches!(e, Event::Body(_, _))));
}
