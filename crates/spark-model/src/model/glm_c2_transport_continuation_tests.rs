// SPDX-License-Identifier: AGPL-3.0-only
//! Serial actual wire replay compared with the real direct-call Gate 2 control.
use super::{fixture::*, transport_test_fixture as wire, verdict_continuation_tests as flow};
use crate::traits::Model;

fn round(
    head: &mut Fixture,
    worker: &mut Fixture,
    control: &mut Fixture,
    tx: &wire::Wire,
    rx: &wire::Wire,
    history: &mut [flow::History; 2],
    accepted: [usize; 2],
    order: [usize; 2],
) {
    for owner in order {
        tx.clear();
        let h = &history[owner];
        let a = accepted[owner];
        let predictions = head
            .model
            .glm_paired_execution()
            .unwrap()
            .verify(&mut head.seqs[owner], &h.issued)
            .unwrap();
        assert_eq!(predictions.len(), 5);
        head.model.ep_broadcast_u32(a as u32).unwrap();
        head.seqs[owner].seq_len = h.base + a + 1;
        head.seqs[owner].tokens.truncate(h.base + a + 1);
        head.model
            .record_glm_mtp_verified(&mut head.seqs[owner], h.base, &h.issued, a)
            .unwrap();
        flow::acknowledge(head, owner, a, false);
        let packets = tx.packets();
        assert_eq!(
            packets,
            vec![
                vec![owner as u32],
                vec![0xfffffff5],
                vec![5],
                h.issued.clone(),
                vec![a as u32]
            ]
        );
        rx.queue(&packets);
        assert!(wire::worker(worker).unwrap());
        rx.done();
        flow::head_verdict(control, owner, h, a);
        flow::acknowledge(control, owner, a, false);
        wire::same_private(head, worker, owner);
        wire::same_private(head, control, owner);
    }
    for owner in order.into_iter().rev() {
        // The direct control's existing byte oracle checks all canonical and
        // speculative rows, plus shifted repair-token/EH pairs. Reuse it.
        flow::continue_owner(control, owner, &mut history[owner], accepted[owner]);
        let h = &history[owner];
        tx.clear();
        head.gpu.clear();
        worker.gpu.clear();
        let drafts = head
            .model
            .glm_paired_execution()
            .unwrap()
            .propose(&mut head.seqs[owner], h.issued[0], h.base, 4, None)
            .unwrap();
        assert_eq!(drafts, h.issued[1..]);
        let packets = tx.packets();
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[2].len(), 8);
        assert_eq!(&packets[2][5..], &[h.base as u32, 4, h.issued[0]]);
        rx.queue(&packets);
        assert!(wire::worker(worker).unwrap());
        rx.done();
        wire::same_private(head, worker, owner);
        wire::same_private(head, control, owner);
        assert_eq!(head.gpu.eh_pairs(), control.gpu.eh_pairs());
        assert_eq!(worker.gpu.eh_pairs(), control.gpu.eh_pairs());
    }
}

#[test]
fn all25_actual_head_worker_transport_pairs_and_continuation() {
    if flow::isolated(
        "transport_continuation_tests::all25_actual_head_worker_transport_pairs_and_continuation",
    ) {
        return;
    }
    for order in [[0, 1], [1, 0]] {
        for a in 0..5 {
            for b in 0..5 {
                let (mut head, _) = flow::prepare(0, order);
                let (mut worker, _) = flow::prepare(1, order);
                let (mut control, mut history) = flow::prepare(0, order);
                let tx = wire::Wire::install(&mut head, 0);
                let rx = wire::Wire::install(&mut worker, 1);
                round(
                    &mut head,
                    &mut worker,
                    &mut control,
                    &tx,
                    &rx,
                    &mut history,
                    [a, b],
                    order,
                );
            }
        }
    }
}

#[test]
fn actual_transport_repeated_unequal_histories_keep_attempts_and_owners_separate() {
    if flow::isolated(
        "transport_continuation_tests::actual_transport_repeated_unequal_histories_keep_attempts_and_owners_separate",
    ) {
        return;
    }
    for first in [[0, 1], [1, 0]] {
        let (mut head, _) = flow::prepare(0, first);
        let (mut worker, _) = flow::prepare(1, first);
        let (mut control, mut history) = flow::prepare(0, first);
        let tx = wire::Wire::install(&mut head, 0);
        let rx = wire::Wire::install(&mut worker, 1);
        for (i, accepted) in [[0, 4], [4, 0], [1, 3], [3, 1], [4, 4], [0, 0]]
            .into_iter()
            .enumerate()
        {
            let order = if i % 2 == 0 {
                first
            } else {
                [first[1], first[0]]
            };
            round(
                &mut head,
                &mut worker,
                &mut control,
                &tx,
                &rx,
                &mut history,
                accepted,
                order,
            );
            assert_eq!(&tx.packets()[2][3..5], &[(i + 2) as u32, 0]);
        }
    }
}
