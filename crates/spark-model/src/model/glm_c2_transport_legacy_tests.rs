// SPDX-License-Identifier: AGPL-3.0-only
//! Real unpaired C1 construction and actual command adapters, not NCCL numerics.
use super::{fixture::*, transport_test_fixture as wire, verdict_continuation_tests as flow};
use crate::traits::Model;
use std::sync::atomic::Ordering;

fn prepared(rank: usize) -> Fixture {
    let mut f = Fixture::new_legacy(rank);
    assert!(f.model.glm_paired_execution().is_none());
    assert_eq!(f.model.levers.max_decode_seqs, 1);
    assert!(f.seqs[1].proposer_state.is_none());
    f.gpu.deterministic_logits.store(true, Ordering::Relaxed);
    f.model
        .prefill(&[1, 2, 3, 4], &mut f.seqs[0], CALLER)
        .unwrap();
    assert_eq!(f.seqs[0].seq_len, 4);
    f.gpu.clear();
    f
}

fn header(v2: bool, command: u32) -> Vec<Vec<u32>> {
    let mut packets = Vec::new();
    if v2 {
        packets.push(vec![0]);
    }
    packets.push(vec![command]);
    packets
}

fn receive_prefix(f: &Fixture, words: usize, payload_bytes: usize) -> Vec<Event> {
    let mut events = Vec::new();
    for _ in 0..words {
        events.push(Event::Sync(DEFAULT));
        events.push(Event::Read(f.model.ep_cmd_buf, 4, DEFAULT));
    }
    events.push(Event::Sync(DEFAULT));
    events.push(Event::Read(
        f.model.buffers.scratch(),
        payload_bytes,
        DEFAULT,
    ));
    events
}

#[test]
fn actual_legacy_c1_e1_four_words_global_hidden_and_receive_order() {
    if flow::isolated(
        "transport_legacy_tests::actual_legacy_c1_e1_four_words_global_hidden_and_receive_order",
    ) {
        return;
    }
    for v2 in [false, true] {
        let mut head = prepared(0);
        let mut worker = prepared(1);
        // Produce a real five-row verifier output before selecting row 2.
        // Plain prefill only final-normalizes its last token into row 0.
        for f in [&mut head, &mut worker] {
            f.model.save_hidden_for_mtp(0, CALLER).unwrap();
            let mut issued = vec![7];
            issued.extend(
                f.model
                    .run_mtp_propose_inner(7, 4, 4, &mut f.seqs[0], None)
                    .unwrap(),
            );
            f.model
                .decode_verify_dflash(&issued, &mut f.seqs[0], CALLER)
                .unwrap();
            f.seqs[0].seq_len -= 2;
            let committed = f.seqs[0].seq_len;
            f.seqs[0].tokens.truncate(committed);
            f.model
                .trim_proposer_state(&mut f.seqs[0], 2, CALLER)
                .unwrap();
            f.model
                .commit_accepted_prefix(&mut f.seqs[0], 3, 5)
                .unwrap();
            f.gpu.clear();
        }
        let position = head.seqs[0].seq_len;
        assert_eq!(position, 7);
        let prefix_tokens = head.seqs[0].tokens.clone();
        assert_eq!(worker.seqs[0].tokens, prefix_tokens);
        let tx = wire::Wire::install(&mut head, 0);
        let rx = wire::Wire::install(&mut worker, 1);
        head.model.ep_protocol_v2 = v2;
        worker.model.ep_protocol_v2 = v2;
        let row = 2;
        let source = head.model.buffers.norm_output().offset(row * ROW_BYTES);
        let hidden = head.gpu.read_span(source, ROW_BYTES);
        assert_eq!(
            worker.gpu.read_span(
                worker.model.buffers.norm_output().offset(row * ROW_BYTES),
                ROW_BYTES,
            ),
            hidden
        );
        let before = flow::private(&head.seqs[0]).seq_len;
        assert_eq!(flow::private(&worker.seqs[0]).seq_len, before);
        head.model.save_hidden_for_mtp(row, CALLER).unwrap();
        let drafts = head
            .model
            .run_mtp_propose_multi(7, position, 4, &mut head.seqs[0], CALLER, None)
            .unwrap();
        assert_eq!(drafts.len(), 4);
        let mut expected = header(v2, 0xffffffe1);
        expected.push(vec![7, position as u32, 4, row as u32]);
        assert_eq!(tx.packets(), expected);
        let mut head_prefix = vec![Event::Copy(
            source,
            head.model.mtp_hidden_save,
            ROW_BYTES,
            DEFAULT,
        )];
        head_prefix.extend(
            (0..1 + usize::from(v2)).map(|_| Event::Upload(head.model.ep_cmd_buf, 4, DEFAULT)),
        );
        head_prefix.push(Event::Upload(head.model.buffers.scratch(), 16, DEFAULT));
        assert!(head.gpu.trace().starts_with(&head_prefix));

        rx.queue(&tx.packets());
        assert!(wire::worker(&mut worker).unwrap());
        rx.done();
        assert_eq!(rx.packets(), expected);
        let mut worker_prefix = receive_prefix(&worker, 1 + usize::from(v2), 16);
        worker_prefix.push(Event::Copy(
            worker.model.buffers.norm_output().offset(row * ROW_BYTES),
            worker.model.mtp_hidden_save,
            ROW_BYTES,
            DEFAULT,
        ));
        assert!(worker.gpu.trace().starts_with(&worker_prefix));
        for f in [&head, &worker] {
            assert_eq!(f.gpu.read_span(f.model.mtp_hidden_save, ROW_BYTES), hidden);
            assert_eq!(f.gpu.eh_pairs().first(), Some(&(7, hidden[0])));
            assert_eq!(flow::private(&f.seqs[0]).seq_len, before + 4);
            assert_eq!(flow::private(&f.seqs[0]).last_num_drafted, 4);
            assert_eq!(f.seqs[0].tokens, prefix_tokens);
            assert_eq!(
                f.gpu
                    .trace()
                    .iter()
                    .filter(|e| matches!(e, Event::Body(_, _)))
                    .count(),
                4
            );
        }
        assert_eq!(head.gpu.eh_pairs(), worker.gpu.eh_pairs());
    }
}

#[test]
fn actual_legacy_f5_width_tokens_acceptance_and_worker_order() {
    if flow::isolated(
        "transport_legacy_tests::actual_legacy_f5_width_tokens_acceptance_and_worker_order",
    ) {
        return;
    }
    for v2 in [false, true] {
        for accepted in 0..5 {
            let mut head = prepared(0);
            let mut worker = prepared(1);
            // Both actual legacy proposers establish the private cursor before F5.
            for f in [&mut head, &mut worker] {
                f.model.save_hidden_for_mtp(2, CALLER).unwrap();
                f.model
                    .run_mtp_propose_inner(7, 4, 4, &mut f.seqs[0], None)
                    .unwrap();
                f.gpu.clear();
            }
            let private_before = flow::private(&worker.seqs[0]).seq_len;
            let tx = wire::Wire::install(&mut head, 0);
            let rx = wire::Wire::install(&mut worker, 1);
            head.model.ep_protocol_v2 = v2;
            worker.model.ep_protocol_v2 = v2;
            let tokens = [7, 6, 5, 4, 3];
            // Generic legacy F5 has no combined transport API: these are the
            // actual Model operations used by the existing scheduler, in order.
            head.model.sync_secondary().unwrap();
            head.model.ep_broadcast_cmd_for_seq(0, 0xfffffff5).unwrap();
            head.model.ep_broadcast_cmd(5).unwrap();
            head.model.ep_broadcast_tokens(&tokens).unwrap();
            assert_eq!(
                head.model
                    .decode_verify_dflash(&tokens, &mut head.seqs[0], CALLER)
                    .unwrap()
                    .len(),
                5
            );
            head.model.ep_broadcast_cmd(accepted as u32).unwrap();
            let mut expected = header(v2, 0xfffffff5);
            expected.extend([vec![5], tokens.to_vec(), vec![accepted as u32]]);
            assert_eq!(tx.packets(), expected);

            rx.queue(&tx.packets());
            assert!(wire::worker(&mut worker).unwrap());
            rx.done();
            assert_eq!(rx.packets(), expected);
            let mut prefix = receive_prefix(&worker, 2 + usize::from(v2), 20);
            prefix.push(Event::WaitEvent(DEFAULT, worker.model.secondary_event));
            let events = worker.gpu.trace();
            assert!(events.starts_with(&prefix));
            let targets: Vec<_> = events
                .iter()
                .filter_map(|e| match e {
                    Event::Target(n, pos, stream) => Some((*n, *pos, *stream)),
                    _ => None,
                })
                .collect();
            assert_eq!(targets, (4..9).map(|p| (1, p, DEFAULT)).collect::<Vec<_>>());
            let target_end = events
                .iter()
                .rposition(|e| matches!(e, Event::Target(_, _, _)))
                .unwrap();
            let accepted_read = events
                .iter()
                .rposition(|e| *e == Event::Read(worker.model.ep_cmd_buf, 4, DEFAULT))
                .unwrap();
            assert!(target_end < accepted_read);
            assert_eq!(events[accepted_read - 1], Event::Sync(DEFAULT));
            assert_eq!(worker.seqs[0].seq_len, 4 + accepted + 1);
            let mut committed = vec![1, 2, 3, 4];
            committed.extend_from_slice(&tokens[..accepted + 1]);
            assert_eq!(worker.seqs[0].tokens, committed);
            assert_eq!(
                flow::private(&worker.seqs[0]).seq_len,
                private_before - (4 - accepted)
            );
            assert!(worker.model.glm_paired_execution().is_none());
        }
    }
}

#[test]
fn actual_unpaired_generic_c2_e1_still_refuses_before_payload_or_body() {
    if flow::isolated(
        "transport_legacy_tests::actual_unpaired_generic_c2_e1_still_refuses_before_payload_or_body",
    ) {
        return;
    }
    let mut head = prepared(0);
    let mut worker = prepared(1);
    let tx = wire::Wire::install(&mut head, 0);
    let rx = wire::Wire::install(&mut worker, 1);
    head.model.levers.max_decode_seqs = 2;
    worker.model.levers.max_decode_seqs = 2;
    let before = flow::private(&head.seqs[0]).seq_len;
    let error = head
        .model
        .run_mtp_propose_multi(7, 4, 4, &mut head.seqs[0], CALLER, None)
        .unwrap_err();
    assert!(error.to_string().contains("max_batch_size=1"));
    assert!(tx.packets().is_empty());
    assert!(head.gpu.trace().is_empty());
    // Only the real preamble is supplied: any unexpected payload receive
    // panics in Wire, so refusal cannot be satisfied by a later parse error.
    let expected = header(true, 0xffffffe1);
    rx.queue(&expected);
    let error = wire::worker(&mut worker).unwrap_err();
    assert!(error.to_string().contains("max_batch_size=1"));
    rx.done();
    assert_eq!(rx.packets(), expected);
    assert_eq!(
        worker.gpu.trace(),
        vec![
            Event::Sync(DEFAULT),
            Event::Read(worker.model.ep_cmd_buf, 4, DEFAULT),
            Event::Sync(DEFAULT),
            Event::Read(worker.model.ep_cmd_buf, 4, DEFAULT),
        ]
    );
    assert_eq!(flow::private(&head.seqs[0]).seq_len, before);
    assert_eq!(flow::private(&worker.seqs[0]).seq_len, before);
    assert!(head.model.glm_paired_execution().is_none());
    assert!(worker.model.glm_paired_execution().is_none());
}
