// SPDX-License-Identifier: AGPL-3.0-only
//! Issued transport failure is terminal even before a verification exists.
use super::{
    fixture::*, transport_boundary_tests::terminal, transport_test_fixture as wire,
    verdict_continuation_tests as flow,
};
use crate::traits::Model;
use std::sync::atomic::Ordering;

fn ready(rank: usize, verify: bool) -> (Fixture, Vec<Vec<u32>>) {
    if verify {
        let (f, h) = flow::prepare(rank, [1, 0]);
        (f, h.into_iter().map(|h| h.issued).collect())
    } else {
        (wire::bootstrapped(rank, [1, 0]), vec![vec![], vec![]])
    }
}

fn execute(
    f: &mut Fixture,
    owner: usize,
    verify: bool,
    tokens: &[u32],
) -> anyhow::Result<Vec<u32>> {
    let position = f.seqs[owner].seq_len;
    let cap = f.model.glm_paired_execution().unwrap();
    if verify {
        cap.verify(&mut f.seqs[owner], tokens)
    } else {
        cap.propose(&mut f.seqs[owner], 7, position, 4, None)
    }
}

#[test]
fn every_head_header_and_payload_broadcast_failure_latches_before_or_after_claim() {
    if flow::isolated(
        "transport_fault_tests::every_head_header_and_payload_broadcast_failure_latches_before_or_after_claim",
    ) {
        return;
    }
    for verify in [false, true] {
        for owner in 0..2 {
            let (mut control, issued) = ready(0, verify);
            let tx = wire::Wire::install(&mut control, 0);
            execute(&mut control, owner, verify, &issued[owner]).unwrap();
            let expected = tx.packets();
            assert_eq!(expected.len(), if verify { 4 } else { 3 });
            for ordinal in 1..=expected.len() {
                let (mut f, issued) = ready(0, verify);
                let tx = wire::Wire::install(&mut f, 0);
                let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
                let positions = f.seqs.each_ref().map(|s| s.seq_len);
                tx.fail.store(ordinal, Ordering::Relaxed);
                f.gpu.clear();
                let error = execute(&mut f, owner, verify, &issued[owner]).unwrap_err();
                assert!(format!("{error:#}").contains("injected command transfer failure"));
                assert_eq!(tx.packets(), expected[..ordinal]);
                assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
                assert_eq!(f.seqs.each_ref().map(|s| s.seq_len), positions);
                assert!(!f.gpu.trace().iter().any(|e| matches!(
                    e,
                    Event::Target(_, _, _) | Event::Body(_, _) | Event::Kv(_, _)
                )));
                terminal(&mut f);
            }
        }
    }
}

#[test]
fn actual_head_command_upload_and_first_local_writer_faults_are_terminal() {
    if flow::isolated(
        "transport_fault_tests::actual_head_command_upload_and_first_local_writer_faults_are_terminal",
    ) {
        return;
    }
    for verify in [false, true] {
        for owner in 0..2 {
            let (mut control, issued) = ready(0, verify);
            wire::Wire::install(&mut control, 0);
            control.gpu.clear();
            execute(&mut control, owner, verify, &issued[owner]).unwrap();
            let trace = control.gpu.trace();
            let mut points: Vec<_> = trace
                .iter()
                .enumerate()
                .filter_map(|(i, event)| match event {
                    Event::Upload(p, n, stream)
                        if *p == control.model.ep_cmd_buf
                            || (*p == control.model.buffers.scratch()
                                && *n == if verify { 20 } else { 32 }) =>
                    {
                        Some((i + 1, Some((*p == control.model.ep_cmd_buf, *n, *stream))))
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(points.len(), if verify { 4 } else { 3 });
            let writer = trace
                .iter()
                .position(|e| {
                    if verify {
                        matches!(e, Event::Target(1, _, DEFAULT))
                    } else {
                        matches!(e, Event::Kv(_, DEFAULT))
                    }
                })
                .unwrap();
            points.push((writer + 1, None));
            for (ordinal, upload) in points {
                let (mut f, issued) = ready(0, verify);
                wire::Wire::install(&mut f, 0);
                let peer = flow::bytes(&f, 1 - owner, flow::private(&f.seqs[1 - owner]).seq_len);
                let peer_pointers = f
                    .head
                    .paired_test_kv_rows(
                        f.seqs[1 - owner].proposer_state.as_ref().unwrap().as_ref(),
                        f.model.gpu.as_ref(),
                        peer.len(),
                    )
                    .unwrap();
                let peer_slab: Vec<_> = (0..6).map(|r| flow::slab(&f, 1 - owner, r)).collect();
                let position = f.seqs[owner].seq_len;
                f.gpu.clear();
                f.gpu.fail.store(ordinal, Ordering::Relaxed);
                let error = execute(&mut f, owner, verify, &issued[owner]).unwrap_err();
                assert!(format!("{error:#}").contains("injected fixture operation failure"));
                let failed = &f.gpu.trace()[ordinal - 1];
                if let Some((command, n, stream)) = upload {
                    assert_eq!(
                        *failed,
                        Event::Upload(
                            if command {
                                f.model.ep_cmd_buf
                            } else {
                                f.model.buffers.scratch()
                            },
                            n,
                            stream
                        )
                    );
                } else if verify {
                    assert_eq!(*failed, Event::Target(1, position, DEFAULT));
                } else {
                    assert!(matches!(failed, Event::Kv(_, DEFAULT)));
                }
                // Global poisoning makes validated row getters unavailable; snapshot
                // raw peer rows before failure and compare their original owned pointers.
                for ((k, v), expected) in peer_pointers.into_iter().zip(peer) {
                    assert_eq!(f.gpu.read_span(k, 1024), expected);
                    assert_eq!(f.gpu.read_span(v, 1024), expected);
                }
                assert_eq!(
                    (0..6)
                        .map(|r| flow::slab(&f, 1 - owner, r))
                        .collect::<Vec<_>>(),
                    peer_slab
                );
                terminal(&mut f);
            }
        }
    }
}

#[test]
fn worker_payload_receive_failure_is_terminal_without_a_verification() {
    if flow::isolated(
        "transport_fault_tests::worker_payload_receive_failure_is_terminal_without_a_verification",
    ) {
        return;
    }
    for verify in [false, true] {
        for owner in 0..2 {
            let (mut head, issued) = ready(0, verify);
            let tx = wire::Wire::install(&mut head, 0);
            execute(&mut head, owner, verify, &issued[owner]).unwrap();
            let packets = tx.packets();
            for ordinal in 3..=packets.len() {
                let (mut f, _) = ready(1, verify);
                let rx = wire::Wire::install(&mut f, 1);
                let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
                rx.queue(&packets);
                rx.fail.store(ordinal, Ordering::Relaxed);
                f.gpu.clear();
                let error = wire::worker(&mut f).unwrap_err();
                assert!(format!("{error:#}").contains("injected command transfer failure"));
                assert_eq!(rx.packets(), packets[..ordinal]);
                assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
                assert!(!f.gpu.trace().iter().any(|e| matches!(
                    e,
                    Event::Target(_, _, _) | Event::Body(_, _) | Event::Kv(_, _)
                )));
                terminal(&mut f);
            }
        }
    }
}

#[test]
fn actual_worker_postheader_reads_completion_and_local_writer_faults_are_terminal() {
    if flow::isolated(
        "transport_fault_tests::actual_worker_postheader_reads_completion_and_local_writer_faults_are_terminal",
    ) {
        return;
    }
    for verify in [false, true] {
        for owner in 0..2 {
            let (mut head, issued) = ready(0, verify);
            let tx = wire::Wire::install(&mut head, 0);
            execute(&mut head, owner, verify, &issued[owner]).unwrap();
            let mut packets = tx.packets();
            if verify {
                packets.push(vec![4]);
            }
            let (mut control, _) = ready(1, verify);
            let rx = wire::Wire::install(&mut control, 1);
            rx.queue(&packets);
            control.gpu.clear();
            assert!(wire::worker(&mut control).unwrap());
            rx.done();
            let trace = control.gpu.trace();
            // First four events are the already-received slot/opcode. Their
            // failures require B2 process containment, not a guessed local owner.
            assert!(matches!(trace[3], Event::Read(_, 4, DEFAULT)));
            let first_writer = trace
                .iter()
                .position(|e| {
                    if verify {
                        matches!(e, Event::Target(1, _, DEFAULT))
                    } else {
                        matches!(e, Event::Kv(_, DEFAULT))
                    }
                })
                .unwrap();
            let mut points: Vec<_> = trace
                .iter()
                .enumerate()
                .skip(4)
                .take_while(|(i, _)| *i < first_writer)
                .filter_map(|(i, e)| match e {
                    Event::Sync(stream) => Some((i + 1, Some((false, 0, *stream)))),
                    Event::Read(p, n, stream) => {
                        Some((i + 1, Some((*p == control.model.ep_cmd_buf, *n, *stream))))
                    }
                    _ => None,
                })
                .collect();
            assert!(points.len() >= 2);
            points.push((first_writer + 1, None));
            for (ordinal, read) in points {
                let (mut f, _) = ready(1, verify);
                let rx = wire::Wire::install(&mut f, 1);
                rx.queue(&packets);
                f.gpu.clear();
                f.gpu.fail.store(ordinal, Ordering::Relaxed);
                let error = wire::worker(&mut f).unwrap_err();
                assert!(format!("{error:#}").contains("injected fixture operation failure"));
                let failed = &f.gpu.trace()[ordinal - 1];
                match read {
                    Some((_, 0, stream)) => assert_eq!(*failed, Event::Sync(stream)),
                    Some((command, n, stream)) => assert_eq!(
                        *failed,
                        Event::Read(
                            if command {
                                f.model.ep_cmd_buf
                            } else {
                                f.model.buffers.scratch()
                            },
                            n,
                            stream
                        )
                    ),
                    None if verify => assert!(matches!(failed, Event::Target(1, _, DEFAULT))),
                    None => assert!(matches!(failed, Event::Kv(_, DEFAULT))),
                }
                terminal(&mut f);
            }
        }
    }
}
