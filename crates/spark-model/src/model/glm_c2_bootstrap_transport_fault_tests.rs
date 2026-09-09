// SPDX-License-Identifier: AGPL-3.0-only
//! Actual issued scalar faults, distinct from preamble/T2 containment.
use super::{bootstrap_transport_tests as control, fixture::*, isolated};
use super::{transport_boundary_tests::terminal, transport_test_fixture as wire};
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug)]
enum Boundary {
    Target,
    Bonus,
    Completion,
}

fn unique(trace: &[Event], event: &Event) -> usize {
    let found: Vec<_> = trace
        .iter()
        .enumerate()
        .filter(|(_, e)| *e == event)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        found.len(),
        1,
        "{event:?} must execute exactly once: {trace:#?}"
    );
    found[0]
}

fn setup(rank: usize, owner: usize) -> (Fixture, std::sync::Arc<wire::Wire>) {
    let mut f = control::prepared(rank, [1, 0]);
    let wire = wire::Wire::install(&mut f, rank);
    if rank == 1 {
        wire.queue(&[vec![owner as u32], vec![5 + owner as u32]]);
    }
    f.gpu.clear();
    (f, wire)
}

fn ordinal(rank: usize, owner: usize, boundary: Boundary) -> usize {
    let (mut f, wire) = setup(rank, owner);
    control::execute(&mut f, owner, rank).unwrap();
    wire.done();
    assert_eq!(f.seqs[owner].seq_len, 5);
    assert_eq!(
        wire.packets(),
        vec![vec![owner as u32], vec![5 + owner as u32]]
    );
    let trace = f.gpu.trace();
    let target = unique(&trace, &Event::Target(1, 4, DEFAULT));
    let bonus = unique(&trace, &control::bonus(&f, owner));
    assert!(target < bonus);
    assert_eq!(trace.get(bonus + 1), Some(&Event::Sync(DEFAULT)));
    match boundary {
        Boundary::Target => target + 1,
        Boundary::Bonus => bonus + 1,
        Boundary::Completion => bonus + 2,
    }
}

#[test]
fn each_head_scalar_word_transfer_failure_is_session_terminal() {
    if isolated(
        "bootstrap_transport_fault_tests::each_head_scalar_word_transfer_failure_is_session_terminal",
    ) {
        return;
    }
    for owner in 0..2 {
        let (mut positive, tx) = setup(0, owner);
        control::execute(&mut positive, owner, 0).unwrap();
        let expected = tx.packets();
        assert_eq!(expected, vec![vec![owner as u32], vec![5 + owner as u32]]);
        for ordinal in 1..=2 {
            let (mut f, tx) = setup(0, owner);
            let before = [control::Snapshot::new(&f, 0), control::Snapshot::new(&f, 1)];
            tx.fail.store(ordinal, Ordering::Relaxed);
            let error = control::execute(&mut f, owner, 0).unwrap_err();
            assert!(format!("{error:#}").contains("injected command transfer failure"));
            assert_eq!(tx.packets(), expected[..ordinal]);
            for slot in 0..2 {
                before[slot].unchanged(&f, slot);
            }
            assert!(!f.gpu.trace().iter().any(|e| matches!(
                e,
                Event::Target(_, _, _) | Event::Body(_, _) | Event::Kv(_, _)
            )));
            tx.clear();
            f.gpu.clear();
            terminal(&mut f);
            assert!(tx.packets().is_empty());
        }
    }
}

#[test]
fn actual_scalar_target_bonus_and_completion_failures_latch_head_and_worker() {
    if isolated(
        "bootstrap_transport_fault_tests::actual_scalar_target_bonus_and_completion_failures_latch_head_and_worker",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            for boundary in [Boundary::Target, Boundary::Bonus, Boundary::Completion] {
                let ordinal = ordinal(rank, owner, boundary);
                let (mut f, wire) = setup(rank, owner);
                let peer = control::Snapshot::new(&f, 1 - owner);
                let expected = match boundary {
                    Boundary::Target => Event::Target(1, 4, DEFAULT),
                    Boundary::Bonus => control::bonus(&f, owner),
                    Boundary::Completion => Event::Sync(DEFAULT),
                };
                f.gpu.fail.store(ordinal, Ordering::Relaxed);
                let error = control::execute(&mut f, owner, rank).unwrap_err();
                assert!(
                    format!("{error:#}").contains("injected fixture operation failure"),
                    "rank{rank} owner{owner} {boundary:?}: {error:#}"
                );
                wire.done();
                let trace = f.gpu.trace();
                assert_eq!(trace.get(ordinal - 1), Some(&expected));
                if matches!(boundary, Boundary::Completion) {
                    assert_eq!(trace.get(ordinal - 2), Some(&control::bonus(&f, owner)));
                }
                assert_eq!(
                    trace.len(),
                    ordinal,
                    "no subsequent operation after {boundary:?}"
                );
                peer.unchanged(&f, 1 - owner);
                wire.clear();
                f.gpu.clear();
                terminal(&mut f);
                assert!(wire.packets().is_empty());
                peer.unchanged(&f, 1 - owner);
            }
        }
    }
}

#[test]
fn each_actual_head_command_upload_failure_stops_before_local_target() {
    if isolated(
        "bootstrap_transport_fault_tests::each_actual_head_command_upload_failure_stops_before_local_target",
    ) {
        return;
    }
    for owner in 0..2 {
        let (mut positive, tx) = setup(0, owner);
        control::execute(&mut positive, owner, 0).unwrap();
        assert_eq!(tx.packets().len(), 2);
        let points: Vec<_> = positive
            .gpu
            .trace()
            .iter()
            .enumerate()
            .filter_map(|(i, e)| match e {
                Event::Upload(p, 4, stream) if *p == positive.model.ep_cmd_buf => {
                    Some((i + 1, *stream))
                }
                _ => None,
            })
            .collect();
        assert_eq!(points.len(), 2);
        for (ordinal, stream) in points {
            let (mut f, tx) = setup(0, owner);
            let peer = control::Snapshot::new(&f, 1 - owner);
            f.gpu.fail.store(ordinal, Ordering::Relaxed);
            let error = control::execute(&mut f, owner, 0).unwrap_err();
            assert!(format!("{error:#}").contains("injected fixture operation failure"));
            let trace = f.gpu.trace();
            assert_eq!(
                trace.get(ordinal - 1),
                Some(&Event::Upload(f.model.ep_cmd_buf, 4, stream))
            );
            assert!(!trace.iter().any(|e| matches!(e, Event::Target(_, _, _))));
            peer.unchanged(&f, 1 - owner);
            tx.clear();
            f.gpu.clear();
            terminal(&mut f);
        }
    }
}
