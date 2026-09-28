// SPDX-License-Identifier: AGPL-3.0-only
//! The actual F1 worker must not drop local sequence owners across an Err.
use super::{fixture::*, transport_test_fixture as wire, verdict_continuation_tests as flow};
use crate::traits::{Model, SequenceState};
use std::sync::atomic::Ordering;

fn slots(f: &mut Fixture) -> [Option<SequenceState>; 2] {
    std::mem::replace(&mut f.seqs, std::array::from_fn(SequenceState::host_only)).map(Some)
}

#[test]
fn actual_f1_retire_and_replacement_failure_keep_caller_slot_and_quarantine_guard() {
    if flow::isolated(
        "f1_ownership_tests::actual_f1_retire_and_replacement_failure_keep_caller_slot_and_quarantine_guard",
    ) {
        return;
    }
    for owner in 0..2 {
        let mut control = Fixture::new(1);
        let rx = wire::Wire::install(&mut control, 1);
        let packets = vec![vec![owner as u32], vec![0xfffffff1]];
        rx.queue(&packets);
        let mut owned = slots(&mut control);
        control.gpu.clear();
        assert!(control.model.ep_worker_step(&mut owned).unwrap());
        rx.done();
        assert_eq!(owned[owner].as_ref().unwrap().slot_idx, owner);
        let trace = control.gpu.trace();
        let syncs: Vec<_> = trace
            .iter()
            .enumerate()
            .skip(4)
            .filter_map(|(i, event)| matches!(event, Event::Sync(DEFAULT)).then_some(i + 1))
            .collect();
        assert!(
            syncs.len() >= 4,
            "free completion plus actual allocation completion controls"
        );
        let boundaries = [syncs[0], syncs[syncs.len() - 2], syncs[syncs.len() - 1]];
        for ordinal in boundaries {
            let mut f = Fixture::new(1);
            let rx = wire::Wire::install(&mut f, 1);
            rx.queue(&packets);
            let mut owned = slots(&mut f);
            let peer_blocks = flow::private(owned[1 - owner].as_ref().unwrap())
                .block_table
                .clone();
            let before_slab = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
            f.gpu.clear();
            f.gpu.fail.store(ordinal, Ordering::Relaxed);
            let error = f.model.ep_worker_step(&mut owned).unwrap_err();
            assert!(format!("{error:#}").contains("injected fixture operation failure"));
            assert_eq!(f.gpu.trace()[ordinal - 1], Event::Sync(DEFAULT));
            assert_eq!(
                f.gpu.trace().len(),
                ordinal,
                "no cleanup follows selected F1 failure"
            );
            assert!(
                owned[owner].is_some(),
                "old sequence must stay in its caller-owned slot on Err"
            );
            let old = owned[owner].as_ref().unwrap();
            assert_eq!(old.slot_idx, owner);
            assert!(old.ssm_slot.as_ref().and_then(|g| g.idx()).is_none());
            assert_eq!(
                flow::private(owned[1 - owner].as_ref().unwrap()).block_table,
                peer_blocks
            );
            assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before_slab);
            drop(owned); // The failed sequence's later Drop must not recycle its target slot.
            assert!(!f.model.ssm_pool.slot_is_free(owner));
            assert!(!f.model.ssm_pool.claim_specific(owner));
            f.model.gpu.synchronize(DEFAULT).unwrap();
            f.gpu.clear();
            assert!(f.model.alloc_sequence().is_err());
            assert!(f.gpu.trace().is_empty());
        }
    }
}

#[test]
fn actual_f1_expected_slot_mismatch_refuses_before_initialization_or_replacement() {
    if flow::isolated(
        "f1_ownership_tests::actual_f1_expected_slot_mismatch_refuses_before_initialization_or_replacement",
    ) {
        return;
    }
    let mut f = Fixture::new(1);
    // Both actual private slots become free (candidate0); real target LIFO
    // order is made0 next. F1 addresses1 and must not build slot0 first.
    for owner in [1, 0] {
        f.model.free_sequence(&mut f.seqs[owner]).unwrap();
    }
    let rx = wire::Wire::install(&mut f, 1);
    let mut owned = slots(&mut f);
    rx.queue(&[vec![1], vec![0xfffffff1]]);
    f.gpu.clear();
    let error = f.model.ep_worker_step(&mut owned).unwrap_err();
    assert!(format!("{error:#}").contains("slot"));
    rx.done();
    assert!(
        owned[1].is_some(),
        "retired addressed owner is not replaced on mismatch"
    );
    assert_eq!(owned[1].as_ref().unwrap().slot_idx, 1);
    assert!(
        owned[1]
            .as_ref()
            .unwrap()
            .ssm_slot
            .as_ref()
            .and_then(|g| g.idx())
            .is_none()
    );
    assert_eq!(
        f.gpu.trace(),
        vec![
            Event::Sync(DEFAULT),
            Event::Read(f.model.ep_cmd_buf, 4, DEFAULT),
            Event::Sync(DEFAULT),
            Event::Read(f.model.ep_cmd_buf, 4, DEFAULT),
        ]
    );
    assert!(
        !f.model.ssm_pool.slot_is_free(0),
        "mismatched actual claim is quarantined"
    );
    drop(owned);
    f.gpu.clear();
    assert!(f.model.alloc_sequence().is_err());
    assert!(f.gpu.trace().is_empty());
}
