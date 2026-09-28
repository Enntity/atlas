// SPDX-License-Identifier: AGPL-3.0-only
//! Actual paired capability and owned fixture; no certificate or GPU numerics.
use super::{fixture::*, isolated};
use crate::model::glm_c2_test_support::wire::Wire;
use crate::speculative::glm_paired_execution::GlmPairedExecution;
use crate::traits::Model;
use std::sync::{Arc, atomic::Ordering};

fn setup(rank: usize) -> (Fixture, Arc<Wire>) {
    let mut f = Fixture::new(rank);
    let wire = Wire::install(&mut f, rank);
    f.gpu.clear();
    (f, wire)
}
fn retained(f: &Fixture, before: &[u8]) {
    assert_eq!(f.gpu.read_span(f.gpu.slab(), SLAB_BYTES), before);
    for (slot, seq) in f.seqs.iter().enumerate() {
        let private = seq
            .proposer_state
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<crate::layers::glm5_mtp::Glm5MtpProposerState>()
            .unwrap();
        assert_eq!(private.seq_len, 0);
        assert_eq!(private.block_table.len(), 128);
        assert_eq!(seq.slot_idx, slot);
        assert_eq!(seq.ssm_slot_idx(), Some(slot));
    }
    assert_eq!(f.gpu.sweeps.load(Ordering::Relaxed), 0);
}

#[test]
fn actual_health_and_distinct_or_aliased_joins() {
    if isolated("quiescence_tests::actual_health_and_distinct_or_aliased_joins") {
        return;
    }
    for rank in 0..2 {
        for alias in [false, true] {
            let (mut f, wire) = setup(rank);
            if alias {
                f.model.secondary_stream = DEFAULT;
            }
            let secondary = f.model.secondary_stream;
            let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
            let cap = f.model.glm_paired_execution().unwrap();
            cap.check_communication_health().unwrap();
            assert_eq!(f.gpu.trace(), [Event::Health(true)]);
            f.gpu.clear();
            cap.quiesce().unwrap();
            let mut expected = vec![
                Event::Health(true),
                Event::Sync(DEFAULT),
                Event::Health(true),
            ];
            if !alias {
                expected.extend([Event::Sync(secondary), Event::Health(true)]);
            }
            assert_eq!(f.gpu.trace(), expected);
            assert!(wire.packets().is_empty());
            retained(&f, &before);
        }
    }
}

#[test]
fn actual_health_or_join_failure_stops_and_sticks() {
    if isolated("quiescence_tests::actual_health_or_join_failure_stops_and_sticks") {
        return;
    }
    for rank in 0..2 {
        for fault in 0..5 {
            let (f, wire) = setup(rank);
            let secondary = f.model.secondary_stream;
            let before = f.gpu.read_span(f.gpu.slab(), SLAB_BYTES);
            let expected = match fault {
                0 => {
                    f.gpu.unhealthy.store(true, Ordering::Relaxed);
                    vec![Event::Health(false)]
                }
                1 => {
                    f.gpu.fail.store(2, Ordering::Relaxed);
                    vec![Event::Health(true), Event::Sync(DEFAULT)]
                }
                2 => {
                    f.gpu
                        .unhealthy_on_sync
                        .store(DEFAULT as usize, Ordering::Relaxed);
                    vec![
                        Event::Health(true),
                        Event::Sync(DEFAULT),
                        Event::Health(false),
                    ]
                }
                3 => {
                    f.gpu.fail.store(4, Ordering::Relaxed);
                    vec![
                        Event::Health(true),
                        Event::Sync(DEFAULT),
                        Event::Health(true),
                        Event::Sync(secondary),
                    ]
                }
                _ => {
                    f.gpu
                        .unhealthy_on_sync
                        .store(secondary as usize, Ordering::Relaxed);
                    vec![
                        Event::Health(true),
                        Event::Sync(DEFAULT),
                        Event::Health(true),
                        Event::Sync(secondary),
                        Event::Health(false),
                    ]
                }
            };
            let cap = f.model.glm_paired_execution().unwrap();
            assert!(cap.quiesce().is_err());
            assert_eq!(f.gpu.trace(), expected);
            assert!(wire.packets().is_empty());
            retained(&f, &before);
            f.gpu.clear();
            f.gpu.unhealthy.store(false, Ordering::Relaxed);
            f.gpu.unhealthy_on_sync.store(0, Ordering::Relaxed);
            // A later good completion cannot turn this failed session into a receipt.
            f.model.synchronize(DEFAULT).unwrap();
            f.gpu.clear();
            assert!(cap.check_communication_health().is_err());
            assert!(cap.quiesce().is_err());
            assert!(f.gpu.trace().is_empty());
            retained(&f, &before);
        }
    }
}

#[test]
fn actual_health_only_failure_never_joins() {
    if isolated("quiescence_tests::actual_health_only_failure_never_joins") {
        return;
    }
    for rank in 0..2 {
        let (f, wire) = setup(rank);
        let cap = f.model.glm_paired_execution().unwrap();
        f.gpu.unhealthy.store(true, Ordering::Relaxed);
        assert!(cap.check_communication_health().is_err());
        assert_eq!(f.gpu.trace(), [Event::Health(false)]);
        f.gpu.unhealthy.store(false, Ordering::Relaxed);
        f.gpu.clear();
        assert!(cap.check_communication_health().is_err());
        assert!(cap.quiesce().is_err());
        assert!(f.gpu.trace().is_empty());
        assert!(wire.packets().is_empty());
    }
}

#[test]
fn actual_nonselected_capture_failed_closed_and_foreign_refusals() {
    if isolated("quiescence_tests::actual_nonselected_capture_failed_closed_and_foreign_refusals") {
        return;
    }
    let legacy = Fixture::new_legacy(0);
    assert!(legacy.model.glm_paired_execution().is_none());
    assert!(GlmPairedExecution::quiesce(&legacy.model).is_err());
    assert!(legacy.gpu.trace().is_empty());
    legacy.model.synchronize(DEFAULT).unwrap();
    assert_eq!(legacy.gpu.trace(), [Event::Sync(DEFAULT)]);
    for rank in 0..2 {
        for fault in 0..6 {
            let (mut f, wire) = setup(rank);
            match fault {
                0 => f.gpu.capturing.store(true, Ordering::Relaxed),
                1 => {
                    f.model
                        .paired_handoff()
                        .unwrap()
                        .fail_session(f.model.gpu.as_ref())
                        .unwrap();
                }
                2 => {
                    f.model
                        .paired_handoff()
                        .unwrap()
                        .close(f.model.gpu.as_ref(), f.model.secondary_stream)
                        .unwrap();
                }
                3 => f.model.comm = None,
                4 => f.model.config.ep_rank = 1 - rank,
                _ => f.model.ep_protocol_v2 = false,
            }
            f.gpu.clear();
            assert!(GlmPairedExecution::quiesce(&f.model).is_err());
            assert!(f.gpu.trace().is_empty());
            assert!(wire.packets().is_empty());
        }
    }
}
