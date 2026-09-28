// SPDX-License-Identifier: AGPL-3.0-only
//! Real slot pools/commit copies and event ordering; the recurrent math is stubbed.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::model::ssm_pool::SsmStatePool;
use crate::traits::Model;
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;
use std::sync::{Arc, atomic::Ordering};

#[path = "glm_c2_verdict_ssm_fixture.rs"]
mod ssm_fixture;
use ssm_fixture::{isolated, prepared};

#[test]
fn actual_k5_refuses_foreign_live_and_intermediate_ssm_bindings_before_work() {
    if isolated("actual_k5_refuses_foreign_live_and_intermediate_ssm_bindings_before_work") {
        return;
    }
    for rank in 0..2 {
        for mutation in 0..6 {
            let (mut f, history) = prepared(rank);
            let state = f.seqs[0].layer_states[1]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .unwrap();
            match mutation {
                0 => state.h_state = f.model.ssm_pool.h_state(0, 1),
                1 => state.conv_state = f.model.ssm_pool.conv_state(0, 1),
                2 => state.h_state_intermediates[0] = f.model.ssm_pool.h_intermediate(0, 1, 0),
                3 => {
                    state.conv_state_intermediates[4] = f.model.ssm_pool.conv_intermediate(0, 1, 4)
                }
                4 => {
                    state.h_state_intermediates.pop();
                }
                _ => {
                    state.conv_state_intermediates.pop();
                }
            }
            let result =
                f.model
                    .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER);
            assert!(
                result.is_err(),
                "actual state/pool pointer mismatch was accepted"
            );
            assert!(
                f.gpu.trace().is_empty(),
                "SSM ownership must fail before target work"
            );
        }
    }
}

#[test]
fn same_index_foreign_pool_guard_does_not_authorize_actual_target_owner() {
    if isolated("same_index_foreign_pool_guard_does_not_authorize_actual_target_owner") {
        return;
    }
    for after_record in [false, true] {
        let (mut f, history) = prepared(1);
        if after_record {
            flow::head_verdict(&mut f, 0, &history[0], 1);
        }
        let foreign = Arc::new(
            SsmStatePool::new(
                &f.model.config,
                2,
                true,
                5,
                4,
                false,
                crate::ssm_reserve::SsmRollbackMode::Snapshot,
                f.model.gpu.as_ref(),
            )
            .unwrap(),
        );
        let foreign_guard = foreign.claim_guarded().unwrap();
        assert_eq!(foreign_guard.idx(), Some(0));
        let actual = f.seqs[0].ssm_slot.replace(foreign_guard).unwrap();
        f.gpu.clear();
        let result = if after_record {
            f.model.commit_accepted_prefix(&mut f.seqs[0], 2, 5)
        } else {
            f.model
                .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
                .map(|_| ())
        };
        assert!(
            result.is_err(),
            "same numeric slot from a foreign pool authorized target work"
        );
        assert!(f.gpu.trace().is_empty());
        f.seqs[0].ssm_slot = Some(actual);
        if after_record {
            f.model
                .commit_accepted_prefix(&mut f.seqs[0], 2, 5)
                .unwrap();
        } else {
            f.model
                .decode_verify_graphed_kgamma(&history[0].issued, &mut f.seqs[0], CALLER)
                .unwrap();
        }
    }
}

#[test]
fn actual_k5_ssm_commit_bytes_and_wait_precede_next_private_consumer() {
    if isolated("actual_k5_ssm_commit_bytes_and_wait_precede_next_private_consumer") {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            for accepted in 0..5 {
                let (mut f, mut history) = prepared(rank);
                flow::head_verdict(&mut f, owner, &history[owner], accepted);
                flow::acknowledge(&mut f, owner, accepted, accepted % 2 == 0);
                let base = history[owner].base;
                assert_eq!(
                    f.gpu.read_span(f.model.ssm_pool.h_state(0, owner), 16),
                    vec![(base + accepted + 0x40) as u8; 16]
                );
                assert_eq!(
                    f.gpu.read_span(f.model.ssm_pool.conv_state(0, owner), 48),
                    vec![(base + accepted + 0x60) as u8; 48]
                );
                assert_eq!(
                    f.gpu
                        .read_span(f.model.ssm_pool.conv_intermediate(0, owner, 4), 48),
                    vec![0xd7; 48],
                    "fifth allocated conv snapshot is not written or read by full accept"
                );
                if accepted < 4 {
                    assert!(f.gpu.trace().contains(&Event::RecordEvent(
                        f.model.secondary_event,
                        f.model.secondary_stream
                    )));
                }
                f.gpu.clear();
                flow::continue_owner(&mut f, owner, &mut history[owner], accepted);
                assert_eq!(
                    f.gpu.trace().first(),
                    Some(&Event::WaitEvent(DEFAULT, f.model.secondary_event))
                );
            }
        }
    }
}

#[test]
fn post_verdict_ssm_destination_tamper_refuses_before_commit_and_restoration_is_valid() {
    if isolated(
        "post_verdict_ssm_destination_tamper_refuses_before_commit_and_restoration_is_valid",
    ) {
        return;
    }
    for rank in 0..2 {
        for conv in [false, true] {
            let (mut f, history) = prepared(rank);
            flow::head_verdict(&mut f, 0, &history[0], 2);
            let state = f.seqs[0].layer_states[1]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .unwrap();
            let original = if conv {
                state.conv_state
            } else {
                state.h_state
            };
            if conv {
                state.conv_state = f.model.ssm_pool.conv_state(0, 1);
            } else {
                state.h_state = f.model.ssm_pool.h_state(0, 1);
            }
            f.gpu.clear();
            assert!(
                f.model
                    .commit_accepted_prefix(&mut f.seqs[0], 3, 5)
                    .is_err(),
                "post-verdict destination mutation reached actual copy"
            );
            assert!(f.gpu.trace().is_empty());
            let state = f.seqs[0].layer_states[1]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .unwrap();
            if conv {
                state.conv_state = original;
            } else {
                state.h_state = original;
            }
            f.model
                .commit_accepted_prefix(&mut f.seqs[0], 3, 5)
                .unwrap();
        }
    }
}

#[test]
fn actual_commit_copy_and_event_failures_quarantine_without_retry() {
    if isolated("actual_commit_copy_and_event_failures_quarantine_without_retry") {
        return;
    }
    let (mut control, history) = prepared(1);
    flow::head_verdict(&mut control, 0, &history[0], 2);
    control.gpu.clear();
    control
        .model
        .commit_accepted_prefix(&mut control.seqs[0], 3, 5)
        .unwrap();
    let trace = control.gpu.trace();
    assert!(trace.iter().any(|e| matches!(e, Event::Copy(..))));
    assert!(trace.iter().any(|e| matches!(e, Event::RecordEvent(..))));
    for fail in 1..=trace.len() {
        let (mut f, history) = prepared(1);
        flow::head_verdict(&mut f, 0, &history[0], 2);
        let peer_h = f.gpu.read_span(f.model.ssm_pool.h_state(0, 1), 16);
        f.gpu.clear();
        f.gpu.fail.store(fail, Ordering::Relaxed);
        assert!(
            f.model
                .commit_accepted_prefix(&mut f.seqs[0], 3, 5)
                .is_err()
        );
        assert_eq!(f.gpu.read_span(f.model.ssm_pool.h_state(0, 1), 16), peer_h);
        f.gpu.clear();
        assert!(
            f.model
                .commit_accepted_prefix(&mut f.seqs[0], 3, 5)
                .is_err()
        );
        assert!(f.gpu.trace().is_empty());
    }
}

#[test]
fn failed_owned_wait_then_early_retirement_error_cannot_release_target_guard_on_drop() {
    if isolated("failed_owned_wait_then_early_retirement_error_cannot_release_target_guard_on_drop")
    {
        return;
    }
    let (mut f, history) = prepared(1);
    flow::head_verdict(&mut f, 0, &history[0], 2);
    flow::acknowledge(&mut f, 0, 2, false);
    let position = f.seqs[0].seq_len;
    f.gpu.clear();
    f.gpu.fail.store(1, Ordering::Relaxed);
    assert!(
        f.model
            .run_mtp_propose_inner(1, position, 4, &mut f.seqs[0], None)
            .is_err()
    );
    assert_eq!(
        f.gpu.trace(),
        vec![Event::WaitEvent(DEFAULT, f.model.secondary_event)]
    );
    f.gpu.clear();
    assert!(f.model.free_sequence(&mut f.seqs[0]).is_err());
    drop(std::mem::replace(
        &mut f.seqs[0],
        crate::traits::SequenceState::host_only(0),
    ));
    assert!(
        !f.model.ssm_pool.slot_is_free(0),
        "dropping failed selected state returned quarantined target slot"
    );
    assert!(!f.model.ssm_pool.claim_specific(0));
    assert!(f.model.ssm_pool.claim_guarded().is_err());
}

#[test]
fn teardown_joins_actual_secondary_even_when_commit_event_record_failed() {
    if isolated("teardown_joins_actual_secondary_even_when_commit_event_record_failed") {
        return;
    }
    let (mut control, history) = prepared(1);
    flow::head_verdict(&mut control, 0, &history[0], 2);
    control.gpu.clear();
    control
        .model
        .commit_accepted_prefix(&mut control.seqs[0], 3, 5)
        .unwrap();
    let fail = control
        .gpu
        .trace()
        .iter()
        .position(|e| matches!(e, Event::RecordEvent(..)))
        .unwrap()
        + 1;
    let failed_commit = || {
        let (mut f, history) = prepared(1);
        flow::head_verdict(&mut f, 0, &history[0], 2);
        f.gpu.clear();
        f.gpu.fail.store(fail, Ordering::Relaxed);
        assert!(
            f.model
                .commit_accepted_prefix(&mut f.seqs[0], 3, 5)
                .is_err()
        );
        // Drained host lifetime: no new admission between dropping owners and
        // model teardown, which must join even an unrecorded secondary write.
        f.seqs = std::array::from_fn(crate::traits::SequenceState::host_only);
        f.gpu.clear();
        f
    };
    let mut f = failed_commit();
    f.model.teardown().unwrap();
    let trace = f.gpu.trace();
    let joined = trace
        .iter()
        .position(|e| *e == Event::Sync(f.model.secondary_stream))
        .expect("actual secondary completion is required, not an old event wait");
    let freed = trace
        .iter()
        .position(|e| matches!(e, Event::Free(_)))
        .unwrap();
    assert!(joined < freed);
    let mut f = failed_commit();
    f.gpu.fail.store(joined + 1, Ordering::Relaxed);
    assert!(f.model.teardown().is_err());
    assert!(!f.gpu.trace().iter().any(|e| matches!(e, Event::Free(_))));
    assert_eq!(f.gpu.sweeps.load(Ordering::Relaxed), 0);
    f.gpu.clear();
    assert!(f.model.teardown().is_err());
    assert!(f.gpu.trace().is_empty());
    assert_eq!(f.gpu.sweeps.load(Ordering::Relaxed), 0);
}

#[test]
fn actual_pending_retirement_waits_secondary_before_target_zero_and_quarantines_failure() {
    if isolated(
        "actual_pending_retirement_waits_secondary_before_target_zero_and_quarantines_failure",
    ) {
        return;
    }
    let (mut control, history) = prepared(1);
    flow::head_verdict(&mut control, 0, &history[0], 2);
    flow::acknowledge(&mut control, 0, 2, false);
    control.gpu.clear();
    control.model.free_sequence(&mut control.seqs[0]).unwrap();
    let trace = control.gpu.trace();
    let wait = trace
        .iter()
        .position(|e| *e == Event::WaitEvent(DEFAULT, control.model.secondary_event))
        .expect("selected actual retirement must join queued secondary commit");
    let zero = trace
        .iter()
        .position(|e| matches!(e, Event::Memset(..)))
        .unwrap();
    assert!(wait < zero);
    let (mut f, history) = prepared(1);
    flow::head_verdict(&mut f, 0, &history[0], 2);
    flow::acknowledge(&mut f, 0, 2, false);
    let peer = f.gpu.read_span(f.model.ssm_pool.h_state(0, 1), 16);
    f.gpu.clear();
    f.gpu.fail.store(wait + 1, Ordering::Relaxed);
    assert!(f.model.free_sequence(&mut f.seqs[0]).is_err());
    assert!(!f.gpu.trace().iter().any(|e| matches!(e, Event::Memset(..))));
    assert!(!f.model.ssm_pool.slot_is_free(0));
    assert!(!f.model.ssm_pool.claim_specific(0));
    assert!(f.model.ssm_pool.claim_guarded().is_err());
    assert_eq!(f.gpu.read_span(f.model.ssm_pool.h_state(0, 1), 16), peer);
    f.gpu.clear();
    f.model.gpu.synchronize(DEFAULT).unwrap();
    assert!(f.model.free_sequence(&mut f.seqs[0]).is_err());
    assert!(!f.model.ssm_pool.slot_is_free(0));
}
