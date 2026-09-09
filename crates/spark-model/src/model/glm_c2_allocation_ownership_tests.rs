// SPDX-License-Identifier: AGPL-3.0-only
//! Actual allocation owns its SlotGuard across fallible initialization.
use super::{fixture::*, verdict_continuation_tests as flow};
use crate::model::ssm_pool::SsmStatePool;
use crate::traits::Model;
use std::collections::BTreeSet;
use std::sync::{Arc, atomic::Ordering};

fn available(rank: usize, owner: usize) -> Fixture {
    let mut f = Fixture::new(rank);
    f.model.free_sequence(&mut f.seqs[owner]).unwrap();
    let mut cfg = f.model.config.clone();
    cfg.layer_types = vec![atlas_core::config::LayerType::LinearAttention];
    cfg.linear_num_key_heads = 1;
    cfg.linear_num_value_heads = 1;
    cfg.linear_key_head_dim = 2;
    cfg.linear_value_head_dim = 2;
    cfg.linear_conv_kernel_dim = 2;
    cfg.mamba_num_heads = 0;
    cfg.mamba_head_dim = 0;
    let pool = Arc::new(
        SsmStatePool::new(
            &cfg,
            2,
            false,
            3,
            4,
            false,
            crate::ssm_reserve::SsmRollbackMode::Snapshot,
            f.model.gpu.as_ref(),
        )
        .unwrap(),
    );
    assert_eq!(
        (pool.num_ssm_layers, pool.h_stored_bytes, pool.conv_bytes),
        (1, 16, 48)
    );
    let first = pool.claim_guarded().unwrap();
    let second = pool.claim_guarded().unwrap();
    let (available, peer) = if first.idx() == Some(owner) {
        (first, second)
    } else {
        (second, first)
    };
    assert_eq!(peer.idx(), Some(1 - owner));
    drop(available);
    f.seqs[owner].ssm_slot = None;
    f.seqs[1 - owner].ssm_slot = Some(peer);
    f.model.ssm_pool = pool;
    f.model.config = cfg;
    f.gpu
        .write_span(f.model.ssm_pool.h_state(0, 1 - owner), &[0x37; 16]);
    f.gpu
        .write_span(f.model.ssm_pool.conv_state(0, 1 - owner), &[0x53; 48]);
    f.gpu.clear();
    f
}

#[test]
fn selected_allocation_each_actual_zero_reset_completion_error_neutralizes_local_guard() {
    if flow::isolated(
        "allocation_ownership_tests::selected_allocation_each_actual_zero_reset_completion_error_neutralizes_local_guard",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let control = available(rank, owner);
            assert_eq!(control.head.paired_test_free_blocks(), 128);
            let mut allocated = control.model.alloc_sequence().unwrap();
            assert_eq!(control.head.paired_test_free_blocks(), 0);
            assert_eq!(allocated.slot_idx, owner);
            assert!(
                allocated
                    .ssm_slot
                    .as_ref()
                    .unwrap()
                    .belongs_to(&control.model.ssm_pool)
            );
            let trace = control.gpu.trace();
            assert_eq!(
                trace,
                vec![
                    Event::Memset(control.model.ssm_pool.h_state(0, owner), 16, DEFAULT),
                    Event::Memset(control.model.ssm_pool.conv_state(0, owner), 48, DEFAULT),
                    Event::Sync(DEFAULT),
                    Event::Memset(control.model.ssm_pool.h_state(0, owner), 16, DEFAULT),
                    Event::Memset(control.model.ssm_pool.conv_state(0, owner), 48, DEFAULT),
                    Event::Sync(DEFAULT),
                ]
            );
            let original: BTreeSet<_> = flow::private(&allocated)
                .block_table
                .iter()
                .copied()
                .collect();
            control.model.free_sequence(&mut allocated).unwrap();
            let reused = control.model.alloc_sequence().unwrap();
            assert_eq!(reused.slot_idx, owner);
            assert_eq!(
                flow::private(&reused)
                    .block_table
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                original
            );
            for ordinal in 1..=trace.len() {
                let f = available(rank, owner);
                let peer_blocks = flow::private(&f.seqs[1 - owner]).block_table.clone();
                f.gpu.fail.store(ordinal, Ordering::Relaxed);
                let result = f.model.alloc_sequence();
                assert!(
                    result.is_err(),
                    "successful control-derived fault must execute"
                );
                let error = result.err().unwrap();
                assert!(format!("{error:#}").contains("injected fixture operation failure"));
                assert_eq!(
                    f.gpu.trace().len(),
                    ordinal,
                    "no operation follows selected failure"
                );
                let failed = &f.gpu.trace()[ordinal - 1];
                match (ordinal - 1) % 3 {
                    0 => assert_eq!(
                        *failed,
                        Event::Memset(f.model.ssm_pool.h_state(0, owner), 16, DEFAULT)
                    ),
                    1 => assert_eq!(
                        *failed,
                        Event::Memset(f.model.ssm_pool.conv_state(0, owner), 48, DEFAULT)
                    ),
                    _ => assert_eq!(*failed, Event::Sync(DEFAULT)),
                }
                assert!(
                    !f.model.ssm_pool.slot_is_free(owner),
                    "local SlotGuard Drop must not recycle failed target owner"
                );
                assert!(!f.model.ssm_pool.claim_specific(owner));
                assert!(f.model.ssm_pool.claim_guarded().is_err());
                assert_eq!(
                    f.gpu.read_span(f.model.ssm_pool.h_state(0, 1 - owner), 16),
                    vec![0x37; 16]
                );
                assert_eq!(
                    f.gpu
                        .read_span(f.model.ssm_pool.conv_state(0, 1 - owner), 48),
                    vec![0x53; 48]
                );
                assert_eq!(flow::private(&f.seqs[1 - owner]).block_table, peer_blocks);
                f.model.gpu.synchronize(DEFAULT).unwrap();
                f.gpu.clear();
                assert!(f.model.alloc_sequence().is_err());
                assert!(f.gpu.trace().is_empty());
            }
        }
    }
}

#[test]
fn initial_selected_allocation_matches_real_private_candidate_to_target_before_gpu() {
    if flow::isolated(
        "allocation_ownership_tests::initial_selected_allocation_matches_real_private_candidate_to_target_before_gpu",
    ) {
        return;
    }
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        for owner in 0..2 {
            f.model.free_sequence(&mut f.seqs[owner]).unwrap();
        }
        // Hold real target0 outside a SequenceState. Both actual private slots
        // are free, so their first candidate is0 while target allocation sees1.
        let first = f.model.ssm_pool.claim_guarded().unwrap();
        let second = f.model.ssm_pool.claim_guarded().unwrap();
        let (target0, target1) = if first.idx() == Some(0) {
            (first, second)
        } else {
            (second, first)
        };
        assert_eq!(target0.idx(), Some(0));
        assert_eq!(target1.idx(), Some(1));
        drop(target1);
        f.gpu.clear();
        let result = f.model.alloc_sequence();
        assert!(
            result.is_err(),
            "initial allocation must bind private0 to target0, not publish target1/private0"
        );
        assert!(
            f.gpu.trace().is_empty(),
            "identity refusal precedes target zero/reset"
        );
        assert!(!f.model.ssm_pool.slot_is_free(1));
        drop(target0); // Never written or failed; this independent guard remains valid.
        f.gpu.clear();
        assert!(
            f.model.alloc_sequence().is_err(),
            "terminal private latch precedes another target claim"
        );
        assert!(f.gpu.trace().is_empty());
    }
}

#[test]
fn actual_private_lease_exhaustion_refuses_before_any_target_initialization() {
    if flow::isolated(
        "allocation_ownership_tests::actual_private_lease_exhaustion_refuses_before_any_target_initialization",
    ) {
        return;
    }
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        // Retain the two real private leases but neutralize/release their actual
        // target guards as a boundary-negative setup. No new lease is invented.
        for seq in &mut f.seqs {
            if let Some(index) = seq.ssm_slot.as_mut().and_then(|guard| guard.take()) {
                f.model.ssm_pool.release_slot(index);
            }
        }
        f.gpu.clear();
        assert!(f.model.alloc_sequence().is_err());
        assert!(
            f.gpu.trace().is_empty(),
            "private budget is checked before target zero/sync"
        );
    }
}

#[test]
fn actual_legacy_allocation_error_retains_existing_guard_reuse_contract() {
    if flow::isolated(
        "allocation_ownership_tests::actual_legacy_allocation_error_retains_existing_guard_reuse_contract",
    ) {
        return;
    }
    let mut f = Fixture::new_legacy(0);
    f.model.free_sequence(&mut f.seqs[0]).unwrap();
    f.gpu.clear();
    let control = f.model.alloc_sequence().unwrap();
    assert_eq!(control.slot_idx, 0);
    let ordinal = f
        .gpu
        .trace()
        .iter()
        .position(|event| matches!(event, Event::Sync(DEFAULT)))
        .unwrap()
        + 1;
    let mut control = control;
    f.model.free_sequence(&mut control).unwrap();
    f.gpu.clear();
    f.gpu.fail.store(ordinal, Ordering::Relaxed);
    assert!(f.model.alloc_sequence().is_err());
    assert!(
        f.model.ssm_pool.slot_is_free(0),
        "legacy RAII failure contract is unchanged"
    );
    let guard = f.model.ssm_pool.claim_guarded().unwrap();
    assert_eq!(guard.idx(), Some(0));
    drop(guard);
}

#[test]
fn actual_target_and_private_body_allocation_callback_errors_quarantine_target() {
    if flow::isolated(
        "allocation_ownership_tests::actual_target_and_private_body_allocation_callback_errors_quarantine_target",
    ) {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            for target in [true, false] {
                let setup = || {
                    let mut f = Fixture::new(rank);
                    f.model.free_sequence(&mut f.seqs[owner]).unwrap();
                    f.gpu
                        .record_state_allocations
                        .store(true, Ordering::Relaxed);
                    f.gpu.clear();
                    f
                };
                let control = setup();
                let mut state = control.model.alloc_sequence().unwrap();
                let trace = control.gpu.trace();
                let ordinal = trace
                    .iter()
                    .position(|event| *event == Event::AllocState(target))
                    .unwrap()
                    + 1;
                assert_eq!(
                    trace
                        .iter()
                        .filter(|event| matches!(event, Event::AllocState(_)))
                        .count(),
                    2
                );
                control.model.free_sequence(&mut state).unwrap();
                let f = setup();
                let private_free = f.head.paired_test_free_blocks();
                let peer = flow::private(&f.seqs[1 - owner]).block_table.clone();
                f.gpu.fail.store(ordinal, Ordering::Relaxed);
                let error = f
                    .model
                    .alloc_sequence()
                    .err()
                    .expect("actual callback must fail");
                assert!(format!("{error:#}").contains("injected fixture operation failure"));
                assert_eq!(f.gpu.trace()[ordinal - 1], Event::AllocState(target));
                assert_eq!(f.gpu.trace().len(), ordinal);
                assert!(!f.model.ssm_pool.slot_is_free(owner));
                assert!(!f.model.ssm_pool.claim_specific(owner));
                assert!(f.model.ssm_pool.claim_guarded().is_err());
                assert_eq!(f.head.paired_test_free_blocks(), private_free);
                assert_eq!(flow::private(&f.seqs[1 - owner]).block_table, peer);
                f.model.gpu.synchronize(DEFAULT).unwrap();
                f.gpu.clear();
                assert!(f.model.alloc_sequence().is_err());
                assert!(f.gpu.trace().is_empty());
            }
        }
    }
}

#[test]
fn actual_legacy_retirement_completion_error_keeps_best_effort_cleanup_order() {
    if flow::isolated(
        "allocation_ownership_tests::actual_legacy_retirement_completion_error_keeps_best_effort_cleanup_order",
    ) {
        return;
    }
    for rank in 0..2 {
        for inject in [false, true] {
            let mut f = Fixture::new_legacy(rank);
            f.model.free_sequence(&mut f.seqs[0]).unwrap();
            let mut seq = f.model.alloc_sequence().unwrap();
            let meta = f
                .model
                .ensure_chunked_prefill_meta(&mut seq, 4, 16)
                .unwrap();
            let pointers = [meta.block_table, meta.seq_len];
            f.gpu.clear();
            if inject {
                f.gpu.fail.store(1, Ordering::Relaxed);
            }
            // Legacy logs its zero-completion error and continues metadata
            // cleanup; this deliberately differs from the selected path.
            f.model.free_sequence(&mut seq).unwrap();
            assert_eq!(
                f.gpu.trace(),
                vec![
                    Event::Sync(DEFAULT),
                    Event::Free(pointers[0]),
                    Event::Free(pointers[1])
                ]
            );
            assert!(seq.ssm_slot_idx().is_none());
            assert!(seq.chunked_prefill_meta.is_none());
            let guard = f.model.ssm_pool.claim_guarded().unwrap();
            assert_eq!(guard.idx(), Some(0));
        }
    }
}
