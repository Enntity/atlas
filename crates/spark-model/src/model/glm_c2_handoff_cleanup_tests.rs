// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Model cleanup boundaries, real paired reserves, real SSM SlotGuards.
//! The tiny SSM pool exercises cleanup only; the fixture body is not an SSM oracle.
use super::{fixture::*, isolated};
use crate::layers::glm5_mtp::Glm5MtpProposerState;
use crate::model::ssm_pool::SsmStatePool;
use crate::speculative::DraftProposer;
use crate::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::collections::BTreeSet;
use std::sync::{Arc, atomic::Ordering};

fn private(seq: &SequenceState) -> &Glm5MtpProposerState {
    seq.proposer_state
        .as_ref()
        .unwrap()
        .as_any()
        .downcast_ref::<Glm5MtpProposerState>()
        .unwrap()
}

fn primed_with_cleanup_pool(bootstrap_peer: bool) -> Fixture {
    let mut f = Fixture::new(1);
    for owner in 0..2 {
        f.model
            .prefill(&[1 + owner as u32, 2, 3, 4], &mut f.seqs[owner], CALLER)
            .unwrap();
    }
    // The cleanup-only pool below deliberately does not match target geometry.
    // Complete the healthy producer first; post-install decode is not valid.
    if bootstrap_peer {
        f.model.decode(6, &mut f.seqs[1], CALLER).unwrap();
    }
    // Install a real minimal pool after producer work: no production geometry
    // check is relaxed, and no SSM numerical computation is represented here.
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
    for owner in 0..2 {
        let guard = pool.claim_guarded().unwrap();
        assert_eq!(guard.idx(), Some(owner));
        f.seqs[owner].ssm_slot = Some(guard);
        f.gpu
            .write_span(pool.h_state(0, owner), &[0x31 + owner as u8; 16]);
        f.gpu
            .write_span(pool.conv_state(0, owner), &[0x51 + owner as u8; 48]);
    }
    f.model.ssm_pool = pool;
    f.gpu.clear();
    f
}

fn cleanup_failure_ordinal(zero: bool) -> usize {
    let mut f = primed_with_cleanup_pool(false);
    let h = f.model.ssm_pool.h_state(0, 0);
    f.model.free_sequence(&mut f.seqs[0]).unwrap();
    f.gpu
        .trace()
        .iter()
        .position(|event| {
            if zero {
                matches!(event, Event::Memset(ptr, 16, DEFAULT) if *ptr == h)
            } else {
                matches!(event, Event::Sync(DEFAULT))
            }
        })
        .expect("actual zero/completion event must execute")
        + 1
}

fn free_count(events: &[Event], ptr: DevicePtr) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, Event::Free(p) if *p == ptr))
        .count()
}

fn healthy_cleanup_control() {
    let mut f = primed_with_cleanup_pool(true);
    let slab = f.gpu.slab();
    let peer_blocks: BTreeSet<_> = private(&f.seqs[1]).block_table.iter().copied().collect();
    let other_blocks: BTreeSet<_> = private(&f.seqs[0]).block_table.iter().copied().collect();
    // The independent control already published, and can retire/reuse its reserve.
    f.model.free_sequence(&mut f.seqs[1]).unwrap();
    f.model.free_sequence(&mut f.seqs[1]).unwrap();
    let target = f.model.ssm_pool.claim_guarded().unwrap();
    assert_eq!(
        target.idx(),
        Some(1),
        "only the healthy target slot returns"
    );
    assert!(f.model.ssm_pool.claim_guarded().is_err());
    assert!(!f.model.ssm_pool.claim_specific(0));
    drop(target);
    let target = f.model.ssm_pool.claim_guarded().unwrap();
    assert_eq!(
        target.idx(),
        Some(1),
        "guard Drop/reclaim cannot reopen slot0"
    );
    assert!(f.model.ssm_pool.claim_guarded().is_err());
    assert!(!f.model.ssm_pool.slot_is_free(0));
    drop(target);
    let mut replacement = f.head.alloc_state(f.model.gpu.as_ref()).unwrap();
    let blocks: BTreeSet<_> = replacement
        .as_any()
        .downcast_ref::<Glm5MtpProposerState>()
        .unwrap()
        .block_table
        .iter()
        .copied()
        .collect();
    assert_eq!(
        blocks, peer_blocks,
        "only the healthy peer reserve may be returned"
    );
    assert!(blocks.is_disjoint(&other_blocks));
    assert!(
        f.head.alloc_state(f.model.gpu.as_ref()).is_err(),
        "no duplicate block return or other live-slot reuse"
    );
    f.head
        .free_state(f.model.gpu.as_ref(), replacement.as_mut())
        .unwrap();
    assert_eq!(
        free_count(&f.gpu.trace(), slab),
        0,
        "per-sequence cleanup never owns the slab"
    );
    let free = f.model.ssm_pool.free_slots.lock();
    assert_eq!(
        free.iter().filter(|&&slot| slot == 1).count(),
        1,
        "healthy SSM slot returns once"
    );
    assert_eq!(
        free.iter().filter(|&&slot| slot == 0).count(),
        0,
        "other live target SSM slot remains held"
    );
}

fn cleanup_failure_quarantines(zero: bool) {
    healthy_cleanup_control();
    let ordinal = cleanup_failure_ordinal(zero);
    let mut f = primed_with_cleanup_pool(false);
    let slab = f.gpu.slab();
    let peer_bytes = f.gpu.read_span(slab.offset(6 * ROW_BYTES), 6 * ROW_BYTES);
    let peer_blocks: BTreeSet<_> = private(&f.seqs[1]).block_table.iter().copied().collect();
    let failed_blocks: BTreeSet<_> = private(&f.seqs[0]).block_table.iter().copied().collect();
    assert_eq!(peer_blocks.len(), 128);
    assert!(peer_blocks.is_disjoint(&failed_blocks));
    f.gpu.fail.store(ordinal, Ordering::Relaxed);
    let result = f.model.free_sequence(&mut f.seqs[0]);
    assert!(
        result.is_err(),
        "selected cleanup must report early {} failure, not erase it with a later sync",
        if zero { "zero_slot" } else { "completion" }
    );
    assert_eq!(free_count(&f.gpu.trace(), slab), 0);
    // A private-KV quarantine alone is insufficient: alloc_sequence claims and
    // zeroes target SSM before allocating its proposer. The old <=1 free-list
    // assertion allowed exactly the unsafe single return of failed slot0.
    assert!(
        !f.model.ssm_pool.slot_is_free(0),
        "failed target SSM slot must not become reusable"
    );
    assert!(
        !f.model.ssm_pool.claim_specific(0),
        "failed target slot must not be claimable by index"
    );
    assert!(
        f.model.ssm_pool.claim_guarded().is_err(),
        "failed slot quarantined and healthy peer still owned: no target slot is free"
    );
    f.gpu.clear();
    f.model.gpu.synchronize(DEFAULT).unwrap();
    // Reaching a later successful completion must not rehabilitate the lease.
    assert!(f.model.free_sequence(&mut f.seqs[0]).is_err());
    assert!(f.head.alloc_state(f.model.gpu.as_ref()).is_err());
    assert!(!f.model.ssm_pool.slot_is_free(0));
    assert!(!f.model.ssm_pool.claim_specific(0));
    assert!(f.model.ssm_pool.claim_guarded().is_err());
    assert_eq!(
        f.gpu.read_span(slab.offset(6 * ROW_BYTES), 6 * ROW_BYTES),
        peer_bytes
    );
    assert_eq!(private(&f.seqs[1]).seq_len, 3);
    assert_eq!(
        private(&f.seqs[1])
            .block_table
            .iter()
            .copied()
            .collect::<BTreeSet<_>>(),
        peer_blocks
    );
    assert_eq!(
        f.gpu.read_span(f.model.ssm_pool.h_state(0, 1), 16),
        vec![0x32; 16]
    );
    assert_eq!(
        f.gpu.read_span(f.model.ssm_pool.conv_state(0, 1), 48),
        vec![0x52; 48]
    );
    assert!(
        !f.model.ssm_pool.slot_is_free(1),
        "peer SlotGuard is still exclusive"
    );
    // This cleanup-only pool can also cause geometry refusal at decode; this
    // assertion alone does not isolate the terminal latch. Valid-pool transport
    // tests prove that cause independently, without incompatible target geometry.
    f.gpu.clear();
    assert!(f.model.decode(6, &mut f.seqs[1], CALLER).is_err());
    assert!(f.model.free_sequence(&mut f.seqs[1]).is_err());
    assert!(f.model.alloc_sequence().is_err());
    assert!(f.gpu.trace().is_empty());
    assert_eq!(f.head.paired_test_free_blocks(), 0);
    assert!(f.model.ssm_pool.claim_guarded().is_err());
    for seq in &mut f.seqs {
        assert!(seq.ssm_slot_idx().is_none());
    }
}

#[test]
fn zero_slot_failure_cannot_be_hidden_by_later_paired_cleanup_sync() {
    if isolated("cleanup_tests::zero_slot_failure_cannot_be_hidden_by_later_paired_cleanup_sync") {
        return;
    }
    cleanup_failure_quarantines(true);
}

#[test]
fn first_cleanup_sync_failure_cannot_be_hidden_by_later_paired_cleanup_sync() {
    if isolated(
        "cleanup_tests::first_cleanup_sync_failure_cannot_be_hidden_by_later_paired_cleanup_sync",
    ) {
        return;
    }
    cleanup_failure_quarantines(false);
}

#[test]
fn actual_model_teardown_closes_retained_head_and_frees_slab_once() {
    if isolated("cleanup_tests::actual_model_teardown_closes_retained_head_and_frees_slab_once") {
        return;
    }
    let mut f = Fixture::new(1);
    let slab = f.gpu.slab();
    let external = f.head.clone();
    // Leave the head slots genuinely reusable before close, so allocation
    // refusal proves closure rather than merely an exhausted two-slot pool.
    for seq in &mut f.seqs {
        f.model.free_sequence(seq).unwrap();
        seq.ssm_slot = None; // Drop the neutralized guard's external pool Arc.
    }
    f.gpu.clear();
    f.model.teardown().unwrap();
    assert_eq!(
        f.gpu.sweeps.load(Ordering::Relaxed),
        1,
        "completed Model teardown must exercise the invocation-only sweep spy"
    );
    assert_eq!(
        free_count(&f.gpu.trace(), slab),
        1,
        "actual Model teardown must own paired slab close"
    );
    let events = f.gpu.trace();
    let free = events
        .iter()
        .position(|event| matches!(event, Event::Free(p) if *p == slab))
        .unwrap();
    assert!(
        events[..free]
            .iter()
            .any(|event| matches!(event, Event::Sync(DEFAULT))),
        "slab free requires successful completion"
    );
    let operations = events.len();
    assert!(external.alloc_state(f.model.gpu.as_ref()).is_err());
    assert_eq!(
        f.gpu.trace().len(),
        operations,
        "closed head rejects before device operations"
    );
    f.model.teardown().unwrap();
    assert_eq!(free_count(&f.gpu.trace(), slab), 1);
    let record = f.gpu.clone();
    drop(f);
    assert_eq!(
        free_count(&record.trace(), slab),
        1,
        "Model drop cannot repeat the explicit slab free"
    );
    drop(external);
    assert_eq!(
        free_count(&record.trace(), slab),
        1,
        "last head Arc drop cannot free through a dead backend"
    );
}

fn teardown_event_ordinal(free: bool) -> usize {
    let mut f = Fixture::new(1);
    detach_target_guards_for_teardown(&mut f);
    let slab = f.gpu.slab();
    f.gpu.clear();
    f.model.teardown().unwrap();
    let events = f.gpu.trace();
    let slab_free = events
        .iter()
        .position(|event| matches!(event, Event::Free(p) if *p == slab))
        .expect("actual paired close must reach a slab free");
    if free {
        slab_free + 1
    } else {
        events[..slab_free]
            .iter()
            .rposition(|event| matches!(event, Event::Sync(DEFAULT)))
            .expect("actual paired close must synchronize before free")
            + 1
    }
}

fn teardown_failure_is_terminal(free: bool) {
    let ordinal = teardown_event_ordinal(free);
    let mut f = Fixture::new(1);
    detach_target_guards_for_teardown(&mut f);
    let external = f.head.clone();
    let slab = f.gpu.slab();
    let before = f.gpu.read_span(slab, SLAB_BYTES);
    f.gpu.fail.store(ordinal, Ordering::Relaxed);
    assert!(
        f.model.teardown().is_err(),
        "actual paired close fault must propagate"
    );
    assert_eq!(free_count(&f.gpu.trace(), slab), usize::from(free));
    // The spy counts real sweep_unreleased calls but deliberately frees no
    // backing bytes. No slab Free event alone cannot prove sweep exclusion.
    if !free {
        assert_eq!(
            f.gpu.sweeps.load(Ordering::Relaxed),
            0,
            "unknown completion must prohibit the backend bulk sweep"
        );
    }
    // Both the failed-free recorder backing and the unsynchronized allocation
    // remain inspectable here; neither is evidence of a native free succeeding.
    assert_eq!(f.gpu.read_span(slab, SLAB_BYTES), before);
    let operations = f.gpu.trace().len();
    assert!(external.alloc_state(f.model.gpu.as_ref()).is_err());
    for seq in &mut f.seqs {
        assert!(
            external
                .free_state(
                    f.model.gpu.as_ref(),
                    seq.proposer_state.as_mut().unwrap().as_mut()
                )
                .is_err()
        );
    }
    assert_eq!(
        f.gpu.trace().len(),
        operations,
        "closed/failed owner operations must be inert"
    );
    f.gpu.fail.store(usize::MAX, Ordering::Relaxed);
    // The remaining ModelResources may finish idempotently; this owner must
    // never retry an uncertain sync or a remove-before-free-error pointer.
    let _ = f.model.teardown();
    assert_eq!(free_count(&f.gpu.trace(), slab), usize::from(free));
    if !free {
        assert_eq!(
            f.gpu.sweeps.load(Ordering::Relaxed),
            0,
            "repeated teardown must not sweep after unknown completion"
        );
    }
    assert!(
        !f.gpu.trace()[operations..]
            .iter()
            .any(|event| matches!(event, Event::Sync(DEFAULT))),
        "paired close must not retry completion after a terminal close fault"
    );
    let record = f.gpu.clone();
    drop(f);
    drop(external);
    assert_eq!(free_count(&record.trace(), slab), usize::from(free));
    if !free {
        assert_eq!(
            record.sweeps.load(Ordering::Relaxed),
            0,
            "dropping the retained head/model must not sweep uncertain owners"
        );
    }
}

// These tests retain private head leases across close, but whole-pool teardown
// requires no external SlotGuard Arc. Neutralize without recycling target slots.
fn detach_target_guards_for_teardown(f: &mut Fixture) {
    for seq in &mut f.seqs {
        let mut guard = seq.ssm_slot.take().unwrap();
        assert_eq!(guard.take(), Some(seq.slot_idx));
    }
}

#[test]
fn actual_model_teardown_sync_failure_invalidates_without_slab_free_or_retry() {
    if isolated(
        "cleanup_tests::actual_model_teardown_sync_failure_invalidates_without_slab_free_or_retry",
    ) {
        return;
    }
    teardown_failure_is_terminal(false);
}

#[test]
fn actual_model_teardown_free_failure_invalidates_without_duplicate_free() {
    if isolated(
        "cleanup_tests::actual_model_teardown_free_failure_invalidates_without_duplicate_free",
    ) {
        return;
    }
    teardown_failure_is_terminal(true);
}
