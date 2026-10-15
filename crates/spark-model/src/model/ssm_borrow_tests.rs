// SPDX-License-Identifier: AGPL-3.0-only

//! Long-copy verify rows borrowed from idle slots, on the real model with the
//! recording mock backend of `prefill_stream_tests` and a KDA-records pool.

// The real model and recording layer of `prefill_stream_tests`.
#[allow(clippy::duplicate_mod)]
#[path = "trait_impl/prefill_stream_test_fixture.rs"]
mod fixture;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use fixture::{DEFAULT, Fixture, Order};

use super::lone_capacity;
use crate::layer::SsmLayerState;
use crate::model::ssm_pool::SsmStatePool;
use crate::ssm_reserve::SsmRollbackMode;
use crate::traits::{Model, SequenceState};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Production's shape: 4 sequence slots of γ = 8 verify rows (7 drafts).
const SLOTS: usize = 4;
const GAMMA: usize = 8;

/// One GLM KDA layer (one 128×128 head: a records row is 1.5 KB) with the
/// verify pools in KDA-records mode, as `--ssm-rollback-mode records` serves.
fn records(rank: usize, ep: usize) -> Fixture {
    let mut f = Fixture::with_tail_split(ep, ep, rank);
    (
        f.model.config.linear_key_head_dim,
        f.model.config.linear_value_head_dim,
    ) = (128, 128);
    f.model.ssm_pool = Arc::new(
        SsmStatePool::new(
            &f.model.config,
            SLOTS,
            true,
            GAMMA,
            GAMMA - 1,
            false,
            SsmRollbackMode::Records,
            f.model.gpu.as_ref(),
        )
        .unwrap(),
    );
    f
}

fn ssm(seq: &SequenceState) -> &SsmLayerState {
    seq.layer_states[0]
        .as_any()
        .downcast_ref::<SsmLayerState>()
        .unwrap()
}

fn rows(seq: &SequenceState) -> Vec<DevicePtr> {
    ssm(seq).conv_state_intermediates.clone()
}

#[test]
fn the_window_is_the_shipped_pool_up_to_the_dummy() {
    let f = records(0, 1);
    let p = &f.model.ssm_pool;
    assert_eq!(p.num_intermediates, GAMMA);
    // The pools are the knob-off size: γ rows for each slot and the dummy.
    let conv_pool = p.conv_intermediate(0, SLOTS, GAMMA).0 - p.conv_intermediate(0, 0, 0).0;
    assert_eq!(conv_pool, ((SLOTS + 1) * GAMMA * p.conv_bytes) as u64);
    // A window runs from its slot's block to the dummy's, never into it.
    assert_eq!(
        [0, 1, 2, 3].map(|s| p.window_rows(s)),
        [32, 24, 16, 8],
        "rows from each slot's block to the dummy"
    );
    for s in 0..SLOTS {
        let end = p.conv_intermediate(0, s, p.window_rows(s));
        assert_eq!(end, p.conv_intermediate(0, SLOTS, 0), "slot {s}");
        let records_end = p
            .kda_records(0, s)
            .offset(p.window_rows(s) * p.kda_record_row_bytes);
        assert_eq!(records_end, p.kda_records(0, SLOTS), "slot {s}");
    }
}

#[test]
fn only_the_records_rollback_borrows() {
    let mut f = records(0, 1);
    f.model.ssm_pool = Arc::new(
        SsmStatePool::new(
            &f.model.config,
            SLOTS,
            true,
            GAMMA,
            GAMMA - 1,
            false,
            SsmRollbackMode::Snapshot,
            f.model.gpu.as_ref(),
        )
        .unwrap(),
    );
    let mut a = f.model.alloc_sequence().unwrap();
    assert_eq!(f.model.ssm_pool.lone_verify_rows(a.slot_idx), GAMMA);
    // Nothing to lend: the shipped verify rows, untouched.
    f.model.prepare_verify_rows(&mut a, 32).unwrap();
    assert_eq!(rows(&a).len(), GAMMA);
}

#[test]
fn rows_are_lent_only_while_the_owner_is_the_only_sequence() {
    let f = records(0, 1);
    let mut a = f.model.alloc_sequence().unwrap();
    let pool = &f.model.ssm_pool;
    assert_eq!(a.slot_idx, 0);
    assert_eq!(pool.lone_verify_rows(0), 32);

    // A request admitted since the proposal: its slot is no longer lent,
    // and a long verify is refused.
    let mut b = f.model.alloc_sequence().unwrap();
    assert_eq!(pool.lone_verify_rows(0), GAMMA);
    let err = f.model.prepare_verify_rows(&mut a, 16).unwrap_err();
    assert!(
        format!("{err:#}").contains("not the only sequence"),
        "{err:#}"
    );
    assert_eq!(rows(&a).len(), GAMMA);
    // Its own γ rows still verify.
    f.model.prepare_verify_rows(&mut a, GAMMA).unwrap();
    assert!(!pool.rows_lent.load(Ordering::Acquire));

    // It finished: the rows are free again.
    f.model.free_sequence(&mut b).unwrap();
    assert_eq!(pool.lone_verify_rows(0), 32);
    f.model.prepare_verify_rows(&mut a, 32).unwrap();
    let want: Vec<DevicePtr> = (0..32).map(|t| pool.conv_intermediate(0, 0, t)).collect();
    assert_eq!(rows(&a), want, "the window continues into slots 1..3");
    assert_eq!(ssm(&a).kda_records, pool.kda_records(0, 0));
    assert!(pool.rows_lent.load(Ordering::Acquire));
    // Never past the window.
    assert!(f.model.prepare_verify_rows(&mut a, 33).is_err());
}

#[test]
fn the_worker_follows_the_head_within_the_window() {
    // The worker keeps a finished sequence's slot claimed until the head
    // reuses it, so it does not check occupancy, only the window.
    let f = records(1, 2);
    let mut a = f.model.alloc_sequence().unwrap();
    let _stale = f.model.alloc_sequence().unwrap();
    f.model.prepare_verify_rows(&mut a, 32).unwrap();
    assert_eq!(rows(&a).len(), 32);
    let err = f.model.prepare_verify_rows(&mut a, 33).unwrap_err();
    assert!(format!("{err:#}").contains("reaches 32"), "{err:#}");
}

#[test]
fn a_partial_accept_rewinds_conv_from_the_borrowed_row() {
    let f = records(0, 1);
    let gpu = f.model.gpu.as_ref();
    let pool = &f.model.ssm_pool;
    let mut a = f.model.alloc_sequence().unwrap();
    let bytes = pool.conv_bytes;
    for t in 0..32 {
        gpu.copy_h2d(&vec![t as u8 + 1; bytes], pool.conv_intermediate(0, 0, t))
            .unwrap();
    }
    f.model.prepare_verify_rows(&mut a, 32).unwrap();
    // 19 drafts + the bonus kept of 32 rows: the state after row 19, which
    // lives in slot 2's block.
    f.model.commit_accepted_prefix(&mut a, 20, 32).unwrap();
    let mut conv = vec![0u8; bytes];
    gpu.copy_d2h(ssm(&a).conv_state, &mut conv).unwrap();
    assert_eq!(conv, vec![20u8; bytes]);
    // Committed: back to the slot's own rows.
    assert_eq!(rows(&a).len(), GAMMA);

    // Knob-off widths are untouched: a γ-row verify rewinds from its own row.
    f.model.prepare_verify_rows(&mut a, GAMMA).unwrap();
    f.model.commit_accepted_prefix(&mut a, 5, GAMMA).unwrap();
    gpu.copy_d2h(ssm(&a).conv_state, &mut conv).unwrap();
    assert_eq!(conv, vec![5u8; bytes]);
}

#[test]
fn a_slot_claimed_after_lent_rows_waits_for_their_commit() {
    let f = records(0, 1);
    let event = f.model.secondary_event;
    let wait = Order::Wait(DEFAULT, event);
    let mut a = f.model.alloc_sequence().unwrap();
    // No rows lent: a claim orders nothing new (as shipped).
    let mut b = f.model.alloc_sequence().unwrap();
    f.model.free_sequence(&mut b).unwrap();
    f.order();
    let mut b = f.model.alloc_sequence().unwrap();
    assert!(!f.order().contains(&wait));
    f.model.free_sequence(&mut b).unwrap();

    f.model.prepare_verify_rows(&mut a, 32).unwrap();
    f.model.commit_accepted_prefix(&mut a, 32, 32).unwrap();
    f.order();
    let _b = f.model.alloc_sequence().unwrap();
    let order = f.order();
    let at = |o: &Order| order.iter().position(|x| x == o);
    assert!(
        matches!((at(&wait), at(&Order::Sync(DEFAULT))), (Some(w), Some(s)) if w < s),
        "{order:?}"
    );
    assert!(!f.model.ssm_pool.rows_lent.load(Ordering::Acquire));
}

#[test]
fn long_copies_verify_past_the_slot_only_on_lent_rows() {
    // (long, own, γ, rows) -> drafts
    assert_eq!(lone_capacity(0, 7, 8, 32), 7, "long copies off");
    assert_eq!(
        lone_capacity(31, 7, 8, 8),
        7,
        "another sequence holds the rows"
    );
    assert_eq!(lone_capacity(31, 7, 8, 32), 31);
    assert_eq!(lone_capacity(15, 7, 8, 16), 15);
    // A window of 32 rows behind a 16-row DFlash hidden-save front.
    let (window, front) = (32, 16);
    assert_eq!(lone_capacity(31, 7, 8, usize::min(window, front)), 15);
}
