// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Model entry points: target/capture/primer ordering, not GPU numerics.
//! Also the default-stream work that must be ordered after a verify commit
//! still in flight on the secondary stream.

#[path = "prefill_stream_test_fixture.rs"]
mod fixture;
use crate::traits::Model;
use fixture::*;

fn isolated(name: &str) -> bool {
    if std::env::var("ATLAS_PREFILL_STREAM_TEST_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("model::trait_impl::prefill_stream_tests::{name}"),
            "--nocapture",
        ])
        .env("ATLAS_PREFILL_STREAM_TEST_CHILD", "1")
        .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
        .env_remove("ATLAS_NO_MTP_EAGER_DRAFTER")
        .env_remove("ATLAS_MTP_CARRY_DRAFTER")
        .env_remove("ATLAS_MTP_ACCEPT_DEBUG")
        .env_remove("ATLAS_DIAG_GEMMA4")
        .env_remove("ATLAS_NEMO_DUMP")
        .env_remove("ATLAS_SSM_SAVE_DUMP")
        .status()
        .unwrap();
    assert!(status.success(), "actual entry-point child failed: {name}");
    true
}

#[derive(Clone, Copy, Debug)]
enum Entry {
    Full,
    Chunk,
    TwoPhase,
}

fn invoke(f: &mut Fixture, entry: Entry) -> anyhow::Result<spark_runtime::gpu::DevicePtr> {
    let tokens = [1, 2, 3, 4];
    match entry {
        Entry::Full => f.model.prefill(&tokens, &mut f.seq, CALLER),
        Entry::Chunk => f
            .model
            .prefill_chunk(&tokens, &mut f.seq, 0, 4, true, CALLER),
        Entry::TwoPhase => f.model.prefill_twophase(&tokens, &mut f.seq, 4, CALLER),
    }
}

#[test]
fn actual_multi_rank_entries_keep_target_capture_and_primer_on_one_stream() {
    if isolated("actual_multi_rank_entries_keep_target_capture_and_primer_on_one_stream") {
        return;
    }
    for (tp, ep, rank) in [(2, 2, 0), (2, 2, 1), (2, 1, 0), (2, 1, 1)] {
        for entry in [Entry::Full, Entry::Chunk, Entry::TwoPhase] {
            let mut f = Fixture::new(tp, ep, rank);
            invoke(&mut f, entry).unwrap();
            assert_eq!(
                f.events(),
                vec![
                    Event::Target(4, DEFAULT),
                    Event::Capture(4, DEFAULT),
                    Event::Primer(3, DEFAULT),
                ],
                "{entry:?}, TP{tp}/EP{ep}, rank{rank}"
            );
            assert_eq!(f.seq.tokens, [1, 2, 3, 4]);
            assert_eq!(f.seq.seq_len, 4);
        }
    }
}

#[test]
fn actual_single_rank_preserves_full_default_and_chunk_caller_streams() {
    if isolated("actual_single_rank_preserves_full_default_and_chunk_caller_streams") {
        return;
    }
    for (entry, expected) in [
        (Entry::Full, DEFAULT),
        (Entry::Chunk, CALLER),
        (Entry::TwoPhase, CALLER),
    ] {
        let mut f = Fixture::new(1, 1, 0);
        invoke(&mut f, entry).unwrap();
        assert_eq!(
            f.events(),
            vec![
                Event::Target(4, expected),
                Event::Capture(4, expected),
                Event::Primer(3, expected),
            ],
            "{entry:?}"
        );
    }
}

#[test]
fn actual_nonfinal_chunks_capture_but_consume_only_after_final_chunk() {
    if isolated("actual_nonfinal_chunks_capture_but_consume_only_after_final_chunk") {
        return;
    }
    for (tp, ep, stream) in [(2, 2, DEFAULT), (1, 1, CALLER)] {
        let mut f = Fixture::new(tp, ep, 0);
        let tokens = [1, 2, 3, 4];
        f.model
            .prefill_chunk(&tokens, &mut f.seq, 0, 2, false, CALLER)
            .unwrap();
        assert_eq!(
            f.events(),
            vec![Event::Target(2, stream), Event::Capture(2, stream)]
        );
        f.model
            .prefill_chunk(&tokens, &mut f.seq, 2, 2, true, CALLER)
            .unwrap();
        assert_eq!(
            f.events(),
            vec![
                Event::Target(2, stream),
                Event::Capture(2, stream),
                Event::Target(2, stream),
                Event::Capture(2, stream),
                Event::Primer(3, stream),
            ]
        );
    }
}

#[test]
fn actual_target_or_capture_errors_never_reach_primer() {
    if isolated("actual_target_or_capture_errors_never_reach_primer") {
        return;
    }
    for entry in [Entry::Full, Entry::Chunk, Entry::TwoPhase] {
        for capture_failure in [false, true] {
            let mut f = Fixture::new(2, 2, 0);
            f.fail(capture_failure);
            assert!(invoke(&mut f, entry).is_err());
            let mut expected = vec![Event::Target(4, DEFAULT)];
            if capture_failure {
                expected.push(Event::Capture(4, DEFAULT));
            }
            assert_eq!(
                f.events(),
                expected,
                "{entry:?}, capture_failure={capture_failure}"
            );
        }
    }
}

#[test]
fn actual_disabled_capture_owner_keeps_target_path_without_primer() {
    if isolated("actual_disabled_capture_owner_keeps_target_path_without_primer") {
        return;
    }
    for entry in [Entry::Full, Entry::Chunk, Entry::TwoPhase] {
        let mut f = Fixture::new(2, 2, 0);
        f.disable_capture();
        invoke(&mut f, entry).unwrap();
        assert_eq!(f.events(), vec![Event::Target(4, DEFAULT)]);
    }
}

/// The default stream waits on the secondary event, then `then` happens.
fn after_commit(order: &[Order], secondary_event: u64, then: &Order) -> bool {
    let wait = Order::Wait(DEFAULT, secondary_event);
    let at = |o| order.iter().position(|x| x == o);
    matches!((at(&wait), at(then)), (Some(w), Some(t)) if w < t)
}

/// A verify commit folds its accepted rows into the SSM slot on the secondary
/// stream. The head orders its next decode after it in the scheduler
/// (`sync_secondary`); no wire command carries that, so the worker orders
/// its own plain decode.
#[test]
fn the_worker_orders_a_plain_decode_after_the_verify_commit() {
    for ep in [1, 2] {
        let mut f = Fixture::new(2, ep, 1);
        let event = f.model.secondary_event;
        let seq = std::mem::replace(&mut f.seq, crate::traits::SequenceState::host_only(0));
        // The fixture's comm delivers zeros: slot 0, token 0, a plain decode.
        let err = f.model.ep_worker_step(&mut [Some(seq)]).unwrap_err();
        assert!(format!("{err:#}").contains("stream fixture"), "{err:#}");
        assert!(after_commit(&f.order(), event, &Order::Decode), "EP{ep}");
    }
}

/// A verify step that finishes a sequence commits on the secondary stream and
/// the sequence is freed (or its slot compacted) straight away: the zero and
/// the slot copy on the default stream must come after that commit.
#[test]
fn freeing_and_compacting_a_slot_come_after_the_verify_commit() {
    let f = Fixture::new(1, 1, 0);
    let event = f.model.secondary_event;
    let mut seq = f.model.alloc_sequence().unwrap();
    f.order();
    f.model.free_sequence(&mut seq).unwrap();
    // The zero is synced on the host right after it is issued.
    assert!(after_commit(&f.order(), event, &Order::Sync(DEFAULT)));

    let mut seq = f.model.alloc_sequence().unwrap();
    let to = seq.slot_idx + 1;
    f.order();
    f.model.compact_sequence(&mut seq, to).unwrap();
    // Likewise the slot copy.
    assert!(after_commit(&f.order(), event, &Order::Sync(DEFAULT)));
}
