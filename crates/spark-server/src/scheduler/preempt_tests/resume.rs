// SPDX-License-Identifier: AGPL-3.0-only

//! Requeue → resume round-trip tests. Split from `preempt_tests.rs` (500-LoC cap).

use super::*;

// ── requeue → resume round trip ──────────────────────────────────────────

#[test]
fn resume_reprefills_exact_history_and_preserves_stream_state() {
    let model = PreemptStubModel::default();
    let (mut a, _rx) = active_seq(3, 6);
    a.disable_mtp = true;
    let last_token = a.last_token;
    let out_before = a.output_tokens.clone();
    let remaining_before = a.remaining;
    let history = a.seq.tokens.clone();

    let p = preempt_requeue(&model, a);
    assert_eq!(p.tokens, history);
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![3]);

    let resumed = resume_preempted_seq(&model, p).expect("resume succeeds");
    // The re-prefill processed EXACTLY the retained history — the pending
    // last_token is decoded next, not re-prefilled and never re-emitted.
    assert_eq!(*model.prefilled.lock().unwrap(), vec![history.clone()]);
    assert_eq!(resumed.seq.tokens, history);
    assert_eq!(resumed.seq.seq_len, history.len());
    assert_eq!(resumed.last_token, last_token);
    assert_eq!(resumed.output_tokens, out_before);
    assert_eq!(resumed.remaining, remaining_before);
    assert!(resumed.disable_mtp);
    assert!(resumed.seq.disable_mtp);
    assert!(!resumed.finished);
    // Starvation guard armed.
    assert_eq!(
        resumed.preempt_immune_until_tokens,
        out_before.len() + PREEMPT_IMMUNITY_TOKENS
    );
}

#[test]
fn swap_resume_restores_native_only_sequence_fence() {
    let model = PreemptStubModel::default();
    let (mut a, _rx) = active_seq(3, 6);
    a.disable_mtp = true;
    let dir = std::env::temp_dir().join(format!(
        "atlas_preempt_swap_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut spill = KvSpillManager::new(dir, 1024 * 1024).unwrap();
    let swapped = match spill_out_sequence(&model, a, &mut spill) {
        Ok(swapped) => swapped,
        Err((_active, error)) => panic!("swap out failed: {error:#}"),
    };
    let resumed = resume_swapped_seq(None, None, &model, swapped, &mut spill).unwrap();
    assert!(resumed.disable_mtp);
    assert!(resumed.seq.disable_mtp);
}

#[test]
fn resume_loop_gates_on_blocks_and_reclaims_from_prefix_cache() {
    let model = PreemptStubModel {
        total_blocks: 100,
        free_blocks: AtomicUsize::new(0),
        reclaimable: AtomicUsize::new(50),
        ..Default::default()
    };
    let (a, _rx) = active_seq(0, 4);
    let p = {
        let mut history_seq = a;
        history_seq.seq.tokens = (0..32).collect(); // 32-token history
        preempt_requeue(&model, history_seq)
    };
    let mut preempted = vec![p];
    let mut active = Vec::new();
    // block_size 16 → needs 32/16+1 = 3 blocks (+1 headroom = 4): free 0,
    // but 50 reclaimable → the loop must ASK the prefix cache and resume.
    resume_preempted_seqs(&model, &mut active, &mut preempted, 8, 16);
    assert_eq!(active.len(), 1);
    assert!(preempted.is_empty());

    // With nothing free AND nothing reclaimable, it stays parked (no error).
    let model2 = PreemptStubModel {
        total_blocks: 100,
        ..Default::default()
    };
    let (a2, mut rx2) = active_seq(0, 4);
    let p2 = preempt_requeue(&model2, a2);
    let mut preempted2 = vec![p2];
    let mut active2 = Vec::new();
    resume_preempted_seqs(&model2, &mut active2, &mut preempted2, 8, 16);
    assert!(active2.is_empty());
    assert_eq!(preempted2.len(), 1);
    assert!(matches!(
        rx2.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
}

#[test]
fn resume_loop_errors_out_a_sequence_that_can_never_fit() {
    let model = PreemptStubModel {
        total_blocks: 2, // pool smaller than the history
        ..Default::default()
    };
    let (a, mut rx) = active_seq(0, 4);
    let p = {
        let mut s = a;
        s.seq.tokens = (0..64).collect();
        preempt_requeue(&model, s)
    };
    let mut preempted = vec![p];
    let mut active = Vec::new();
    resume_preempted_seqs(&model, &mut active, &mut preempted, 8, 16);
    assert!(preempted.is_empty() && active.is_empty());
    // The client is told, not left hanging forever.
    assert!(rx.try_recv().expect("error delivered").is_err());
}
