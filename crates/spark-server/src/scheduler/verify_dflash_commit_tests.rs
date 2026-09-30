// SPDX-License-Identifier: AGPL-3.0-only

//! The DFlash verify tails commit the accepted rows BEFORE they emit.
//!
//! An emit can finish the sequence (stop token, budget, guard) and return.
//! `finish_sequence` then caches `seq.tokens` together with the sequence's
//! live recurrent state, and the worker rank has already committed the step.
//! A tail that returned before its commit left the head's state at the
//! pre-verify position while `seq.tokens` and the worker were past it.

use super::lifecycle_tests::{COMMITS, StubModel};
use super::sched_ctx::SchedCtx;
use super::test_support::{EOS, test_seq};
use super::types::ActiveSeq;
use super::verify_dflash_batch_step::apply_dflash_accept;
use super::verify_dflash_step::verify_dflash_tail;

const PRE: usize = 40;

/// A live sequence as `decode_verify_dflash` leaves it: `seq_len` and
/// `seq.tokens` advanced by the whole `k`-row verify block.
fn verified(k: usize) -> ActiveSeq {
    let (mut a, _rx) = test_seq(vec![10], 20, None, PRE + k);
    a.finished = false;
    a.min_tokens = 0;
    a.seq.tokens = (0..(PRE + k) as u32).collect();
    a
}

fn commits() -> Vec<(usize, usize, usize)> {
    COMMITS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Run the per-sequence tail over `drafts` against the target's `argmax`.
fn tail(a: &mut ActiveSeq, drafts: &[u32], argmax: &[u32]) -> Option<usize> {
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    let tokens: Vec<u32> = std::iter::once(a.last_token)
        .chain(drafts.iter().copied())
        .collect();
    verify_dflash_tail(
        &StubModel::default(),
        a,
        &sched,
        drafts,
        drafts.len(),
        &ctx,
        true,
        &tokens,
        argmax.to_vec(),
        false,
        0.0,
        true,
    )
}

#[test]
fn tail_commits_when_the_bonus_token_finishes_the_sequence() {
    // Two of four drafts accepted; the bonus (row 2) is the stop token.
    let mut a = verified(5);
    assert_eq!(tail(&mut a, &[1, 2, 3, 4], &[1, 2, EOS[0], 9, 9]), None);
    assert!(a.finished, "the stop token must finish the sequence");
    assert_eq!(a.seq.tokens.len(), PRE + 3);
    assert_eq!(a.seq.seq_len, PRE + 3);
    assert_eq!(commits(), vec![(PRE + 3, 3, 5)]);
}

#[test]
fn tail_commits_when_an_accepted_draft_finishes_the_sequence() {
    // All four drafts accepted; draft 1 is the stop token, so the emit loop
    // returns with two accepted rows still un-emitted. The state still has to
    // hold every accepted row, exactly as `seq.tokens` and the worker do.
    let mut a = verified(5);
    let drafts = [1, EOS[0], 3, 4];
    assert_eq!(tail(&mut a, &drafts, &[1, EOS[0], 3, 4, 7]), None);
    assert!(a.finished);
    assert_eq!(a.output_tokens, vec![10, 1, EOS[0]]);
    assert_eq!(commits(), vec![(PRE + 5, 5, 5)]);
}

#[test]
fn tail_commits_once_when_the_sequence_continues() {
    let mut a = verified(5);
    assert_eq!(tail(&mut a, &[1, 2, 3, 4], &[1, 2, 8, 9, 9]), Some(4));
    assert!(!a.finished);
    assert_eq!(a.last_token, 8);
    assert_eq!(commits(), vec![(PRE + 3, 3, 5)]);
}

#[test]
fn batched_accept_commits_when_an_emit_finishes_the_sequence() {
    let sched = SchedCtx::for_test();
    let accept = |drafts: &[u32], argmax: &[u32]| {
        let mut a = verified(4);
        apply_dflash_accept(
            &StubModel::default(),
            &mut a,
            &sched,
            drafts,
            argmax,
            3,
            true,
        );
        (a.finished, commits())
    };
    // The bonus is the stop token.
    assert_eq!(
        accept(&[1, 2, 3], &[1, EOS[0], 9, 9]),
        (true, vec![(PRE + 2, 2, 4)])
    );
    // An accepted draft is the stop token.
    assert_eq!(
        accept(&[EOS[0], 2, 3], &[EOS[0], 2, 3, 7]),
        (true, vec![(PRE + 4, 4, 4)])
    );
    // No finish: still exactly one commit.
    assert_eq!(
        accept(&[1, 2, 3], &[1, 5, 9, 9]),
        (false, vec![(PRE + 2, 2, 4)])
    );
}
