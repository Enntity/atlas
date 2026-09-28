// SPDX-License-Identifier: AGPL-3.0-only
//! Logical emission positions must not rewind the already committed model state.
use super::{emit_token, emit_token_at_position};
use crate::api::StreamEvent;
use crate::scheduler::{sched_ctx::SchedCtx, test_support::test_seq, types::ResponseSink};

#[test]
fn committed_prefix_does_not_end_emission_at_its_first_earlier_row() {
    let (mut a, _) = test_seq(vec![10], 20, None, 13);
    a.finished = false;
    a.min_tokens = 0;
    a.seq.tokens = vec![1; 13];
    let canonical = a.seq.tokens.clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    a.sink = ResponseSink::Streaming(tx);
    let mut sched = SchedCtx::for_test();
    sched.limits.max_seq_len = 14;
    for (position, token) in (10..=13).zip(40..=43) {
        emit_token_at_position(&mut a, token, None, &sched, position);
        assert_eq!(a.finished, position == 13);
        assert_eq!(a.seq.seq_len, 13, "emission must not rewind target state");
        assert_eq!(a.seq.tokens, canonical);
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(t)) if t == token));
    }
    assert_eq!(a.output_tokens, [10, 40, 41, 42, 43]);
    assert_eq!(a.remaining, 16);
}

#[test]
fn suppressed_eos_uses_logical_row_but_cannot_bypass_a_real_ceiling() {
    for output_limit in [false, true] {
        // The existing emitter consumes generation budget while thinking too.
        let budget = if output_limit { 1 } else { 20 };
        let (mut a, _) = test_seq(vec![10], budget, None, 13);
        a.finished = false;
        a.min_tokens = 100;
        a.eos_tokens = vec![900];
        a.inside_thinking = true;
        a.think_ended = false;
        a.thinking_budget = Some(100);
        let mut sched = SchedCtx::for_test();
        sched.limits.max_seq_len = 14;
        emit_token_at_position(&mut a, 900, None, &sched, 10);
        assert_eq!(a.finished, output_limit);
        assert_eq!(a.seq.seq_len, 13);
        if !output_limit {
            emit_token_at_position(&mut a, 900, None, &sched, 13);
            assert!(
                a.finished,
                "logical context ceiling still wins over EOS suppression"
            );
            assert_eq!(a.seq.seq_len, 13);
        }
    }
}

#[test]
fn ordinary_wrapper_retains_existing_emission_accounting() {
    for ceiling in [0, 12, 20] {
        for budget in [1, 4] {
            for token in [42, 900] {
                let mut sched = SchedCtx::for_test();
                sched.limits.max_seq_len = ceiling;
                let (mut legacy, _) = test_seq(vec![10], budget, None, 12);
                let (mut explicit, _) = test_seq(vec![10], budget, None, 12);
                for a in [&mut legacy, &mut explicit] {
                    a.finished = false;
                    a.min_tokens = 0;
                    a.eos_tokens = vec![900];
                }
                emit_token(&mut legacy, token, None, &sched);
                emit_token_at_position(&mut explicit, token, None, &sched, 12);
                assert_eq!(legacy.output_tokens, explicit.output_tokens);
                assert_eq!(legacy.remaining, explicit.remaining);
                assert_eq!(legacy.finished, explicit.finished);
                assert_eq!(legacy.guard_stop, explicit.guard_stop);
                assert_eq!(legacy.thinking_tokens, explicit.thinking_tokens);
                assert_eq!(legacy.seq.seq_len, explicit.seq.seq_len);
            }
        }
    }
}
