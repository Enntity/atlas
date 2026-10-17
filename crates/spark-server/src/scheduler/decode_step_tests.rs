// SPDX-License-Identifier: AGPL-3.0-only

//! F4 speculative-lane thinking-budget tests: the draft-side eligibility
//! gate and the commit-side truncation composition (a forced draft that
//! crosses the budget must truncate at the governed position and defer the
//! boundary emission to serial).

use super::{exl3_spec_eligible, exl3_spec_processors_idle};
use crate::scheduler::emit_step::emit_token;
use crate::scheduler::types::ActiveSeq;
use crate::scheduler::{
    sched_ctx::SchedCtx,
    test_support::{RespRx, test_seq},
};

const CLOSE: u32 = 901;
const FENCE: u32 = 902;

/// The receiver is returned so callers keep it alive: a dropped receiver is
/// a client disconnect, which `emit_token` retires before committing.
fn thinking_seq(count: u32, budget: Option<u32>) -> (ActiveSeq, RespRx) {
    let (mut a, rx) = test_seq(vec![10], 64, None, 12);
    a.finished = false;
    a.inside_thinking = true;
    a.think_end_token = Some(CLOSE);
    a.think_start_token = Some(900);
    a.thinking_tokens = count;
    a.thinking_budget = budget;
    (a, rx)
}

#[test]
fn spec_eligible_refuses_an_exhausted_thinking_budget() {
    let (mut a, _rx_a) = thinking_seq(4, Some(4));
    assert!(
        !exl3_spec_eligible(&a),
        "an exhausted budget must defer to serial"
    );
    a.thinking_tokens = 3;
    assert!(exl3_spec_eligible(&a), "headroom must stay speculative");
    // Content steps (outside thinking) are gated by no budget.
    a.inside_thinking = false;
    a.thinking_tokens = 1_000;
    assert!(exl3_spec_eligible(&a));
    // None = unlimited: never refused for the budget.
    a.inside_thinking = true;
    a.thinking_budget = None;
    assert!(exl3_spec_eligible(&a));
}

#[test]
fn forced_draft_crossing_the_budget_truncates_and_defers_to_serial() {
    let (mut a, _rx_a) = thinking_seq(2, Some(4));
    let budget = a.think_budget_state();
    // Forced draft: 8 candidate emissions crossing budget 4 from count 2.
    let draft = [50, 51, 52, 53, 54, 55, 56, 57];
    let kept = budget.plain_emission_len(&draft);
    assert_eq!(
        kept, 2,
        "only the two pre-boundary tokens are plain argmaxes"
    );
    let sched = SchedCtx::for_test();
    for &tok in &draft[..kept] {
        emit_token(&mut a, tok, None, &sched);
    }
    assert_eq!(
        a.thinking_tokens, 4,
        "accounting lands exactly on the budget"
    );
    assert!(a.force_end_thinking, "the trip token must arm the governor");
    assert_eq!(
        a.output_tokens,
        vec![10, 50, 51],
        "no token past the boundary"
    );
    assert!(!a.finished);
    assert!(
        !exl3_spec_eligible(&a),
        "the boundary emission must be deferred to serial"
    );
}

#[test]
fn natural_think_end_keeps_the_content_tail() {
    let (mut a, _rx_a) = thinking_seq(2, Some(4));
    let budget = a.think_budget_state();
    let draft = [50, 51, CLOSE, 60, 61];
    assert_eq!(
        budget.plain_emission_len(&draft),
        5,
        "a natural think-end exits thinking and unlocks the content tail"
    );
    let sched = SchedCtx::for_test();
    for &tok in &draft {
        emit_token(&mut a, tok, None, &sched);
    }
    assert_eq!(a.thinking_tokens, 4);
    assert!(!a.inside_thinking && a.think_ended);
    assert!(!a.force_end_thinking, "the close resets the governor");
    assert_eq!(a.output_tokens, vec![10, 50, 51, CLOSE, 60, 61]);
}

#[test]
fn spec_emission_toggles_the_code_fence_for_counted_thinking_tokens() {
    let (mut a, _rx_a) = thinking_seq(0, None);
    let mut sched = SchedCtx::for_test();
    sched.limits.code_fence_token = Some(FENCE);
    emit_token(&mut a, FENCE, None, &sched);
    assert!(
        a.in_code_fence,
        "a counted fence token must toggle the fence"
    );
    emit_token(&mut a, 50, None, &sched);
    assert!(a.in_code_fence, "ordinary thinking tokens leave it alone");
    emit_token(&mut a, CLOSE, None, &sched);
    assert!(
        !a.in_code_fence,
        "leaving thinking resets fence parity like serial decode"
    );
}

#[test]
fn spec_emission_decay_mirrors_the_serial_budget_on_reentry() {
    let (mut a, _rx) = test_seq(vec![10], 64, None, 12);
    a.finished = false;
    a.think_start_token = Some(900);
    a.think_end_token = Some(CLOSE);
    a.spontaneous_think_budget = 256;
    a.think_watchdog_fires = 2;
    let sched = SchedCtx::for_test();
    emit_token(&mut a, 900, None, &sched);
    assert!(a.inside_thinking);
    assert_eq!(
        a.thinking_budget,
        Some(64),
        "spontaneous entry must apply the serial decay (256 >> 2)"
    );
}

#[test]
fn budget_state_is_per_sequence() {
    let (a, _rx_a) = thinking_seq(2, Some(4));
    let (mut b, _rx_b) = thinking_seq(0, Some(16));
    b.thinking_tokens = 0;
    let sa = a.think_budget_state();
    let sb = b.think_budget_state();
    let draft = [50, 51, 52, 53, 54, 55, 56, 57];
    assert_eq!(sa.plain_emission_len(&draft), 2);
    assert_eq!(
        sb.plain_emission_len(&draft),
        8,
        "each sequence is truncated against its own accounting"
    );
}

#[test]
fn mid_step_crossing_at_the_discriminating_budget() {
    // The k=3 lane shape (<=4 emissions per step) at budget 63: the step
    // starts at count 61 and its stream would emit thinking tokens
    // #62..#65, so #63 (the arming token) is kept and #64..#65 — the first
    // governed position and after — are dropped. This is the exact
    // mid-step crossing the budget-64 GPU arm could not exhibit (there the
    // boundary landed on a step end).
    let (mut a, _rx_a) = thinking_seq(61, Some(63));
    let budget = a.think_budget_state();
    let stream = [50, 51, 52, 53];
    let kept = budget.plain_emission_len(&stream);
    assert_eq!(kept, 2, "keep through the arming token only");
    assert_eq!(stream.len() - kept, 2, "two governed emissions dropped");
    let sched = SchedCtx::for_test();
    for &tok in &stream[..kept] {
        emit_token(&mut a, tok, None, &sched);
    }
    assert_eq!(a.thinking_tokens, 63);
    assert!(a.force_end_thinking);
    assert!(!exl3_spec_eligible(&a), "boundary emission goes to serial");
}

#[test]
fn observed_budget_64_case_stream_ends_at_the_boundary() {
    // The GPU-observed inert case: budget 64, the step's stream ends exactly
    // at the arming token (count 62 + 2 emissions = #63, #64). Nothing is
    // dropped, so patched == unpatched there — the fix is silent without a
    // mid-step crossing, which is why engagement telemetry exists.
    let (mut a, _rx_a) = thinking_seq(62, Some(64));
    let budget = a.think_budget_state();
    let stream = [50, 51];
    let kept = budget.plain_emission_len(&stream);
    assert_eq!(kept, 2, "boundary on a step end drops nothing");
    assert_eq!(stream.len() - kept, 0);
    let sched = SchedCtx::for_test();
    for &tok in &stream[..kept] {
        emit_token(&mut a, tok, None, &sched);
    }
    assert_eq!(a.thinking_tokens, 64);
    assert!(a.force_end_thinking, "the governor still arms for serial");
}

#[test]
fn confidence_and_unlimited_thinking_use_the_serial_pipeline() {
    let (mut a, _rx_a) = thinking_seq(10, Some(1000));
    let mut sched = SchedCtx::for_test();
    sched.watchdog.confidence_early_stop = true;
    assert!(!exl3_spec_processors_idle(&a, &sched));
    sched.watchdog.confidence_early_stop = false;
    assert!(exl3_spec_processors_idle(&a, &sched));
    a.thinking_budget = None;
    assert!(!exl3_spec_processors_idle(&a, &sched));
    a.inside_thinking = false;
    assert!(exl3_spec_processors_idle(&a, &sched));
}

#[test]
fn post_close_raw_think_token_is_never_committed() {
    let (mut a, _rx_a) = thinking_seq(2, Some(8));
    a.inside_thinking = false;
    a.think_ended = true;
    a.tool_call_start_token = Some(903);
    let policy = a.think_budget_state();
    for tok in [900, CLOSE, 903] {
        assert_eq!(policy.processor_free_emission_len(&[tok, 50]), 0);
        assert_eq!(policy.processor_free_emission_len(&[50, tok, 51]), 1);
    }
}
