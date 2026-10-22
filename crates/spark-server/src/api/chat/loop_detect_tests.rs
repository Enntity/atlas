// SPDX-License-Identifier: AGPL-3.0-only

//! #55: the spinning detector must never hard-mask `<tool_call>`.
//!
//! Quinn's tool-eval-bench TC-07 under context pressure: the harness
//! fills the context with short pure-text ASSISTANT turns, so under the
//! old scan `recent_short >= 5` fired on the FIRST real turn and the
//! spinning arm hard-masked the tool-call opener the model had just
//! announced. The model re-announced until `finish_reason: length`.
//! With the mask off, the sweep scores 100% — the signal was pure noise
//! in these scenarios.
//!
//! Two layers of protection, each with its own test:
//!   1. only ENGINE-generated turns count toward `recent_short`
//!      (`register_engine_turn`; injected filler breaks the run);
//!   2. when spinning DOES fire (the model really is spinning), it
//!      drives only the soft bias decay — `suppress_tool_call` stays
//!      false, so `sampling_setup`'s decay and `raw_pick_masked`/the
//!      tool_during_think −12.0 bias see a non-suppressed turn.

use super::loop_detect::check_loops;
use crate::ir::{ContentPart, Message, Role};

fn text_turn(t: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![ContentPart::Text(t.to_string())],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn user_turn(t: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentPart::Text(t.to_string())],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

/// A unique session id per test: the registry is a process-wide static,
/// so parallel tests sharing one SESSION would pollute each other.
fn session() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0x5555);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[test]
fn filler_turns_never_trip_spinning() {
    // 8 injected filler turns: short pure-text assistant messages the
    // harness put in the history. None was engine-generated.
    let filler: Vec<Message> = (0..8)
        .map(|i| {
            text_turn(&format!("(ref #2a-2864-000{i}) [id:2a-2864-000{i}]"))
        })
        .collect();
    let mut messages = filler;
    messages.push(user_turn("Find the Q3 budget report and email the total."));

    let out = check_loops(&messages, true, 0, session());
    assert!(
        !out.suppress_tool_call,
        "filler alone must not suppress tool calls"
    );
    assert_eq!(
        out.tool_call_repeat_count, 0,
        "filler alone must not seed the bias decay"
    );
}

#[test]
fn engine_generated_short_turns_still_fire_the_soft_decay() {
    // 5 consecutive short engine-generated pure-text turns: a genuine
    // spin. The detector fires — but only as bias decay, never a mask.
    let s = session();
    for i in 0..5 {
        super::loop_detect::register_engine_turn(s, &format!("thinking… {i}"));
    }
    let messages: Vec<Message> = (0..5).map(|i| text_turn(&format!("thinking… {i}"))).collect();

    let out = check_loops(&messages, true, 0, s);
    assert!(
        !out.suppress_tool_call,
        "#55: spinning drives only the soft bias, never the hard mask"
    );
    assert!(
        out.tool_call_repeat_count >= 5,
        "spinning must drive the bias decay via tool_call_repeat_count, got {}",
        out.tool_call_repeat_count
    );
}

#[test]
fn a_foreign_turn_breaks_the_engine_turn_run() {
    let s = session();
    for i in 0..5 {
        super::loop_detect::register_engine_turn(s, &format!("spin {i}"));
    }
    let mut messages: Vec<Message> =
        (0..5).map(|i| text_turn(&format!("spin {i}"))).collect();
    // One filler turn the harness injected (never registered): breaks the run.
    messages.push(text_turn("(ref #2a) [id:2a] injected filler"));
    let out = check_loops(&messages, true, 0, s);
    assert!(
        !out.suppress_tool_call,
        "the unregistered newest turn must break the run"
    );
    assert_eq!(out.tool_call_repeat_count, 0);
}

#[test]
fn suppress_verdict_keeps_its_hard_mask() {
    // The Suppress verdict (signature-based, genuinely repeated calls)
    // keeps its P1-5 hard mask: three identical succeeding calls. Only
    // the spinning arm was softened.
    use crate::ir::message::ToolCall;
    let call = || {
        Message {
            role: Role::Assistant,
            content: vec![],
            tool_calls: vec![ToolCall {
                id: "c".into(),
                name: "search".into(),
                arguments: serde_json::json!({"query": "quarterly budget spreadsheet totals Q3 2026 revenue expenses office supplies travel"}),
            }],
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        }
    };
    let messages = vec![user_turn("x"), call(), call(), call()];
    let out = check_loops(&messages, true, 0, session());
    assert!(
        out.suppress_tool_call,
        "the Suppress verdict must keep its hard mask (repeated succeeding calls)"
    );
}
