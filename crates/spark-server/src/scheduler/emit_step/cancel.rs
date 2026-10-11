// SPDX-License-Identifier: AGPL-3.0-only

//! Cooperative-cancellation retirement shared by the emit and decode paths.

use super::*;

/// Cooperative cancellation only marks retirement; lifecycle owns state cleanup.
/// An already-issued forward cannot be undone here. Preserve finish-reason and
/// hard-limit metadata rather than inventing a new cancellation reason.
///
/// Also the checkpoint for a BLOCKING request's client stop strings: the
/// sink's `StopStringWatch` scans the tokens emitted since the last call and
/// a hit retires the sequence exactly like a streaming stop-string cancel
/// (lifecycle derives "stop"; the handler truncates at the match). Streaming
/// requests match in the API layer and arrive here through `cancel_flag`.
pub(in crate::scheduler) fn retire_if_cancelled(a: &mut ActiveSeq) -> bool {
    let cancelled = a
        .cancel_flag
        .as_ref()
        .is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire))
        || a.sink.receiver_closed();
    // Not gated on `!a.finished`: like the cancel flag, a hit must keep
    // retiring at every later checkpoint of the same step (the non-spec
    // decode path rechecks after sampling and would otherwise commit the
    // row's placeholder token). The watch makes the hit sticky.
    let stop_string_hit = !cancelled
        && match &mut a.sink {
            ResponseSink::Blocking {
                stop_watch: Some(w),
                ..
            } => w.observe(&a.output_tokens, a.inside_thinking),
            _ => false,
        };
    if cancelled || stop_string_hit {
        a.finished = true;
        true
    } else {
        false
    }
}
