// SPDX-License-Identifier: AGPL-3.0-only

//! Client `stop` string matching — the single source of truth shared by
//! every path that honours a request's stop strings:
//!
//! - streaming hold-back (`chat_stream::handle_token::apply_stop_string_holdback`),
//! - blocking final truncation (`inference_impl::strip_stop_sequences_matched`),
//! - the scheduler-side early stop for blocking requests ([`StopStringWatch`],
//!   carried on `ResponseSink::Blocking`), which is what makes a non-streaming
//!   request actually STOP generating once a multi-token stop string appears
//!   instead of running to EOS / `max_tokens` and truncating afterwards.
//!
//! Deliberately free of `crate::` imports so the pure logic stays trivially
//! unit-testable.

use std::sync::Arc;

/// Find the client stop string that completes earliest in
/// `text[search_from..]`.
///
/// vLLM semantics (`vllm/v1/engine/detokenizer.py:397` `check_stop_strings`):
/// the stop string whose END is earliest wins — what token-by-token arrival
/// would have hit first — and ties on the same end are broken by stop-list
/// order. Returns `(byte start of the match in text, matched stop string)`;
/// the caller truncates at the start (vLLM default
/// `include_stop_str_in_output=False`).
///
/// Empty stop strings are ignored (vLLM rejects them at validation,
/// `sampling_params.py:763`; matching one would end every response at byte 0).
/// `search_from` is clamped and snapped down to a char boundary, so a
/// byte-offset window computed by the caller can never panic.
pub(crate) fn find_stop_string<'s>(
    text: &str,
    search_from: usize,
    stops: &'s [String],
) -> Option<(usize, &'s str)> {
    let from = text.floor_char_boundary(search_from.min(text.len()));
    let hay = &text[from..];
    let mut best: Option<(usize, usize, &'s str)> = None; // (start, end, stop)
    for s in stops.iter().filter(|s| !s.is_empty()) {
        // For a single needle the first occurrence is also the one that
        // ends first, so one `find` per stop string suffices.
        let Some(rel) = hay.find(s.as_str()) else {
            continue;
        };
        let start = from + rel;
        let end = start + s.len();
        // Strict `<`: on equal ends the earlier-listed stop keeps the win.
        if best.is_none_or(|(_, best_end, _)| end < best_end) {
            best = Some((start, end, s.as_str()));
        }
    }
    best.map(|(start, _, s)| (start, s))
}

/// Map a token id that ended the sequence to the client stop string it came
/// from. Single-token client stops are merged into the scheduler's EOS set
/// (`tokenize_stop_sequences`), so a hit ends the turn without the token ever
/// reaching the text matchers; this recovers the original string so the
/// Anthropic `stop_sequence` echo (and `matched_stop` in general) is set.
/// First-listed stop wins when several strings encode to the same id.
pub(crate) fn stop_string_for_token(single_token_stops: &[(u32, String)], id: u32) -> Option<&str> {
    single_token_stops
        .iter()
        .find(|(t, _)| *t == id)
        .map(|(_, s)| s.as_str())
}

/// The client stop string to report for a finished sequence: the text match
/// (earliest stop string found in the content) if any, else the single-token
/// client stop whose id ended the sequence (`stop_token_id` — the scheduler's
/// EOS-set terminal token, see `InferenceResponse::stop_token_id`). A model
/// EOS that is not a client stop maps to `None`.
pub(crate) fn resolve_matched_stop(
    text_match: Option<String>,
    stop_token_id: Option<u32>,
    single_token_stops: &[(u32, String)],
) -> Option<String> {
    text_match.or_else(|| {
        stop_token_id
            .and_then(|id| stop_string_for_token(single_token_stops, id))
            .map(str::to_string)
    })
}

/// Pure stop-string accumulator + hold-back algorithm. Returns the
/// bytes that should be forwarded to the client this delta; the
/// remainder (≤ `buffer_len` bytes) stays withheld inside
/// `accumulated_content` until the next call or until `handle_done`
/// flushes the tail at stream close.
///
/// Mirrors vLLM's `IncrementalDetokenizer.update`
/// (`vllm/v1/engine/detokenizer.py`):
/// 1. Append `new_chars` to the accumulator.
/// 2. Search the accumulator for any stop string.
/// 3a. On hit, truncate the accumulator AND the emittable delta at
///     the match position (Atlas never echoes the stop literal).
/// 3b. On miss, hold back the last `buffer_len` bytes; emit
///     everything between the previously emitted offset and the
///     hold-back boundary, snapped to a valid UTF-8 char boundary.
///
/// Pre/postconditions:
/// - `*triggered` must be `false` on entry (callers gate on this).
/// - On match, `*triggered` is flipped to `true` and the accumulator
///   is truncated to the prefix that precedes the stop string.
/// - On miss, `*triggered` stays `false`.
/// Returns `(emitted_text, matched_stop_string)`. When a stop string
/// matches, `matched_stop_string` is `Some(that string)` so the caller
/// can record it for the Anthropic `stop_sequence` echo.
pub(crate) fn apply_stop_string_holdback(
    new_chars: &str,
    stop_strings: &[String],
    buffer_len: usize,
    accumulated_content: &mut String,
    emitted_len: &mut usize,
    triggered: &mut bool,
) -> (String, Option<String>) {
    debug_assert!(!*triggered, "caller must gate on !triggered");
    accumulated_content.push_str(new_chars);

    // Bounded search window: only the suffix that could contain a stop
    // string straddling the newly appended chars can hold a *new* match —
    // every prior call already full-scanned (and found nothing in) the
    // content before this window. A match can begin at earliest
    // `max_stop_len - 1` bytes before the new chars; we also back up over
    // the held-back `buffer_len` bytes for margin. This keeps per-token
    // cost O(new + buffer + max_stop) instead of O(total), turning the
    // whole-response scan from O(n²) into O(n).
    let max_stop_len = stop_strings.iter().map(String::len).max().unwrap_or(0);
    let search_start = {
        let raw = accumulated_content
            .len()
            .saturating_sub(new_chars.len() + buffer_len + max_stop_len);
        accumulated_content.floor_char_boundary(raw)
    };
    // Earliest-COMPLETING stop string, ties by request order — the matcher
    // shared with the blocking path (`stop_match::find_stop_string`, vLLM
    // `check_stop_strings`), so both report the same `matched_stop`.
    let matched = find_stop_string(accumulated_content, search_start, stop_strings)
        .map(|(pos, s)| (pos, s.to_string()));

    if let Some((pos, stop_str)) = matched {
        accumulated_content.truncate(pos);
        let emit_start = (*emitted_len).min(pos);
        let out = accumulated_content[emit_start..pos].to_string();
        *emitted_len = pos;
        *triggered = true;
        return (out, Some(stop_str));
    }

    // No match: hold back the last `buffer_len` bytes. Snap to a UTF-8
    // boundary so the emitted prefix is always valid Rust `str` and
    // the held-back tail never contains a partial codepoint.
    let acc_len = accumulated_content.len();
    let raw_emit_end = acc_len.saturating_sub(buffer_len);
    let emit_end = accumulated_content.floor_char_boundary(raw_emit_end);
    let emit_start = (*emitted_len).min(emit_end);
    let out = accumulated_content[emit_start..emit_end].to_string();
    *emitted_len = emit_end;
    (out, None)
}

/// Owned hold-back state for a plain-text stream (legacy `/v1/completions`):
/// [`apply_stop_string_holdback`] per delta plus the finish-time tail flush.
pub(crate) struct StopHoldback {
    stops: Vec<String>,
    buffer_len: usize,
    acc: String,
    emitted: usize,
    triggered: bool,
}

impl StopHoldback {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        let buffer_len = stops
            .iter()
            .map(String::len)
            .max()
            .map_or(0, |m| m.saturating_sub(1));
        Self {
            stops,
            buffer_len,
            acc: String::new(),
            emitted: 0,
            triggered: false,
        }
    }

    /// A stop string has matched; later deltas must be dropped.
    pub(crate) fn triggered(&self) -> bool {
        self.triggered
    }

    /// Text of `delta` that may be sent now (empty while held back or once
    /// a match cut it). Pass-through when no stop strings are configured.
    pub(crate) fn push(&mut self, delta: &str) -> String {
        if self.stops.is_empty() {
            return delta.to_string();
        }
        if self.triggered {
            return String::new();
        }
        apply_stop_string_holdback(
            delta,
            &self.stops,
            self.buffer_len,
            &mut self.acc,
            &mut self.emitted,
            &mut self.triggered,
        )
        .0
    }

    /// Held-back tail to release at stream end (empty after a match).
    pub(crate) fn finish(&mut self) -> String {
        if self.triggered || self.emitted >= self.acc.len() {
            return String::new();
        }
        let tail = self.acc[self.emitted..].to_string();
        self.emitted = self.acc.len();
        tail
    }
}

/// Decoder used by [`StopStringWatch`] (token ids → text, special tokens
/// skipped — the same decode the blocking handlers apply to the final output).
pub(crate) type StopDecodeFn = dyn Fn(&[u32]) -> String + Send + Sync;

/// Extra tokens of look-back beyond the longest stop string's byte length.
/// Every non-special token decodes to at least one byte, so `max_stop_len`
/// tokens of look-back already cover any match ending in the new tokens; the
/// margin absorbs a split UTF-8 lead-in at the window start and a few
/// zero-width (skipped special) tokens.
const WATCH_LOOKBACK_MARGIN: usize = 8;

/// Scheduler-side incremental stop-string detector for BLOCKING requests.
///
/// Streaming requests detect stop strings in the API layer and flip the
/// sequence's `cancel_flag`; a blocking request has no per-token consumer, so
/// without this the scheduler kept generating past a multi-token stop. The
/// scheduler calls [`Self::observe`] at its cooperative-cancellation
/// checkpoints (`emit_step::retire_if_cancelled`) — the same points a
/// streaming cancel is honoured — and finishes the sequence on a hit. The
/// handler's final [`find_stop_string`] pass over the full decoded content
/// stays authoritative for the truncation offset and the matched string.
///
/// Matching is restricted to content (tokens outside a thinking block),
/// mirroring the streaming hold-back, which only runs in the content phase,
/// and vLLM's reasoning stop guard (stops dormant while `<think>` is open,
/// `vllm/v1/engine/detokenizer.py:206-217`).
///
/// Not mirrored (in either Atlas path): vLLM also keeps stops dormant for the
/// first `min_tokens` output tokens (`detokenizer.py:194-196`).
pub struct StopStringWatch {
    stops: Vec<String>,
    decode: Arc<StopDecodeFn>,
    /// `output_tokens` prefix length already scanned.
    scanned: usize,
    /// First output index eligible for matching (end of the latest
    /// observed thinking span).
    content_start: usize,
    lookback_tokens: usize,
    /// Sticky: once a stop string matched, every later checkpoint must keep
    /// reporting it (a decode step re-checks after sampling and must not
    /// commit the row's placeholder token).
    hit: bool,
}

impl StopStringWatch {
    /// `None` when there is nothing to watch (no non-empty stop strings).
    pub(crate) fn new(stops: &[String], decode: Arc<StopDecodeFn>) -> Option<Self> {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let max_len = stops.iter().map(String::len).max()?;
        Some(Self {
            stops,
            decode,
            scanned: 0,
            content_start: 0,
            lookback_tokens: max_len + WATCH_LOOKBACK_MARGIN,
            hit: false,
        })
    }

    /// Scan the tokens appended since the previous call. Returns `true` when a
    /// stop string appears in the content, i.e. generation should end — and
    /// keeps returning `true` on every later call (like a set cancel flag).
    ///
    /// `inside_thinking` is the sequence's current reasoning state: while it
    /// is set, tokens are skipped (and become the content floor). A shrinking
    /// `output_tokens` (rollback re-steer) is tolerated.
    pub(crate) fn observe(&mut self, output_tokens: &[u32], inside_thinking: bool) -> bool {
        if self.hit {
            return true;
        }
        let len = output_tokens.len();
        self.scanned = self.scanned.min(len);
        self.content_start = self.content_start.min(len);
        if inside_thinking {
            self.scanned = len;
            self.content_start = len;
            return false;
        }
        if len == self.scanned {
            return false;
        }
        let start = self
            .scanned
            .saturating_sub(self.lookback_tokens)
            .max(self.content_start);
        self.scanned = len;
        let text = (self.decode)(&output_tokens[start..]);
        self.hit = find_stop_string(&text, 0, &self.stops).is_some();
        self.hit
    }
}

#[cfg(test)]
mod tests;
