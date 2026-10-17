// SPDX-License-Identifier: AGPL-3.0-only

//! Client `stop` string matching — the single source of truth shared by
//! every path that honours a request's stop strings:
//!
//! - streaming hold-back ([`apply_stop_string_holdback`]: chat stream and
//!   legacy completions stream),
//! - blocking final truncation ([`truncate_at_stop`]),
//! - the scheduler-side early stop for blocking requests ([`StopStringWatch`],
//!   carried on `ResponseSink::Blocking`), which is what makes a non-streaming
//!   request actually STOP generating once a multi-token stop string appears
//!   instead of running to EOS / `max_tokens` and truncating afterwards.
//!
//! Deliberately free of `crate::` imports so the pure logic stays trivially
//! unit-testable.

/// Request-time validation of the client stop parameters, mirroring vLLM's
/// `SamplingParams._verify_args` (`vllm/sampling_params.py:729-763`): an
/// empty stop string is rejected (matching one would end every response at
/// byte 0), and so is `min_tokens > max_tokens`. Returns
/// `(offending parameter, message)` for the caller's 400 envelope.
pub(crate) fn validate_stop_params(
    stops: &[String],
    min_tokens: usize,
    max_tokens: usize,
) -> Result<(), (&'static str, String)> {
    if stops.iter().any(String::is_empty) {
        return Err(("stop", "stop cannot contain an empty string.".to_string()));
    }
    if min_tokens > max_tokens {
        return Err((
            "min_tokens",
            format!(
                "min_tokens must be less than or equal to max_tokens={max_tokens}, got {min_tokens}."
            ),
        ));
    }
    Ok(())
}

/// The request's stop-token set: single-token client stop strings
/// (`tokenize_stop_sequences`) plus the client `stop_token_ids`, sorted and
/// deduplicated (vLLM `dict.fromkeys`, `sampling_params.py:627`). The
/// scheduler merges this into the model EOS set, so a hit ends the sequence
/// exactly like EOS: the token is kept in `output_tokens` (counted in
/// `completion_tokens`), reported as `stop_token_id`, never streamed, and
/// held back by the `min_tokens` EOS suppression.
pub(crate) fn merge_stop_token_ids(mut tokens: Vec<u32>, stop_token_ids: &[u32]) -> Vec<u32> {
    tokens.extend_from_slice(stop_token_ids);
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// One client stop-string match: byte range in the searched text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StopHit<'s> {
    pub start: usize,
    pub end: usize,
    pub stop: &'s str,
}

impl StopHit<'_> {
    /// Where the caller truncates: before the match (vLLM default), or after
    /// it under `include_stop_str_in_output` (`detokenizer.py:441-447`).
    pub(crate) fn cut(&self, include_stop_str: bool) -> usize {
        if include_stop_str {
            self.end
        } else {
            self.start
        }
    }
}

/// Find the client stop string that completes earliest in
/// `text[search_from..]`, considering only matches that END past `floor`.
///
/// vLLM semantics (`vllm/v1/engine/detokenizer.py:397` `check_stop_strings`):
/// the stop string whose END is earliest wins — what token-by-token arrival
/// would have hit first — and ties on the same end are broken by stop-list
/// order.
///
/// `floor` is the byte length of the text produced by the first `min_tokens`
/// output tokens (0 = no floor): vLLM keeps stops dormant over that text by
/// advancing `stop_check_offset` past it (`detokenizer.py:194-196`), so a
/// match counts only if it ends after it — one straddling the floor does.
///
/// Empty stop strings are ignored (rejected at validation,
/// [`validate_stop_params`]; matching one would end every response at byte
/// 0). `search_from` is clamped and snapped down to a char boundary, so a
/// byte-offset window computed by the caller can never panic.
pub(crate) fn find_stop_hit<'s>(
    text: &str,
    search_from: usize,
    floor: usize,
    stops: &'s [String],
) -> Option<StopHit<'s>> {
    let from = text.floor_char_boundary(search_from.min(text.len()));
    let mut best: Option<StopHit<'s>> = None;
    for s in stops.iter().filter(|s| !s.is_empty()) {
        // First position a match ending past `floor` can start at. Snapping
        // down may admit an occurrence that ends at or before the floor;
        // step past it (only possible within the first stop-length bytes).
        let mut at = (floor + 1).saturating_sub(s.len()).max(from);
        at = text.floor_char_boundary(at.min(text.len()));
        let start = loop {
            let Some(rel) = text[at..].find(s.as_str()) else {
                break None;
            };
            let start = at + rel;
            if start + s.len() > floor {
                break Some(start);
            }
            at = start + s.chars().next().map_or(1, char::len_utf8);
        };
        let Some(start) = start else {
            continue;
        };
        let end = start + s.len();
        // Strict `<`: on equal ends the earlier-listed stop keeps the win.
        if best.is_none_or(|b| end < b.end) {
            best = Some(StopHit {
                start,
                end,
                stop: s.as_str(),
            });
        }
    }
    best
}

/// [`find_stop_hit`] without a floor, as `(match start, matched stop)`.
pub(crate) fn find_stop_string<'s>(
    text: &str,
    search_from: usize,
    stops: &'s [String],
) -> Option<(usize, &'s str)> {
    find_stop_hit(text, search_from, 0, stops).map(|h| (h.start, h.stop))
}

/// Blocking final truncation: cut `text` at the earliest client stop string
/// ending past `floor` ([`find_stop_hit`]) — before it, or after it under
/// `include_stop_str_in_output`. Searches the WHOLE text: a blocking request
/// stops only at the next scheduler checkpoint after the stop string appears
/// ([`StopStringWatch`]), and an EOS / `max_tokens` finish may contain one
/// mid-text. Returns the text and the matched stop string.
pub(crate) fn truncate_at_stop(
    mut text: String,
    stops: &[String],
    floor: usize,
    include_stop_str: bool,
) -> (String, Option<String>) {
    match find_stop_hit(&text, 0, floor, stops) {
        Some(hit) => {
            let stop = hit.stop.to_string();
            text.truncate(hit.cut(include_stop_str));
            (text, Some(stop))
        }
        None => (text, None),
    }
}

/// Map a `min_tokens` floor measured on the full decoded output (`full_len`
/// bytes, of which the first `floor_full` come from the first `min_tokens`
/// tokens) onto a post-processed `content` string of `content_len` bytes.
/// Content (reasoning split off, leading whitespace trimmed) is a suffix of
/// the full decode, so the floor keeps its distance from the END. A floor
/// that falls inside the reasoning maps to 0. Approximate only where content
/// post-processing rewrote bytes after the floor (think-marker scrubbing).
pub(crate) fn content_floor(content_len: usize, full_len: usize, floor_full: usize) -> usize {
    content_len.saturating_sub(full_len.saturating_sub(floor_full))
}

/// The client stop token whose text belongs in the output: under
/// `include_stop_str_in_output` vLLM detokenizes the stop-terminating token
/// (`detokenizer.py:178-185`; special tokens still decode to nothing). Atlas
/// does so for CLIENT stops only — a single-token stop string or a
/// `stop_token_ids` entry, i.e. `client_stop_tokens` — and never for a model
/// EOS, whose text (e.g. a non-special `</assistant>`) is template markup.
pub(crate) fn included_stop_token(
    include_stop_str: bool,
    stop_token_id: Option<u32>,
    client_stop_tokens: &[u32],
) -> Option<u32> {
    stop_token_id.filter(|id| include_stop_str && client_stop_tokens.contains(id))
}

/// Context tokens [`token_text_after`] decodes before the stop token.
const STOP_TEXT_CONTEXT: usize = 4;

/// The text `id` adds after `context` (the tokens generated before it),
/// decoded in context as the full-output decode would render it: a lone
/// SentencePiece word-start piece drops its leading space, and a byte-level
/// piece completing a UTF-8 sequence decodes to U+FFFD on its own. Falls
/// back to the lone decode when the context decode is not a prefix.
pub(crate) fn token_text_after(
    decode: impl Fn(&[u32]) -> String,
    context: &[u32],
    id: u32,
) -> String {
    let ctx = &context[context.len().saturating_sub(STOP_TEXT_CONTEXT)..];
    let before = decode(ctx);
    let mut with = ctx.to_vec();
    with.push(id);
    match decode(&with).strip_prefix(before.as_str()) {
        Some(text) => text.to_string(),
        None => decode(&[id]),
    }
}

/// Bytes a stream withholds so a stop string split across deltas never leaks
/// its prefix: the longest stop minus one, and none when the stop string is
/// returned anyway (vLLM `stop_buffer_length`, `detokenizer.py:154-157`).
pub(crate) fn holdback_len(stops: &[String], include_stop_str: bool) -> usize {
    if include_stop_str {
        return 0;
    }
    stops
        .iter()
        .map(String::len)
        .max()
        .map_or(0, |m| m.saturating_sub(1))
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

/// Per-request inputs of [`apply_stop_string_holdback`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct HoldbackParams<'a> {
    pub stops: &'a [String],
    /// Bytes withheld on a miss ([`holdback_len`]).
    pub buffer_len: usize,
    /// vLLM `include_stop_str_in_output`: emit through the END of a match.
    pub include_stop_str: bool,
}

/// Mutable hold-back cursor of one stream.
#[derive(Debug, Default)]
pub(crate) struct HoldbackState {
    /// Content seen so far (truncated at the cut on a match).
    pub acc: String,
    /// `acc` prefix already sent to the client.
    pub emitted: usize,
    /// `acc` prefix produced by the first `min_tokens` tokens: no stop
    /// string may END inside it ([`find_stop_hit`]).
    pub floor: usize,
    /// A stop string matched; later deltas are dropped by the caller.
    pub triggered: bool,
}

/// Pure stop-string accumulator + hold-back algorithm. Returns the
/// bytes that should be forwarded to the client this delta; the
/// remainder (≤ `buffer_len` bytes) stays withheld inside `st.acc`
/// until the next call or until the stream's finish flushes the tail.
///
/// Mirrors vLLM's `IncrementalDetokenizer.update`
/// (`vllm/v1/engine/detokenizer.py`):
/// 1. Append `new_chars` to the accumulator. When the token that produced
///    them is within the first `min_tokens` (`below_min_tokens`), raise the
///    floor to the new end: stops stay dormant over that text.
/// 2. Search the accumulator for a stop string ending past the floor.
/// 3a. On hit, truncate the accumulator AND the emittable delta at the
///     match start — or its end under `include_stop_str_in_output`.
/// 3b. On miss, hold back the last `buffer_len` bytes; emit
///     everything between the previously emitted offset and the
///     hold-back boundary, snapped to a valid UTF-8 char boundary.
///
/// `st.triggered` must be `false` on entry (callers gate on it); a match
/// flips it. Returns `(emitted_text, matched_stop_string)`, the latter for
/// the Anthropic `stop_sequence` echo.
pub(crate) fn apply_stop_string_holdback(
    new_chars: &str,
    below_min_tokens: bool,
    p: HoldbackParams<'_>,
    st: &mut HoldbackState,
) -> (String, Option<String>) {
    debug_assert!(!st.triggered, "caller must gate on !triggered");
    st.acc.push_str(new_chars);
    if below_min_tokens {
        st.floor = st.acc.len();
    }

    // Bounded search window: only the suffix that could contain a stop
    // string straddling the newly appended chars can hold a *new* match —
    // every prior call already full-scanned (and found nothing in) the
    // content before this window. A match can begin at earliest
    // `max_stop_len - 1` bytes before the new chars; we also back up over
    // the held-back `buffer_len` bytes for margin. This keeps per-token
    // cost O(new + buffer + max_stop) instead of O(total), turning the
    // whole-response scan from O(n²) into O(n).
    let max_stop_len = p.stops.iter().map(String::len).max().unwrap_or(0);
    let search_start = {
        let raw = st
            .acc
            .len()
            .saturating_sub(new_chars.len() + p.buffer_len + max_stop_len);
        st.acc.floor_char_boundary(raw)
    };
    // Earliest-COMPLETING stop string, ties by request order — the matcher
    // shared with the blocking path ([`truncate_at_stop`], vLLM
    // `check_stop_strings`), so both report the same `matched_stop`.
    let matched = find_stop_hit(&st.acc, search_start, st.floor, p.stops)
        .map(|hit| (hit.cut(p.include_stop_str), hit.stop.to_string()));

    if let Some((cut, stop_str)) = matched {
        st.acc.truncate(cut);
        let emit_start = st.emitted.min(cut);
        let out = st.acc[emit_start..cut].to_string();
        st.emitted = cut;
        st.triggered = true;
        return (out, Some(stop_str));
    }

    // No match: hold back the last `buffer_len` bytes. Snap to a UTF-8
    // boundary so the emitted prefix is always valid Rust `str` and
    // the held-back tail never contains a partial codepoint.
    let raw_emit_end = st.acc.len().saturating_sub(p.buffer_len);
    let emit_end = st.acc.floor_char_boundary(raw_emit_end);
    let emit_start = st.emitted.min(emit_end);
    let out = st.acc[emit_start..emit_end].to_string();
    st.emitted = emit_end;
    (out, None)
}

/// Owned hold-back state for a plain-text stream (legacy `/v1/completions`):
/// [`apply_stop_string_holdback`] per token delta plus the finish-time tail
/// flush.
pub(crate) struct StopHoldback {
    stops: Vec<String>,
    buffer_len: usize,
    include_stop_str: bool,
    min_tokens: usize,
    /// Token deltas pushed so far (one [`Self::push`] per sampled token).
    tokens: usize,
    st: HoldbackState,
}

impl StopHoldback {
    pub(crate) fn new(stops: Vec<String>, min_tokens: usize, include_stop_str: bool) -> Self {
        Self {
            buffer_len: holdback_len(&stops, include_stop_str),
            stops,
            include_stop_str,
            min_tokens,
            tokens: 0,
            st: HoldbackState::default(),
        }
    }

    /// A stop string has matched; later deltas must be dropped.
    pub(crate) fn triggered(&self) -> bool {
        self.st.triggered
    }

    /// Text of one sampled token's `delta` that may be sent now (empty while
    /// held back or once a match cut it). Pass-through when no stop strings
    /// are configured. Call once per token, even for an empty delta: the
    /// call count drives the `min_tokens` floor.
    pub(crate) fn push(&mut self, delta: &str) -> String {
        self.tokens += 1;
        if self.stops.is_empty() {
            return delta.to_string();
        }
        if self.st.triggered {
            return String::new();
        }
        let p = HoldbackParams {
            stops: &self.stops,
            buffer_len: self.buffer_len,
            include_stop_str: self.include_stop_str,
        };
        apply_stop_string_holdback(delta, self.tokens <= self.min_tokens, p, &mut self.st).0
    }

    /// Held-back tail to release at stream end (empty after a match).
    pub(crate) fn finish(&mut self) -> String {
        if self.st.triggered || self.st.emitted >= self.st.acc.len() {
            return String::new();
        }
        let tail = self.st.acc[self.st.emitted..].to_string();
        self.st.emitted = self.st.acc.len();
        tail
    }
}

mod watch;
#[cfg(test)]
pub(crate) use watch::StopDecodeFn;
pub use watch::StopStringWatch;

#[cfg(test)]
mod tests;
