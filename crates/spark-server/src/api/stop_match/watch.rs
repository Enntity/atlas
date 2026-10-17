// SPDX-License-Identifier: AGPL-3.0-only

//! [`StopStringWatch`]: the scheduler-side stop-string detector for blocking
//! requests (split from `stop_match` for the 500-LoC cap). Same matcher as
//! every other stop path ([`super::find_stop_hit`]).

use std::sync::Arc;

use super::find_stop_hit;

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
/// handler's final [`super::truncate_at_stop`] pass over the full decoded content
/// stays authoritative for the truncation offset and the matched string.
///
/// Matching is restricted to content (tokens outside a thinking block),
/// mirroring the streaming hold-back, which only runs in the content phase,
/// and vLLM's reasoning stop guard (stops dormant while `<think>` is open,
/// `vllm/v1/engine/detokenizer.py:206-217`).
///
/// Like vLLM, stops stay dormant over the first `min_tokens` output tokens
/// (`detokenizer.py:194-196`): a match must end past their text.
pub struct StopStringWatch {
    stops: Vec<String>,
    decode: Arc<StopDecodeFn>,
    min_tokens: usize,
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
    pub(crate) fn new(
        stops: &[String],
        min_tokens: usize,
        decode: Arc<StopDecodeFn>,
    ) -> Option<Self> {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        if stops.is_empty() {
            tracing::trace!(target: "stop_match", "StopStringWatch: no non-empty stop strings, returning None");
            return None;
        }
        let max_len = stops.iter().map(String::len).max()?;
        tracing::debug!(target: "stop_match", "StopStringWatch: new stops={:?} min_tokens={} max_len={} lookback_tokens={}", stops, min_tokens, max_len, max_len + WATCH_LOOKBACK_MARGIN);
        Some(Self {
            stops,
            decode,
            min_tokens,
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
            tracing::trace!(target: "stop_match", "observe: already hit, returning true");
            return true;
        }
        let len = output_tokens.len();
        tracing::trace!(target: "stop_match", "observe: len={} scanned={} content_start={} inside_thinking={} min_tokens={}", len, self.scanned, self.content_start, inside_thinking, self.min_tokens);
        self.scanned = self.scanned.min(len);
        self.content_start = self.content_start.min(len);
        if inside_thinking {
            tracing::trace!(target: "stop_match", "observe: inside_thinking=true, skipping content");
            self.scanned = len;
            self.content_start = len;
            return false;
        }
        if len == self.scanned {
            tracing::trace!(target: "stop_match", "observe: no new tokens");
            return false;
        }
        if len <= self.min_tokens {
            // Every token so far is within `min_tokens`: nothing can match.
            tracing::trace!(target: "stop_match", "observe: len <= min_tokens, skipping");
            self.scanned = len;
            return false;
        }
        let start = self
            .scanned
            .saturating_sub(self.lookback_tokens)
            .max(self.content_start);
        self.scanned = len;
        let text = (self.decode)(&output_tokens[start..]);
        let floor = if self.min_tokens > start {
            (self.decode)(&output_tokens[start..self.min_tokens]).len()
        } else {
            0
        };
        let hit = find_stop_hit(&text, 0, floor, &self.stops).is_some();
        self.hit = hit;
        tracing::debug!(target: "stop_match", "observe: start={} text_len={} floor={} hit={} text={:?}", start, text.len(), floor, hit, text);
        self.hit
    }
}
