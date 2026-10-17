// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence thinking-budget accounting for the F4 speculative lane.
//!
//! [`ThinkBudgetState`] is a point-in-time snapshot of the serial decode
//! path's per-sequence thinking accounting, taken by the scheduler where a
//! speculative step is dispatched. The speculative commit replays it over
//! the candidate emission stream to find how many leading tokens the serial
//! governor would emit as plain argmaxes, and truncates its committed prefix
//! there (`glm_exl3_dflash::spec`); the truncated tail is emitted by the next
//! — serial — step exactly as a serial decode would, so the forced `</think>`
//! and its sentence-boundary deferral stay entirely in the serial governor.
//!
//! Mirrored semantics (SSOTs, both in spark-server):
//! - counting + lazy arming: the one post-pick commit rule
//!   (`scheduler/emit_step/think_commit.rs::advance_thinking`, run by
//!   `emit_token` and by serial decode alike) increments
//!   `thinking_tokens` once per counted thinking token and arms
//!   `force_end_thinking` when the count reaches `thinking_budget`, AFTER
//!   that token is emitted. So the token that reaches the budget is a plain
//!   argmax; the NEXT token is governed.
//! - the governed position:
//!   `scheduler/logit_processors/forced_think_end.rs` runs before sampling
//!   while `inside_thinking && force_end_thinking` and may replace the token
//!   with `think_end_token`, so a raw-argmax speculative step must not pick
//!   a token there. A `think_end_token` at a governed position is exempt:
//!   the injector forces exactly that token, so it is always safe to keep.
//! - spontaneous `think_start` entry: `think_commit::think_gate` sets
//!   `thinking_budget = Some((spontaneous_think_budget >> fires.min(4)).max(8))`
//!   without resetting `thinking_tokens`.
//!
//! Conservative deviations (both fail toward EARLIER truncation, never past
//! the budget): a native GLM EOS-during-thinking
//! (`glm_tool_boundary::native_eos_while_thinking`, which is not counted) and
//! the GLM native tool opener
//! (`first_token_thinking::apply_native_tool_boundary`, which exits thinking)
//! are treated as ordinary thinking tokens here. Each can only shorten the
//! kept prefix; the dropped tail is re-drafted on the next step.

use anyhow::{Result, ensure};

/// Sentinel for an absent `Option<u32>` in the fixed-width wire encoding.
const NONE_WORD: u32 = u32::MAX;

/// Snapshot of the per-sequence thinking-budget accounting one speculative
/// step must mirror at commit time. Copied onto the EP wire verbatim so both
/// TP ranks truncate the commit identically (rank symmetry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkBudgetState {
    /// `ActiveSeq::inside_thinking`.
    pub inside_thinking: bool,
    /// `ActiveSeq::thinking_tokens` (cumulative, never reset).
    pub thinking_tokens: u32,
    /// `ActiveSeq::thinking_budget` — `None` = unlimited (never arms).
    pub thinking_budget: Option<u32>,
    /// `ActiveSeq::force_end_thinking` (true = the step is already governed
    /// and must not speculate at all).
    pub force_end_thinking: bool,
    /// `ActiveSeq::spontaneous_think_budget` (undecayed; the decay is
    /// applied on entry from `think_watchdog_fires`, mirroring `emit_token`).
    pub spontaneous_think_budget: u32,
    /// `ActiveSeq::think_watchdog_fires`.
    pub think_watchdog_fires: u32,
    /// `ActiveSeq::think_start_token`.
    pub think_start_token: Option<u32>,
    /// `ActiveSeq::think_end_token`.
    pub think_end_token: Option<u32>,
    /// Structural picks are handed to serial sampling before any commit.
    pub tool_call_start_token: Option<u32>,
}

impl ThinkBudgetState {
    /// Fixed-width wire encoding length (`to_words`/`from_words`).
    pub const WORDS: usize = 9;

    /// Encode for the EP per-step broadcast (head → workers).
    pub fn to_words(self) -> [u32; Self::WORDS] {
        [
            u32::from(self.inside_thinking),
            self.thinking_tokens,
            self.thinking_budget.unwrap_or(NONE_WORD),
            u32::from(self.force_end_thinking),
            self.spontaneous_think_budget,
            self.think_watchdog_fires,
            self.think_start_token.unwrap_or(NONE_WORD),
            self.think_end_token.unwrap_or(NONE_WORD),
            self.tool_call_start_token.unwrap_or(NONE_WORD),
        ]
    }

    /// Decode the EP per-step broadcast. Fails fast on a word the head could
    /// not have sent (PCND: no implicit default for a corrupt wire value).
    pub fn from_words(words: [u32; Self::WORDS]) -> Result<Self> {
        let [
            inside,
            count,
            budget,
            armed,
            spontaneous,
            fires,
            start,
            end,
            tool,
        ] = words;
        ensure!(
            inside <= 1 && armed <= 1,
            "think budget wire: invalid bool words {inside} {armed}"
        );
        Ok(Self {
            inside_thinking: inside == 1,
            thinking_tokens: count,
            thinking_budget: (budget != NONE_WORD).then_some(budget),
            force_end_thinking: armed == 1,
            spontaneous_think_budget: spontaneous,
            think_watchdog_fires: fires,
            think_start_token: (start != NONE_WORD).then_some(start),
            think_end_token: (end != NONE_WORD).then_some(end),
            tool_call_start_token: (tool != NONE_WORD).then_some(tool),
        })
    }

    /// Prefix whose raw argmax picks need no structural-token processor.
    /// Stop before every thinking/tool boundary, even a natural close: serial
    /// sampling owns its masks and transitions. This also prevents a later
    /// row in the same verify block from reopening freshly closed thinking.
    pub fn processor_free_emission_len(&self, stream: &[u32]) -> usize {
        let structural = stream
            .iter()
            .position(|&tok| {
                [
                    self.think_start_token,
                    self.think_end_token,
                    self.tool_call_start_token,
                ]
                .contains(&Some(tok))
            })
            .unwrap_or(stream.len());
        self.plain_emission_len(&stream[..structural])
    }

    /// Number of leading `stream` tokens the serial governor would emit as
    /// plain argmaxes — the longest prefix a speculative step may commit.
    ///
    /// A token is kept iff, at its sampling position, the forced-`</think>`
    /// injector would not act (see the module docs); the per-token
    /// accounting mirrors `emit_token`'s think branch exactly.
    pub fn plain_emission_len(&self, stream: &[u32]) -> usize {
        let mut inside = self.inside_thinking;
        let mut count = self.thinking_tokens;
        let mut budget = self.thinking_budget;
        let mut armed = self.force_end_thinking;
        for (i, &tok) in stream.iter().enumerate() {
            let end = self.think_end_token == Some(tok);
            // Governed position: the injector may replace this token with
            // `think_end_token`. Keeping a natural `think_end_token` is safe
            // (the injector forces exactly that token).
            if inside && armed && !end {
                return i;
            }
            if !inside && self.think_start_token == Some(tok) {
                // Spontaneous entry (emit_token): new budget, count kept.
                inside = true;
                let decay_shift = self.think_watchdog_fires.min(4);
                budget = Some((self.spontaneous_think_budget >> decay_shift).max(8));
            } else if !inside && end {
                // Stray `think_end_token` outside thinking: skipped.
            } else if inside && end {
                inside = false;
                armed = false;
            } else if inside {
                count = count.saturating_add(1);
                if let Some(b) = budget
                    && count >= b
                    && !armed
                {
                    armed = true;
                }
            }
        }
        stream.len()
    }
}

#[cfg(test)]
mod tests {
    use super::ThinkBudgetState;

    const START: u32 = 100;
    const END: u32 = 101;

    fn inside(count: u32, budget: u32) -> ThinkBudgetState {
        ThinkBudgetState {
            inside_thinking: true,
            thinking_tokens: count,
            thinking_budget: Some(budget),
            force_end_thinking: false,
            spontaneous_think_budget: 256,
            think_watchdog_fires: 0,
            think_start_token: Some(START),
            think_end_token: Some(END),
            tool_call_start_token: None,
        }
    }

    fn outside(count: u32) -> ThinkBudgetState {
        ThinkBudgetState {
            inside_thinking: false,
            thinking_budget: None,
            ..inside(count, 256)
        }
    }

    #[test]
    fn structural_picks_must_be_left_to_serial_processors() {
        let mut s = outside(0);
        s.tool_call_start_token = Some(102);
        for special in [START, END, 102] {
            assert_eq!(s.processor_free_emission_len(&[2, special, 3]), 1);
            assert_eq!(s.processor_free_emission_len(&[special, 3]), 0);
        }
        assert_eq!(s.processor_free_emission_len(&[2, 3, 4]), 3);
    }

    #[test]
    fn thinking_end_boundary_and_budget_both_limit_the_prefix() {
        let s = inside(2, 4);
        assert_eq!(s.processor_free_emission_len(&[2, END, START, 3]), 1);
        assert_eq!(s.processor_free_emission_len(&[2, 3, 4, END]), 2);
    }

    #[test]
    fn full_stream_kept_below_the_budget() {
        assert_eq!(
            inside(0, 16).plain_emission_len(&[2, 3, 4, 5, 6, 7, 8, 9]),
            8
        );
    }

    #[test]
    fn truncates_after_the_token_that_arms_the_budget() {
        // count 2 of budget 4: two thinking tokens fit; the third would be
        // the first governed position (the second token arms the budget).
        assert_eq!(
            inside(2, 4).plain_emission_len(&[2, 3, 4, 5, 6, 7, 8, 9]),
            2
        );
    }

    #[test]
    fn forced_draft_dropping_the_bonus_at_the_boundary() {
        // k=7 step: 8 candidate emissions crossing budget 4 from count 3.
        // Only the trip token (count 3 -> 4) is plain; the bonus is dropped.
        let draft = [2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(inside(3, 4).plain_emission_len(&draft), 1);
    }

    #[test]
    fn natural_think_end_keeps_the_content_tail() {
        // The budget arms at the second token, but the third token is a
        // natural `think_end` (exempt) which exits thinking; the content
        // tail is ungoverned and kept whole.
        let stream = [2, 3, END, 7, 8, 9];
        assert_eq!(inside(2, 4).plain_emission_len(&stream), 6);
    }

    #[test]
    fn governed_think_end_is_kept() {
        // The token at the first governed position IS `think_end`: the
        // injector forces exactly it, so it is kept and thinking exits.
        let stream = [2, END, 7, 8];
        assert_eq!(inside(3, 4).plain_emission_len(&stream), 4);
    }

    #[test]
    fn stray_think_end_outside_thinking_is_skipped_not_counted() {
        let stream = [END, 2, 3];
        assert_eq!(outside(0).plain_emission_len(&stream), 3);
    }

    #[test]
    fn spontaneous_entry_keeps_the_arming_trip_token() {
        // Cumulative count 10 with the spontaneous budget 8 already passed:
        // entry is plain, the first thinking token trips the arming after
        // being emitted, the next is governed.
        let mut s = outside(10);
        s.spontaneous_think_budget = 8;
        let stream = [5, START, 20, 21];
        assert_eq!(s.plain_emission_len(&stream), 3);
    }

    #[test]
    fn spontaneous_entry_with_headroom() {
        // Entry with cumulative count 5 and budget 8: three thinking tokens
        // fit after `think_start` (counts 6..8, the third trips the arming);
        // the fifth stream token is governed.
        let mut s = outside(5);
        s.spontaneous_think_budget = 8;
        let stream = [START, 20, 21, 22, 23];
        assert_eq!(s.plain_emission_len(&stream), 4);
    }

    #[test]
    fn watchdog_decay_tightens_the_spontaneous_budget() {
        // fires=1 -> budget 256>>1 = 128 on entry (emit_token twin). From
        // count 127 the first thinking token arms; the next is governed.
        let mut s = outside(127);
        s.think_watchdog_fires = 1;
        let stream = [START, 20, 21];
        assert_eq!(s.plain_emission_len(&stream), 2);
    }

    #[test]
    fn already_armed_governs_the_first_token() {
        let mut s = inside(4, 4);
        s.force_end_thinking = true;
        assert_eq!(s.plain_emission_len(&[2, 3]), 0);
    }

    #[test]
    fn unlimited_budget_never_governs() {
        let mut s = inside(1_000_000, 4);
        s.thinking_budget = None;
        assert_eq!(s.plain_emission_len(&[2, 3, 4, 5]), 4);
    }

    #[test]
    fn content_stream_is_unbounded() {
        assert_eq!(outside(0).plain_emission_len(&[2, 3, 4, 5, 6]), 5);
    }

    #[test]
    fn word_round_trip() {
        let mut s = inside(7, 12);
        s.tool_call_start_token = Some(102);
        assert_eq!(ThinkBudgetState::from_words(s.to_words()).unwrap(), s);
        let none = ThinkBudgetState {
            thinking_budget: None,
            think_start_token: None,
            think_end_token: None,
            ..s
        };
        assert_eq!(ThinkBudgetState::from_words(none.to_words()).unwrap(), none);
    }

    #[test]
    fn from_words_rejects_invalid_bools() {
        let mut words = inside(0, 4).to_words();
        words[0] = 2;
        assert!(ThinkBudgetState::from_words(words).is_err());
    }
}
