// SPDX-License-Identifier: AGPL-3.0-only

//! LoopCD `skip-thinking`: whether each row of a forward sits in a thinking
//! message (oMLX `loopcd.thinking_flags`, jundot/omlx#5640, Apache-2.0). A
//! row's flag follows the last control token among the sequence's tokens up
//! to and including the row's input. The `Model` entry points know both: the
//! sequence's history (`SequenceState::tokens`) and the forward's input
//! tokens, back to back in row order. Each entry stamps its rows before the
//! forward, outside any graph capture, with a blocking upload (the previous
//! forward's head has finished: its entry returned what it sampled); the next
//! final norm takes the stamp, and a head without one is guided throughout
//! (`loopcd.rs` says so once).
//!
//! A prefill samples only its last row, so every row of a prompt's forward
//! carries the flag after the chunk's last token.

use std::sync::atomic::Ordering;

use anyhow::Result;
use atlas_core::config::{LOOPCD_FIRST_CONTROL_TOKEN, loopcd_thinking_after};

use super::types::TransformerModel;
use crate::traits::SequenceState;

/// Whether the tokens so far leave the sequence in a thinking message.
pub(crate) fn thinking_after(history: &[u32]) -> bool {
    history
        .iter()
        .rev()
        .find(|&&t| t >= LOOPCD_FIRST_CONTROL_TOKEN)
        .is_some_and(|&t| loopcd_thinking_after(false, t))
}

/// The flags of rows laid out back to back: per run, its sequence's history
/// and its rows' input tokens. Rows past the runs (graph padding) are 0.
pub(crate) fn row_flags(runs: &[(&[u32], &[u32])], rows: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows];
    let mut row = 0;
    for (history, inputs) in runs {
        let mut thinking = thinking_after(history);
        for &t in *inputs {
            thinking = loopcd_thinking_after(thinking, t);
            if let Some(f) = out.get_mut(row) {
                *f = u8::from(thinking);
            }
            row += 1;
        }
    }
    out
}

impl TransformerModel {
    fn loopcd_skips_thinking(&self) -> bool {
        self.loopcd
            .as_deref()
            .is_some_and(|lc| lc.flags().is_some())
    }

    fn loopcd_upload_flags(&self, flags: &[u8]) -> Result<()> {
        let Some(lc) = self.loopcd.as_deref() else {
            return Ok(());
        };
        let Some((dev, cap)) = lc.flags() else {
            return Ok(());
        };
        let n = flags.len().min(cap);
        self.gpu.copy_h2d(&flags[..n], dev)?;
        lc.stamped.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// One sequence's rows: `inputs` after its history.
    pub(super) fn loopcd_stamp_seq(&self, seq: &SequenceState, inputs: &[u32]) -> Result<()> {
        if !self.loopcd_skips_thinking() {
            return Ok(());
        }
        let flags = row_flags(&[(&seq.tokens, inputs)], inputs.len());
        self.loopcd_upload_flags(&flags)
    }

    /// Several sequences' rows back to back: `ks[i]` input tokens each.
    pub(super) fn loopcd_stamp_batch(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &[&mut SequenceState],
    ) -> Result<()> {
        if !self.loopcd_skips_thinking() {
            return Ok(());
        }
        let mut runs = Vec::with_capacity(seqs.len());
        let mut at = 0;
        for (seq, &k) in seqs.iter().zip(ks) {
            let end = (at + k).min(tokens.len());
            runs.push((seq.tokens.as_slice(), &tokens[at..end]));
            at = end;
        }
        let rows = crate::layers::inkling2::layer::MAX_VERIFY_ROWS * seqs.len().max(1);
        self.loopcd_upload_flags(&row_flags(&runs, rows.max(at)))
    }

    /// A prompt's forward up to `end` tokens: every row the last one's flag.
    pub(super) fn loopcd_stamp_prompt(&self, prompt: &[u32], end: usize) -> Result<()> {
        if !self.loopcd_skips_thinking() {
            return Ok(());
        }
        let Some((_, cap)) = self.loopcd.as_deref().and_then(|lc| lc.flags()) else {
            return Ok(());
        };
        let flag = u8::from(thinking_after(&prompt[..end.min(prompt.len())]));
        self.loopcd_upload_flags(&vec![flag; cap])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::LOOPCD_THINKING_TOKEN as THINK;

    const END: u32 = LOOPCD_FIRST_CONTROL_TOKEN;

    #[test]
    fn history_decides_until_a_row_brings_a_control_token() {
        assert!(!thinking_after(&[1, 2, 3]));
        assert!(thinking_after(&[END, THINK, 5, 6]));
        assert!(!thinking_after(&[THINK, 5, END, 6]));
        let flags = row_flags(&[(&[THINK, 4], &[5, END, 6]), (&[1], &[THINK])], 6);
        assert_eq!(flags, [1, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn a_row_repeating_the_last_committed_token_changes_nothing() {
        // a verify's first input may already be the history's last token
        for history in [&[THINK][..], &[END][..], &[THINK, 7][..]] {
            let last = *history.last().unwrap();
            let once = row_flags(&[(&history[..history.len() - 1], &[last])], 1);
            let twice = row_flags(&[(history, &[last])], 1);
            assert_eq!(once, twice);
        }
    }
}
