// SPDX-License-Identifier: AGPL-3.0-only

//! LoopCD for looped Inkling 2 checkpoints: "Decoding Looped Transformers
//! Better for (Almost) Free" (arXiv 2610.02185), LoopCD-Logits with the
//! adaptive rule, as oMLX serves it (jundot/omlx#5640, Apache-2.0:
//! `omlx/patches/inkling2/loopcd.py`).
//!
//! The first loop pass's state (its hidden output and the block stack it
//! wrote) also runs the coda layers, with its own KV and conv state, then the
//! final block residual, final norm and head. Both logit rows are softcapped,
//! and the served logits are `final + w * (final - reference)` with
//! `w = MAX_STRENGTH * (1 - (p1 - p2))` from the final row's top-two
//! probability margin. [`LoopCdMode::SkipThinking`] leaves rows inside a
//! thinking message unguided and keeps every row's control-token
//! probabilities.

use anyhow::{Result, bail};

/// The guidance weight at a zero top-two margin (the setting oMLX and the
/// paper's author use for Inkling 2 Mini).
pub const LOOPCD_MAX_STRENGTH: f32 = 0.5;
/// Inkling 2's first control token: ids at or past it are message structure.
pub const LOOPCD_FIRST_CONTROL_TOKEN: u32 = 199_998;
/// The control token that opens a thinking message.
pub const LOOPCD_THINKING_TOKEN: u32 = 200_008;

/// `--loopcd`: whether the server guides Inkling 2's logits with the first
/// loop pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopCdMode {
    /// Plain decoding: the engine runs exactly as without LoopCD.
    #[default]
    Off,
    /// Every row guided.
    On,
    /// Rows in a thinking message unguided; control-token probabilities kept.
    SkipThinking,
}

impl LoopCdMode {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "off" => Self::Off,
            "on" => Self::On,
            "skip-thinking" => Self::SkipThinking,
            other => bail!("--loopcd {other:?}: expected off, on or skip-thinking"),
        })
    }
}

/// Whether a row whose input is `token` sits in a thinking message, given
/// whether the row before did: the last control token decides.
pub fn loopcd_thinking_after(before: bool, token: u32) -> bool {
    if token >= LOOPCD_FIRST_CONTROL_TOKEN {
        token == LOOPCD_THINKING_TOKEN
    } else {
        before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_parse_and_refuse_typos() {
        assert_eq!(LoopCdMode::parse("off").unwrap(), LoopCdMode::Off);
        assert_eq!(LoopCdMode::parse("on").unwrap(), LoopCdMode::On);
        assert_eq!(
            LoopCdMode::parse("skip-thinking").unwrap(),
            LoopCdMode::SkipThinking
        );
        assert!(LoopCdMode::parse("yes").is_err());
        assert_eq!(LoopCdMode::default(), LoopCdMode::Off);
    }

    #[test]
    fn the_last_control_token_decides_thinking() {
        let mut thinking = false;
        let mut seen = Vec::new();
        for t in [
            5,
            LOOPCD_THINKING_TOKEN,
            7,
            8,
            LOOPCD_FIRST_CONTROL_TOKEN,
            9,
        ] {
            thinking = loopcd_thinking_after(thinking, t);
            seen.push(thinking);
        }
        assert_eq!(seen, [false, true, true, true, false, false]);
    }
}
