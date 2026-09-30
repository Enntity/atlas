// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_DEBUG_SLOT0`: a debug-only switch for one acceptance
//! experiment, never a release default. Unset (`asis`) it adds no GPU
//! operation and changes nothing.
//!
//! The prompt capture writes every prefill pass's rows from accumulator slot
//! 0 on (`try_dflash_prefill_capture_layer`), so the tail pass of a split
//! prompt overwrites the head of the first pass — including the row of
//! prompt position 0, an outlier by norm. The variants tell "is that row in
//! the drafter's context" apart from "where do the newest rows sit". Both act
//! only on a window that starts at prompt position 0 (the prompt fits it).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::DflashProposerState;
use super::first_append::strict_env;

/// Strict startup switch: an unknown value is an error, never a default.
const DEBUG_SLOT0_ENV: &str = "ATLAS_DFLASH_DEBUG_SLOT0";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DebugSlot0 {
    /// Today's behaviour, byte for byte.
    #[default]
    AsIs,
    /// After prefill, a slot 0 stamped prompt position 0 is zero-filled.
    Zero,
    /// Once a pass has written prompt position 0 into slot 0, the later
    /// passes of that prefill skip their write to slot 0, and only that one.
    Keep,
}

impl DebugSlot0 {
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        Ok(match raw {
            None | Some("asis") => Self::AsIs,
            Some("zero") => Self::Zero,
            Some("keep") => Self::Keep,
            Some(other) => {
                anyhow::bail!("{DEBUG_SLOT0_ENV}={other:?} is not one of asis|zero|keep")
            }
        })
    }

    /// Startup only; checked on every rank before any weight load.
    pub fn from_env() -> Result<Self> {
        Self::parse(strict_env(DEBUG_SLOT0_ENV)?.as_deref())
    }

    /// Read at head construction and logged once.
    pub fn for_head() -> Result<Self> {
        let variant = Self::from_env()?;
        tracing::info!("DFlash debug slot 0: {variant:?} ({DEBUG_SLOT0_ENV}=asis|zero|keep)");
        Ok(variant)
    }
}

/// The per-sequence side of the switch (`DflashProposerState::slot0`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Slot0 {
    /// Frozen from the head at `alloc_state`.
    pub variant: DebugSlot0,
    /// `keep`: `Some(n)` once a pass of the running prefill has written
    /// prompt position 0 into slot 0; `n` writes to slot 0 were skipped since.
    pub kept: Option<usize>,
}

impl Slot0 {
    /// `keep`: how many leading rows (0 or 1) a prefill pass leaves unwritten.
    /// The pass is about to write prompt position `pos` into accumulator slot
    /// `slot`; `fits` says the whole prompt fits the window. Called once per
    /// capture layer, so every layer of a pass gets the same answer.
    pub(crate) fn keep_skip(&mut self, fits: bool, pos: usize, slot: usize) -> usize {
        if self.variant != DebugSlot0::Keep || !fits || slot != 0 {
            return 0;
        }
        if pos == 0 {
            self.kept.get_or_insert(0);
            return 0;
        }
        self.kept.as_mut().map_or(0, |skipped| {
            *skipped += 1;
            1
        })
    }
}

impl DflashProposerState {
    /// The end of a prefill, right after `seed_prefill_ctx`. `stream` is the
    /// one the capture ran on, so a zero-fill lands after the last pass's own
    /// writes. One log line per prefill unless the switch is unset.
    pub(crate) fn end_prefill_slot0(&mut self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let kept = self.slot0.kept.take();
        let stamp = self.ctx_positions.first().copied();
        match self.slot0.variant {
            DebugSlot0::AsIs => {}
            DebugSlot0::Zero => {
                let zeroed = stamp == Some(0);
                if zeroed {
                    gpu.memset_async(self.ctx_hidden_acc, 0, self.ctx_slot_bytes, stream)?;
                }
                tracing::info!("DFlash debug slot 0: zero, stamp {stamp:?}, zero-filled {zeroed}");
            }
            DebugSlot0::Keep => tracing::info!(
                "DFlash debug slot 0: keep, stamp {stamp:?}, position-0 row written {}, \
                 later writes skipped {}",
                kept.is_some(),
                kept.unwrap_or(0),
            ),
        }
        Ok(())
    }
}
