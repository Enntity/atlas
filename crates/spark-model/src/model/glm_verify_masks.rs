// SPDX-License-Identifier: AGPL-3.0-only
//! Per-row grammar bitmasks for a strict structured-output GLM verify
//! (`ATLAS_GLM_STRICT_SPEC=1`).
//!
//! The head walks the grammar matcher through the drafted tokens and builds
//! one bitmask per verify row (vLLM #14702, "Enable Speculative Decoding with
//! Structured Outputs"; #44297 for the reasoning boundary). On the TP2 vocab
//! split each rank takes the argmax of its own half of the vocabulary, so rank
//! 1, which holds no grammar state, needs the masks too: `stage_row_masks`
//! uploads them on the head and broadcasts them, once per verify, into the same device
//! buffer on both ranks, and the next split-head verify applies them before
//! each rank's partial argmax (`glm_vocab_split`). Unmasked verifies never
//! touch any of this.
//!
//! Wire contract (both ranks, in this order): the head sends the width word
//! with [`MASKED_VERIFY`] set, the tokens, then this broadcast of
//! `rows * words` 32-bit words; the worker receives the same three.
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::types::TransformerModel;

/// Flag on the generic-verify width word: per-row masks follow the tokens.
pub const MASKED_VERIFY: u32 = 1 << 16;
/// Widest masked verify (the worker's generic-verify width bound).
const MAX_ROWS: usize = 32;

/// The worker's reading of a generic-verify width word: `(rows, masked)`.
pub(super) fn split_verify_width(word: u32) -> (usize, bool) {
    ((word & !MASKED_VERIFY) as usize, word & MASKED_VERIFY != 0)
}

/// The mask buffer (allocated on first use) and the armed row count.
#[derive(Default)]
pub(crate) struct VerifyRowMasks {
    buf: parking_lot::Mutex<Option<DevicePtr>>,
    armed: AtomicUsize,
}

impl VerifyRowMasks {
    /// Free the buffer at teardown.
    pub(super) fn release(&self, gpu: &dyn spark_runtime::gpu::GpuBackend) -> Result<()> {
        match self.buf.lock().take() {
            Some(ptr) => gpu.free(ptr),
            None => Ok(()),
        }
    }
}

impl TransformerModel {
    /// 32-bit words per row mask.
    pub(crate) fn verify_mask_words(&self) -> usize {
        self.config.vocab_size.div_ceil(32)
    }

    /// Stage `rows` row masks for the next verify on every rank. The head
    /// passes them (`rows * verify_mask_words()` words); the worker passes
    /// `None` and receives them. The broadcast is synchronous.
    pub(crate) fn stage_row_masks(&self, rows: usize, masks: Option<&[u32]>) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows),
            "masked verify rows {rows} out of range"
        );
        let words = self.verify_mask_words();
        let bytes = rows * words * 4;
        let buf = {
            let mut slot = self.verify_row_masks.buf.lock();
            match *slot {
                Some(ptr) => ptr,
                None => *slot.insert(self.gpu.alloc(MAX_ROWS * words * 4)?),
            }
        };
        if let Some(masks) = masks {
            ensure!(
                masks.len() == rows * words,
                "masked verify: {} mask words for {rows} rows of {words}",
                masks.len()
            );
            // SAFETY: a live &[u32] reinterpreted as its own bytes (align 1).
            let bytes_view: &[u8] =
                unsafe { std::slice::from_raw_parts(masks.as_ptr().cast::<u8>(), bytes) };
            self.gpu.copy_h2d(bytes_view, buf)?;
        }
        if let Some(comm) = self.comm.as_ref().filter(|c| c.world_size() > 1) {
            comm.broadcast(buf.0, bytes, 0)?;
        }
        self.verify_row_masks.armed.store(rows, Ordering::Release);
        Ok(())
    }

    /// Take the staged masks for a verify of `rows` rows, if any were staged.
    /// A staged count that does not match is a protocol error.
    pub(crate) fn take_verify_row_masks(&self, rows: usize) -> Result<Option<DevicePtr>> {
        let armed = self.verify_row_masks.armed.swap(0, Ordering::AcqRel);
        if armed == 0 {
            return Ok(None);
        }
        ensure!(
            armed == rows,
            "masked verify staged {armed} rows for a {rows}-row verify"
        );
        Ok(*self.verify_row_masks.buf.lock())
    }
}

#[cfg(test)]
mod tests {
    use super::{MASKED_VERIFY, split_verify_width};

    #[test]
    fn the_width_word_round_trips_rows_and_the_mask_flag() {
        // Unmasked verifies send the bare width, as before this flag existed.
        for rows in [2usize, 9, 32] {
            assert_eq!(split_verify_width(rows as u32), (rows, false));
            assert_eq!(
                split_verify_width(rows as u32 | MASKED_VERIFY),
                (rows, true)
            );
        }
    }
}
