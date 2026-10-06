// SPDX-License-Identifier: AGPL-3.0-only
//! Layer side of the qwen4_exp mid-chunk prefix-cache checkpoint
//! (`ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1`; the model side is
//! `model::trait_impl::prefill_b::qwen4exp_ckpt`).
//!
//! Without it a prefix-caching prompt's last chunk runs as two passes, cut one
//! block below the prompt end, so the first pass can end where the next turn
//! restores (the "tail split"); with prefix caching off it runs as one pass,
//! and the two shapes give different bits. With it the last chunk is one pass
//! -- the caching-off numerics -- and the checkpoint is captured INSIDE that
//! pass at `cp`, the last 64-token chunk boundary of the pass at or below the
//! cut:
//! * each GDN layer's recurrent state: the state spine's FP32 entry state of
//!   chunk `cp / 64` (`gated_delta_rule_chunk_delta_h_*_cap`), or, on the
//!   token-sequential warm-replay recurrence, the state after a split at `cp`;
//! * each GDN layer's conv window: the conv split at `cp`
//!   (`conv1d_prefill_capture`, the existing mid-chunk machinery);
//! * the PLE conv carry: the PLE conv split at `cp` (this module's PLE hook);
//! * the PLE token history and the QSA indexer keys: rebuilt at `cp` by the
//!   model from the tokens and the pooled block keys.
//!
//! Every split here is exact by construction (the conv carries and the
//! token-sequential recurrence chain their state across a split; the spine
//! twin only adds a store), so the pass's outputs are unchanged by the
//! capture. The per-thread state below lives for one forward pass on the
//! thread that runs it, set and drained by the model around `forward_layers`.

use spark_runtime::gpu::DevicePtr;
use std::cell::Cell;

/// `ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1`.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// The spine's chunk length.
pub const CHUNK: usize = 64;

thread_local! {
    /// A checkpoint pass is running on this thread.
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    /// Armed by a GDN layer, consumed by the spine launch: (chunk, dst).
    static SPINE: Cell<Option<(u32, u64)>> = const { Cell::new(None) };
    /// GDN layers whose recurrent state was captured this pass.
    static H_DONE: Cell<usize> = const { Cell::new(0) };
    /// PLE conv capture: (row of the PLE forward, dst), and whether it landed.
    static PLE: Cell<Option<(usize, u64)>> = const { Cell::new(None) };
    static PLE_DONE: Cell<bool> = const { Cell::new(false) };
}

/// Start a checkpoint pass: the PLE conv carry at forward row `ple_row` goes
/// to `ple_dst`.
pub fn begin(ple_row: usize, ple_dst: DevicePtr) {
    ACTIVE.with(|c| c.set(true));
    SPINE.with(|c| c.set(None));
    H_DONE.with(|c| c.set(0));
    PLE.with(|c| c.set(Some((ple_row, ple_dst.0))));
    PLE_DONE.with(|c| c.set(false));
}

/// End the pass: (GDN layers captured, PLE conv captured).
pub fn end() -> (usize, bool) {
    ACTIVE.with(|c| c.set(false));
    SPINE.with(|c| c.set(None));
    PLE.with(|c| c.set(None));
    (H_DONE.with(Cell::get), PLE_DONE.with(Cell::get))
}

/// A checkpoint pass is running on this thread.
pub fn active() -> bool {
    ACTIVE.with(Cell::get)
}

/// Ask the next GDN state-spine launch to store chunk `chunk`'s FP32 entry
/// state at `dst` (its `_cap` twin).
pub fn arm_spine(chunk: u32, dst: DevicePtr) {
    SPINE.with(|c| c.set(Some((chunk, dst.0))));
}

/// The armed spine capture, if any (taken: one launch serves it).
pub fn take_spine() -> Option<(u32, DevicePtr)> {
    SPINE.with(Cell::take).map(|(c, d)| (c, DevicePtr(d)))
}

/// A GDN layer's recurrent state landed.
pub fn h_captured() {
    H_DONE.with(|c| c.set(c.get() + 1));
}

/// The PLE conv capture point of this pass, if any: (forward row, dst).
pub fn ple_capture() -> Option<(usize, DevicePtr)> {
    if !active() {
        return None;
    }
    PLE.with(Cell::get).map(|(r, d)| (r, DevicePtr(d)))
}

/// The PLE conv carry landed.
pub fn ple_captured() {
    PLE_DONE.with(|c| c.set(true));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pass_counts_its_captures_and_ends_clean() {
        assert!(!active() && ple_capture().is_none());
        begin(128, DevicePtr(0x1000));
        assert!(active());
        assert_eq!(ple_capture(), Some((128, DevicePtr(0x1000))));
        arm_spine(2, DevicePtr(0x2000));
        assert_eq!(take_spine(), Some((2, DevicePtr(0x2000))));
        assert_eq!(take_spine(), None, "one launch takes the arm");
        h_captured();
        h_captured();
        ple_captured();
        assert_eq!(end(), (2, true));
        assert!(!active() && ple_capture().is_none() && take_spine().is_none());
    }
}

#[cfg(test)]
#[path = "qwen4exp_ckpt_gpu_tests.rs"]
mod gpu_tests;
