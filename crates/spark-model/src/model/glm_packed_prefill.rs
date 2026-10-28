// SPDX-License-Identifier: AGPL-3.0-only

//! Packed multi-prompt prefill (`ATLAS_GLM_PACKED_PREFILL=1`, default off).
//!
//! # Why
//!
//! A prefill pass at a few dozen rows is bound by the routed-expert weights
//! it reads: GLM-5.3-Flash holds ~83 GB of experts per rank under EP2, and 50
//! tokens x top-8 over 288 experts touch ~75% of them, ~270 ms of a ~350 ms
//! pass. Eight short prompts prefilled one after another read most of the
//! experts eight times. Packed into one traversal (400 rows touch ~100%) they
//! read them once.
//!
//! # What runs
//!
//! New prompts of at most `ATLAS_GLM_PACKED_PREFILL_MAX_TOKENS` tokens
//! (default 512) with no restorable prefix-cache state are laid out back to
//! back in the arena (`layer::glm_packed`). Each prompt gets exactly the
//! metadata, block reservation, embedding, DFlash capture and last-row LM
//! head a cold single-prompt prefill gives it; the layers run their row-local
//! work once over all rows and each prompt's recurrence and attention over
//! its own rows. The prompts skip the prefix-cache lookup and insert
//! (a prompt that has restorable state takes the ordinary path instead).
//!
//! # Exactness
//!
//! With the switch on a prompt runs this path whether it is alone (C1) or
//! packed with others (C4/C8), and every row-count dependent choice inside a
//! packed pass is made from the prompt's own row count or pinned
//! (`packed_scope`), so its output does not depend on its company.
//! `ATLAS_GLM_PACKED_PREFILL_CHECK=1` logs one hash line per prompt to prove
//! it on hardware (`PACKED_CHECK`).
//!
//! # EP
//!
//! One worker command, `0xFFFFFFE8`: preamble slot of the first prompt, then
//! N, `slots[N]`, `lengths[N]` and every prompt's tokens in one bulk broadcast.
//! The head decides the packing; the worker follows it.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::layer::glm_packed::GlmPackedOwner;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::traits::{Model, SequenceState};

pub(super) const EP_CMD_GLM_PACKED_PREFILL: u32 = 0xFFFF_FFE8;

/// Largest prompt the packing takes by default.
pub const DEFAULT_MAX_TOKENS: usize = 512;
/// The FlashKDA recurrence starts at 2048 rows and borrows the packed
/// buffer; a packed prompt stays below it.
pub const MAX_TOKENS_CAP: usize = 2047;
/// Most prompts in one packed pass.
pub const MAX_PROMPTS: usize = 16;

/// `ATLAS_GLM_PACKED_PREFILL=1`. Read once per process.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PACKED_PREFILL").as_deref() == Ok("1"))
}

/// `ATLAS_GLM_PACKED_PREFILL_MAX_TOKENS` (default 512, at most 2047).
pub fn max_tokens() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        parse_max_tokens(
            std::env::var("ATLAS_GLM_PACKED_PREFILL_MAX_TOKENS")
                .ok()
                .as_deref(),
        )
    })
}

pub(crate) fn parse_max_tokens(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .clamp(2, MAX_TOKENS_CAP)
}

/// `ATLAS_GLM_PACKED_PREFILL_CHECK=1`: per-prompt hash lines.
fn check_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PACKED_PREFILL_CHECK").as_deref() == Ok("1"))
}

/// Arena rows of each prompt: back to back from row 0.
pub(crate) fn row_offsets(lens: &[usize]) -> Vec<usize> {
    lens.iter()
        .scan(0usize, |next, &len| {
            let at = *next;
            *next += len;
            Some(at)
        })
        .collect()
}

/// Bytes of one prompt's metadata block (positions [x3 under MRoPE], then
/// slots), 256-aligned like the fused chunk's.
pub(crate) fn meta_block_bytes(rows: usize, mrope: bool) -> usize {
    let streams = if mrope { 3 } else { 1 };
    let slot_offset = (rows * 4 * streams + 7) & !7;
    (slot_offset + rows * 8).div_ceil(256) * 256
}

/// Scratch layout of a packed pass: the MoE top-k staging at 0 (as for one
/// chunk of `total` rows), the joint positions/slots, then one metadata block
/// per prompt. Returns `(joint_at, owners_at, end)`.
pub(crate) fn scratch_layout(
    lens: &[usize],
    top_k: usize,
    mrope: bool,
) -> (usize, Vec<usize>, usize) {
    let total: usize = lens.iter().sum();
    let align = |b: usize| b.div_ceil(256) * 256;
    let joint_at = align(total * top_k * 4 * 2);
    let mut at = joint_at + meta_block_bytes(total, mrope);
    let owners = lens
        .iter()
        .map(|&rows| {
            let here = at;
            at += meta_block_bytes(rows, mrope);
            here
        })
        .collect();
    (joint_at, owners, at)
}

#[path = "glm_packed_prefill_impl.rs"]
mod packed_impl;

#[cfg(test)]
#[path = "glm_packed_prefill_tests.rs"]
mod tests;
