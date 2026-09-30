// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_FIRST_APPEND`: what `propose()` appends to the drafter
//! context on top of the prefill capture and the verify commits.
//!
//! `propose()` appends row 0 of the model-global capture buffer unless a
//! commit already covered it. On the first propose after prefill nothing of
//! this sequence is in that buffer yet: the slot gets the previous request's
//! last verify row (zeros after boot; under concurrent traffic, whichever
//! sequence wrote last), so drafts — and with them accept counts and verify
//! widths — depend on request history. `Legacy` keeps that, byte for byte.
//! The other variants read the global row only when this sequence's own
//! decode or verify wrote it, and differ in what the first propose appends.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::speculative::ProposerState;

/// Strict startup switch: an unknown value is an error, never a default.
const FIRST_APPEND_ENV: &str = "ATLAS_DFLASH_FIRST_APPEND";

/// The slot the first propose after prefill appends (stamped `position - 1`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FirstAppend {
    /// Row 0 of the model-global capture buffer, whoever wrote it.
    #[default]
    Legacy,
    /// Nothing: the context is the prefill capture alone.
    None,
    /// This request's own last prompt position, captured during its prefill.
    Own,
    /// A zero row (what `Legacy` gives the first request after boot).
    Zero,
}

impl FirstAppend {
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        Ok(match raw {
            None | Some("legacy") => Self::Legacy,
            Some("none") => Self::None,
            Some("own") => Self::Own,
            Some("zero") => Self::Zero,
            Some(other) => {
                anyhow::bail!("{FIRST_APPEND_ENV}={other:?} is not one of legacy|none|own|zero")
            }
        })
    }

    /// Read once at head construction and logged once.
    pub fn from_env() -> Result<Self> {
        let raw = match std::env::var(FIRST_APPEND_ENV) {
            Ok(raw) => Some(raw),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => anyhow::bail!("{FIRST_APPEND_ENV}: {error}"),
        };
        let variant = Self::parse(raw.as_deref())?;
        tracing::info!(
            "DFlash first context append: {variant:?} ({FIRST_APPEND_ENV}=legacy|none|own|zero)"
        );
        Ok(variant)
    }
}

/// Where the propose-side append takes its slot from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AppendSource {
    Row(DevicePtr),
    Zero,
}

/// Pure source decision for the propose-side append; `None` = no append.
///
/// `suppressed`: a commit already appended this capture (or the debug switch
/// is set). `stack`: the caller's model-global capture row. `first`: this is
/// the propose right after prefill. `own_capture`: `stack` was written by
/// this sequence's own latest decode or verify.
pub(super) fn decode_append_source(
    variant: FirstAppend,
    suppressed: bool,
    stack: Option<DevicePtr>,
    room: bool,
    first: bool,
    own_capture: bool,
    own_row: Option<DevicePtr>,
) -> Option<AppendSource> {
    let stack = stack.filter(|_| !suppressed && room)?;
    match variant {
        FirstAppend::Legacy => Some(AppendSource::Row(stack)),
        _ if own_capture => Some(AppendSource::Row(stack)),
        FirstAppend::Own if first => own_row.map(AppendSource::Row),
        FirstAppend::Zero if first => Some(AppendSource::Zero),
        _ => None,
    }
}

/// The model-global capture row now holds this sequence's own latest token
/// (single-sequence decode), so its next propose may append it. No-op for
/// other proposers.
pub(crate) fn note_own_capture(state: &mut Option<Box<dyn ProposerState>>) {
    if let Some(dstate) = state
        .as_mut()
        .and_then(|s| s.as_any_mut().downcast_mut::<DflashProposerState>())
    {
        dstate.own_capture = true;
    }
}

impl BlockDiffusionDraftHead {
    /// The propose-side context append: at most one slot at `ctx_len`. The
    /// state's one-shot markers are consumed whether or not it appends.
    pub(super) fn append_decode_ctx(
        &self,
        dstate: &mut DflashProposerState,
        target_hidden_stack: Option<DevicePtr>,
        position: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        // EAGLE-fix: a commit that already appended this capture (in EAGLE
        // order) sets the one-shot flag so it is not appended twice.
        let eagle_skip = std::mem::take(&mut dstate.skip_next_decode_append);
        let first = dstate.first_append_at.take() == Some(position);
        let own_capture = std::mem::take(&mut dstate.own_capture);
        let Some(source) = decode_append_source(
            self.startup.diagnostics.first_append,
            self.startup.diagnostics.no_decode_append || eagle_skip,
            target_hidden_stack,
            dstate.ctx_len < dstate.max_ctx_len,
            first,
            own_capture,
            dstate.own_row,
        ) else {
            return Ok(());
        };
        let slot = dstate
            .ctx_hidden_acc
            .offset(dstate.ctx_len * dstate.ctx_slot_bytes);
        match source {
            AppendSource::Row(row) => {
                gpu.copy_d2d_async(row, slot, dstate.ctx_slot_bytes, stream)?
            }
            AppendSource::Zero => gpu.memset_async(slot, 0, dstate.ctx_slot_bytes, stream)?,
        }
        // Phase I (v2): stamp this slot's TRUE absolute position, fixed
        // forever. The just-decoded token sits at `position - 1` (the
        // full-rebuild formula assigns slot ctx_len the position
        // (position - (ctx_len+1)) + ctx_len == position - 1). Keeping
        // ctx_positions parallel to ctx_len lets precompute rope each
        // slot by its own fixed position instead of a sliding base.
        debug_assert_eq!(dstate.ctx_positions.len(), dstate.ctx_len);
        dstate.ctx_positions.push(position.saturating_sub(1) as i32);
        dstate.ctx_len += 1;
        Ok(())
    }
}
