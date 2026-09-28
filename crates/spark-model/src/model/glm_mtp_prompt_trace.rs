// SPDX-License-Identifier: AGPL-3.0-only
//! Model-owned prompt capture provenance. No public raw-pointer constructor.
use super::types::TransformerModel;
use crate::layers::glm5_mtp::hidden_trace;
use crate::traits::Model;
use crate::{layer::ForwardContext, traits::SequenceState};
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
pub(crate) struct Capture {
    base: DevicePtr,
    capacity: usize,
    rows: usize,
    generation: u64,
    slot: usize,
}
impl Capture {
    fn checked(
        base: DevicePtr,
        capacity: usize,
        rows: usize,
        generation: u64,
        slot: usize,
    ) -> Result<Self> {
        ensure!(
            (2..=256).contains(&rows)
                && rows <= capacity
                && capacity <= 2044
                && generation != 0
                && base.0 != 0
                && base.0 % 2 == 0
                && base.0.checked_add((capacity * 8192) as u64).is_some(),
            "GLM prompt capture owner/extent"
        );
        Ok(Self {
            base,
            capacity,
            rows,
            generation,
            slot,
        })
    }
    pub(crate) fn rows(self) -> usize {
        self.rows
    }
    pub(crate) fn identity(self) -> (usize, u64) {
        (self.slot, self.generation)
    }
    pub(crate) fn matches(self, base: DevicePtr, bytes: usize, start: usize, rows: usize) -> bool {
        start.checked_add(rows).is_some_and(|end| end <= self.rows)
            && base == self.base.offset(start * 8192)
            && bytes == rows * 8192
    }
    pub(crate) fn same_owner(
        self,
        base: DevicePtr,
        bytes: usize,
        generation: u64,
        prompt: usize,
    ) -> bool {
        self.base == base
            && self.capacity * 8192 == bytes
            && self.generation == generation
            && self.rows == prompt
    }
}

impl TransformerModel {
    pub(super) fn arm_glm_prompt_trace(
        &self,
        seq: &mut SequenceState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !hidden_trace::prompt_selected(seq) {
            return Ok(false);
        }
        hidden_trace::spend_prompt(seq)?;
        // Request depth/grammar do not exist in SequenceState at prefill.
        // Validate supported storage now; arm_prepared validates actual future
        // arguments before emitting any source evidence.
        ensure!(
            crate::speculative::glm_repair_policy::enabled(),
            "GLM prompt trace requires active repair"
        );
        crate::speculative::glm_repair_policy::validate_environment()?;
        ensure!(
            self.mtp_slot_draft_capacity(seq.slot_idx) >= 4,
            "GLM prompt trace insufficient actual verify capacity"
        );
        ensure!(
            self.mtp_prefill_capacity >= 4,
            "GLM prompt trace insufficient repair capture capacity"
        );
        ensure!(
            !super::trait_impl::drafter_prefill::eager_drafter_disabled(),
            "GLM prompt trace requires eager primer"
        );
        ensure!(
            self.proposer.is_some(),
            "GLM prompt trace missing proposer owner"
        );
        let generation = self.mtp_prefill_capture_gen.load(Ordering::Relaxed);
        ensure!(
            generation == seq.mtp_capture_gen
                && self.mtp_prefill_capture_len.load(Ordering::Relaxed) == seq.prompt_len
                && seq.seq_len == seq.prompt_len
                && seq.tokens.len() == seq.prompt_len,
            "GLM prompt capture live generation/coverage"
        );
        let capture = Capture::checked(
            self.mtp_prefill_hidden,
            self.mtp_prefill_capacity,
            seq.prompt_len,
            generation,
            seq.slot_idx,
        )?;
        let owner_ctx = ForwardContext {
            comm: self.comm_ref(),
            midchunk_capture: None,
            ..*ctx
        };
        hidden_trace::arm_prompt(seq, capture, &owner_ctx, stream, || {
            self.hidden_trace_adapter_ownership()
        })?;
        Ok(true)
    }
}

#[cfg(test)]
pub(crate) fn fixture_capture(
    base: DevicePtr,
    capacity: usize,
    rows: usize,
    generation: u64,
    slot: usize,
) -> Result<Capture> {
    // Unit fixtures exercise the same sealed checked constructor. Actual model
    // ownership and error propagation are separately tested at its caller.
    Capture::checked(base, capacity, rows, generation, slot)
}
