// SPDX-License-Identifier: AGPL-3.0-only
//! Bootstrap H[P] is captured at the successful target producer, not propose.
use super::*;

impl Pool {
    fn bootstrap_target_plan(
        &self,
        state: &Glm5MtpProposerState,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        ctx: &ForwardContext,
    ) -> Result<usize> {
        self.scratch_idle()?;
        let index = self.matches_request(state, input, ctx)?;
        self.tail_span(state, ctx.gpu)?;
        let data = input.data();
        let primer = data
            .prompt
            .checked_sub(1)
            .context("paired bootstrap prompt is empty")?;
        ensure!(
            data.position == data.prompt
                && state.seq_len == primer
                && (token as usize) < ctx.config.vocab_size
                && self.slots[index].bonus.is_none()
                && self.slots[index].pending_target.is_none(),
            "paired target requires exactly one bootstrap decode after eager P-1"
        );
        Ok(index)
    }
    pub(super) fn matches_request(
        &self,
        state: &Glm5MtpProposerState,
        input: &crate::model::GlmPairedInput<'_>,
        ctx: &ForwardContext,
    ) -> Result<usize> {
        self.validate(state, ctx.gpu)?;
        let lease = state
            .paired
            .as_ref()
            .context("paired target lease missing")?;
        let slot = &self.slots[lease.slot];
        let binding = slot
            .binding
            .as_ref()
            .context("paired target has no eager primer")?;
        let data = input.data();
        ensure!(
            !slot.retiring
                && binding.sequence_slot == data.slot
                && binding.capture_generation == data.capture_generation
                && binding.prompt == data.prompt
                && binding.capture == data.capture.ptr
                && binding.normalized == data.normalized.ptr
                && data.normalized.ptr == ctx.buffers.norm_output()
                && data.normalized.bytes == ctx.buffers.sizes().norm_output
                && data.normalized.bytes >= ROW_BYTES,
            "paired producer owner/position/storage changed"
        );
        Ok(lease.slot)
    }
}

impl Glm5MtpHead {
    pub(super) fn paired_validate_target(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("paired target requires actual GLM state")?;
        self.validate_paired_live(state, ctx.gpu)?;
        self.paired
            .as_ref()
            .context("paired pool absent")?
            .lock()
            .bootstrap_target_plan(state, input, token, ctx)?;
        Ok(())
    }
    pub(super) fn paired_begin_target(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired target requires actual GLM state")?;
        self.validate_paired_live(state, ctx.gpu)?;
        let mut pool = self.paired.as_ref().context("paired pool absent")?.lock();
        let index = pool.bootstrap_target_plan(state, input, token, ctx)?;
        pool.slots[index].pending_target = Some(token);
        pool.slots[index].writing = true;
        Ok(())
    }

    pub(super) fn paired_publish_target(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired target requires actual GLM state")?;
        let owner = self.paired.as_ref().context("paired pool absent")?;
        let (index, destination) = {
            let pool = owner.lock();
            let index = pool.matches_request(state, input, ctx)?;
            let data = input.data();
            ensure!(
                stream == ctx.gpu.default_stream()
                    && data.position
                        == data
                            .prompt
                            .checked_add(1)
                            .context("paired target position overflow")?
                    && data.tokens.last() == Some(&token)
                    && state.seq_len == data.prompt - 1
                    && pool.slots[index].pending_target == Some(token)
                    && pool.slots[index].writing
                    && pool.slots[index].bonus.is_none(),
                "paired target completion is stale or duplicate"
            );
            (index, pool.slab.offset(index * SLOT_BYTES + 5 * ROW_BYTES))
        };
        let result = (|| {
            ctx.gpu
                .copy_d2d_async(input.data().normalized.ptr, destination, ROW_BYTES, stream)?;
            ctx.gpu.synchronize(stream)?;
            let mut pool = owner.lock();
            pool.matches_request(state, input, ctx)?;
            let slot = &mut pool.slots[index];
            slot.bonus = Some(HiddenView {
                generation: slot.generation,
                position: input.data().prompt,
                row: 5,
                rows: 1,
            });
            slot.writing = false;
            Ok(())
        })();
        if result.is_err() {
            owner.lock().slots[index].failed = true;
            state.repair = repair_state::RepairPhase::Failed;
        }
        result
    }

    pub(super) fn paired_quarantine(
        &self,
        state: &mut dyn ProposerState,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired quarantine requires actual GLM state")?;
        let mut pool = self.paired.as_ref().context("paired pool absent")?.lock();
        let lease = state
            .paired
            .as_ref()
            .context("paired quarantine lease absent")?;
        ensure!(
            pool.backend == gpu as *const dyn GpuBackend as *const () as usize
                && std::sync::Arc::ptr_eq(&pool.identity, &lease.owner)
                && lease.slot < 2
                && pool.slots[lease.slot].generation == lease.generation,
            "foreign paired quarantine lease"
        );
        pool.slots[lease.slot].failed = true;
        if pool
            .verification
            .as_ref()
            .is_some_and(|v| v.owns(lease.slot))
        {
            pool.producer_failed = true;
        }
        state.repair = repair_state::RepairPhase::Failed;
        Ok(())
    }
}
