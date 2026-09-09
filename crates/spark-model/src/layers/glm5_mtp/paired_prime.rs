// SPDX-License-Identifier: AGPL-3.0-only
//! Actual eager KV writer followed by ordered prompt-tail detachment.
use super::*;
use crate::speculative::glm_repair::GlmPairedHandoff;
use kv_rows_plan::DeviceSpan;

impl Pool {
    pub(super) fn tail_span(
        &self,
        state: &Glm5MtpProposerState,
        gpu: &dyn GpuBackend,
    ) -> Result<DeviceSpan> {
        self.validate(state, gpu)?;
        let lease = state.paired.as_ref().context("paired tail lease missing")?;
        let slot = &self.slots[lease.slot];
        ensure!(
            !slot.retiring && !slot.writing,
            "paired tail unavailable during writer/retirement"
        );
        let binding = slot.binding.as_ref().context("paired request not bound")?;
        let view = slot.tail.as_ref().context("paired tail not published")?;
        ensure!(
            binding.sequence_slot < 2
                && binding.capture_generation != 0
                && !binding.capture.is_null()
                && !binding.normalized.is_null()
                && view.generation == lease.generation
                && view.position + 1 == binding.prompt
                && view.row == 0
                && view.rows == 1,
            "paired tail identity/position mismatch"
        );
        Ok(DeviceSpan {
            ptr: self
                .slab
                .offset(lease.slot * SLOT_BYTES + view.row * ROW_BYTES),
            bytes: view.rows * ROW_BYTES,
        })
    }
}

impl GlmPairedHandoff for Glm5MtpHead {
    fn validate_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        state: &dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.paired_validate_verify(input, tokens, state, ctx)
    }
    fn validate_propose(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        seed: u32,
        state: &dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(u64, u64)> {
        self.paired_validate_propose(input, seed, state, ctx, stream)
    }
    fn validate_session(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let pool = self
            .paired
            .as_ref()
            .context("paired session pool missing")?
            .lock();
        pool.backend(gpu)?;
        ensure!(
            !pool.producer_failed && !pool.slots.iter().any(|slot| slot.failed),
            "paired session is terminal"
        );
        Ok(())
    }
    fn validate_allocation(&self, gpu: &dyn GpuBackend) -> Result<usize> {
        let pool = self
            .paired
            .as_ref()
            .context("paired allocation pool missing")?
            .lock();
        pool.claim_candidate(gpu).map(|(index, _)| index)
    }
    fn fail_session(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut pool = self
            .paired
            .as_ref()
            .context("paired transport pool missing")?
            .lock();
        pool.backend(gpu)?;
        pool.producer_failed = true;
        for slot in &mut pool.slots {
            slot.failed = true;
        }
        Ok(())
    }
    fn commit_target(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        committed: usize,
        width: usize,
        completed: bool,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.paired_commit_target(input, committed, width, completed, state, ctx)
    }
    fn begin_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.paired_begin_verify(input, tokens, state, ctx)
    }
    fn publish_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        predictions: &[u32],
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.paired_publish_verify(input, tokens, predictions, state, ctx)
    }
    fn record_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        base: usize,
        tokens: &[u32],
        accepted: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.paired_record_verify(input, base, tokens, accepted, state, ctx)
    }
    fn validate_cold(
        &self,
        state: &dyn ProposerState,
        sequence_slot: usize,
        prompt: usize,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("paired preflight requires actual GLM state")?;
        self.validate_paired_live(state, ctx.gpu)?;
        let pool = self.paired.as_ref().context("paired pool missing")?.lock();
        pool.scratch_idle()?;
        let slot = &pool.slots[state.paired.as_ref().expect("validated lease").slot];
        ensure!(
            pool.rank == ctx.config.ep_rank
                && sequence_slot < 2
                && prompt > 0
                && prompt < pool.context
                && state.seq_len == 0
                && state.last_num_drafted == 0
                && slot.binding.is_none()
                && pool.slots.iter().all(|s| !s.active
                    || s.binding
                        .as_ref()
                        .is_none_or(|binding| binding.sequence_slot != sequence_slot)),
            "paired preflight requires unbound cold owner on the actual rank"
        );
        Ok(())
    }
    fn retire(&self, state: &mut dyn ProposerState, gpu: &dyn GpuBackend) -> Result<Option<usize>> {
        self.paired_retire(state, gpu)
    }
    fn close(&self, gpu: &dyn GpuBackend, secondary_stream: u64) -> Result<()> {
        self.paired_close(gpu, secondary_stream)
    }
    fn propose_owned(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.paired_propose_owned(input, token, state, ctx, stream)
    }
    fn validate_decode(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.paired_validate_target(input, token, state, ctx)
    }
    fn begin_decode(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.paired_begin_target(input, token, state, ctx)
    }
    fn publish_decode(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.paired_publish_target(input, token, state, ctx, stream)
    }
    fn quarantine(&self, state: &mut dyn ProposerState, gpu: &dyn GpuBackend) -> Result<()> {
        self.paired_quarantine(state, gpu)
    }
    fn prime(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired primer requires actual GLM state")?;
        self.validate_paired_live(state, ctx.gpu)?;
        let data = input.data();
        ensure!(
            stream == ctx.gpu.default_stream()
                && !ctx.graph_capture
                && !ctx.gpu.stream_is_capturing(stream),
            "paired primer requires actual eager default stream"
        );
        ensure!(
            data.position == data.prompt
                && data.tokens.len() == data.prompt
                && data.prompt > 0
                && state.seq_len == 0
                && state.last_num_drafted == 0,
            "paired primer requires cold complete prompt and uninitialized logical KV"
        );
        let source = DeviceSpan {
            ptr: data.capture.ptr,
            bytes: data.capture.bytes,
        };
        let needed = data
            .prompt
            .checked_mul(ROW_BYTES)
            .context("paired prompt extent overflow")?;
        ensure!(
            !source.ptr.is_null() && source.ptr.0.is_multiple_of(2) && needed <= source.bytes,
            "paired primer capture capacity invalid"
        );
        source.end()?;
        ensure!(
            data.normalized.ptr == ctx.buffers.norm_output()
                && data.normalized.bytes == ctx.buffers.sizes().norm_output,
            "paired normalized owner differs from actual context"
        );
        let forbidden = {
            let cache = self.kv_cache.lock();
            self.validate_kv_blocks(&cache, &state.block_table)?;
            self.validate_kv_inputs(
                if data.prompt == 1 {
                    data.tokens
                } else {
                    &data.tokens[1..]
                },
                source,
                ctx,
                &cache,
            )?
        };
        let owner = self.paired.as_ref().context("paired pool missing")?;
        let index = state.paired.as_ref().context("paired lease missing")?.slot;
        let destination;
        {
            let mut pool = owner.lock();
            pool.scratch_idle()?;
            pool.validate(state, ctx.gpu)?;
            ensure!(
                data.prompt < pool.context
                    && pool.slots[index].binding.is_none()
                    && pool.slots.iter().all(|s| s
                        .binding
                        .as_ref()
                        .is_none_or(|b| b.sequence_slot != data.slot)),
                "duplicate/oversized paired primer"
            );
            ensure!(
                !source.overlaps(DeviceSpan {
                    ptr: pool.slab,
                    bytes: SLAB_BYTES
                })?,
                "paired prompt source aliases handoff slab"
            );
            for span in &forbidden {
                ensure!(
                    !span.overlaps(DeviceSpan {
                        ptr: pool.slab,
                        bytes: SLAB_BYTES
                    })?,
                    "paired handoff slab aliases actual writer scratch/cache"
                );
            }
            destination = pool.slab.offset(index * SLOT_BYTES);
            pool.slots[index].writing = true;
        }
        let result = (|| {
            let rows = self.prefill_kv_batched(data.tokens, source.ptr, state, ctx, stream)?;
            ensure!(
                rows == data.prompt - 1,
                "paired eager primer did not write exact P-1 rows"
            );
            ctx.gpu.copy_d2d_async(
                source.ptr.offset((data.prompt - 1) * ROW_BYTES),
                destination,
                ROW_BYTES,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            let mut pool = owner.lock();
            pool.validate(state, ctx.gpu)?;
            let slot = &mut pool.slots[index];
            ensure!(!slot.retiring, "paired primer retired before completion");
            slot.binding = Some(Binding {
                sequence_slot: data.slot,
                capture_generation: data.capture_generation,
                prompt: data.prompt,
                capture: data.capture.ptr,
                normalized: data.normalized.ptr,
            });
            slot.tail = Some(HiddenView {
                generation: slot.generation,
                position: data.prompt - 1,
                row: 0,
                rows: 1,
            });
            slot.writing = false;
            pool.tail_span(state, ctx.gpu)?;
            Ok(())
        })();
        if result.is_err() {
            owner.lock().slots[index].failed = true;
            state.repair = repair_state::RepairPhase::Failed;
        }
        result
    }
}
