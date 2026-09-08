// SPDX-License-Identifier: AGPL-3.0-only
//! Detach accepted rows and bonus only from the matching actual producer.
use super::*;

impl Glm5MtpHead {
    pub(in crate::layers::glm5_mtp) fn paired_acknowledge(
        &self,
        accepted: usize,
        state: &mut Glm5MtpProposerState,
    ) -> Result<()> {
        let pool = self
            .paired
            .as_ref()
            .context("paired acknowledgement pool missing")?
            .lock();
        let lease = state
            .paired
            .as_ref()
            .context("paired acknowledgement lease missing")?;
        ensure!(
            !pool.closed
                && std::sync::Arc::ptr_eq(&pool.identity, &lease.owner)
                && lease.slot < 2
                && lease.slab == pool.slab,
            "foreign paired acknowledgement lease"
        );
        let slot = &pool.slots[lease.slot];
        ensure!(
            slot.active
                && !slot.failed
                && !slot.retiring
                && !slot.writing
                && slot.generation == lease.generation
                && slot.blocks == state.block_table,
            "stale or failed paired acknowledgement"
        );
        state.repair.acknowledge(accepted)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn paired_commit_target(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        committed: usize,
        width: usize,
        completed: bool,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired commit state missing")?;
        self.validate_paired_live(state, ctx.gpu)?;
        let mut pool = self
            .paired
            .as_ref()
            .context("paired commit pool missing")?
            .lock();
        ensure!(
            !pool.producer_failed,
            "paired verification terminally failed before commit"
        );
        let index = pool.matches_request(state, input, ctx)?;
        let repair_state::RepairPhase::Pending(pending) = state.repair else {
            anyhow::bail!("paired target commit has no owned verdict");
        };
        pool.pending_input(index, input, pending)?;
        ensure!(
            width == 5
                && pending
                    .plan
                    .bonus_hidden_row()
                    .and_then(|a| a.checked_add(1))
                    == Some(committed)
                && pending.plan.state().target_position() == input.data().position
                && pending.cached_rows == state.seq_len
                && !pool.slots[index].commit_queued,
            "paired target commit count/position changed or duplicate"
        );
        if completed {
            pool.slots[index].commit_queued = true;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn paired_record_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        base: usize,
        tokens: &[u32],
        accepted: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired verdict state missing")?;
        self.validate_paired_live(state, ctx.gpu)?;
        let owner = self.paired.as_ref().context("paired pool missing")?;
        let (index, destination, pending) = {
            let pool = owner.lock();
            let index = pool.matches_request(state, input, ctx)?;
            let record = pool.verification_owner(index, tokens)?;
            ensure!(accepted <= 4, "paired verdict accepted count exceeds four");
            let end = base
                .checked_add(accepted + 1)
                .context("paired verdict end overflow")?;
            ensure!(
                record.produced
                    && record.issued.base == base
                    && record.normalized == input.data().normalized.ptr
                    && input.data().position == end
                    && input.data().tokens.get(base..end) == Some(&tokens[..accepted + 1]),
                "paired verdict differs from actual verified committed prefix"
            );
            ensure!(
                input.data().tokens.get(..base) == Some(pool.slots[index].issued_prefix.as_slice()),
                "paired verdict differs from actual verified committed prefix"
            );
            let mut pending = state.repair;
            pending.record(
                record.generation,
                record.generation,
                base,
                tokens,
                accepted,
                end,
                state.seq_len,
                5,
            )?;
            (index, pool.slab.offset(index * SLOT_BYTES), pending)
        };
        let stream = ctx.gpu.default_stream();
        owner.lock().slots[index].writing = true;
        let result = (|| {
            if accepted > 0 {
                ctx.gpu.copy_d2d_async(
                    input.data().normalized.ptr,
                    destination.offset(ROW_BYTES),
                    accepted * ROW_BYTES,
                    stream,
                )?;
            }
            ctx.gpu.copy_d2d_async(
                input.data().normalized.ptr.offset(accepted * ROW_BYTES),
                destination.offset(5 * ROW_BYTES),
                ROW_BYTES,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            let mut pool = owner.lock();
            pool.matches_request(state, input, ctx)?;
            pool.verification_owner(index, tokens)?;
            let slot = &mut pool.slots[index];
            slot.bonus = Some(HiddenView {
                generation: slot.generation,
                position: base + accepted,
                row: 5,
                rows: 1,
            });
            slot.writing = false;
            slot.issued = None;
            slot.commit_queued = false;
            state.repair = pending;
            pool.verification = None;
            Ok(())
        })();
        if result.is_err() {
            let mut pool = owner.lock();
            pool.slots[index].failed = true;
            pool.producer_failed = true;
            state.repair = repair_state::RepairPhase::Failed;
        }
        result
    }
}
