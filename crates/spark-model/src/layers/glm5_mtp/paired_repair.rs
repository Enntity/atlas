// SPDX-License-Identifier: AGPL-3.0-only
//! Accepted true pairs and next proposal consume only detached owner rows.
use super::*;
use crate::speculative::glm_pair_plan::{Limits, Profile};
use kv_rows_plan::DeviceSpan;

impl Pool {
    pub(super) fn pending_input(
        &self,
        index: usize,
        input: &crate::model::GlmPairedInput<'_>,
        pending: repair_state::PendingRepair,
    ) -> Result<()> {
        let accepted = pending
            .plan
            .bonus_hidden_row()
            .context("paired pending bonus row missing")?;
        let position = pending.plan.state().target_position();
        let base = position
            .checked_sub(accepted + 1)
            .context("paired pending base underflow")?;
        ensure!(
            input.data().position == position
                && input.data().tokens.get(..base)
                    == Some(self.slots[index].issued_prefix.as_slice())
                && input.data().tokens.get(base..position) == Some(&pending.tokens[..accepted + 1]),
            "paired pending committed token prefix changed"
        );
        Ok(())
    }

    fn limits(&self) -> Result<Limits> {
        Limits::new(
            Profile {
                sequences: 1,
                drafts: 4,
                continuous: true,
                grammar: false,
                adaptive_depth: false,
                catchup: false,
                carry: false,
                prefix_reuse: false,
            },
            self.context,
            self.blocks_per_slot * 16,
            4,
        )
    }
}

impl Glm5MtpHead {
    pub(super) fn paired_repair_owned(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let owner = self.paired.as_ref().context("paired repair pool missing")?;
        let (index, pending, source, bonus, next) = self.repair_plan(input, state, ctx)?;
        owner.lock().slots[index].writing = true;
        state.repair = repair_state::RepairPhase::Failed;
        let result = (|| {
            // This event comes only from the same actual model's sealed input.
            // It orders queued target SSM commit; it is not a CPU completion.
            ctx.gpu
                .stream_wait_event(stream, input.data().secondary_event)?;
            if let Some(write) = pending.plan.write() {
                self.write_kv_rows(
                    &pending.tokens[write.token_start()..write.token_start() + write.rows()],
                    source,
                    write.cache_start(),
                    &state.block_table,
                    ctx,
                    stream,
                )?;
            }
            state.seq_len = pending.plan.state().cache_rows();
            state.last_num_drafted = 0;
            state.repair = repair_state::RepairPhase::Proposed(next);
            {
                let mut pool = owner.lock();
                pool.matches_request(state, input, ctx)?;
                pool.slots[index].writing = false;
                pool.slots[index].proposing = true;
                pool.slots[index].commit_queued = false;
            }
            let drafts = self.propose(
                token,
                bonus,
                input.data().position,
                4,
                state,
                ctx,
                stream,
                None,
                None,
                None,
            )?;
            let mut pool = owner.lock();
            pool.slots[index].proposing = false;
            pool.issue(
                index,
                input.data().position,
                token,
                &drafts,
                input.data().tokens,
            )?;
            Ok(drafts)
        })();
        if result.is_err() {
            owner.lock().slots[index].failed = true;
            state.repair = repair_state::RepairPhase::Failed;
        }
        result
    }
    pub(super) fn repair_plan(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        state: &Glm5MtpProposerState,
        ctx: &ForwardContext,
    ) -> Result<(
        usize,
        repair_state::PendingRepair,
        DeviceSpan,
        DevicePtr,
        crate::speculative::glm_pair_plan::ProposalPlan,
    )> {
        let owner = self.paired.as_ref().context("paired repair pool missing")?;
        let (index, pending, source, bonus, next) = {
            let pool = owner.lock();
            pool.scratch_idle()?;
            let index = pool.matches_request(state, input, ctx)?;
            pool.next_attempt(index)?;
            let slot = &pool.slots[index];
            let repair_state::RepairPhase::Pending(record) = state.repair else {
                anyhow::bail!("paired repair has no owned verdict");
            };
            let accepted = record
                .plan
                .bonus_hidden_row()
                .context("paired bonus row missing")?;
            let pending = state
                .repair
                .pending(slot.generation, input.data().position, accepted)?;
            pool.pending_input(index, input, pending)?;
            ensure!(
                slot.commit_queued && state.seq_len == pending.cached_rows && !slot.proposing,
                "paired repair precedes target commit or private cursor changed"
            );
            let view = slot.bonus.as_ref().context("paired repair bonus absent")?;
            ensure!(
                view.generation == slot.generation
                    && view.row == 5
                    && view.rows == 1
                    && view.position.checked_add(1) == Some(input.data().position),
                "paired repair bonus position/identity changed"
            );
            let next = pool.limits()?.propose(
                pending.plan.state(),
                slot.generation,
                input.data().position,
                pending.plan.state().cache_rows(),
                4,
            )?;
            (
                index,
                pending,
                DeviceSpan {
                    ptr: pool.slab.offset(index * SLOT_BYTES + ROW_BYTES),
                    bytes: accepted * ROW_BYTES,
                },
                pool.slab.offset(index * SLOT_BYTES + 5 * ROW_BYTES),
                next,
            )
        };
        {
            let cache = self.kv_cache.lock();
            self.validate_kv_blocks(&cache, &state.block_table)?;
            if let Some(write) = pending.plan.write() {
                self.plan_kv_rows(
                    &pending.tokens[write.token_start()..write.token_start() + write.rows()],
                    source,
                    write.cache_start(),
                    &state.block_table,
                    ctx,
                    &cache,
                )?;
            }
        }
        self.proposal_metadata(state, next, ctx)?;
        Ok((index, pending, source, bonus, next))
    }
}
