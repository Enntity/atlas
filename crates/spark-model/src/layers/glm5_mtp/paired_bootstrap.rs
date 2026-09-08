// SPDX-License-Identifier: AGPL-3.0-only
//! Real missing-pair KV write and proposal consume sealed per-request rows.
use super::*;
use crate::speculative::glm_pair_plan::{EagerTailInput, Limits, Profile};
use kv_rows_plan::DeviceSpan;

impl Glm5MtpHead {
    pub(super) fn paired_propose_owned(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .context("paired proposal requires actual GLM state")?;
        self.validate_paired_live(state, ctx.gpu)?;
        ensure!(
            stream == ctx.gpu.default_stream()
                && !ctx.graph_capture
                && (token as usize) < ctx.config.vocab_size,
            "paired proposal requires eager default stream and valid token"
        );
        let owner = self.paired.as_ref().context("paired pool missing")?;
        let data = input.data();
        let (index, tail, bonus, finish, next) = {
            let pool = owner.lock();
            let index = pool.matches_request(state, input, ctx)?;
            let slot = &pool.slots[index];
            ensure!(
                matches!(state.repair, repair_state::RepairPhase::Capture)
                    && state.last_num_drafted == 0
                    && !slot.proposing,
                "paired bootstrap was already consumed or is not ready"
            );
            let tail = pool.tail_span(state, ctx.gpu)?;
            let view = slot
                .bonus
                .as_ref()
                .context("paired bootstrap lacks producer-time H[P]")?;
            ensure!(
                view.generation == slot.generation
                    && view.position == data.prompt
                    && view.row == 5
                    && view.rows == 1
                    && data.tokens.get(data.prompt) == slot.pending_target.as_ref(),
                "paired bootstrap bonus/token identity changed"
            );
            // This plan describes ONE request's four-draft transaction, not
            // model/server admission; no C2 guard is inferred from it.
            let limits = Limits::new(
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
                pool.context,
                pool.blocks_per_slot * 16,
                4,
            )?;
            let finish = limits.bootstrap_eager_tail(EagerTailInput {
                generation: slot.generation,
                prompt_tokens: data.prompt,
                target_position: data.position,
                token_rows: data.tokens.len(),
                cached_rows: state.seq_len,
                tail_position: slot.tail.as_ref().context("paired tail missing")?.position,
            })?;
            let next = limits.propose(
                finish.state(),
                slot.generation,
                data.position,
                finish.state().cache_rows(),
                4,
            )?;
            (
                index,
                tail,
                DeviceSpan {
                    ptr: pool.slab.offset(index * SLOT_BYTES + 5 * ROW_BYTES),
                    bytes: ROW_BYTES,
                },
                finish,
                next,
            )
        };
        {
            let cache = self.kv_cache.lock();
            self.validate_kv_inputs(
                &data.tokens[data.prompt..data.prompt + 1],
                tail,
                ctx,
                &cache,
            )?;
            self.validate_kv_blocks(&cache, &state.block_table)?;
        }
        owner.lock().slots[index].writing = true;
        state.repair = repair_state::RepairPhase::Failed;
        let result = (|| {
            let write = finish
                .write()
                .context("paired bootstrap missing write plan")?;
            self.write_kv_rows(
                &data.tokens[write.token_start()..write.token_start() + 1],
                tail,
                write.cache_start(),
                &state.block_table,
                ctx,
                stream,
            )?;
            state.seq_len = finish.state().cache_rows();
            state.last_num_drafted = 0;
            state.repair = repair_state::RepairPhase::Proposed(next);
            {
                let mut pool = owner.lock();
                pool.matches_request(state, input, ctx)?;
                pool.slots[index].writing = false;
                pool.slots[index].proposing = true;
            }
            let drafts = self.propose(
                token,
                bonus.ptr,
                data.position,
                4,
                state,
                ctx,
                stream,
                None,
                None,
                None,
            )?;
            owner.lock().slots[index].proposing = false;
            Ok(drafts)
        })();
        if result.is_err() {
            owner.lock().slots[index].failed = true;
            state.repair = repair_state::RepairPhase::Failed;
        }
        result
    }

    pub(in crate::layers::glm5_mtp) fn authorize_paired_propose(
        &self,
        state: &Glm5MtpProposerState,
        source: DevicePtr,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.validate_paired_live(state, ctx.gpu)?;
        let Some(owner) = &self.paired else {
            return Ok(());
        };
        let pool = owner.lock();
        let lease = state
            .paired
            .as_ref()
            .context("paired proposal lease missing")?;
        ensure!(
            pool.slots[lease.slot].proposing
                && source == pool.slab.offset(lease.slot * SLOT_BYTES + 5 * ROW_BYTES),
            "paired proposal must consume its private prepared bonus view"
        );
        Ok(())
    }
}
