// SPDX-License-Identifier: AGPL-3.0-only
//! Immutable proposal plans reused by actual owned writers.
use super::*;

impl Glm5MtpHead {
    pub(super) fn check_owned_proposal(
        &self,
        token: u32,
        state: &Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.validate_paired_live(state, ctx.gpu)?;
        ensure!(
            stream == ctx.gpu.default_stream()
                && !ctx.graph_capture
                && !ctx.gpu.stream_is_capturing(stream)
                && (token as usize) < ctx.config.vocab_size,
            "paired proposal requires eager default stream and valid token"
        );
        let bonus = {
            let pool = self.paired.as_ref().context("paired pool missing")?.lock();
            pool.scratch_idle()?;
            let index = state
                .paired
                .as_ref()
                .context("paired proposal lease missing")?
                .slot;
            kv_rows_plan::DeviceSpan {
                ptr: pool.slab.offset(index * SLOT_BYTES + 5 * ROW_BYTES),
                bytes: ROW_BYTES,
            }
        };
        let cache = self.kv_cache.lock();
        self.validate_kv_inputs(&[token], bonus, ctx, &cache)?;
        Ok(())
    }

    pub(super) fn paired_validate_propose(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("paired proposal requires actual GLM state")?;
        self.check_owned_proposal(token, state, ctx, stream)?;
        if matches!(state.repair, repair_state::RepairPhase::Pending(_)) {
            self.repair_plan(input, state, ctx)?;
        } else {
            self.bootstrap_plan(input, state, ctx)?;
        }
        Ok(())
    }

    pub(super) fn proposal_metadata(
        &self,
        state: &Glm5MtpProposerState,
        next: crate::speculative::glm_pair_plan::ProposalPlan,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let end = next.speculative_cache_end();
        let last = end
            .checked_sub(1)
            .context("paired speculative end missing")?;
        let cache = self.kv_cache.lock();
        self.validate_kv_blocks(&cache, &state.block_table)?;
        let bs = cache.block_size();
        let physical = *state
            .block_table
            .get(last / bs)
            .context("paired speculative block missing")?;
        let slot = (physical as usize)
            .checked_mul(bs)
            .and_then(|n| n.checked_add(last % bs))
            .context("paired speculative slot overflow")?;
        let position = next
            .position()
            .checked_add(3)
            .context("paired draft position overflow")?;
        crate::layers::mtp_meta::pack_mtp_attn_meta(
            u32::try_from(position)?,
            i64::try_from(slot)?,
            i32::try_from(end)?,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        Ok(())
    }
}
