// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence DFlash paged-context preparation shared by serial and batch proposal.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::layer::ForwardContext;

pub(super) const DFLASH_BLOCK_SIZE: usize = 16;

pub(super) fn dflash_table_capacity(max_ctx_len: usize, gamma: usize) -> usize {
    (max_ctx_len + gamma + 1).div_ceil(DFLASH_BLOCK_SIZE)
}

pub(super) fn dflash_blocks_needed(ctx_len: usize, gamma: usize) -> usize {
    (ctx_len + gamma + 1).div_ceil(DFLASH_BLOCK_SIZE)
}

impl BlockDiffusionDraftHead {
    pub(super) fn prepare_paged_state(
        &self,
        position: usize,
        state: &mut DflashProposerState,
        ctx: &ForwardContext,
        stream: u64,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<(DevicePtr, u32)> {
        let eagle_skip = state.skip_next_decode_append;
        state.skip_next_decode_append = false;
        let skip_append = std::env::var("ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND")
            .ok()
            .as_deref()
            == Some("1");
        if !skip_append
            && !eagle_skip
            && let Some(latest) = target_hidden_stack
            && state.ctx_len < state.max_ctx_len
        {
            ctx.gpu.copy_d2d_async(
                latest,
                state
                    .ctx_hidden_acc
                    .offset(state.ctx_len * state.ctx_slot_bytes),
                state.ctx_slot_bytes,
                stream,
            )?;
            debug_assert_eq!(state.ctx_positions.len(), state.ctx_len);
            state.ctx_positions.push(position.saturating_sub(1) as i32);
            state.ctx_len += 1;
        }

        anyhow::ensure!(
            std::env::var("ATLAS_DFLASH_OPTION_B").ok().as_deref() == Some("1"),
            "the fixed GLM appliance requires DFlash Option B"
        );
        self.ensure_paged_capacity(state, ctx)?;

        let committed = if std::env::var("ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE")
            .ok()
            .as_deref()
            == Some("1")
        {
            0
        } else {
            state.ctx_committed.min(state.ctx_len)
        };
        if state.ctx_len > committed {
            anyhow::ensure!(self.ctx_window > 0, "DFlash context scratch has zero rows");
            let mut start = committed;
            while start < state.ctx_len {
                let count = (state.ctx_len - start).min(self.ctx_window);
                crate::layers::ops::fill_slots_from_block_table(
                    ctx.gpu,
                    self.kernels.fill_slots,
                    self.scratch.slot_mapping_dev,
                    state.block_table_dev.expect("allocated above"),
                    start as u32,
                    count as u32,
                    DFLASH_BLOCK_SIZE as u32,
                    stream,
                )?;
                self.precompute_ctx_kv(
                    state.ctx_hidden_acc,
                    start,
                    count,
                    &state.ctx_positions[start..start + count],
                    self.scratch.slot_mapping_dev,
                    ctx,
                    stream,
                    true,
                )?;
                start += count;
            }
            state.ctx_committed = state.ctx_len;
        }
        state.ctx_count_drafter = state.ctx_len;
        let context_count = if std::env::var("ATLAS_DFLASH_OPTION_B_NO_CTX")
            .ok()
            .as_deref()
            == Some("1")
        {
            0
        } else {
            state.ctx_count_drafter as u32
        };
        Ok((
            state.block_table_dev.expect("allocated above"),
            context_count,
        ))
    }

    /// Grow a sequence's physical drafter KV lease only as its live context
    /// grows. The device block-table allocation is full-capacity but tiny
    /// (~8 KiB at 32K context); the expensive KV blocks are leased on demand.
    pub(super) fn ensure_paged_capacity(
        &self,
        state: &mut DflashProposerState,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let table_capacity = dflash_table_capacity(state.max_ctx_len, self.gamma);
        let required = dflash_blocks_needed(state.ctx_len, self.gamma);
        anyhow::ensure!(
            required <= table_capacity,
            "DFlash context requires {required} blocks, table capacity is {table_capacity}"
        );

        if state.block_table_dev.is_none() {
            state.block_table_dev = Some(ctx.gpu.alloc(table_capacity * size_of::<u32>())?);
        }
        if required <= state.block_table.len() {
            state.max_ctx_count_drafter = state.block_table.len() * DFLASH_BLOCK_SIZE;
            return Ok(());
        }

        let old_len = state.block_table.len();
        let delta = required - old_len;
        let new_blocks = self
            .kv_cache
            .lock()
            .try_alloc_blocks(delta)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "DFlash paged KV exhausted growing from {old_len} to {required} blocks"
                )
            })?;
        state.block_table.extend_from_slice(&new_blocks);
        let bytes = state
            .block_table
            .iter()
            .flat_map(|block| block.to_le_bytes())
            .collect::<Vec<_>>();
        if let Err(error) = ctx
            .gpu
            .copy_h2d(&bytes, state.block_table_dev.expect("allocated above"))
        {
            self.kv_cache.lock().free_blocks(&new_blocks);
            state.block_table.truncate(old_len);
            return Err(error);
        }
        state.max_ctx_count_drafter = state.block_table.len() * DFLASH_BLOCK_SIZE;
        tracing::debug!(
            blocks = state.block_table.len(),
            slots = state.max_ctx_count_drafter,
            ctx_len = state.ctx_len,
            "DFlash drafter KV lease grown"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{dflash_blocks_needed, dflash_table_capacity};

    #[test]
    fn short_requests_do_not_reserve_full_context_tables() {
        assert_eq!(dflash_table_capacity(32_768, 8), 2_049);
        assert_eq!(dflash_blocks_needed(0, 8), 1);
        assert_eq!(dflash_blocks_needed(34, 8), 3);
        assert_eq!(dflash_blocks_needed(32_768, 8), 2_049);
    }
}
