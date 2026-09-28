// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn sparse_ep_reduce_enabled(ep_world_size: usize, has_comm: bool, has_kernel: bool) -> bool {
    ep_world_size > 1 && has_comm && has_kernel
}

impl MoeLayer {
    fn use_sparse_ep_reduce(&self, ctx: &ForwardContext) -> bool {
        sparse_ep_reduce_enabled(
            ctx.config.ep_world_size,
            ctx.comm.is_some(),
            self.moe_unpermute_reduce_ep.0 != 0,
        )
    }

    pub(super) fn prepare_ep_prefill_outputs(
        &self,
        total_expanded: u32,
        inter: u32,
        hidden: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let force_zero = std::env::var("ATLAS_MOE_PREFILL_ZERO").ok().as_deref() == Some("1");
        if self.use_sparse_ep_reduce(ctx) && !force_zero {
            return Ok(());
        }
        if ctx.comm.is_none() && !force_zero {
            return Ok(());
        }
        let rows = total_expanded as usize;
        let gate_bytes = rows * inter as usize * 2;
        ctx.gpu
            .memset_async(ctx.buffers.expert_gate_out(), 0, gate_bytes, stream)?;
        ctx.gpu
            .memset_async(ctx.buffers.expert_up_out(), 0, gate_bytes, stream)?;
        ctx.gpu.memset_async(
            ctx.buffers.expert_down_out(),
            0,
            rows * hidden as usize * 2,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn unpermute_ep_prefill(
        &self,
        expert_output: DevicePtr,
        output: DevicePtr,
        token_to_perm: DevicePtr,
        topk_ids: DevicePtr,
        topk_weights: DevicePtr,
        hidden: u32,
        tokens: u32,
        topk: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.use_sparse_ep_reduce(ctx) {
            let (local_start, local_end) = ctx.config.local_expert_range();
            return ops::moe_unpermute_reduce_indexed_ep(
                ctx.gpu,
                self.moe_unpermute_reduce_ep,
                expert_output,
                output,
                token_to_perm,
                topk_ids,
                topk_weights,
                hidden,
                tokens,
                topk,
                local_start as u32,
                local_end as u32,
                stream,
            );
        }
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce,
            expert_output,
            output,
            token_to_perm,
            topk_weights,
            hidden,
            tokens,
            topk,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::sparse_ep_reduce_enabled;

    #[test]
    fn sparse_reduce_requires_real_ep_communication_and_kernel() {
        assert!(sparse_ep_reduce_enabled(2, true, true));
        assert!(!sparse_ep_reduce_enabled(1, true, true));
        assert!(!sparse_ep_reduce_enabled(2, false, true));
        assert!(!sparse_ep_reduce_enabled(2, true, false));
    }
}
