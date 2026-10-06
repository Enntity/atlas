// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE=1`: the q38 local-route unpermute of a
//! qwen4_exp SP chunk, row range by row range, with the MoE reduce-scatter
//! pipelined behind it (`layers::qwen4exp_sp_pipe::compute_and_reduce_scatter`).
//!
//! `moe_q38_unpermute_local` is one CTA per token reading only that token's
//! routes, so a row range launched apart (pointers advanced to its first
//! token) writes exactly the bytes the whole-chunk launch writes there.

use super::*;

impl MoeLayer {
    /// Unpermute this chunk into `ptrs[1]` and reduce-scatter it, the peer's
    /// window first. `ptrs` = [expert_down_out, output, token_to_perm,
    /// topk_ids, topk_weights]; `dims` = [hidden, top_k]. `Ok(false)`: not
    /// taken (switch off, the q38 unpermute does not serve, or the pair cannot
    /// pipeline); nothing was launched.
    pub(super) fn q38_unpermute_reduce_scatter(
        &self,
        sp: crate::layers::glm_sp::SpRows,
        ptrs: [DevicePtr; 5],
        dims: [u32; 2],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let [expert_out, output, token_to_perm, topk_ids, topk_weights] = ptrs;
        let [hidden, topk] = dims;
        let inter = ctx.config.routed_inter_local() as u32;
        if !crate::layers::qwen4exp_sp_pipe::rs_requested()
            || !hidden.is_multiple_of(8)
            || hidden / 8 > 1024
            || topk_ids.is_null()
            || !self.q38_routed_serves(hidden, inter, ctx)
        {
            return Ok(false);
        }
        let route = |r0: usize| (r0 * topk as usize * 4) as u64;
        crate::layers::qwen4exp_sp_pipe::compute_and_reduce_scatter(
            sp,
            output,
            hidden as usize,
            ctx,
            stream,
            |r0, n, s| {
                let served = self.try_q38_unpermute(
                    expert_out,
                    output.offset(r0 * hidden as usize * 2),
                    DevicePtr(token_to_perm.0 + route(r0)),
                    DevicePtr(topk_ids.0 + route(r0)),
                    DevicePtr(topk_weights.0 + route(r0)),
                    [hidden, n as u32, topk],
                    ctx,
                    s,
                )?;
                anyhow::ensure!(served, "q38 unpermute refused a row range it served");
                Ok(())
            },
        )
    }
}
