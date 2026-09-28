// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_prefill_finish — the tail of `forward_prefill`
//! (routed-expert LoRA fold → unpermute+weighted reduce → shared-expert
//! blend → EP all-reduce). Split out under the 500-LoC cap; pure move.

use super::*;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    #[allow(unused_assignments)]
    pub(super) fn forward_prefill_finish(
        &self,
        input: DevicePtr,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        total_expanded: u32,
        token_to_perm: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        h: u32,
        n: u32,
        top_k: u32,
        num_tokens: usize,
        has_shared: bool,
        use_overlap: bool,
        sp: Option<crate::layers::glm_sp::SpRows>,
        split: bool,
        overlap_shared_reduce: bool,
        shared_in: DevicePtr,
        shared_n: u32,
        defer_shared_hc: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    t0 = Some(std::time::Instant::now());
                }
            };
        }

        // Feature-1: fold the routed-expert down_proj LoRA deltas onto the sorted
        // `expert_down_out` BEFORE the unpermute + weighted reduce, so the router
        // weight multiplies base+delta (PEFT semantics). x = the post-SiLU sorted
        // activations. No-op unless routed-expert deltas are installed.
        self.apply_expert_lora_prefill_down(
            ctx.buffers.expert_gate_out(),
            expert_down_out,
            expert_offsets,
            sorted_token_ids,
            total_expanded,
            ctx,
            stream,
        )?;

        // 7. Unpermute + weighted reduce: scatter sorted outputs to token order
        let output = ctx.buffers.moe_output();
        self.unpermute_ep_prefill(
            expert_down_out,
            output,
            token_to_perm,
            indices_dev,
            weights_dev,
            h,
            n,
            top_k,
            ctx,
            stream,
        )?;

        // 8. Blend shared expert: output += sigmoid(dot(input, gate)) * shared
        // Skip when has_shared == false (no shared expert in this model config).
        // EP fix: defer shared expert blend until AFTER all-reduce to avoid doubling.
        let is_ep_prefill = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        if has_shared && !is_ep_prefill {
            let shared_down_out = ctx.buffers.attn_output();
            if use_overlap {
                ctx.gpu.stream_wait_event(stream, self.event_b)?;
            }
            super::dump::dump_routed_only(ctx.gpu, stream, output, n, h)?;
            super::dump::dump_shared_out(ctx.gpu, stream, shared_down_out, n, h)?;
            super::dump::dump_shared_gate(
                ctx.gpu,
                stream,
                input,
                self.weights.shared_expert_gate.weight,
                n,
                h,
            )?;
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                n,
                stream,
            )?;
        }
        super::dump::dump_moe_out(ctx.gpu, stream, output, n, h)?;
        prof_step!("unpermute_blend");

        // The routed result is now complete. Starting the shared expert here
        // lets its GEMMs run beside the EP collective instead of beside the
        // bandwidth-heavy routed GEMMs. event_a makes the auxiliary stream wait
        // for this point; event_b is joined immediately before the shared blend.
        if overlap_shared_reduce {
            self.run_shared_expert_prefill(
                shared_in,
                shared_n,
                h,
                ctx.config.shared_expert_intermediate_size as u32,
                self.prefill_stream,
                stream,
                true,
                ctx,
            )?;
        }

        // `ATLAS_GLM_SHARED_TP_SPLIT=1`: each rank already blended its partial
        // shared-expert columns before this point (see forward_prefill).
        if split {
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                ctx.buffers.attn_output(),
                input,
                self.weights.shared_expert_gate.weight,
                h,
                n,
                stream,
            )?;
        }

        // EP all-reduce
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            let _t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            if let Some(sp) = sp {
                sp.reduce_scatter(output, h as usize, ctx, stream)?;
            } else if ctx.graph_capture {
                comm.all_reduce(output.0, num_tokens * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
            }
            if let Some(t0) = _t0 {
                ctx.gpu.synchronize(stream)?;
                tracing::info!(
                    "  EP allreduce (moe out) N={}: {}µs",
                    num_tokens,
                    t0.elapsed().as_micros(),
                );
            }
            // Add shared expert ONCE after all-reduce (prevents EP doubling)
            if has_shared && !defer_shared_hc && !split {
                let shared_down_out = ctx.buffers.attn_output();
                if use_overlap || overlap_shared_reduce {
                    ctx.gpu.stream_wait_event(stream, self.event_b)?;
                }
                ops::moe_batched_blend(
                    ctx.gpu,
                    self.moe_batched_blend,
                    sp.map_or(output, |sp| sp.local(output, h as usize)),
                    shared_down_out,
                    shared_in,
                    self.weights.shared_expert_gate.weight,
                    h,
                    shared_n,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}
