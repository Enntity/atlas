// SPDX-License-Identifier: AGPL-3.0-only

//! GLM K=5 verification built from the proven K2/K3 routed paths plus one
//! exact-M=5 shared-expert pass.

use super::*;

impl MoeLayer {
    /// Returns the buffer containing five output rows. The optimized arm is
    /// intentionally restricted to GLM's unified NVFP4, ungated-shared layout.
    pub fn forward_k5(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let optimized = self.lora.is_none()
            && self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.has_mixed_bf16_shared_expert()
            && matches!(
                self.experts_scale_kind,
                crate::weight_map::WeightQuantFormat::Nvfp4
            )
            && self.use_t_layout_for_decode()
            && self.weights.shared_expert_gate.weight.is_null()
            && !self.weights.shared_expert.gate_proj.is_null()
            && !self.weights.shared_expert.up_proj.is_null()
            && !self.weights.shared_expert.down_proj.is_null()
            && self.w4a16_batchm.kernel(5).0 != 0;

        if !optimized {
            self.forward_batched(input, 5, ctx, stream)?;
            return Ok(ctx.buffers.moe_output());
        }

        let h = ctx.config.hidden_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let gate_out = ctx.buffers.logits();
        let up_out = ctx.buffers.ssm_qkvz();
        let shared_down = ctx.buffers.attn_output();
        let sh_gate = &self.weights.shared_expert.gate_proj;
        let sh_up = &self.weights.shared_expert.up_proj;
        let sh_down = &self.weights.shared_expert.down_proj;
        let batch5 = self.w4a16_batchm.kernel(5);

        ops::w4a16_gemv_batchm(
            ctx.gpu,
            batch5,
            input,
            sh_gate,
            gate_out,
            5,
            shared_inter,
            h,
            stream,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            batch5,
            input,
            sh_up,
            up_out,
            5,
            shared_inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_silu_mul,
            gate_out,
            up_out,
            gate_out,
            5 * shared_inter,
            stream,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            batch5,
            gate_out,
            sh_down,
            shared_down,
            5,
            h,
            shared_inter,
            stream,
        )?;

        // Run routed experts through their fused small-M kernels. Consume each
        // temporary moe_output before the next group overwrites it.
        self.forward_k2_routed_only(input, ctx, stream)?;
        ctx.gpu
            .copy_d2d_async(ctx.buffers.moe_output(), input, 2 * h as usize * 2, stream)?;
        let row3 = input.offset(2 * h as usize * 2);
        self.forward_k3_routed_only(row3, ctx, stream)?;
        ctx.gpu
            .copy_d2d_async(ctx.buffers.moe_output(), row3, 3 * h as usize * 2, stream)?;
        ops::residual_add(
            ctx.gpu,
            self.residual_add,
            input,
            shared_down,
            5 * h,
            stream,
        )?;
        Ok(input)
    }
}
