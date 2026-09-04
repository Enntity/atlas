// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-quantized native-FP4 routed-expert helpers.

use super::*;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prequant_fp4_gate_up(
        &self,
        expert_input: DevicePtr,
        gate: &ExpertPtrTable,
        up: &ExpertPtrTable,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        max_m_tiles: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // expert_down_out is not consumed until after gate/up and has ample
        // capacity for packed token-major A followed by its group scales.
        let a_packed = ctx.buffers.expert_down_out();
        let a_scale = a_packed.offset(n as usize * h as usize / 2);
        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.quantize_nvfp4_k,
            expert_input,
            a_packed,
            a_scale,
            n,
            h,
            stream,
        )?;
        let grouped_kernel = if self.nvfp4_vecscale && self.moe_w4a4_prequant_t_k64_vecscale.0 != 0
        {
            self.moe_w4a4_prequant_t_k64_vecscale
        } else {
            self.moe_w4a4_prequant_t_k64
        };
        for (weight, output) in [(gate, expert_gate_out), (up, expert_up_out)] {
            ops::moe_w4a4_grouped_gemm_prequant_n128(
                ctx.gpu,
                grouped_kernel,
                a_packed,
                a_scale,
                weight.packed_ptrs,
                weight.scale_ptrs,
                weight.scale2_vals,
                output,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                max_m_tiles,
                stream,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prequant_fp4_down(
        &self,
        expert_up_out: DevicePtr,
        down: &ExpertPtrTable,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        total_expanded: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        max_m_tiles: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Up output is dead after SiLU·mul; reuse it for packed down A and
        // scales. No allocation or persistent memory is introduced.
        let a_packed = expert_up_out;
        let a_scale = a_packed.offset(total_expanded as usize * inter as usize / 2);
        let grouped_kernel = if self.nvfp4_vecscale && self.moe_w4a4_prequant_t_k64_vecscale.0 != 0
        {
            self.moe_w4a4_prequant_t_k64_vecscale
        } else {
            self.moe_w4a4_prequant_t_k64
        };
        ops::moe_w4a4_grouped_gemm_prequant_n128(
            ctx.gpu,
            grouped_kernel,
            a_packed,
            a_scale,
            down.packed_ptrs,
            down.scale_ptrs,
            down.scale2_vals,
            expert_down_out,
            expert_offsets,
            DevicePtr(0),
            num_experts,
            h,
            inter,
            max_m_tiles,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn fused_silu_prequant_fp4_down(
        &self,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        total_expanded: u32,
        inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Writing compact output directly into expert_up_out would race its
        // still-live BF16 rows across independently scheduled row CTAs. Stage
        // in expert_down_out, then compact-copy into the now-dead up buffer.
        let staged_packed = ctx.buffers.expert_down_out();
        let packed_bytes = total_expanded as usize * inter as usize / 2;
        let scale_bytes = total_expanded as usize * inter as usize / 16;
        let staged_scale = staged_packed.offset(packed_bytes);
        let out_bf16 = if self.lora.is_some() {
            expert_gate_out
        } else {
            DevicePtr::NULL
        };
        ops::silu_mul_quant_nvfp4(
            ctx.gpu,
            self.silu_mul_quant_nvfp4_k,
            expert_gate_out,
            expert_up_out,
            staged_packed,
            staged_scale,
            out_bf16,
            total_expanded,
            inter,
            stream,
        )?;
        ctx.gpu.copy_d2d_async(
            staged_packed,
            expert_up_out,
            packed_bytes + scale_bytes,
            stream,
        )
    }
}
