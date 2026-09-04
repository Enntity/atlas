// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-quantized native-FP4 routed-expert helpers.

use super::*;

#[derive(Clone, Copy)]
pub(super) struct CompactMoeWorklist {
    pub worklist: DevicePtr,
    pub total_tiles: DevicePtr,
    pub max_tiles: u32,
}

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
        compact: Option<CompactMoeWorklist>,
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
        if let Some(work) = compact
            && std::env::var("ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP").as_deref() == Ok("1")
        {
            let fused_kernel = if self.nvfp4_vecscale
                && self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0 != 0
            {
                self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up
            } else {
                self.moe_w4a4_prequant_t_k64_compact_gate_up
            };
            if fused_kernel.0 != 0 {
                return ops::moe_w4a4_grouped_gemm_prequant_compact_gate_up_n128(
                    ctx.gpu,
                    fused_kernel,
                    a_packed,
                    a_scale,
                    gate.packed_ptrs,
                    gate.scale_ptrs,
                    gate.scale2_vals,
                    expert_gate_out,
                    up.packed_ptrs,
                    up.scale_ptrs,
                    up.scale2_vals,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    work.worklist,
                    work.total_tiles,
                    work.max_tiles,
                    stream,
                );
            }
        }
        for (weight, output) in [(gate, expert_gate_out), (up, expert_up_out)] {
            if let Some(work) = compact {
                let compact_kernel = if self.nvfp4_vecscale
                    && self.moe_w4a4_prequant_t_k64_vecscale_compact.0 != 0
                {
                    self.moe_w4a4_prequant_t_k64_vecscale_compact
                } else {
                    self.moe_w4a4_prequant_t_k64_compact
                };
                ops::moe_w4a4_grouped_gemm_prequant_compact_n128(
                    ctx.gpu,
                    compact_kernel,
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
                    work.worklist,
                    work.total_tiles,
                    work.max_tiles,
                    stream,
                )?;
            } else {
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
