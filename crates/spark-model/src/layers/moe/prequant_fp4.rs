// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-quantized native-FP4 routed-expert helpers.

use super::*;

#[derive(Clone, Copy)]
pub(super) struct CompactMoeWorklist {
    pub worklist: DevicePtr,
    pub total_tiles: DevicePtr,
    pub max_tiles: u32,
}

fn c3_grouped_shape(config: &atlas_core::config::ModelConfig, rows: u32, decode_rows: u32) -> bool {
    (rows == 3 || (rows % 3 == 0 && rows <= 12 && OWNER_ROWS.with(|r| r.get()) == rows))
        && decode_rows >= 3
        && glm_grouped_shape(config)
}

thread_local! {
    /// Rows of an owner-batched long-context K3 verify FFN in progress, else 0.
    static OWNER_ROWS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Run `f` as the joint FFN of an owner-batched K3 verify of `rows` rows: the
/// C3 grouped arithmetic (router, routed and shared) is applied row-for-row as
/// in a single-owner K3 verify, with each routed expert read once for all rows.
pub(crate) fn with_owner_rows<R>(rows: u32, f: impl FnOnce() -> R) -> R {
    OWNER_ROWS.with(|r| r.set(rows));
    let out = f();
    OWNER_ROWS.with(|r| r.set(0));
    out
}

pub(super) fn c4_grouped_shape(
    config: &atlas_core::config::ModelConfig,
    rows: u32,
    decode_rows: u32,
) -> bool {
    rows == 4 && decode_rows >= 4 && glm_grouped_shape(config)
}

pub(super) fn glm_grouped_shape(config: &atlas_core::config::ModelConfig) -> bool {
    config.model_type == "glm5_next"
        && config.hidden_size == 4096
        && config.moe_intermediate_size == 2048
        && config.shared_expert_intermediate_size == 2048
        && config.num_experts == 288
        && config.num_experts_per_tok == 8
        && config.tp_world_size == 2
        && config.ep_world_size == 2
        && config.scoring_func == "sigmoid"
}

pub(super) fn compact_gate_up_worklist_bytes(rows: u32, top_k: u32, inter: u32) -> usize {
    16 + rows as usize * top_k as usize * inter.div_ceil(128) as usize * 8
}

impl MoeLayer {
    /// C3-only prototype: stateless MoE work is independent of sequence
    /// ownership. Keep the control's router and shared expert, changing only
    /// routed GEMV to the established prequantized native-FP4 grouped path.
    pub(super) fn glm_c3_grouped(&self, ctx: &ForwardContext, rows: u32) -> bool {
        std::env::var("ATLAS_GLM_C3_GROUPED_MOE").as_deref() == Ok("1")
            && c3_grouped_shape(ctx.config, rows, ctx.levers.max_decode_seqs)
            && self.glm_grouped_resources(ctx)
            && self.w4a16_gemv_batch3.0 != 0
    }

    pub(super) fn glm_c4_grouped(&self, ctx: &ForwardContext, rows: u32) -> bool {
        crate::model::glm_c4::enabled(&ctx.config.model_type)
            && std::env::var("ATLAS_GLM_C4_GROUPED_MOE").as_deref() == Ok("1")
            && c4_grouped_shape(ctx.config, rows, ctx.levers.max_decode_seqs)
            && ctx.attn_metadata.is_some_and(|m| m.num_seqs == 4)
            && self.glm_grouped_resources(ctx)
            && self.w4a16_batchm.kernel(4).0 != 0
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
    }

    pub(super) fn glm_grouped_resources(&self, ctx: &ForwardContext) -> bool {
        self.glm_native_moe_resources(ctx)
            && self.nvfp4_prequant_moe
            && self.nvfp4_fused_silu_quant
            && self.moe_w4a4_prequant_t_k64.0 != 0
            && self.moe_w4a4_prequant_t_k64_compact.0 != 0
            && self.moe_build_tile_worklist_k.0 != 0
            // Prequantization reuses expert_down_out before routed down.
            // Remote expert rows remain scratch, so reduction must skip them.
            && self.moe_unpermute_reduce_ep.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && self.silu_mul_quant_nvfp4_k.0 != 0
    }

    pub(super) fn glm_native_moe_resources(&self, ctx: &ForwardContext) -> bool {
        ctx.comm.is_some()
            && self.lora.is_none()
            && self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.has_mixed_bf16_shared_expert()
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && !self.nvfp4_mmq_layout
            && self.use_btile_or_t_prefill()
            && self.gate_fp8.is_none()
            && self.gate_nvfp4.is_none()
            && self.correction_bias_dev.is_some()
            && self.tid2eid_dev.is_none()
            && self.pre_expert_norm.is_none()
            && self.weights.shared_expert_gate.weight.is_null()
            && !self.weights.shared_expert.gate_proj.is_null()
            && !self.weights.shared_expert.up_proj.is_null()
            && !self.weights.shared_expert.down_proj.is_null()
            && !super::forward_prefill_routed::grouped_cutlass_gate_up_enabled()
            && !super::forward_prefill_routed::grouped_cutlass_down_enabled()
    }

    /// Same batch-three shared projections as forward_k3, with destinations
    /// owned by the grouped pipeline until its final post-EP shared blend.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn c3_shared_expert(
        &self,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_out: DevicePtr,
        h: u32,
        inter: u32,
        rows: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Every W4A16 batch tier runs the same per-row FMA chain (bit-identical
        // to batch3 and to M x w4a16_gemv, per w4a16_batch_bitparity_microtest),
        // so several owners' shared-expert rows share one weight read while each
        // row keeps the single-owner K3 arithmetic.
        anyhow::ensure!(rows % 3 == 0 && rows > 0, "C3 shared expert rows {rows}");
        let (h, inter) = (h as usize, inter as usize);
        let mut row = 0usize;
        while row < rows as usize {
            let left = rows as usize - row;
            let wide = if left <= 8 { left } else if left <= 12 { 6 } else { 8 };
            let kernel = self.w4a16_batchm.kernel(wide as u32);
            let take = if wide == 3 || kernel.0 == 0 { 3 } else { wide };
            let (x, g, u, d) = (
                input.offset(row * h * 2),
                gate_out.offset(row * inter * 2),
                up_out.offset(row * inter * 2),
                down_out.offset(row * h * 2),
            );
            let gemv = |x: DevicePtr, weight: &QuantizedWeight, out: DevicePtr, n: usize, k: usize| {
                if take == 3 {
                    ops::w4a16_gemv_batch3(
                        ctx.gpu,
                        self.w4a16_gemv_batch3,
                        x,
                        weight,
                        out,
                        n as u32,
                        k as u32,
                        stream,
                    )
                } else {
                    ops::w4a16_gemv_batchm(
                        ctx.gpu,
                        kernel,
                        x,
                        weight,
                        out,
                        take as u32,
                        n as u32,
                        k as u32,
                        stream,
                    )
                }
            };
            gemv(x, &self.weights.shared_expert.gate_proj, g, inter, h)?;
            gemv(x, &self.weights.shared_expert.up_proj, u, inter, h)?;
            ops::silu_mul(ctx.gpu, self.moe_silu_mul, g, u, g, (take * inter) as u32, stream)?;
            gemv(g, &self.weights.shared_expert.down_proj, d, h, inter)?;
            row += take;
        }
        Ok(())
    }

    /// C3 router logits in scalar `dense_gemm` numerics. Short batches use
    /// the bit-identical row-parallel kernel (`ATLAS_GLM_ROUTER_ROWS=0`
    /// reverts; `ATLAS_GLM_ROUTER_ROWS_CHECK=1` compares both byte for byte).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn c3_router_logits(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        rows: u32,
        num_experts: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        static MODE: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
        let (enabled, check) = *MODE.get_or_init(|| {
            (
                std::env::var("ATLAS_GLM_ROUTER_ROWS").as_deref() != Ok("0"),
                std::env::var("ATLAS_GLM_ROUTER_ROWS_CHECK").as_deref() == Ok("1"),
            )
        });
        let scalar = |out: DevicePtr| {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm,
                router_in,
                &self.weights.gate,
                out,
                rows,
                num_experts,
                h,
                stream,
            )
        };
        if !enabled || rows > 32 || self.dense_gemm_router_rows.0 == 0 {
            return scalar(gate_logits);
        }
        ops::dense_gemm_router_rows(
            ctx.gpu,
            self.dense_gemm_router_rows,
            router_in,
            &self.weights.gate,
            gate_logits,
            rows,
            num_experts,
            h,
            stream,
        )?;
        if check {
            let bytes = (rows * num_experts) as usize * 2;
            let mut fast = vec![0u8; bytes];
            ctx.gpu.copy_d2h(gate_logits, &mut fast)?;
            scalar(gate_logits)?;
            let mut reference = vec![0u8; bytes];
            ctx.gpu.copy_d2h(gate_logits, &mut reference)?;
            let diff = fast.iter().zip(&reference).filter(|(a, b)| a != b).count();
            anyhow::ensure!(
                diff == 0,
                "router rows kernel differs from dense_gemm in {diff}/{bytes} bytes (rows={rows})"
            );
        }
        Ok(())
    }

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
        mode: super::forward_pair_verify::PrefillMode,
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
            && (mode.is_verify_group()
                || std::env::var("ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP").as_deref() == Ok("1")
                || self.glm_c3_grouped(ctx, n)
                || self.glm_c2_grouped(ctx, n)
                || self.independent_grouped(ctx, n)
                || self.glm_c4_grouped(ctx, n))
        {
            let fused_kernel = if self.nvfp4_vecscale
                && self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0 != 0
            {
                self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up
            } else {
                self.moe_w4a4_prequant_t_k64_compact_gate_up
            };
            if fused_kernel.0 != 0 {
                return self.m16_gate_up.run(
                    super::gate_up_m16::GateUpCall {
                        rows: n,
                        n: inter,
                        k: h,
                        experts: num_experts,
                        work,
                        native_resources: self.m16_gate_up.enabled()
                            && self.glm_grouped_resources(ctx),
                        vector: self.nvfp4_vecscale
                            && self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0 != 0,
                        original: fused_kernel,
                        outputs: [expert_gate_out, expert_up_out],
                        sorted_tokens: sorted_token_ids,
                        gate_table: gate.packed_ptrs,
                    },
                    ctx,
                    stream,
                    |kernel| {
                        ops::moe_w4a4_grouped_gemm_prequant_compact_gate_up_n128(
                            ctx.gpu,
                            kernel,
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
                        )
                    },
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

#[cfg(test)]
#[path = "prequant_fp4_tests.rs"]
mod c3_tests;
