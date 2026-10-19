// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_MOE=1`: Qwen3.8-Flash-Next's MoE prefill on its
//! own kernels:
//!
//! * the router logits ([`MoeLayer::try_q38_router`]): the checkpoint's BF16
//!   weight on the tile GEMM (0.82 ms a layer at 16K);
//! * the shared expert ([`MoeLayer::try_q38_shared`]): 4.88 -> 3.41 ms;
//! * the routed experts ([`MoeLayer::try_q38_routed_prefill`]): the
//!   reference's W4A4 (`forward_prefill_w4a4`), or W4A16 on the BF16 TC
//!   kernels (`forward_prefill_tcp`), both on the checkpoint's own planes.
//!
//! The routed chain takes only the shape it serves: NVFP4 experts, SiLU, no
//! LoRA on the experts, none of the alternative gate/up or down arms
//! selected. Remote experts (EP) have null tables and are skipped, and the
//! local unpermute ([`MoeLayer::try_q38_unpermute`]) never reads their rows.
//!
//! `ATLAS_QWEN4EXP_PREFILL_MOE_W2=1` runs the shared expert's GEMMs on their
//! `moe_q38w_*` twins (2 x 4 warp grid, 16-byte dequant stores),
//! byte-identical; see [`w2_requested`].

use super::*;
use crate::layers::ops::qwen4exp_rowinv::bf16_proj;
use spark_runtime::kernel_args::KernelLaunch;

/// `ATLAS_QWEN4EXP_PREFILL_MOE=1`, implied by `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ`.
pub(crate) fn q38_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let flag = super::forward_prefill_routed::env_flag;
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_MOE") || bf16_proj())
}

/// `ATLAS_QWEN4EXP_PREFILL_MOE_W2=1` (with `_MOE`): the shared expert's
/// GEMMs on the `moe_q38w_*` twins -- the same k32 MMAs per output on a 2 x 4
/// warp grid with 16-byte dequant stores, every byte identical
/// (`scripts/dev/qwen4exp_moe_prefill_bench.cu`; GB10, 16000 tokens: the
/// shared expert 3.42 -> 3.18 ms).
/// It also takes the router below 32 rows (byte-identical logits at 1..31
/// rows; a 30-row pass: 0.31 -> 0.13 ms a layer).
pub(crate) fn w2_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::forward_prefill_routed::env_flag("ATLAS_QWEN4EXP_PREFILL_MOE_W2"))
}

/// `ATLAS_QWEN4EXP_PREFILL_SP_SHARED=1`; see [`MoeLayer::q38_sp_rows`].
pub(crate) fn sp_shared_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::forward_prefill_routed::env_flag("ATLAS_QWEN4EXP_PREFILL_SP_SHARED"))
}

/// The q38 entry `name`, or its `moe_q38w_*` twin under `_MOE_W2`.
fn q38_entry(name: &'static str) -> &'static str {
    if !w2_requested() {
        return name;
    }
    match name {
        "moe_q38_dense_gate_up_silu" => "moe_q38w_dense_gate_up_silu",
        "moe_q38_dense_down" => "moe_q38w_dense_down",
        other => other,
    }
}

impl MoeLayer {
    /// The router logits of a prefill chunk, when the q38 arm serves them:
    /// `dense_gemm_bf16_pipelined` (128-row tiles, one in-order m16n8k16
    /// chain per output) over the checkpoint's BF16 router weight, so a row's
    /// logits do not depend on the rows beside it. GB10, 16000 tokens x 512
    /// experts: 0.82 ms a layer, against 6.44 for the scalar-order GEMM the
    /// other models' BF16 routers take. Below 32 rows only under `_MOE_W2`,
    /// as before. `Ok(false)` launched nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_q38_router(
        &self,
        router_in: DevicePtr,
        logits: DevicePtr,
        n: u32,
        n_out: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !q38_requested()
            || ctx.config.model_type != "qwen4_exp"
            || self.gate_nvfp4.is_some()
            || self.gate_fp8.is_some()
            || (n < 32 && !w2_requested())
            || !h.is_multiple_of(16)
        {
            return Ok(false);
        }
        let k_gemm = ctx.gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
        super::forward_prefill_route_sp::q38_router_gemm(
            ctx.gpu,
            k_gemm,
            [router_in, self.weights.gate.weight, logits],
            [n, n_out, h],
            stream,
        )?;
        Ok(true)
    }

    /// The q38 shared-expert arm takes this layer's shape (row count aside).
    fn q38_shared_shape_serves(&self, h: u32, inter: u32, ctx: &ForwardContext) -> bool {
        q38_requested()
            && ctx.config.model_type == "qwen4_exp"
            && self.shared_gate_t.is_some()
            && self.shared_up_t.is_some()
            && self.shared_down_t.is_some()
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.shared_gate_fp8.is_none()
            && self.shared_up_fp8.is_none()
            && self.shared_down_fp8.is_none()
            && self.bf16_shared_expert.is_none()
            && !self.m5_projections.shared.enabled()
            && !self.gelu_activation
            && self.lora.is_none()
            && h.is_multiple_of(128)
            && inter.is_multiple_of(64)
    }

    /// `ATLAS_QWEN4EXP_PREFILL_SP_SHARED=1`: under the qwen4_exp SP split,
    /// the shared expert runs only this rank's rows instead of every row
    /// (`SpRows::full_shared` off), when the q38 arm serves it. Its kernels
    /// give every row the same arithmetic whatever the row count -- 128-row
    /// tiles, one in-order k32 chain per output, elementwise E4M3 input -- so
    /// the halves computed apart equal the whole chunk byte for byte
    /// (`scripts/dev/qwen4exp_moe_prefill_bench.cu`: 16016 split at 8192,
    /// 12000 at 6144, 5000 at 2048; q38 and W2). GB10, 16000 rows: 3.2 ms a
    /// layer -> about half on each rank. The MXFP8 shared expert splits too
    /// (row-invariant at every row count: `shared_mxfp8.rs`).
    pub(super) fn q38_sp_rows(
        &self,
        sp: crate::layers::glm_sp::SpRows,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
    ) -> crate::layers::glm_sp::SpRows {
        let tiers_skip =
            |rows: usize| crate::layers::w4a16_gemv_tiers::tc_kernel(rows as u32).0 == 0;
        let split = sp.full_shared
            && sp_shared_requested()
            && (self.mxfp8_shared_expert.is_some()
                || (self.q38_shared_shape_serves(h, inter, ctx)
                    && tiers_skip(sp.rows)
                    && tiers_skip(sp.total())));
        crate::layers::glm_sp::SpRows {
            full_shared: sp.full_shared && !split,
            ..sp
        }
    }

    /// The shared expert of a prefill chunk, when the q38 arm serves it: the
    /// default's transposed arm (`w4a16_gemm_t` gate, up and down around
    /// `moe_silu_mul`) as `moe_q38_a_to_e4m3`, `moe_q38_dense_gate_up_silu`
    /// and `moe_q38_dense_down`, the same arithmetic as the routed pair --
    /// byte-identical `shared_down_out`. GB10, 16000 tokens: 4.88 -> 3.41
    /// ms a layer. `out` = [gate scratch, up scratch, down output] (the E4M3
    /// activation and input are staged in the first two). `Ok(false)`
    /// launched nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_q38_shared(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        out: [DevicePtr; 3],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (Some(sg), Some(su), Some(sd)) =
            (&self.shared_gate_t, &self.shared_up_t, &self.shared_down_t)
        else {
            return Ok(false);
        };
        let sizes = ctx.buffers.sizes();
        let serves = self.q38_shared_shape_serves(h, inter, ctx)
            && sizes.ssm_deinterleaved >= n as usize * inter as usize
            && sizes.ssm_qkvz >= n as usize * h as usize;
        if !serves || out[0] != ctx.buffers.ssm_deinterleaved() || out[1] != ctx.buffers.ssm_qkvz()
        {
            return Ok(false);
        }
        let gpu = ctx.gpu;
        let k_a8 = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_a_to_e4m3");
        let k_gu = crate::layers::try_kernel(
            gpu,
            "moe_prefill_q38",
            q38_entry("moe_q38_dense_gate_up_silu"),
        );
        let k_dn =
            crate::layers::try_kernel(gpu, "moe_prefill_q38", q38_entry("moe_q38_dense_down"));
        if k_a8.0 == 0 || k_gu.0 == 0 || k_dn.0 == 0 {
            return Ok(false);
        }
        let (act8, a8) = (out[0], out[1]);
        let cells = n * h;
        KernelLaunch::new(gpu, k_a8)
            .grid([(cells / 4).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(a8)
            .arg_u32(cells)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_gu)
            .grid([inter / 64, n.div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(a8)
            .arg_ptr(sg.weight)
            .arg_ptr(sg.weight_scale)
            .arg_f32(sg.weight_scale_2)
            .arg_ptr(su.weight)
            .arg_ptr(su.weight_scale)
            .arg_f32(su.weight_scale_2)
            .arg_ptr(act8)
            .arg_u32(n)
            .arg_u32(inter)
            .arg_u32(h)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_dn)
            .grid([h / 128, n.div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(act8)
            .arg_ptr(sd.weight)
            .arg_ptr(sd.weight_scale)
            .arg_f32(sd.weight_scale_2)
            .arg_ptr(out[2])
            .arg_u32(n)
            .arg_u32(h)
            .arg_u32(inter)
            .launch(stream)?;
        Ok(true)
    }

    /// Run the routed gate/up, SiLU and down of a prefill chunk on the q38
    /// kernels when they serve it; `Ok(false)` launched nothing.
    /// The unpermute + top-k reduce of a q38 chunk: `moe_q38_unpermute_local`
    /// sums only the routes of this rank's experts (all of them without EP),
    /// in slot order -- byte-identical to `moe_unpermute_reduce_indexed` over
    /// zeroed remote rows, without the 1.2 GB of clears a 16K layer needed
    /// for them (`prepare_ep_prefill_outputs`) and with 16-byte loads (GB10,
    /// 16000 tokens: 4.19 -> 2.17 ms). `dims` = [hidden, tokens, top_k].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_q38_unpermute(
        &self,
        expert_output: DevicePtr,
        output: DevicePtr,
        token_to_perm: DevicePtr,
        topk_ids: DevicePtr,
        topk_weights: DevicePtr,
        dims: [u32; 3],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let [hidden, tokens, topk] = dims;
        let inter = ctx.config.routed_inter_local() as u32;
        if !hidden.is_multiple_of(8)
            || hidden / 8 > 1024
            || topk_ids.is_null()
            || !self.q38_routed_serves(hidden, inter, ctx)
        {
            return Ok(false);
        }
        let k = crate::layers::try_kernel(ctx.gpu, "moe_prefill_q38", "moe_q38_unpermute_local");
        if k.0 == 0 {
            return Ok(false);
        }
        let (start, end) = if ctx.config.ep_world_size > 1 {
            let (s, e) = ctx.config.local_expert_range();
            (s as u32, e as u32)
        } else {
            (0, ctx.config.num_experts as u32)
        };
        KernelLaunch::new(ctx.gpu, k)
            .grid([tokens, 1, 1])
            .block([hidden / 8, 1, 1])
            .arg_ptr(expert_output)
            .arg_ptr(output)
            .arg_ptr(token_to_perm)
            .arg_ptr(topk_ids)
            .arg_ptr(topk_weights)
            .arg_u32(hidden)
            .arg_u32(tokens)
            .arg_u32(topk)
            .arg_u32(start)
            .arg_u32(end)
            .launch(stream)?;
        Ok(true)
    }

    /// Whether the q38 routed chain serves this layer's prefill: NVFP4
    /// experts in their checkpoint planes, SiLU, no expert LoRA, none of the
    /// alternative gate/up or down arms, the TC kernels' shape (their W4A16
    /// arm must be able to take any layer). Also read before the grid is
    /// sized: the kernels stride their row tiles, so they do not need the
    /// exact per-expert tile count, nor the host round trip
    /// (`ATLAS_MOE_PREFILL_EXACT_TILES`) that buys it.
    pub(super) fn q38_routed_serves(&self, h: u32, inter: u32, ctx: &ForwardContext) -> bool {
        let fp8_down = super::forward_prefill_routed::prefill_fp8_down();
        q38_requested()
            && ctx.config.model_type == "qwen4_exp"
            && self.gate_ptrs_t.is_none()
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && !self.btile_storage.is_published()
            && !self.nvfp4_mmq_layout
            && !self.nvfp4_prequant_moe
            && !self.gateup_fp4
            && !self.down_fp4
            && !self.nvfp4_gate_up_m128
            && !self.nvfp4_down_m32
            && !self.gelu_activation
            && self.lora.is_none()
            && !fp8_down
            && !super::forward_prefill_routed::grouped_cutlass_gate_up_enabled()
            && h == 2560
            && inter == 640
    }

    /// The routed gate/up, SiLU and down of a prefill chunk when the q38
    /// chain serves it: W4A16 on the BF16 TC kernels under
    /// `ATLAS_QWEN4EXP_PREFILL_MOE_BF16` (decode numerics), else the
    /// reference's W4A4 where the layer ships its input scales, else W4A16.
    /// `Ok(false)` launched nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_q38_routed_prefill(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        rows_per_expert_grid: u32,
        total_expanded: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !self.q38_routed_serves(h, inter, ctx) {
            return Ok(false);
        }
        let dims = (h, inter, num_experts, rows_per_expert_grid);
        if !super::tcp_requested()
            && self.try_w4a4_routed_prefill(
                expert_input,
                expert_offsets,
                sorted_token_ids,
                [n, total_expanded],
                dims,
                ctx,
                stream,
            )?
        {
            return Ok(true);
        }
        let served = self.tcp_routed_prefill(
            expert_input,
            expert_offsets,
            sorted_token_ids,
            dims,
            ctx,
            stream,
        )?;
        anyhow::ensure!(
            served,
            "qwen4_exp prefill MoE: the target lacks qwen4exp_moe_tcp"
        );
        Ok(true)
    }
}
