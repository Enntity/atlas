// SPDX-License-Identifier: AGPL-3.0-only

//! Routed grouped-GEMM phase of `MoeLayer::forward_prefill`.
//! Covers grid sizing, grouped gate/up GEMM, SiLU, and grouped down GEMM.

use super::*;

/// Whether the single-launch CUTLASS grouped NVFP4 path is enabled. The
/// model-neutral name is used by new integrations; retain the Holo variable
/// as a compatibility alias for existing recipes.
pub(super) fn grouped_cutlass_gate_up_enabled() -> bool {
    env_flag("ATLAS_MOE_GROUPED_CUTLASS") || env_flag("ATLAS_HOLO_MOE_GROUPED_CUTLASS")
}

pub(super) fn grouped_cutlass_down_enabled() -> bool {
    env_flag("ATLAS_MOE_GROUPED_CUTLASS") || env_flag("ATLAS_HOLO_MOE_GROUPED_DOWN")
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

impl MoeLayer {
    /// Routed-expert grouped-GEMM path: upper-bound grid sizing → grouped
    /// gate+up GEMM → SiLU+mul → grouped down GEMM.
    ///
    /// Writes routed expert outputs into `ctx.buffers.expert_down_out()`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::unnecessary_unwrap)]
    pub(super) fn run_routed_grouped_gemm(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        num_tokens: usize,
        ne: usize,
        t0: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    *t0 = Some(std::time::Instant::now());
                }
            };
        }

        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        // Default to the absolute worst case (one expert receives every routed
        // token) to prevent silent truncation. An opt-in load-factor cap lets
        // Holo experiments trade that safety margin for fewer empty expert
        // tiles after validating the router histogram.
        let worst_case_m_tiles = (num_tokens * top_k as usize).div_ceil(64).max(1) as u32;
        // Default-on for NVFP4 experts ONLY; opt-in ("=1") everywhere else.
        //
        // Reads the REAL expert offsets instead of the worst-case bound above, so it
        // cannot truncate — that bound exists only to avoid this D2H copy+sync. On
        // NVFP4 the trade is strongly positive: 120.7 ms of cold TTFT on the 35B by
        // leave-one-out, and without it the rest of the fast-MoE stack buys nothing
        // at all (690.94 ms vs 688.12 with no flags set).
        //
        // ★ On FP8 experts the same sync is a LOSS, and defaulting it on globally
        // regressed the ttft-warm gate's TAIL — measured on that gate's own recipe
        // (qwen3.6-35b-a3b-fp8-bf16head), one variable:
        //     exact_tiles on   p90 +4.9%  (limit +5.0%)  <- 0.1% from failing
        //     exact_tiles off  p90 -5.0%
        // Median barely moved either way (+0.1% vs -0.9%), so only the tail shows it.
        // The win was measured on NVFP4; scope the default to where it was measured.
        let exact_tiles = match std::env::var("ATLAS_MOE_PREFILL_EXACT_TILES")
            .ok()
            .as_deref()
        {
            Some("0") => false,
            Some("1") => true,
            _ => self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4,
        } && worst_case_m_tiles > 1
            && !ctx.graph_capture;
        // Keep the host copy when exact sizing already paid for it. The
        // CUTLASS grouped path also needs these offsets to build its problem
        // list; copying them again would introduce a second stream-draining
        // D2H boundary in every MoE layer.
        let mut exact_eoff: Option<Vec<i32>> = None;
        let max_m_tiles = if exact_tiles {
            let mut offsets = vec![0u8; (ne + 1) * 4];
            ctx.gpu
                .copy_d2h_on_stream(expert_offsets, &mut offsets, stream)?;
            let eoff: Vec<i32> = offsets
                .chunks_exact(4)
                .map(|raw| i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
                .collect();
            let mut prev = 0u32;
            let mut max_rows = 0u32;
            for &cur in eoff.iter().skip(1) {
                let cur = cur as u32;
                max_rows = max_rows.max(cur.saturating_sub(prev));
                prev = cur;
            }
            exact_eoff = Some(eoff);
            max_rows.div_ceil(64).max(1).min(worst_case_m_tiles)
        } else {
            std::env::var("ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&factor| factor > 0)
                .map(|factor| {
                    let capped_rows = avg_per_expert.saturating_mul(factor);
                    worst_case_m_tiles.min(capped_rows.div_ceil(64).max(1) as u32)
                })
                .unwrap_or(worst_case_m_tiles)
        };
        super::dump::dump_expert_load(
            ctx.gpu,
            stream,
            expert_offsets,
            ne,
            num_tokens,
            avg_per_expert,
            max_m_tiles,
        );
        prof_step!("grid_setup");

        let total_expanded = n * top_k;
        // GLM K=5 has only 40 routes across 288 experts. Its prequantized
        // native-FP4 kernels otherwise launch a dense expert grid where most
        // CTAs immediately exit. Keep this exact-shape gate explicit while the
        // compact scheduler is evaluated. GLM's correction-bias router excludes
        // the FP32-routing path, leaving its workspace dead here. It provides
        // 16 KiB even for a one-row arena: enough for the K=5 gate/up worklist
        // (640 items + counter) without allocations.
        let compact_k5 = std::env::var("ATLAS_GLM_K5_COMPACT_MOE").as_deref() == Ok("1")
            && ctx.config.model_type == "glm5_next"
            && n == 5
            && total_expanded == 40
            && num_experts == 288
            && h == 4096
            && inter == 2048
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.nvfp4_prequant_moe
            && self.moe_w4a4_prequant_t_k64.0 != 0
            && self.moe_w4a4_prequant_t_k64_compact.0 != 0
            && self.moe_build_tile_worklist_k.0 != 0;

        // 5. Grouped gate+up GEMM — cp.async pipelined FP8-MMA K64 (transposed).
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // EP remote experts return without writing, so the destination must be
        // zeroed before dispatch. In non-EP, `moe_sort_by_expert` produces a
        // dense token_to_perm over exactly [0, total_expanded), and grouped
        // kernels write every row that can be referenced by unpermute_reduce.
        // Skipping the memset removes ~138 MB/layer of scratch clears on Holo.
        self.prepare_ep_prefill_outputs(total_expanded, inter, h, ctx, stream)?;
        // Host expert_offsets from the CUTLASS gate_up, reused by down to skip
        // a second D2H + host-blocking synchronize.
        let mut cutlass_eoff: Option<Vec<i32>> = None;
        if max_m_tiles > 0 {
            // CUTLASS grouped NVFP4 gate_up reads the ORIGINAL [N,K/2] tables
            // (CUTLASS B is ColumnMajor = K-contiguous), NOT the Atlas
            // transposed ones, so it must be reachable when gate_ptrs_t is
            // absent — that is exactly the originals-only layout a
            // checkpoint-native model runs in.
            if self.nvfp4_mmq_layout {
                self.run_nvfp4_mmq_gate_up(
                    expert_input,
                    expert_gate_out,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    total_expanded,
                    h,
                    inter,
                    num_experts,
                    max_m_tiles,
                    ctx,
                    stream,
                )?;
            } else if grouped_cutlass_gate_up_enabled() && self.cutlass_grouped_host.is_some() {
                // ── SINGLE-LAUNCH CUTLASS grouped NVFP4 gate_up
                // (ATLAS_HOLO_MOE_GROUPED_CUTLASS=1) ── one
                // GemmUniversalMode::kGrouped launch over all active experts in
                // place of the per-expert collective loop. Weights: the load-time
                // host snapshot (`cutlass_grouped_host`) of the decode
                // `gate_ptrs`/`up_ptrs` packed `[N,K/2]` (CUTLASS ColumnMajor B) +
                // swizzled SFB + real per-expert scale2 (epilogue alpha). The token
                // gather is FUSED into the kernel's per-group A-pack (lever 2): pass
                // token-major expert_input + sorted_token_ids directly, no separate
                // permute pass. Writes C_gate/C_up in the sorted layout so
                // silu+down+unpermute are unchanged.
                cutlass_eoff = Some(ops::moe_grouped_gate_up_cutlass(
                    ctx.gpu,
                    exact_eoff.as_deref(),
                    self.cutlass_grouped_host.as_ref().expect("checked above"),
                    expert_input,
                    sorted_token_ids,
                    expert_gate_out,
                    expert_up_out,
                    expert_offsets,
                    inter,
                    h,
                    stream,
                )?);
            } else if let (Some(gp), Some(up)) = (&self.gate_ptrs_t, &self.up_ptrs_t) {
                if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                    // ── ARM-2 Phase-K: native-MXFP4 (E8M0) fused gate_up ──
                    // Leading branch so E8M0 routed experts NEVER reach the
                    // NVFP4-only cutlass/fp4/m128 sub-paths below (structurally
                    // off for the V4 serve, but the branch makes it provable).
                    assert!(
                        self.moe_fused_gate_up_t_k64_e8m0.0 != 0,
                        "ARM-2: routed experts Mxfp4E8m0 but fused_gate_up_t_k64_e8m0 unresolved"
                    );
                    ops::moe_w4a16_fused_gate_up_k64_n128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64_e8m0,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles,
                        stream,
                    )?;
                } else if self.nvfp4_prequant_moe && self.moe_w4a4_prequant_t_k64.0 != 0 {
                    let compact = if compact_k5 || self.glm_c3_grouped(ctx, n) {
                        let total_tiles = ctx.buffers.moe_router_in_f32();
                        let worklist = total_tiles.offset(16);
                        let n_tiles = inter.div_ceil(128);
                        anyhow::ensure!(
                            ctx.buffers.sizes().moe_router_in_f32
                                >= super::prequant_fp4::compact_gate_up_worklist_bytes(
                                    n, top_k, inter
                                ),
                            "compact native-FP4 gate/up worklist exceeds router scratch"
                        );
                        ops::moe_build_tile_worklist(
                            ctx.gpu,
                            self.moe_build_tile_worklist_k,
                            expert_offsets,
                            gp.packed_ptrs,
                            worklist,
                            total_tiles,
                            num_experts,
                            n_tiles,
                            64,
                            stream,
                        )?;
                        Some(super::prequant_fp4::CompactMoeWorklist {
                            worklist,
                            total_tiles,
                            max_tiles: total_expanded * n_tiles,
                        })
                    } else {
                        None
                    };
                    self.prequant_fp4_gate_up(
                        expert_input,
                        gp,
                        up,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        n,
                        h,
                        inter,
                        num_experts,
                        max_m_tiles,
                        compact,
                        ctx,
                        stream,
                    )?;
                } else if self.gateup_fp4 && self.moe_fused_gate_up_t_k64_fp4.0 != 0 {
                    // ── FUSED FP4 gate_up (ATLAS_HOLO_MOE_GATEUP_FP4) ──
                    // Block-scaled FP4 over the SHARED FAST_MOE=full [K/2,N] tables
                    // (gate_ptrs_t/up_ptrs_t — the SAME bytes the FP8 fused path
                    // reads, selected here only by kernel handle, so NO extra MoE
                    // memory). The kernel loads them coalesced K-major and re-gathers
                    // N-major on-chip (FP4_TRANSPOSE). gp/up carry the REAL per-expert
                    // scale2 (applied at writeback) — not the legacy hardcoded 1.0.
                    // Single launch, grid z = num_experts; writes C_gate/C_up in the
                    // same sorted layout as FP8 so silu+down+unpermute are unchanged.
                    ops::moe_w4a16_fused_gate_up_k64_n128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64_fp4,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles,
                        stream,
                    )?;
                } else {
                    // Block D #3 dispatch: M=128 path needs the env var on AND
                    // the kernel actually loaded (try_kernel returns 0 on
                    // models that don't ship it). max_m_tiles_m128 = ceil(...
                    // /128) instead of /64; reuse the same upper bound by
                    // halving (each m128 tile covers 2 m64 tiles).
                    let use_m128 =
                        self.nvfp4_gate_up_m128 && self.moe_fused_gate_up_t_k64_m128.0 != 0;
                    if use_m128 {
                        let max_m_tiles_m128 = max_m_tiles.div_ceil(2).max(1);
                        ops::moe_w4a16_fused_gate_up_k64_m128(
                            ctx.gpu,
                            self.moe_fused_gate_up_t_k64_m128,
                            expert_input,
                            gp.packed_ptrs,
                            gp.scale_ptrs,
                            gp.scale2_vals,
                            up.packed_ptrs,
                            up.scale_ptrs,
                            up.scale2_vals,
                            expert_gate_out,
                            expert_up_out,
                            expert_offsets,
                            sorted_token_ids,
                            num_experts,
                            inter,
                            h,
                            max_m_tiles_m128,
                            stream,
                        )?;
                    } else {
                        ops::moe_w4a16_fused_gate_up_k64_n128(
                            ctx.gpu,
                            self.moe_fused_gate_up_t_k64,
                            expert_input,
                            gp.packed_ptrs,
                            gp.scale_ptrs,
                            gp.scale2_vals,
                            up.packed_ptrs,
                            up.scale_ptrs,
                            up.scale2_vals,
                            expert_gate_out,
                            expert_up_out,
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
            } else {
                // ARM-2 Phase-K straggler net: V4 native builds gate_ptrs_t, so
                // E8M0 never reaches this non-transposed fallback. If it does,
                // panic (a real finding) rather than run NVFP4-on-E8M0 garbage.
                self.experts_scale_kind.expect(
                    crate::weight_map::WeightQuantFormat::Nvfp4,
                    "prefill non-transposed gate_up fallback (no E8M0 variant wired)",
                );
                let (gp, up) = (&self.gate_ptrs, &self.up_ptrs);
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    gp.packed_ptrs,
                    gp.scale_ptrs,
                    gp.scale2_vals,
                    expert_gate_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    up.packed_ptrs,
                    up.scale_ptrs,
                    up.scale2_vals,
                    expert_up_out,
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
        prof_step!("grouped_gate_up");

        // 6. Activation+mul for routed experts + grouped down GEMM (K64 pipelined).
        let expert_down_out = ctx.buffers.expert_down_out();
        if max_m_tiles > 0 {
            if self.nvfp4_mmq_layout {
                self.run_nvfp4_mmq_silu_down(
                    expert_input,
                    expert_gate_out,
                    expert_up_out,
                    expert_down_out,
                    expert_offsets,
                    sorted_token_ids,
                    total_expanded,
                    h,
                    inter,
                    num_experts,
                    max_m_tiles,
                    ctx,
                    stream,
                )?;
                prof_step!("grouped_silu_down");
                return Ok(());
            }
            // Feature-1: fold the routed-expert gate/up_proj LoRA deltas onto the
            // sorted `expert_gate_out`/`expert_up_out` BEFORE `silu_mul` consumes
            // them in place (x = token-major `expert_input`, gathered per sorted
            // row). No-op unless gate/up deltas are installed. Covers all five
            // nvfp4 gate_up sub-branches (all wrote sorted BF16 gate/up).
            self.apply_expert_lora_prefill_gateup(
                expert_gate_out,
                expert_up_out,
                expert_input,
                expert_offsets,
                sorted_token_ids,
                total_expanded,
                ctx,
                stream,
            )?;
            let fused_nvfp4_down = self.nvfp4_prequant_moe
                && self.nvfp4_fused_silu_quant
                && self.silu_mul_quant_nvfp4_k.0 != 0;
            if fused_nvfp4_down {
                self.fused_silu_prequant_fp4_down(
                    expert_gate_out,
                    expert_up_out,
                    total_expanded,
                    inter,
                    ctx,
                    stream,
                )?;
            } else {
                ops::silu_mul(
                    ctx.gpu,
                    self.moe_act_mul,
                    expert_gate_out,
                    expert_up_out,
                    expert_gate_out,
                    total_expanded * inter,
                    stream,
                )?;
            }
            // ── FP4 down (ATLAS_HOLO_MOE_DOWN_FP4) ── single block-scaled FP4
            // MMA per k64 tile (mxf4nvf4.scale_vec::4X.m16n8k64), reading the
            // post-SiLU intermediate (expert_gate_out) and the per-expert FP4
            // down tables. Same sorted layout + null sorted_token_ids as the
            // FP8/w4a16 down kernels, so unpermute downstream is unchanged.
            // Compounds with the FP4 gate_up path to run the whole FFN at FP4.
            // CUTLASS grouped down reads the ORIGINAL [N,K/2] table, so like
            // gate_up it must be reachable without down_ptrs_t.
            if grouped_cutlass_gate_up_enabled()
                && let Some(down_host) = self
                    .cutlass_grouped_host
                    .as_ref()
                    .and_then(|t| t.down.as_ref())
                && grouped_cutlass_down_enabled()
            {
                // ── CUTLASS grouped NVFP4 down (ATLAS_HOLO_MOE_GROUPED_CUTLASS
                //    + ATLAS_HOLO_MOE_GROUPED_DOWN) ──
                // A = post-SiLU expert_gate_out, already expert-contiguous (the grouped
                // gate_up wrote it sorted), so NO gather. Weights = the load-time host
                // snapshot of decode down_ptrs packed [N=hidden,K/2] + swizzled SFB +
                // real scale2. Writes expert_down_out in the sorted layout (unpermute
                // downstream unchanged).
                ops::moe_grouped_down_cutlass(
                    ctx.gpu,
                    cutlass_eoff.as_deref(),
                    down_host,
                    expert_gate_out,
                    expert_down_out,
                    expert_offsets,
                    h,
                    inter,
                    stream,
                )?;
            } else if let Some(dp) = &self.down_ptrs_t {
                if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                    // ── ARM-2 Phase-K: native-MXFP4 (E8M0) grouped down ──
                    // Leading branch (bypasses NVFP4-only cutlass/fp4/fp8_down).
                    assert!(
                        self.moe_grouped_gemm_t_k64_e8m0.0 != 0,
                        "ARM-2: routed experts Mxfp4E8m0 but grouped_gemm_t_k64_e8m0 unresolved"
                    );
                    ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                        ctx.gpu,
                        self.moe_grouped_gemm_t_k64_e8m0,
                        expert_gate_out,
                        dp.packed_ptrs,
                        dp.scale_ptrs,
                        dp.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        DevicePtr(0),
                        num_experts,
                        h,
                        inter,
                        max_m_tiles,
                        stream,
                    )?;
                } else if self.nvfp4_prequant_moe && self.moe_w4a4_prequant_t_k64.0 != 0 {
                    if !fused_nvfp4_down {
                        let a_packed = expert_up_out;
                        let a_scale = a_packed.offset(total_expanded as usize * inter as usize / 2);
                        ops::quantize_bf16_to_nvfp4(
                            ctx.gpu,
                            self.quantize_nvfp4_k,
                            expert_gate_out,
                            a_packed,
                            a_scale,
                            total_expanded,
                            inter,
                            stream,
                        )?;
                    }
                    self.prequant_fp4_down(
                        expert_up_out,
                        dp,
                        expert_down_out,
                        expert_offsets,
                        total_expanded,
                        h,
                        inter,
                        num_experts,
                        max_m_tiles,
                        ctx,
                        stream,
                    )?;
                } else if self.down_fp4 && self.moe_down_t_k64_fp4.0 != 0 {
                    // ── FP4 down (ATLAS_HOLO_MOE_DOWN_FP4) over the SHARED down_ptrs_t
                    // [K/2,N] table (real per-expert scale2; coalesced K-major load +
                    // on-chip DN4_TRANSPOSE). Same sorted layout + null
                    // sorted_token_ids as the FP8/w4a16 down kernels, so unpermute is
                    // unchanged. No extra MoE memory (shared table).
                    ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                        ctx.gpu,
                        self.moe_down_t_k64_fp4,
                        expert_gate_out,
                        dp.packed_ptrs,
                        dp.scale_ptrs,
                        dp.scale2_vals,
                        expert_down_out,
                        expert_offsets,
                        DevicePtr(0),
                        num_experts,
                        h,
                        inter,
                        max_m_tiles,
                        stream,
                    )?;
                } else {
                    let fp8_down = std::env::var("ATLAS_MOE_PREFILL_FP8_DOWN").ok().as_deref()
                        == Some("1")
                        && self.moe_fp8_grouped_gemm_t.0 != 0
                        && self.bf16_to_fp8_k.0 != 0;
                    if fp8_down {
                        ops::bf16_to_fp8(
                            ctx.gpu,
                            self.bf16_to_fp8_k,
                            expert_gate_out,
                            expert_up_out,
                            total_expanded * inter,
                            stream,
                        )?;
                        ops::moe_fp8_grouped_gemm_ptrtable_n128(
                            ctx.gpu,
                            self.moe_fp8_grouped_gemm_t,
                            expert_up_out,
                            dp.packed_ptrs,
                            dp.scale_ptrs,
                            dp.scale2_vals,
                            expert_down_out,
                            expert_offsets,
                            DevicePtr(0),
                            num_experts,
                            h,
                            inter,
                            max_m_tiles,
                            stream,
                        )?;
                    } else {
                        if self.nvfp4_down_m32 && self.moe_grouped_gemm_t_k64_m32.0 != 0 {
                            ops::moe_w4a16_grouped_gemm_ptrtable_k64_m32_n128(
                                ctx.gpu,
                                self.moe_grouped_gemm_t_k64_m32,
                                expert_gate_out,
                                dp.packed_ptrs,
                                dp.scale_ptrs,
                                dp.scale2_vals,
                                expert_down_out,
                                expert_offsets,
                                DevicePtr(0),
                                num_experts,
                                h,
                                inter,
                                max_m_tiles,
                                stream,
                            )?;
                        } else {
                            ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                                ctx.gpu,
                                self.moe_grouped_gemm_t_k64,
                                expert_gate_out,
                                dp.packed_ptrs,
                                dp.scale_ptrs,
                                dp.scale2_vals,
                                expert_down_out,
                                expert_offsets,
                                DevicePtr(0),
                                num_experts,
                                h,
                                inter,
                                max_m_tiles,
                                stream,
                            )?;
                        }
                    }
                }
            } else {
                // ARM-2 Phase-K straggler net (see gate_up fallback above).
                self.experts_scale_kind.expect(
                    crate::weight_map::WeightQuantFormat::Nvfp4,
                    "prefill non-transposed down fallback (no E8M0 variant wired)",
                );
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_gate_out,
                    self.down_ptrs.packed_ptrs,
                    self.down_ptrs.scale_ptrs,
                    self.down_ptrs.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_silu_down");

        Ok(())
    }
}
