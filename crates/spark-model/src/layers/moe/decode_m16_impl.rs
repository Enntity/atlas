// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl MoeLayer {
    /// Whether a routed FFN of `rows` rows is a verify-decode batch: the K5
    /// compact shape or one of the exact grouped decode entries.
    pub(in crate::layers::moe) fn verify_decode_batch(
        &self,
        compact_k5: bool,
        ctx: &ForwardContext,
        rows: u32,
    ) -> bool {
        compact_k5
            || self.glm_c2_grouped(ctx, rows)
            || self.glm_c3_grouped(ctx, rows)
            || self.glm_c4_grouped(ctx, rows)
            || self.independent_grouped(ctx, rows)
    }

    /// `ATLAS_GLM_MOE_STREAM_NOSYNC`: whether a routed FFN of `rows` rows
    /// runs the stream twins on the prequant path whatever its kind, so
    /// nothing reads a host copy of its expert offsets. This mirrors the
    /// dispatch of `run_routed_grouped_gemm`: no MMQ layout, no CUTLASS
    /// grouped GEMM, the transposed NVFP4 tables and the prequant kernel,
    /// then [`Self::prequant_tiles`] taking [`Self::decode_m16_grid`], whose
    /// gate/up and down both launch over its grid.
    pub(in crate::layers::moe) fn offsets_unread(&self, rows: u32, h: u32, inter: u32) -> bool {
        self.decode_m16.nosync && self.stream_m16_prequant(rows, h, inter)
    }

    /// Whether a routed FFN of `rows` rows can take the stream twins' W4A16
    /// kernels (`ATLAS_GLM_FP4L_DECODE_A16`): loaded, and the batch one the
    /// stream twins take.
    pub(in crate::layers::moe) fn a16_stream_fits(&self, rows: u32, h: u32, inter: u32) -> bool {
        self.decode_m16.a16.is_some()
            && rows <= WIDE_MAX_ROWS
            && self.stream_m16_prequant(rows, h, inter)
    }

    /// Whether a routed FFN of `rows` rows takes the stream twins' prequant path.
    pub(in crate::layers::moe) fn stream_m16_prequant(
        &self,
        rows: u32,
        h: u32,
        inter: u32,
    ) -> bool {
        // A packed prefill keeps the K128W grid at every row count.
        !crate::layers::glm_packed_scope::active()
            && self.decode_m16.stream()
            && self.m16_grid_fits(rows, self.decode_m16.max_rows(), h, inter)
            && !self.nvfp4_mmq_layout
            && !(super::forward_prefill_routed::grouped_cutlass_gate_up_enabled()
                && self.cutlass_grouped_host.is_some())
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.prequant_moe()
            && self.moe_w4a4_prequant_t_k64.0 != 0
    }

    /// Whether a prefill routed FFN of `rows` tokens takes the K128W
    /// prequant path in both projections (`prequant_tiles` handing back the
    /// K128W grid, `prequant_fp4_gate_up` / `prequant_grouped` launching over
    /// it), which sizes its work from the device prefix and never reads the
    /// hottest expert's tile count: the exact-tile host read is dead there.
    pub(in crate::layers::moe) fn k128w_prequant(
        &self,
        rows: u32,
        h: u32,
        inter: u32,
        decode_batch: bool,
    ) -> bool {
        // The batches another lane takes.
        let cutlass = (super::forward_prefill_routed::grouped_cutlass_gate_up_enabled()
            && self.cutlass_grouped_host.is_some())
            || super::forward_prefill_routed::grouped_cutlass_down_enabled();
        let elsewhere = decode_batch
            || self.stream_m16_prequant(rows, h, inter)
            || self.btile_storage.is_published()
            || self.nvfp4_mmq_layout
            || cutlass;
        !elsewhere
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.prequant_moe()
            && self.moe_w4a4_prequant_t_k64.0 != 0
            && self.moe_w4a4_prequant_t_k128w.grid.0 != 0
            && self.moe_mtile_prefix_k.0 != 0
            && h.is_multiple_of(256)
            && inter.is_multiple_of(256)
    }

    /// Whether the M16 kernels can take `rows` routed rows (at most
    /// `max_rows`): loaded, the shape tiled, the worklist builder present and
    /// the fused SiLU quantization the one serving would run.
    fn m16_grid_fits(&self, rows: u32, max_rows: u32, h: u32, inter: u32) -> bool {
        self.decode_m16.loaded()
            && m16_shape(rows, max_rows, h, inter)
            && self.moe_build_tile_worklist_k.0 != 0
            && self.nvfp4_fused_silu_quant
            && self.silu_mul_quant_nvfp4_k.0 != 0
            && self.lora.is_none()
    }

    /// The row tiles of one prequant routed FFN of `rows` rows. A verify
    /// decode batch (`decode`) takes the M16 grid when selected, else the
    /// K128W grid under `ATLAS_GLM_MOE_DECODE_K128W`, else the compact M64
    /// worklist; any other batch the stream twins' grid when they are loaded
    /// and fit, else the K128W grid when it is loaded.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::moe) fn prequant_tiles(
        &self,
        decode: bool,
        expert_offsets: DevicePtr,
        local_ptrs: DevicePtr,
        [rows, top_k]: [u32; 2],
        [h, inter]: [u32; 2],
        num_experts: u32,
        prerouted: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(Option<CompactMoeWorklist>, Option<MtileGrid>)> {
        let total_expanded = rows * top_k;
        let k128w = || {
            self.mtile_grid(
                expert_offsets,
                local_ptrs,
                total_expanded,
                num_experts,
                ctx,
                stream,
            )
        };
        let m16 =
            if !crate::layers::glm_packed_scope::active() && (decode || self.decode_m16.stream()) {
                self.decode_m16_grid(
                    expert_offsets,
                    local_ptrs,
                    rows,
                    total_expanded,
                    [h, inter],
                    num_experts,
                    prerouted,
                    ctx,
                    stream,
                )?
            } else {
                None
            };
        self.decode_m16.note(ctx.gpu, rows, m16.is_some());
        if m16.is_some() {
            return Ok((None, m16));
        }
        if !decode {
            return Ok((None, k128w()?));
        }
        if self.decode_m16.k128w {
            let grid = k128w()?;
            if ctx
                .gpu
                .op_cache()
                .first_shape("moe:decode_k128w", rows, grid.is_some() as u32, 0)
            {
                tracing::info!(
                    "ATLAS_GLM_MOE_DECODE_K128W {}: rows={rows}",
                    if grid.is_some() {
                        "engaged"
                    } else {
                        "not used (K128W kernels not loaded)"
                    }
                );
            }
            if grid.is_some() {
                return Ok((None, grid));
            }
        }
        let total_tiles = ctx.buffers.moe_router_in_f32();
        let worklist = total_tiles.offset(16);
        let n_tiles = inter.div_ceil(128);
        anyhow::ensure!(
            ctx.buffers.sizes().moe_router_in_f32
                >= super::prequant_fp4::compact_gate_up_worklist_bytes(rows, top_k, inter),
            "compact native-FP4 gate/up worklist exceeds router scratch"
        );
        ops::moe_build_tile_worklist(
            ctx.gpu,
            self.moe_build_tile_worklist_k,
            expert_offsets,
            local_ptrs,
            worklist,
            total_tiles,
            num_experts,
            n_tiles,
            64,
            stream,
        )?;
        let compact = CompactMoeWorklist {
            worklist,
            total_tiles,
            max_tiles: total_expanded * n_tiles,
        };
        Ok((Some(compact), None))
    }

    /// The row-tile grid of the M16 kernels over a verify batch of `rows`
    /// rows (`total_expanded` sorted rows), when they are loaded and the
    /// fused SiLU quantization they apply is the one serving would run. Its
    /// "prefix" is the compact worklist scratch with one N tile per routed
    /// local expert, which the twins index by grid row. The grid bound is
    /// host-static: there are at most `total_expanded` routed experts.
    /// `prerouted`: `glm_moe_route_select` already built that worklist.
    #[allow(clippy::too_many_arguments)]
    fn decode_m16_grid(
        &self,
        expert_offsets: DevicePtr,
        local_ptrs: DevicePtr,
        rows: u32,
        total_expanded: u32,
        [h, inter]: [u32; 2],
        num_experts: u32,
        prerouted: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<MtileGrid>> {
        let k = self.decode_m16;
        if !self.m16_grid_fits(rows, k.max_rows(), h, inter) {
            return Ok(None);
        }
        let total_tiles = ctx.buffers.moe_router_in_f32();
        anyhow::ensure!(
            ctx.buffers.sizes().moe_router_in_f32 >= 16 + total_expanded as usize * 8,
            "M16 decode worklist exceeds router scratch"
        );
        if !prerouted {
            ops::moe_build_tile_worklist(
                ctx.gpu,
                self.moe_build_tile_worklist_k,
                expert_offsets,
                local_ptrs,
                total_tiles.offset(16),
                total_tiles,
                num_experts,
                1,
                64,
                stream,
            )?;
        }
        let grid_only = |grid| ops::K128wKernel {
            grid,
            persist: KernelHandle(0),
        };
        let [gate_up_silu, down] = k.pair(rows, self.a16_stream());
        // The persistent twins stride over the worklist's items themselves.
        let schedule = match k.ctas {
            0 => ops::K128wSchedule::Grid {
                bound: total_expanded.min(num_experts),
            },
            ctas => ops::K128wSchedule::Stride { ctas },
        };
        Ok(Some(MtileGrid {
            prefix: total_tiles,
            schedule,
            rows: total_expanded,
            gate_up_silu: grid_only(gate_up_silu),
            down: grid_only(down),
        }))
    }
}
