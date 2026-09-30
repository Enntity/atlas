// SPDX-License-Identifier: AGPL-3.0-only

//! M16 twins of the K128W routed-expert kernels for GLM verify decode
//! (`ATLAS_GLM_MOE_DECODE_M16=1`, default off).
//!
//! A verify step routes at most [`MAX_ROWS`] rows, and top-k picks an expert
//! at most once per row, so no expert holds more than one m16 MMA slab. The
//! compact k64 gate/up, `silu_mul_quant_nvfp4` and dense K128 down that
//! decode launches otherwise pad every tile to 64 rows; the twins compute
//! the same MMAs per output element over 16-row tiles with the SiLU in the
//! gate/up epilogue, so the bytes are identical (`moe_decode_bench`).

use super::prequant_fp4::MtileGrid;
use super::*;

/// Rows of a routed FFN batch the M16 tiles cover: one m16 MMA slab.
pub(super) const MAX_ROWS: u32 = 16;

/// The M16 fused gate/up and down kernels; both null unless the flag is on
/// and the target ships them.
#[derive(Clone, Copy)]
pub(super) struct DecodeM16 {
    gate_up_silu: KernelHandle,
    down: KernelHandle,
}

impl DecodeM16 {
    pub(super) fn new(gpu: &dyn GpuBackend, config: &atlas_core::config::ModelConfig) -> Self {
        let requested = std::env::var("ATLAS_GLM_MOE_DECODE_M16").as_deref() == Ok("1");
        let on = requested && config.model_type == "glm5_next";
        let kernel = |name| super::super::try_kernel_gated(on, gpu, "moe_w4a16", name);
        let mut this = Self {
            gate_up_silu: kernel("glm_moe_decode_m16_gate_up_silu_k128w"),
            down: kernel("glm_moe_decode_m16_k128w"),
        };
        if !this.loaded() {
            // Both or neither: half a pair never launches.
            this.gate_up_silu = KernelHandle(0);
        }
        if requested && gpu.op_cache().once("moe:decode_m16") {
            if this.loaded() {
                tracing::info!(
                    "ATLAS_GLM_MOE_DECODE_M16: M16 routed gate/up+SiLU and down for verify decode"
                );
            } else if on {
                tracing::warn!("ATLAS_GLM_MOE_DECODE_M16=1 ignored: target lacks the M16 kernels");
            }
        }
        this
    }

    fn loaded(&self) -> bool {
        self.gate_up_silu.0 != 0 && self.down.0 != 0
    }
}

/// Whether `rows` routed rows of `[inter, h]` experts fit the M16 tiles: one
/// slab of rows, K128 stages, and 128 gate/up and 256 down columns per tile.
pub(super) fn m16_shape(rows: u32, h: u32, inter: u32) -> bool {
    (1..=MAX_ROWS).contains(&rows) && inter.is_multiple_of(128) && h.is_multiple_of(256)
}

impl MoeLayer {
    /// The row-tile grid of the M16 kernels over a verify batch of `rows`
    /// rows (`total_expanded` sorted rows), when they are loaded and the
    /// fused SiLU quantization they apply is the one serving would run. Its
    /// "prefix" is the compact worklist scratch with one N tile per routed
    /// local expert, which the twins index by grid row. The grid bound is
    /// host-static: there are at most `total_expanded` routed experts.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_m16_grid(
        &self,
        expert_offsets: DevicePtr,
        local_ptrs: DevicePtr,
        rows: u32,
        total_expanded: u32,
        [h, inter]: [u32; 2],
        num_experts: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<MtileGrid>> {
        let k = self.decode_m16;
        if !k.loaded()
            || !m16_shape(rows, h, inter)
            || self.moe_build_tile_worklist_k.0 == 0
            || !self.nvfp4_fused_silu_quant
            || self.silu_mul_quant_nvfp4_k.0 == 0
            || self.lora.is_some()
        {
            return Ok(None);
        }
        let total_tiles = ctx.buffers.moe_router_in_f32();
        anyhow::ensure!(
            ctx.buffers.sizes().moe_router_in_f32 >= 16 + total_expanded as usize * 8,
            "M16 decode worklist exceeds router scratch"
        );
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
        let grid_only = |grid| ops::K128wKernel {
            grid,
            persist: KernelHandle(0),
        };
        Ok(Some(MtileGrid {
            prefix: total_tiles,
            schedule: ops::K128wSchedule::Grid {
                bound: total_expanded.min(num_experts),
            },
            rows: total_expanded,
            gate_up_silu: grid_only(k.gate_up_silu),
            down: grid_only(k.down),
        }))
    }
}
