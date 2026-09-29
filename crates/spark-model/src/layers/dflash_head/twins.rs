// SPDX-License-Identifier: AGPL-3.0-only

//! Quantized drafter twins on the tensor-core GEMV tiers.
//!
//! main's drafter NVFP4 twins serve only blocks of up to four rows through
//! the scalar `w4a16_gemv_batch4`, so a γ=8 DFlash2 drafter (GLM-5.3) read
//! its BF16 projections every proposal. These levers serve every row block
//! from smaller twins instead. All of them change drafts only — the target
//! verifies every token — so they can move acceptance, never output:
//!
//! - `ATLAS_DFLASH_NVFP4_TC=1`: the layer projections' NVFP4 twins on the
//!   tensor-core `w4a16_gemv_tc{8,16,32}` tiers (`ATLAS_W4A16_TC=1` resolves
//!   them), in 32-row pieces for batched proposals.
//! - `ATLAS_DFLASH_MXFP8=1`: MXFP8 twins (E4M3 + one E8M0 scale per 32
//!   values, lossless into BF16 MMA operands) of q/o/gate/up/down on
//!   `mxfp8_gemv_tc{8,16,32}`. With `ATLAS_DFLASH_NVFP4_TC=1` the NVFP4 tiers
//!   serve the layers instead and no MXFP8 layer twin is built.
//!
//! Row blocks the tiers cannot take fall back to the BF16 weights.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::BlockDiffusionDraftHead;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Mxfp8Weight, QuantizedWeight};

/// Startup-resolved twin state beyond main's per-layer NVFP4/FP8 fields.
#[derive(Default)]
pub struct DflashTwins {
    /// `ATLAS_DFLASH_NVFP4_TC=1`: layer projections take the NVFP4 twins on
    /// the tensor-core tiers at any row count (main: only up to 4 rows).
    pub nvfp4_tc: bool,
}

/// MXFP8 twins of a layer's five large projections (`ATLAS_DFLASH_MXFP8=1`);
/// k/v stay BF16 (the context precompute reads them through the fused K/V).
pub struct LayerMxfp8 {
    pub q_proj: Mxfp8Weight,
    pub o_proj: Mxfp8Weight,
    pub gate_proj: Mxfp8Weight,
    pub up_proj: Mxfp8Weight,
    pub down_proj: Mxfp8Weight,
}

fn env_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

impl BlockDiffusionDraftHead {
    /// Install the twins the `ATLAS_DFLASH_*` levers request, after the
    /// Phase G FP8 install. main's NVFP4 install is skipped when MXFP8 serves
    /// the layers (its twins would never be read at γ > 4).
    pub(super) fn install_twins(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let mxfp8 = env_on("ATLAS_DFLASH_MXFP8");
        let nvfp4_tc = env_on("ATLAS_DFLASH_NVFP4_TC");
        if mxfp8 && !nvfp4_tc {
            self.install_mxfp8_layers(gpu)?;
        } else {
            self.try_install_nvfp4(gpu)?;
        }
        self.twins.nvfp4_tc =
            nvfp4_tc && matches!(self.quant, super::DflashQuantization::Nvfp4Weights);
        if nvfp4_tc && !self.twins.nvfp4_tc {
            tracing::warn!(
                "ATLAS_DFLASH_NVFP4_TC=1 but the drafter NVFP4 twins were not installed \
                 (ATLAS_NO_DFLASH_DRAFTER_NVFP4, drafter FP8, or missing quantizer)"
            );
        }
        tracing::info!(
            "DFlash twins: NVFP4 tensor-core layers {}, MXFP8 layers {}",
            self.twins.nvfp4_tc,
            self.layers.iter().filter(|l| l.mx.is_some()).count(),
        );
        Ok(())
    }

    fn install_mxfp8_layers(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let h = self.hidden_size;
        let q_dim = self.num_q_heads * self.head_dim;
        let inter = self.intermediate_size;
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let mx = LayerMxfp8 {
                q_proj: self.quantize_mxfp8(gpu, &layer.q_proj, q_dim, h)?,
                o_proj: self.quantize_mxfp8(gpu, &layer.o_proj, h, q_dim)?,
                gate_proj: self.quantize_mxfp8(gpu, &layer.gate_proj, inter, h)?,
                up_proj: self.quantize_mxfp8(gpu, &layer.up_proj, inter, h)?,
                down_proj: self.quantize_mxfp8(gpu, &layer.down_proj, h, inter)?,
            };
            self.layers[i].mx = Some(mx);
        }
        gpu.synchronize(gpu.default_stream())
    }

    /// E4M3 `[n, k]` + E8M0 `[n, k/32]` twin of a BF16 `[n, k]` weight.
    pub(super) fn quantize_mxfp8(
        &self,
        gpu: &dyn GpuBackend,
        w: &DenseWeight,
        n: usize,
        k: usize,
    ) -> Result<Mxfp8Weight> {
        let quant = self.kernels.mxfp8_quantize;
        anyhow::ensure!(
            quant.0 != 0 && self.kernels.mxfp8_gemv.iter().all(|k| k.0 != 0),
            "ATLAS_DFLASH_MXFP8 twins need the mxfp8_gemv kernels (GLM-5.3 kernel target)"
        );
        anyhow::ensure!(
            w.weight.0 != 0,
            "DFlash MXFP8 twin of a dropped BF16 weight"
        );
        let data = gpu.alloc(n * k)?;
        let scales = gpu.alloc(n * k / ops::MXFP8_BLOCK)?;
        ops::mxfp8_quantize(
            gpu,
            quant,
            w.weight,
            data,
            scales,
            n,
            k,
            gpu.default_stream(),
        )?;
        Ok(Mxfp8Weight { data, scales })
    }

    /// `out[m, n] = input[m, k] · Wᵀ` from an NVFP4 twin on the tensor-core
    /// tiers, in 32-row pieces. `Ok(false)` (nothing launched) when a tier
    /// covering a piece is not resolved (`ATLAS_W4A16_TC` unset, or a target
    /// without the tiers).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn nvfp4_tc_rows(
        &self,
        gpu: &dyn GpuBackend,
        w: &QuantizedWeight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let tier = crate::layers::w4a16_gemv_tiers::tc_kernel;
        let pieces = || (0..m).step_by(32).map(move |r0| (r0, (m - r0).min(32)));
        if m == 0 || !pieces().all(|(_, rows)| tier(rows).0 != 0) {
            return Ok(false);
        }
        for (r0, rows) in pieces() {
            ops::w4a16_gemv_batchm(
                gpu,
                tier(rows),
                input.offset(r0 as usize * k as usize * 2),
                w,
                out.offset(r0 as usize * n as usize * 2),
                rows,
                n,
                k,
                stream,
            )?;
        }
        Ok(true)
    }

    /// `out[m, n] = input[m, k] · Wᵀ` from an MXFP8 twin (up to 32 rows).
    /// `Ok(false)` when the row count or the kernels rule it out.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mxfp8_rows(
        &self,
        gpu: &dyn GpuBackend,
        w: &Mxfp8Weight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let kernel = match m {
            1..=8 => self.kernels.mxfp8_gemv[0],
            9..=16 => self.kernels.mxfp8_gemv[1],
            17..=32 => self.kernels.mxfp8_gemv[2],
            _ => return Ok(false),
        };
        if kernel.0 == 0 {
            return Ok(false);
        }
        ops::mxfp8_gemv(
            gpu, kernel, input, w.data, w.scales, out, m, n, k, n, stream,
        )?;
        Ok(true)
    }
}
