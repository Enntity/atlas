// SPDX-License-Identifier: AGPL-3.0-only

//! FP4-Local's GDN prefill projections as MXFP8 W8A8 GEMMs
//! (`ATLAS_QWEN4EXP_PREFILL_W8A8`, `ops::qwen4exp_w8a8`): qkvz and out_proj
//! multiply the decode twins' E4M3 bytes (`init_mxfp8.rs`) on cuBLASLt's
//! block-scaled tensor cores, and the BF16 copies they replace are freed:
//! 57.7 MB a layer on a TP2 rank (qkvz 8192 x 2560, out_proj 2560 x 3072),
//! ~2.1 GB over the 36 GDN layers.
//!
//! Nothing reads those BF16 copies afterwards. Decode reads the twins on
//! every arm (`init_mxfp8.rs`), and every prefill call reaches the W8A8 arm
//! first (`trait_prefill_proj.rs`, `trait_prefill_helper.rs`), the SP
//! reduce-scatter pipe's row ranges included. `in_proj_a` / `in_proj_b`
//! (`ssm.in_proj_ba`) are not touched: the fused BA + gates kernel reads them
//! as BF16. The freed slots are NULL (`out_proj_dense` stays `Some`, so the
//! decode dispatch that asks for a dense out_proj before it reaches the twin
//! keeps its shape): a reader missed here fails its NULL check or faults,
//! never reads freed memory.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3SsmLayer;
use super::exact_rows::GdnProj;
use crate::layer::ForwardContext;
use crate::layers::ops::{
    MX_QUANT_KERNEL, MXFP8_ACT_MODULE, MxW8a8, W8a8Swap, qwen4exp_w8a8, undo_on_err,
};

/// The W8A8 operands of the two projections, and the activation quant.
pub(crate) struct GdnW8a8 {
    qkvz: MxW8a8,
    out: MxW8a8,
    quant: KernelHandle,
}

impl Qwen3SsmLayer {
    /// Install the W8A8 prefill over the MXFP8 decode twins
    /// (`install_mxfp8_decode` first), its GEMMs tuned to `m_max` rows a pass,
    /// and take the BF16 qkvz and out_proj out of the layer, for the loader to
    /// free. Load time; synchronizes `stream`.
    pub(crate) fn install_w8a8_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        m_max: usize,
        stream: u64,
    ) -> Result<W8a8Swap> {
        let mx = self
            .gdn_mx
            .as_mut()
            .context("W8A8 GDN prefill needs the MXFP8 decode twins")?;
        let out_dense = self
            .out_proj_dense
            .as_mut()
            .context("W8A8 GDN prefill: no BF16 out_proj to replace")?;
        ensure!(
            mx.w8a8.is_none()
                && !self.ssm.in_proj_qkvz.weight.is_null()
                && !out_dense.weight.is_null(),
            "W8A8 GDN prefill: already installed, or the BF16 copies are gone"
        );
        let quant = gpu.kernel(MXFP8_ACT_MODULE, MX_QUANT_KERNEL)?;
        let qkvz = MxW8a8::from_twin(gpu, &mx.qkvz.0, mx.qkvz.1, m_max, stream)?;
        let out = undo_on_err(
            MxW8a8::from_twin(gpu, &mx.out.0, mx.out.1, m_max, stream),
            || qkvz.free(gpu),
        )?;
        let allocated = qkvz.owned_bytes() + out.owned_bytes();
        let bf16 = vec![
            (
                std::mem::replace(&mut self.ssm.in_proj_qkvz.weight, DevicePtr::NULL),
                qkvz.n * qkvz.k * 2,
            ),
            (
                std::mem::replace(&mut out_dense.weight, DevicePtr::NULL),
                out.n * out.k * 2,
            ),
        ];
        mx.w8a8 = Some(GdnW8a8 { qkvz, out, quant });
        Ok(W8a8Swap { allocated, bf16 })
    }

    /// Whether the layer carries the MXFP8 decode twins W8A8 reads.
    pub(crate) fn has_mxfp8_twins(&self) -> bool {
        self.gdn_mx.is_some()
    }

    fn w8a8(&self) -> Option<&GdnW8a8> {
        self.gdn_mx.as_ref()?.w8a8.as_ref()
    }

    /// `rows` contiguous rows of `input` through `proj`'s W8A8 GEMM into
    /// `output` (contiguous at the projection's width). `Ok(false)`, nothing
    /// launched, on a layer without W8A8.
    pub(crate) fn w8a8_proj(
        &self,
        ctx: &ForwardContext,
        proj: GdnProj,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<bool> {
        let Some(w) = self.w8a8() else {
            return Ok(false);
        };
        if rows == 0 {
            return Ok(true);
        }
        let wt = match proj {
            GdnProj::Qkvz => &w.qkvz,
            GdnProj::Out => &w.out,
        };
        let act =
            qwen4exp_w8a8::activation(ctx.gpu, w.quant, ctx.buffers, input, rows, wt.k, stream)?;
        wt.gemm(act, output, rows, stream)?;
        Ok(true)
    }

    /// Arm the collapse that writes rows `[row0, row0 + rows)` of the
    /// qkvz input `x` (`[total, hidden]`) to emit their MXFP8 image too
    /// (`ops::qwen4exp_w8a8::Producer`). `None` on a layer without W8A8.
    pub(super) fn w8a8_producer(
        &self,
        ctx: &ForwardContext,
        x: DevicePtr,
        total: usize,
        row0: usize,
        rows: usize,
    ) -> Option<qwen4exp_w8a8::Producer> {
        self.w8a8()?;
        let h = ctx.config.hidden_size;
        qwen4exp_w8a8::Producer::arm(ctx.buffers, x, total, row0, rows, h)
    }
}
