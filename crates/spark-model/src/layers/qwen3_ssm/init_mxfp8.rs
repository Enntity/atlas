// SPDX-License-Identifier: AGPL-3.0-only

//! The checkpoint's own MXFP8 GDN projections as the decode copy.
//!
//! `nvidia/Qwen3.8-Flash-Next-FP4-Local` ships `in_proj_qkv`, `in_proj_z` and
//! `out_proj` as MXFP8 (E4M3 + one UE8M0 scale per 32 K values). The loader
//! dequantizes them to BF16 exactly for prefill (cuBLASLt / tensor-core GEMMs,
//! and the parity reference) and installs the checkpoint's bytes here, sliced
//! to the rank but never requantized, for decode. Same shape as the 128-block
//! FP8 option (`init_fp8.rs`, `ATLAS_QWEN4EXP_FP8_GDN=1`) minus its rounding:
//! `mxfp8_gemv_tc*` turns each E4M3 byte times its power-of-two scale into
//! the exact BF16 operand the dequantized copy holds, so the MXFP8 and BF16
//! GEMVs differ only in FP32 summation order. Under
//! `ATLAS_QWEN4EXP_PREFILL_W8A8` (`w8a8.rs`) prefill reads these bytes too,
//! as W8A8 GEMMs, and the BF16 copies are freed.
//!
//! Bytes: 1.03 a weight against BF16's 2. Per TP2 rank that is 29.5 MB a
//! layer against 57.7 MB, ~1.0 GB less read per decode step over the 36 GDN
//! layers.
//!
//! EVERY decode arm reads this copy: serial decode (`ssm_forward`,
//! `ssm_out_proj`, one row) and every batched one (verify, multi-sequence,
//! `ATLAS_QWEN4EXP_BATCH_FAST`) through `exact_rows_proj`. The tiers are
//! row-invariant (`ops::mxfp8_gemv_rows`), so a verify row reproduces the
//! serial token's bytes, which is what keeps spec decode equal to serial.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3SsmLayer;
use super::exact_rows::GdnProj;
use crate::layers::ops;
use crate::weight_map::Mxfp8Weight;

/// The two twins, their `[n, k]`, the `mxfp8_gemv_tc{8,16,32}` tiers, and,
/// under `ATLAS_QWEN4EXP_PREFILL_W8A8`, the prefill's W8A8 operands over the
/// same bytes (`w8a8.rs`).
pub(crate) struct GdnMxfp8 {
    pub(super) qkvz: (Mxfp8Weight, [usize; 2]),
    pub(super) out: (Mxfp8Weight, [usize; 2]),
    tiers: [KernelHandle; 3],
    pub(super) w8a8: Option<super::w8a8::GdnW8a8>,
}

impl Qwen3SsmLayer {
    /// Install the MXFP8 decode copies of a BF16 GDN build: `qkvz` the local
    /// `[Q|K|V|Z]` rows `[qkvz, hidden]`, `out` the rank's `out_proj` columns
    /// `[hidden, value_dim]`, each with its `[n, k]`. Load time only; the
    /// BF16 copies stay for prefill.
    pub(crate) fn install_mxfp8_decode(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        qkvz: (Mxfp8Weight, [usize; 2]),
        out: (Mxfp8Weight, [usize; 2]),
    ) -> Result<()> {
        // Only the sequential BF16 build has every decode arm routed here;
        // a requantized (NVFP4 / FP8) or packed copy would be a second
        // decode format beside this one.
        ensure!(
            self.sequential_qkvz
                && !self.ssm.in_proj_qkvz.weight.is_null()
                && self.out_proj_dense.is_some()
                && self.qkvz_nvfp4.is_none()
                && self.qkvz_fp8w.is_none()
                && self.out_proj_fp8w.is_none()
                && self.qkvz_q2.is_none(),
            "install_mxfp8_decode: not a sequential BF16 GDN build"
        );
        let h = config.hidden_size;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let want = ([config.ssm_qkvz_size(), h], [h, value_dim]);
        ensure!(
            (qkvz.1, out.1) == want,
            "install_mxfp8_decode: twins {:?} / {:?}, layer wants {:?} / {:?}",
            qkvz.1,
            out.1,
            want.0,
            want.1,
        );
        let tiers = [
            gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc8")?,
            gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc16")?,
            gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc32")?,
        ];
        self.gdn_mx = Some(GdnMxfp8 {
            qkvz,
            out,
            tiers,
            w8a8: None,
        });
        Ok(())
    }

    /// `rows` contiguous rows of `input` through `proj`'s MXFP8 copy into
    /// `output` (rows contiguous at the projection's width). `Ok(false)`,
    /// nothing launched, when the layer has no MXFP8 copy.
    pub(super) fn mxfp8_rows_proj(
        &self,
        gpu: &dyn GpuBackend,
        proj: GdnProj,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<bool> {
        let Some(mx) = self.gdn_mx.as_ref() else {
            return Ok(false);
        };
        let (w, [n, k]) = match proj {
            GdnProj::Qkvz => &mx.qkvz,
            GdnProj::Out => &mx.out,
        };
        ops::mxfp8_gemv_rows(gpu, &mx.tiers, w, input, output, rows, *n, *k, *n, stream)?;
        Ok(true)
    }
}
