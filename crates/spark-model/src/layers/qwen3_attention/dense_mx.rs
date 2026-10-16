// SPDX-License-Identifier: AGPL-3.0-only

//! A checkpoint's own MXFP8 attention projections as the decode copy of the
//! BF16 (`--attn-proj-dtype bf16`-shaped) build.
//!
//! `nvidia/Qwen3.8-Flash-Next-FP4-Local` ships q/k/v/o_proj of its 12 QSA
//! layers as MXFP8. Requantizing their BF16 dequant to NVFP4, the default
//! for a BF16 checkpoint, would be a lossy round trip the checkpoint did not
//! make, so the loader builds the exact BF16 copy (prefill) and installs the
//! MXFP8 bytes here, keyed by WHICH of the layer's BF16 slots they stand for
//! (`attn.{q,k,v}_proj`, `o_dense_bf16`), not by its pointer: under
//! `ATLAS_QWEN4EXP_PREFILL_W8A8` prefill reads these bytes too (`w8a8.rs`)
//! and the BF16 copies are freed, their slots NULL. Every dense decode site
//! asks [`Qwen3AttentionLayer::dense_gemv_or_mx`] (one row) or
//! [`Qwen3AttentionLayer::mx_rows`] (rows) and gets the twin when one
//! exists: serial decode (`attention_forward*`), the multi-sequence / verify
//! Q/K/V (`multi_seq/qkv_mx.rs`, ahead of every other arm) and O
//! (`multi_seq/attn/o_proj.rs`) then compute every row with the same
//! row-invariant tiers.
//!
//! Bytes against BF16: 1.03 a weight against 2, ~13.6 MB a layer less on a
//! TP2 rank. Against the NVFP4 requantization a BF16 checkpoint takes, it is
//! ~0.47 bytes a weight MORE: the price of decoding the checkpoint's math.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3AttentionLayer;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Mxfp8Weight};

/// The MXFP8 twins of Q, K, V and O (in that order, each with its `[n, k]`;
/// empty without), and their W8A8 prefill operands (`w8a8.rs`).
#[derive(Default)]
pub(crate) struct DenseMx {
    pub(super) twins: Vec<([usize; 2], Mxfp8Weight)>,
    pub(super) w8a8: Option<super::w8a8::AttnW8a8>,
}

impl DenseMx {
    pub(crate) fn is_empty(&self) -> bool {
        self.twins.is_empty()
    }
}

impl Qwen3AttentionLayer {
    /// Install the MXFP8 twins of this layer's BF16 Q, K, V (`attn.*_proj`)
    /// and O (`o_dense_bf16`), in that order, each with its `[n, k]`. Load
    /// time only.
    pub(crate) fn install_dense_mxfp8(
        &mut self,
        [q, k, v, o]: [(Mxfp8Weight, [usize; 2]); 4],
    ) -> Result<()> {
        ensure!(
            self.mxfp8_gemv_k.iter().all(|k| k.0 != 0),
            "MXFP8 attention twins but the target lacks mxfp8_gemv_tc{{8,16,32}}"
        );
        // Decode must reach the dense arms: no NVFP4 / FP8 / packed copy
        // ahead of them, and a BF16 O for the `o_dense_bf16` arm.
        let o_bf16 = self.o_dense_bf16.map(|o| o.weight);
        let own = [
            Some(self.attn.q_proj.weight),
            Some(self.attn.k_proj.weight),
            Some(self.attn.v_proj.weight),
            o_bf16,
        ];
        ensure!(
            own.iter().all(|w| w.is_some_and(|p| !p.is_null()))
                && self.q_weight.is_none()
                && self.k_weight.is_none()
                && self.v_weight.is_none()
                && self.o_weight.is_none(),
            "MXFP8 attention twins need the BF16 Q/K/V/O build (--attn-proj-dtype bf16 shape)"
        );
        self.dense_mx.twins = [q, k, v, o].into_iter().map(|(mx, nk)| (nk, mx)).collect();
        Ok(())
    }

    /// Which twin `w` stands for: the slot of the layer's own field it is
    /// (Q, K, V, O), by address, since the BF16 pointer may be NULL by now.
    pub(super) fn mx_slot(&self, w: &DenseWeight) -> Option<usize> {
        let own = [
            Some(&self.attn.q_proj),
            Some(&self.attn.k_proj),
            Some(&self.attn.v_proj),
            self.o_dense_bf16.as_ref(),
        ];
        own.iter()
            .position(|o| o.is_some_and(|o| std::ptr::eq(o, w)))
    }

    fn mx_twin(&self, w: &DenseWeight, n: u32, k: u32) -> Result<Option<&Mxfp8Weight>> {
        let Some((nk, mx)) = self.mx_slot(w).and_then(|i| self.dense_mx.twins.get(i)) else {
            return Ok(None);
        };
        ensure!(
            *nk == [n as usize, k as usize],
            "MXFP8 twin {nk:?} asked for [{n}, {k}]"
        );
        Ok(Some(mx))
    }

    /// `rows` contiguous rows of `input` (`[rows, k]`) through BF16 weight
    /// `w`'s MXFP8 twin into `output` (rows `out_stride` elements apart);
    /// `Ok(false)`, nothing launched, when `w` has none.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mx_rows(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &DenseWeight,
        output: DevicePtr,
        rows: usize,
        n: u32,
        k: u32,
        out_stride: usize,
        stream: u64,
    ) -> Result<bool> {
        let Some(mx) = self.mx_twin(w, n, k)? else {
            return Ok(false);
        };
        let (n, k) = (n as usize, k as usize);
        ops::mxfp8_gemv_rows(
            gpu,
            &self.mxfp8_gemv_k,
            mx,
            input,
            output,
            rows,
            n,
            k,
            out_stride,
            stream,
        )?;
        Ok(true)
    }

    /// `ops::dense_gemv` (one row), or its MXFP8 twin when `w` has one.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_gemv_or_mx(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &DenseWeight,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.mx_rows(gpu, input, w, output, 1, n, k, n as usize, stream)? {
            return Ok(());
        }
        ops::dense_gemv(gpu, self.dense_gemv_k, input, w, output, n, k, stream)
    }
}
