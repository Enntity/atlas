// SPDX-License-Identifier: AGPL-3.0-only

//! The PLE conv of one prefill span, with the qwen4_exp mid-chunk checkpoint
//! (`layers::qwen4exp_ckpt`) split. Split out of `layer.rs` for the 500-LoC
//! cap.

use super::*;

impl PleLayer {
    /// `ple_conv` over the span's `n` rows starting at forward row `base`
    /// (scratch row `srow`).
    /// When a checkpoint pass captures the carry at a forward row inside the
    /// span (or at its end), the conv runs as two launches split there and
    /// the carry at the split is copied out: the carry chains across launches
    /// exactly as across spans and per-row verify launches, so every output
    /// byte is unchanged.
    pub(super) fn conv_span(
        &self,
        st: &PleSeqState,
        srow: usize,
        base: usize,
        n: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let c = self.hc_mult * self.hidden;
        let conv = |r0: usize, rows: usize| -> Result<()> {
            if rows == 0 {
                return Ok(());
            }
            let off = (srow + r0) * c * 4; // [T, c] FP32
            ops::ple_conv(
                gpu,
                self.conv_k,
                self.gated_normed.offset(off),
                self.gated.offset(off),
                self.conv1d.weight,
                st.conv,
                self.out.offset(off),
                rows as u32,
                c as u32,
                self.k_size as u32,
                self.dilation as u32,
                stream,
            )
        };
        let cap = crate::layers::qwen4exp_ckpt::ple_capture()
            .filter(|&(row, _)| row > base && row <= base + n);
        let Some((row, dst)) = cap else {
            return conv(0, n);
        };
        let split = row - base;
        conv(0, split)?;
        gpu.copy_d2d_async(st.conv, dst, self.conv_bytes(), stream)?;
        crate::layers::qwen4exp_ckpt::ple_captured();
        conv(split, n - split)
    }
}
