// SPDX-License-Identifier: AGPL-3.0-only

//! FP4-Local's QSA attention prefill projections as MXFP8 W8A8 GEMMs
//! (`ATLAS_QWEN4EXP_PREFILL_W8A8`, `ops::qwen4exp_w8a8`): q(+gate), k, v and
//! o multiply the decode twins' E4M3 bytes (`dense_mx.rs`) on cuBLASLt's
//! block-scaled tensor cores, and the BF16 copies they replace are freed:
//! 49.8 MB a layer on a TP2 rank (q 6144 x 2560, k and v 256 x 2560, o
//! 2560 x 3072), ~0.6 GB over the 12 layers.
//!
//! q, k and v read one activation image of their shared input (from the
//! attention-site collapse where it emitted one, else one quant pass); o
//! quantizes the attention output. Under the SP reduce-scatter pipe
//! (`ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE`) o runs by row ranges with the
//! exchange behind it, as the FP8 arm it stands in for does; per-row quant
//! and in-order kernels keep every byte the unsplit GEMM's.
//!
//! Nothing reads the freed BF16 copies: decode finds the twins by slot
//! (`dense_mx.rs`), and every prefill projection reaches its W8A8 arm before
//! a BF16 one (`prefill/paged_qkv.rs`, `prefill/cache_skip_qkv.rs`,
//! `prefill/paged_oproj.rs`). The QSA indexer's `index_qk_proj` is its own
//! BF16 weight, untouched. LoRA folds its delta beside the BF16 arms only,
//! so a layer with an adapter refuses W8A8 at its first prefill.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops::{
    MX_QUANT_KERNEL, MXFP8_ACT_MODULE, MxAct, MxW8a8, W8a8Swap, qwen4exp_w8a8, undo_on_err,
};
use crate::layers::qwen4exp_sp_pipe as pipe;

/// The W8A8 operands of Q, K, V, O (`dense_mx` slot order) and the
/// activation quant.
pub(crate) struct AttnW8a8 {
    proj: [MxW8a8; 4],
    quant: KernelHandle,
}

const LORA: &str = "ATLAS_QWEN4EXP_PREFILL_W8A8 projections carry no LoRA delta; \
                    set ATLAS_QWEN4EXP_PREFILL_W8A8=0 to serve an adapter";

impl Qwen3AttentionLayer {
    /// Install the W8A8 prefill over the MXFP8 decode twins
    /// (`install_dense_mxfp8` first), its GEMMs tuned to `m_max` rows a pass,
    /// and take the BF16 Q, K, V and O out of the layer, for the loader to
    /// free. Load time; synchronizes `stream`.
    pub(crate) fn install_w8a8_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        m_max: usize,
        stream: u64,
    ) -> Result<W8a8Swap> {
        let o_bf16 = self
            .o_dense_bf16
            .as_mut()
            .context("W8A8 attention prefill: no BF16 o_proj to replace")?;
        let slots = [
            &mut self.attn.q_proj.weight,
            &mut self.attn.k_proj.weight,
            &mut self.attn.v_proj.weight,
            &mut o_bf16.weight,
        ];
        ensure!(
            self.dense_mx.twins.len() == 4
                && self.dense_mx.w8a8.is_none()
                && slots.iter().all(|p| !p.is_null()),
            "W8A8 attention prefill needs the MXFP8 decode twins and the BF16 Q/K/V/O"
        );
        let quant = gpu.kernel(MXFP8_ACT_MODULE, MX_QUANT_KERNEL)?;
        let mut proj: Vec<MxW8a8> = Vec::with_capacity(4);
        for (nk, twin) in &self.dense_mx.twins {
            let w = undo_on_err(MxW8a8::from_twin(gpu, twin, *nk, m_max, stream), || {
                // Free them all; the first failure is the one logged.
                proj.drain(..)
                    .map(|w| w.free(gpu))
                    .fold(Ok(()), Result::and)
            })?;
            proj.push(w);
        }
        let proj: [MxW8a8; 4] = proj.try_into().expect("four twins");
        let allocated = proj.iter().map(MxW8a8::owned_bytes).sum();
        let bf16 = slots
            .into_iter()
            .zip(&proj)
            .map(|(p, w)| (std::mem::replace(p, DevicePtr::NULL), w.n * w.k * 2))
            .collect();
        self.dense_mx.w8a8 = Some(AttnW8a8 { proj, quant });
        Ok(W8a8Swap { allocated, bf16 })
    }

    fn w8a8(&self) -> Result<Option<&AttnW8a8>> {
        let w = self.dense_mx.w8a8.as_ref();
        ensure!(w.is_none() || self.lora.is_none(), LORA);
        Ok(w)
    }

    /// The MXFP8 image of the q/k/v input `normed` (`[n, hidden]`); `None`
    /// on a layer without W8A8.
    pub(crate) fn w8a8_qkv_act(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<Option<MxAct>> {
        let Some(w) = self.w8a8()? else {
            return Ok(None);
        };
        let (gpu, k) = (ctx.gpu, w.proj[0].k);
        let act =
            qwen4exp_w8a8::activation(gpu, w.quant, ctx.buffers, normed, n as usize, k, stream)?;
        Ok(Some(act))
    }

    /// Projection `slot` (0 Q, 1 K, 2 V) of `act`'s `n` rows into `out`.
    pub(crate) fn w8a8_qkv(
        &self,
        slot: usize,
        act: MxAct,
        out: DevicePtr,
        [n, out_dim]: [u32; 2],
        stream: u64,
    ) -> Result<()> {
        let w = self.w8a8()?.context("W8A8 q/k/v without W8A8 weights")?;
        let p = &w.proj[slot];
        ensure!(
            p.n == out_dim as usize,
            "W8A8 projection {slot}: [{}, {}] asked for {out_dim} outputs",
            p.n,
            p.k
        );
        p.gemm(act, out, n as usize, stream)
    }

    /// The o_proj of `attn_out` (`[n, kd]`) into `o_out` (`[n, h]`), its SP
    /// reduce-scatter piped behind it when offered. `Ok(false)`, nothing
    /// launched, on a layer without W8A8.
    pub(crate) fn w8a8_oproj(
        &self,
        attn_out: DevicePtr,
        o_out: DevicePtr,
        [n, h, kd]: [u32; 3],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(w) = self.w8a8()? else {
            return Ok(false);
        };
        let (o, (h, kd)) = (&w.proj[3], (h as usize, kd as usize));
        ensure!(
            [o.n, o.k] == [h, kd],
            "W8A8 o_proj [{}, {}] asked for [{h}, {kd}]",
            o.n,
            o.k
        );
        let rows = |r0: usize, m: usize, s: u64| -> Result<()> {
            let input = attn_out.offset(r0 * kd * 2);
            let act = qwen4exp_w8a8::activation(ctx.gpu, w.quant, ctx.buffers, input, m, kd, s)?;
            o.gemm(act, o_out.offset(r0 * h * 2), m, s)
        };
        let sp = crate::layers::glm_sp::current().filter(|sp| sp.total() == n as usize);
        if let Some(sp) = sp.filter(|_| pipe::rs_offered())
            && pipe::compute_and_reduce_scatter(sp, o_out, h, ctx, stream, &rows, |_| Ok(()))?
        {
            pipe::rs_took();
            return Ok(true);
        }
        rows(0, n as usize, stream)?;
        Ok(true)
    }

    /// Arm the collapse that writes rows `[row0, row0 + rows)` of the q/k/v
    /// input `x` (`[total, hidden]`) to emit their MXFP8 image too
    /// (`ops::qwen4exp_w8a8::Producer`). `None` on a layer without W8A8.
    pub(super) fn w8a8_producer(
        &self,
        ctx: &ForwardContext,
        x: DevicePtr,
        total: usize,
        row0: usize,
        rows: usize,
    ) -> Option<qwen4exp_w8a8::Producer> {
        self.dense_mx.w8a8.as_ref()?;
        let h = ctx.config.hidden_size;
        qwen4exp_w8a8::Producer::arm(ctx.buffers, x, total, row0, rows, h)
    }
}
