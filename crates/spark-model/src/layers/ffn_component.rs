// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// FFN component: MoE (expert routing), dense SwiGLU, or None (standalone attention).
#[allow(clippy::large_enum_variant)]
pub enum FfnComponent {
    Moe(MoeLayer),
    Exl3(Box<glm5_exl3_ffn::Glm5Exl3FfnLayer>),
    Dense(DenseFfnLayer),
    /// No FFN — used by Nemotron-H standalone attention layers.
    None,
}

impl FfnComponent {
    /// Release native EXL3 owners during model teardown.
    pub fn release_exl3(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        use atlas_core::scope::ModelResource;
        match self {
            Self::Exl3(layer) => layer.release(gpu),
            _ => Ok(()),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// True for a plain dense (SwiGLU) FFN. Wide-batch verify paths gate their
    /// `forward_prefill` fast path on this: batching reads dense weights once
    /// (big win at N=17), but on a 256-expert MoE the grouped-GEMM is a net
    /// loss at small batch (per-expert M~1 + sort/permute overhead), so MoE
    /// keeps its per-token loop.
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_)) || matches!(self, Self::Exl3(layer) if layer.is_dense())
    }

    /// True when this MoE FFN can serve DECODE through the grouped read-once
    /// GEMM (forward_prefill) instead of the pairwise per-slot loop. The
    /// is_dense() comment above asserts grouped is "a net loss at small batch"
    /// on a 256-expert MoE, but that was never measured for decode CONCURRENCY
    /// (n=4) where the pairwise path re-reads ~14-20 distinct experts as 40
    /// per-slot CTAs. Native-NVFP4-routed only (forward_prefill's unconditional
    /// grouped path); dense/none are false.
    pub fn moe_grouped_decode_ok(&self) -> bool {
        match self {
            Self::Moe(m) => m.grouped_decode_ok(),
            _ => false,
        }
    }

    /// ATLAS_FP32_ROUTING active for this FFN (MoE only; false otherwise).
    pub fn fp32_routing_active(&self) -> bool {
        match self {
            Self::Moe(m) => m.fp32_routing_active(),
            _ => false,
        }
    }

    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match self {
            Self::Exl3(layer) => layer.forward_rows(input, 1, ctx, stream),
            Self::Moe(m) => m.forward(input, ctx, stream),
            Self::Dense(d) => d.forward(input, ctx, stream),
            Self::None => Ok(input),
        }
    }

    /// [`Self::forward`] for a single-token mHC decode layer: the MoE may run
    /// the layer's low-rank `hc_post` itself (`hc_post`, fused with its last
    /// kernel under `ATLAS_QWEN4EXP_DECODE_FUSE`). Returns the output and
    /// whether the post ran; when it did not, the caller still runs it.
    pub fn forward_hc_post(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        hc_post: Option<ops::qwen4exp_decode_fuse::MoeHcPost>,
        stream: u64,
    ) -> Result<(DevicePtr, bool)> {
        match self {
            Self::Moe(m) => m.forward_hc_post(input, ctx, hc_post, stream),
            _ => Ok((self.forward(input, ctx, stream)?, false)),
        }
    }

    /// `rows` rows of `input` -> `moe_output` rows, each byte-identical to
    /// [`Self::forward`] on that row, with the MoE's per-row collectives
    /// batched (`MoeLayer::forward_rows`). `None` where that does not hold
    /// (a dense FFN, or a MoE whose decode takes a grouped prefill arm): the
    /// caller keeps its per-row `forward` loop. The flag: `hc_post` ran
    /// (`MoeLayer::forward_rows`); if not, the caller runs the post.
    pub fn forward_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        hc_post: Option<ops::qwen4exp_decode_fuse::MoeHcPost>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, bool)>> {
        match self {
            Self::Moe(m) => m.forward_rows(input, rows, hc_post, ctx, stream),
            _ => Ok(None),
        }
    }

    /// [`Self::forward_rows`] over the first `active` of `rows` rows of a
    /// padded batch, with the padding rows' outputs zeroed: a padding row
    /// (zero hidden, dummy state, discarded logits) then injects nothing
    /// into its highway, where `forward` on its zero input also produced
    /// zero but cost a full row of expert reads (`hc_post` only unpadded).
    pub fn forward_rows_padded(
        &self,
        input: DevicePtr,
        rows: usize,
        active: usize,
        hc_post: Option<ops::qwen4exp_decode_fuse::MoeHcPost>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, bool)>> {
        let active = active.min(rows);
        if active == 0 {
            return Ok(None);
        }
        let hc_post = hc_post.filter(|_| active == rows);
        let Some((out, posted)) = self.forward_rows(input, active, hc_post, ctx, stream)? else {
            return Ok(None);
        };
        if active < rows {
            let row = ctx.config.hidden_size * 2;
            ctx.gpu
                .memset_async(out.offset(active * row), 0, (rows - active) * row, stream)?;
        }
        Ok(Some((out, posted)))
    }

    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Exl3(layer) => layer.forward_rows(input, 2, ctx, stream).map(|_| ()),
            Self::Moe(m) => m.forward_k2(input, ctx, stream),
            Self::Dense(d) => d.forward_k2(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Exl3(layer) => layer.forward_rows(input, 3, ctx, stream).map(|_| ()),
            Self::Moe(m) => m.forward_k3(input, ctx, stream),
            Self::Dense(d) => d.forward_k3(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    /// Whether the K=m (m<=8) batched-GEMV verify FFN is available (dense
    /// only — MoE / missing batch4/batch8 kernel / non-NVFP4 weights →
    /// false). Lets callers gate branch entry BEFORE computing the pre-FFN
    /// norm, so there is no half-done fallthrough to `forward_prefill`.
    pub fn can_forward_km(&self, m: u32) -> bool {
        matches!(self, Self::Dense(d) if d.can_forward_km(m))
    }

    /// K=m (m=4..8) verify FFN via batched GEMV (dense only). Returns
    /// `false` when the path is unavailable (MoE / missing batchm kernel /
    /// non-NVFP4 weights) so the caller can fall back to `forward_prefill`.
    pub fn try_forward_km(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Dense(d) if d.can_forward_km(m) => {
                d.forward_km(input, m, ctx, stream)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Exl3(layer) => layer
                .forward_rows(input, num_tokens, ctx, stream)
                .map(|_| ()),
            Self::Moe(m) => m.forward_prefill(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_prefill(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Exl3(layer) => layer
                .forward_rows(input, num_tokens, ctx, stream)
                .map(|_| ()),
            Self::Moe(m) => m.forward_batched(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_token_major_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Exl3(layer) => layer
                .forward_rows(input, num_tokens, ctx, stream)
                .map(|_| ()),
            Self::Moe(m) => m.forward_token_major_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_atomic_c4_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Exl3(layer) => layer
                .forward_rows(input, num_tokens, ctx, stream)
                .map(|_| ()),
            Self::Moe(m) => m.forward_atomic_c4_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }
}
