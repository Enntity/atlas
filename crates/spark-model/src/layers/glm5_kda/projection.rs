// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Glm5KdaLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Memory-tight GLM KDA projection.
///
/// The checkpoint-native BF16 matrix is quantized once at load time and then
/// released. Decode uses W4A16 GEMV; prefill uses W4A16 GEMM.
pub struct Glm5Projection {
    pub nvfp4: QuantizedWeight,
    /// Optional `[K/2, N]` twin for Atlas' tensor-core prefill GEMMs.
    pub prefill_nvfp4_t: Option<QuantizedWeight>,
}

pub struct Glm5KdaWeights {
    pub q_proj: Glm5Projection,
    pub k_proj: Glm5Projection,
    pub v_proj: Glm5Projection,
    pub b_proj: DenseWeight,
    pub f_a_proj: DenseWeight,
    pub f_b_proj: DenseWeight,
    pub g_a_proj: DenseWeight,
    pub g_b_proj: DenseWeight,
    pub conv: DenseWeight,
    pub a_log: DenseWeight,
    pub dt_bias: DenseWeight,
    pub o_norm: DenseWeight,
    pub o_proj: Glm5Projection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProjectionPath {
    PrefillTransposedM128,
    PrefillBase,
    DecodeGemv,
}

impl ProjectionPath {
    pub(super) fn for_forward(decode: bool, has_transposed: bool, tokens: u32) -> Self {
        if decode {
            Self::DecodeGemv
        } else if has_transposed && tokens > 128 {
            Self::PrefillTransposedM128
        } else {
            Self::PrefillBase
        }
    }
}

impl Glm5KdaLayer {
    /// Project a short speculative-verification batch with the decode kernels
    /// that amortize one weight read across the candidate rows.  Atlas only
    /// ships tuned NVFP4 variants for M=2/3/4 today; larger draft batches
    /// retain the correct small-GEMM fallback.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_hot_verify(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if (2..=4).contains(&m) {
            self.project_hot_multi_decode(input, weight, output, m, n, k, ctx, stream)
        } else {
            self.project_hot(input, weight, output, m, n, k, false, ctx, stream)
        }
    }

    /// BF16 side projections already have a general batch-M GEMV kernel, so
    /// speculative verification need not fall through to a prefill GEMM.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_dense_verify(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if m == 1 {
            self.project_dense(input, weight, output, m, n, k, ctx, stream)
        } else {
            self.project_dense_multi_decode(input, weight, output, m, n, k, ctx, stream)
        }
    }

    /// Read one NVFP4 projection once for two or three concurrent decode rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_hot_multi_decode(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match m {
            2 => ops::w4a16_gemv_batch2(
                ctx.gpu,
                self.w4a16_gemv_batch2_k,
                input,
                &weight.nvfp4,
                output,
                n,
                k,
                stream,
            ),
            3 => ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3_k,
                input,
                &weight.nvfp4,
                output,
                n,
                k,
                stream,
            ),
            4 => {
                let kernel = self.w4a16_gemv_batchm.kernel(m);
                anyhow::ensure!(
                    kernel.0 != 0,
                    "GLM KDA M=4 decode requires w4a16_gemv_batch4"
                );
                ops::w4a16_gemv_batchm(
                    ctx.gpu,
                    kernel,
                    input,
                    &weight.nvfp4,
                    output,
                    m,
                    n,
                    k,
                    stream,
                )
            }
            _ => anyhow::bail!("GLM KDA multi-decode projection requires M=2..=4, got {m}"),
        }
    }

    /// Read one BF16 projection once for all concurrent decode rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_dense_multi_decode(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::dense_gemv_batchm(
            ctx.gpu,
            self.dense_gemv_batchm_k,
            input,
            weight,
            output,
            m,
            n,
            k,
            n,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_hot(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        decode: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match ProjectionPath::for_forward(decode, weight.prefill_nvfp4_t.is_some(), m) {
            ProjectionPath::DecodeGemv => ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                input,
                &weight.nvfp4,
                output,
                n,
                k,
                stream,
            ),
            ProjectionPath::PrefillTransposedM128 => ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128_k,
                input,
                weight.prefill_nvfp4_t.as_ref().unwrap(),
                output,
                m,
                n,
                k,
                stream,
            ),
            ProjectionPath::PrefillBase => ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                input,
                &weight.nvfp4,
                output,
                m,
                n,
                k,
                stream,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_dense(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if m == 1 {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                input,
                weight,
                output,
                n,
                k,
                stream,
            )
        } else {
            ops::dense_gemm_prefill(
                ctx.gpu,
                self.dense_gemm_k,
                self.dense_gemm_pipelined_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm_kda_selects_nvfp4_gemv_for_decode_and_gemm_for_prefill() {
        assert_eq!(
            ProjectionPath::for_forward(true, true, 1000),
            ProjectionPath::DecodeGemv
        );
        assert_eq!(
            ProjectionPath::for_forward(false, true, 1000),
            ProjectionPath::PrefillTransposedM128
        );
        assert_eq!(
            ProjectionPath::for_forward(false, true, 128),
            ProjectionPath::PrefillBase
        );
        assert_eq!(
            ProjectionPath::for_forward(false, false, 1000),
            ProjectionPath::PrefillBase
        );
    }
}
