// SPDX-License-Identifier: AGPL-3.0-only

//! GLM paged prefill row math: W_uk query absorb and the W_uv + o_proj
//! output.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::{MlaWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// W_uk absorb of `rows` q_b outputs into `nq` per-head latent queries.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_absorb_queries(
        &self,
        q_full: DevicePtr,
        q_absorbed: DevicePtr,
        rows: u32,
        nq: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mla = self.mla.as_ref().expect("GLM absorb without MLA");
        let (hd, kv_lora) = (mla.nope as u32, mla.kv_lora_rank as u32);
        self.glm_head_gemm(
            q_full,
            mla.w_uk_t.weight,
            q_absorbed,
            [rows, nq, hd, kv_lora, nq * hd, nq * kv_lora],
            ctx,
            stream,
        )
    }

    /// Per-head MLA GEMM `c[:, h*n..] = a[:, h*k..] · weight_hᵀ` over `g`
    /// heads: its MXFP8 twin (`ATLAS_GLM_MLA_KVB_MXFP8=1`) for up to 16 rows,
    /// else the BF16 grouped GEMM.
    fn glm_head_gemm(
        &self,
        a: DevicePtr,
        weight: DevicePtr,
        c: DevicePtr,
        [rows, g, k, n, a_stride, c_stride]: [u32; 6],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if rows <= ops::MXFP8_GROUPED_MAX_M
            && let Some((_, mx)) = self.mla_mx.iter().find(|(w, _)| *w == weight)
        {
            return ops::mxfp8_gemv_grouped(
                ctx.gpu,
                &self.mxfp8_gemv_grouped_k,
                a,
                mx.data,
                mx.scales,
                c,
                rows,
                g,
                k,
                n,
                a_stride,
                c_stride,
                stream,
            );
        }
        ops::glm_paged_grouped_gemm_mla(
            ctx.gpu,
            self.grouped_gemm_mla_k,
            &ctx.config.model_type,
            a,
            weight,
            c,
            rows,
            g,
            k,
            n,
            a_stride,
            c_stride,
            stream,
        )
    }

    /// W_uv then the row-parallel o_proj for `rows` latent attention rows
    /// of `nq` local heads into `h` hidden columns.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn paged_glm_output(
        &self,
        mla: &MlaWeights,
        latent: DevicePtr,
        out: DevicePtr,
        [rows, nq, h]: [u32; 3],
        ctx: &ForwardContext,
        stream: u64,
        accelerated: bool,
    ) -> Result<()> {
        let (kv_lora, v_dim) = (mla.kv_lora_rank as u32, mla.v_dim as u32);
        let v_extracted = ctx.buffers.qkv_output();
        self.glm_head_gemm(
            latent,
            mla.w_uv.weight,
            v_extracted,
            [rows, nq, kv_lora, v_dim, nq * kv_lora, nq * v_dim],
            ctx,
            stream,
        )?;
        self.paged_glm_projection(
            v_extracted,
            &mla.wo,
            out,
            rows,
            h,
            nq * v_dim,
            ctx,
            stream,
            accelerated,
        )
    }
}
