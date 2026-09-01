// SPDX-License-Identifier: AGPL-3.0-only

//! Shared GLM projection and hyperconnection helpers.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::types::{Glm5Layer, GlmHcWeights};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantWeight};

impl Glm5Layer {
    pub(super) fn contract_hc_for_output_or_dflash(
        &self,
        hidden: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !should_contract_hc(
            self.layer_idx,
            ctx.config.num_hidden_layers,
            &ctx.config.dflash_capture_layers,
        ) {
            return Ok(());
        }
        // SGLang's GLM DFlash contract captures the completed output of
        // target layer k and then hc_contracts the four mHC streams. Atlas
        // keeps those streams in a separate FP32 highway, while `hidden` is
        // merely the most recent hc_pre scratch at intermediate layers.
        // Materialize the arithmetic mean here so the generic capture hook
        // that runs immediately after this layer sees the trained feature.
        ops::glm53_hc_mean(
            ctx.gpu,
            self.kernels.hc_mean,
            ctx.buffers.hc_streams(),
            hidden,
            rows as u32,
            ctx.config.hidden_size as u32,
            ctx.config.hc_mult as u32,
            stream,
        )
    }

    pub(super) fn tp_sum(
        &self,
        value: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if ctx.config.tp_world_size <= 1 {
            return Ok(());
        }
        let comm = ctx
            .comm
            .ok_or_else(|| anyhow::anyhow!("GLM TP2 forward requires a communicator"))?;
        let bytes = rows * ctx.config.hidden_size * 2;
        // Keep GLM's collective on the layer's compute stream in both eager
        // execution and CUDA-graph capture. The generic two-rank path uses a
        // legacy/communication stream plus a local BF16-add kernel. Calling
        // that path while the compute stream is being captured leaves the TP
        // reduction outside the captured graph, so replay can consume an
        // unreduced shard. NCCL collectives issued on the capturing stream are
        // graph nodes and replay in the same layer order on both fixed ranks.
        //
        // GLM verifies 8 draft rows per sequence, so its two-Spark payloads
        // are 72-295 KiB at C1-C4; native NCCL is also the simpler eager path
        // for this fixed appliance.
        comm.all_reduce_direct(value.0, bytes, stream)
    }

    pub(super) fn project(
        &self,
        input: DevicePtr,
        weight: &QuantWeight,
        output: DevicePtr,
        rows: usize,
        n: usize,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let QuantWeight::Dense(dense) = weight else {
            anyhow::bail!(
                "GLM-5.3 EXL3 mixed-precision contract requires native dense projections"
            );
        };
        let (rows, n, k) = (rows as u32, n as u32, k as u32);
        if rows == 1 {
            return ops::dense_gemv(
                ctx.gpu,
                self.kernels.dense_gemv,
                input,
                dense,
                output,
                n,
                k,
                stream,
            );
        }
        // GLM's K=9 verifier and ragged prefill are both native BF16. On
        // GB10 cuBLASLt is ~3x the hand-written mma.sync kernels and its plan
        // is cached by shape, so this is the sole multi-row projection path.
        ops::cublas_bf16_proj_dense(input, dense.weight, output, rows, n, k, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_pre_norm(
        &self,
        weights: &GlmHcWeights,
        norm_weight: &DenseWeight,
        hidden: DevicePtr,
        normed: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.hc_pre_norm_at(weights, norm_weight, hidden, normed, 0, rows, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_pre_norm_at(
        &self,
        weights: &GlmHcWeights,
        norm_weight: &DenseWeight,
        hidden: DevicePtr,
        normed: DevicePtr,
        row_offset: usize,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc_mult = ctx.config.hc_mult;
        let h = ctx.config.hidden_size;
        let streams = ctx
            .buffers
            .hc_streams()
            .offset(row_offset * hc_mult * h * size_of::<f32>());
        let post = ctx
            .buffers
            .hc_post()
            .offset(row_offset * hc_mult * size_of::<f32>());
        let comb = ctx
            .buffers
            .hc_comb()
            .offset(row_offset * hc_mult * hc_mult * size_of::<f32>());
        if weights.use_tensor_core {
            ensure!(
                h == 4096 && hc_mult == 4,
                "GLM tensor-core mHC requires hidden=4096 and hc_mult=4"
            );
            let workspace = ctx.buffers.glm_workspace();
            let layout = ctx.buffers.glm_layout();
            ensure!(!workspace.is_null(), "GLM mHC workspace is not allocated");
            ensure!(
                row_offset.saturating_add(rows) <= layout.max_batch_tokens,
                "GLM mHC row range {row_offset}+{rows} exceeds workspace capacity {}",
                layout.max_batch_tokens
            );
            let sqsum = workspace.offset(layout.mhc_sqsum + row_offset * size_of::<f32>());
            let mix = workspace
                .offset(layout.mhc_mix + row_offset * (2 + hc_mult) * hc_mult * size_of::<f32>());
            ops::glm53_hc_pre_sqsum(
                ctx.gpu,
                self.kernels.hc_pre_sqsum,
                streams,
                sqsum,
                rows as u32,
                stream,
            )?;
            spark_runtime::cublaslt::tf32_gemm_act_weight_t(
                streams.0,
                weights.function_tf32.weight.0,
                mix.0,
                rows as u32,
                24,
                16384,
                stream,
            )?;
            return ops::glm53_hc_pre_finish_norm(
                ctx.gpu,
                self.kernels.hc_pre_finish_norm,
                streams,
                mix,
                sqsum,
                weights.scale.weight,
                weights.base.weight,
                hidden,
                norm_weight.weight,
                normed,
                post,
                comb,
                rows as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            );
        }
        self.hc_pre_at(weights, hidden, row_offset, rows, ctx, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.kernels.rms_norm,
            hidden,
            norm_weight,
            normed,
            rows as u32,
            h as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )
    }

    pub(super) fn hc_pre_at(
        &self,
        weights: &GlmHcWeights,
        hidden: DevicePtr,
        row_offset: usize,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc_mult = ctx.config.hc_mult;
        let h = ctx.config.hidden_size;
        let streams = ctx
            .buffers
            .hc_streams()
            .offset(row_offset * hc_mult * h * size_of::<f32>());
        let post = ctx
            .buffers
            .hc_post()
            .offset(row_offset * hc_mult * size_of::<f32>());
        let comb = ctx
            .buffers
            .hc_comb()
            .offset(row_offset * hc_mult * hc_mult * size_of::<f32>());
        if weights.use_tensor_core {
            ensure!(
                ctx.config.hidden_size == 4096 && ctx.config.hc_mult == 4,
                "GLM tensor-core mHC requires hidden=4096 and hc_mult=4"
            );
            let workspace = ctx.buffers.glm_workspace();
            let layout = ctx.buffers.glm_layout();
            ensure!(!workspace.is_null(), "GLM mHC workspace is not allocated");
            ensure!(
                rows <= layout.max_batch_tokens,
                "GLM mHC rows {rows} exceed workspace capacity {}",
                layout.max_batch_tokens
            );
            let sqsum = workspace.offset(layout.mhc_sqsum + row_offset * size_of::<f32>());
            let mix = workspace
                .offset(layout.mhc_mix + row_offset * (2 + hc_mult) * hc_mult * size_of::<f32>());
            ops::glm53_hc_pre_sqsum(
                ctx.gpu,
                self.kernels.hc_pre_sqsum,
                streams,
                sqsum,
                rows as u32,
                stream,
            )?;
            spark_runtime::cublaslt::tf32_gemm_act_weight_t(
                streams.0,
                weights.function_tf32.weight.0,
                mix.0,
                rows as u32,
                24,
                16384,
                stream,
            )?;
            return ops::glm53_hc_pre_finish_norm(
                ctx.gpu,
                self.kernels.hc_pre_finish_norm,
                streams,
                mix,
                sqsum,
                weights.scale.weight,
                weights.base.weight,
                hidden,
                DevicePtr::NULL,
                DevicePtr::NULL,
                post,
                comb,
                rows as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            );
        }
        ops::hc_pre(
            ctx.gpu,
            self.kernels.hc_pre,
            streams,
            weights.function.weight,
            weights.scale.weight,
            weights.base.weight,
            hidden,
            post,
            comb,
            rows as u32,
            ctx.config.hidden_size as u32,
            ctx.config.hc_mult as u32,
            ctx.config.hc_sinkhorn_iters as u32,
            ctx.config.rms_norm_eps as f32,
            ctx.config.hc_eps,
            stream,
        )
    }

    pub(super) fn hc_post(
        &self,
        output: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.hc_post_at(output, 0, rows, ctx, stream)
    }

    pub(super) fn hc_post_at(
        &self,
        output: DevicePtr,
        row_offset: usize,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc_mult = ctx.config.hc_mult;
        let h = ctx.config.hidden_size;
        let streams = ctx
            .buffers
            .hc_streams()
            .offset(row_offset * hc_mult * h * size_of::<f32>());
        let post = ctx
            .buffers
            .hc_post()
            .offset(row_offset * hc_mult * size_of::<f32>());
        let comb = ctx
            .buffers
            .hc_comb()
            .offset(row_offset * hc_mult * hc_mult * size_of::<f32>());
        ops::hc_post(
            ctx.gpu,
            self.kernels.hc_post,
            output,
            streams,
            post,
            comb,
            streams,
            rows as u32,
            ctx.config.hidden_size as u32,
            ctx.config.hc_mult as u32,
            stream,
        )
    }

    pub(super) fn upload_state_table(
        &self,
        values: &[u64],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let workspace = ctx.buffers.glm_workspace();
        let layout = ctx.buffers.glm_layout();
        ensure!(!workspace.is_null(), "GLM workspace is not allocated");
        ensure!(
            size_of_val(values) <= layout.state_ptrs_stride,
            "GLM state table exceeds workspace"
        );
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let destination = workspace.offset(layout.state_table_offset(self.layer_idx)?);
        if !ctx.graph_capture {
            ctx.gpu.copy_h2d_async(&bytes, destination, stream)?;
        }
        Ok(destination)
    }
}

fn should_contract_hc(
    layer_idx: usize,
    num_hidden_layers: usize,
    dflash_capture_layers: &[usize],
) -> bool {
    layer_idx + 1 == num_hidden_layers || dflash_capture_layers.contains(&layer_idx)
}

#[cfg(test)]
mod tests {
    use super::should_contract_hc;

    #[test]
    fn glm_hc_contracts_at_every_dflash_feature_boundary_and_final_output() {
        let capture = [5, 14, 24, 33, 42];
        for layer in capture {
            assert!(should_contract_hc(layer, 45, &capture));
        }
        assert!(should_contract_hc(44, 45, &capture));
        assert!(!should_contract_hc(43, 45, &capture));
    }
}
