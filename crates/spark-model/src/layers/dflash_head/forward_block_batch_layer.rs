// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer halves for the fixed-width native DFlash2 batch graph.

use anyhow::Result;

use super::{BlockDiffusionDraftHead, DflashLayer};
use crate::layer::ForwardContext;

impl BlockDiffusionDraftHead {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_block_batch_pre(
        &self,
        layer: &DflashLayer,
        rows: u32,
        sequence_rows: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;
        let gpu = ctx.gpu;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let projection_input = if let Some(conv) = layer.attention_conv.as_ref() {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                self.scratch.norm_buf,
                &conv.kernel_projection,
                self.scratch.mlp_intermediate,
                rows,
                self.dflash2_conv_groups as u32 * 4,
                h,
                stream,
            )?;
            ops::dflash2_dynamic_conv2(
                gpu,
                self.kernels.dflash2_dynamic_conv,
                self.scratch.norm_buf,
                self.scratch.mlp_intermediate,
                conv.base_kernel.weight,
                self.scratch.stream_acc,
                rows,
                sequence_rows,
                h,
                self.dflash2_conv_groups as u32,
                0,
                stream,
            )?;
            self.scratch.stream_acc
        } else {
            self.scratch.norm_buf
        };
        for (weight, output, width) in [
            (&layer.q_proj, self.scratch.q_buf, q_dim),
            (&layer.k_proj, self.scratch.k_buf, kv_dim),
            (&layer.v_proj, self.scratch.v_buf, kv_dim),
        ] {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                projection_input,
                weight,
                output,
                rows,
                width,
                h,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.q_buf,
            &layer.q_norm,
            self.scratch.q_buf,
            rows * self.num_q_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.k_buf,
            &layer.k_norm,
            self.scratch.k_buf,
            rows * self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            self.scratch.q_buf,
            self.scratch.k_buf,
            self.scratch.position_ids,
            rows,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_block_batch_post(
        &self,
        layer: &DflashLayer,
        rows: u32,
        sequence_rows: u32,
        h: u32,
        q_dim: u32,
        inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;
        let gpu = ctx.gpu;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            rows,
            h,
            q_dim,
            stream,
        )?;
        let attention = if let Some(conv) = layer.attention_conv.as_ref() {
            ops::dflash2_dynamic_conv2(
                gpu,
                self.kernels.dflash2_dynamic_conv,
                self.scratch.stream_acc,
                self.scratch.mlp_intermediate,
                conv.base_kernel.weight,
                self.scratch.attn_out,
                rows,
                sequence_rows,
                h,
                self.dflash2_conv_groups as u32,
                1,
                stream,
            )?;
            self.scratch.attn_out
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attention,
            rows * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_input = if let Some(conv) = layer.mlp_conv.as_ref() {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                self.scratch.norm_buf,
                &conv.kernel_projection,
                self.scratch.q_buf,
                rows,
                self.dflash2_conv_groups as u32 * 4,
                h,
                stream,
            )?;
            ops::dflash2_dynamic_conv2(
                gpu,
                self.kernels.dflash2_dynamic_conv,
                self.scratch.norm_buf,
                self.scratch.q_buf,
                conv.base_kernel.weight,
                self.scratch.stream_acc,
                rows,
                sequence_rows,
                h,
                self.dflash2_conv_groups as u32,
                0,
                stream,
            )?;
            self.scratch.stream_acc
        } else {
            self.scratch.norm_buf
        };
        for (weight, output) in [
            (&layer.gate_proj, self.scratch.mlp_intermediate),
            (&layer.up_proj, self.scratch.mlp_up),
        ] {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                mlp_input,
                weight,
                output,
                rows,
                inter,
                h,
                stream,
            )?;
        }
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            rows * inter,
            stream,
        )?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            rows,
            h,
            inter,
            stream,
        )?;
        let mlp = if let Some(conv) = layer.mlp_conv.as_ref() {
            ops::dflash2_dynamic_conv2(
                gpu,
                self.kernels.dflash2_dynamic_conv,
                self.scratch.stream_acc,
                self.scratch.q_buf,
                conv.base_kernel.weight,
                self.scratch.norm_buf,
                rows,
                sequence_rows,
                h,
                self.dflash2_conv_groups as u32,
                1,
                stream,
            )?;
            self.scratch.norm_buf
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp,
            rows * h,
            stream,
        )
    }
}
