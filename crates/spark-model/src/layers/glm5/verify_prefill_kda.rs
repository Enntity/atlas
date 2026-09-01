// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-projection KDA attention for heterogeneous verify + prompt rows.
//!
//! The two lanes need different state kernels: verifier rows write rollback
//! images while prompt rows run FlashKDA.  Their dense projections are the
//! same model weights, however, so streaming those weights twice defeats the
//! purpose of heterogeneous dispatch.  This path projects `[verify | prompt]`
//! once, runs the two state kernels on pointer-offset slices, then applies one
//! output projection.  Recurrent state and convolution history remain fully
//! sequence-private.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::decode::{KDA_HEADS, KDA_WIDTH};
use super::types::{Glm5Layer, KdaWeights};
use crate::layer::{ForwardContext, KdaLayerState, LayerState};
use crate::layers::ops;
use crate::layers::ops::glm5_flash_kda::{
    Glm53FlashKdaPrefillArgs, glm53_flash_kda_prefill, glm53_flash_kda_workspace_size,
};

const KDA_DIM: usize = 128;

impl Glm5Layer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn kda_forward_verify_with_prefill(
        &self,
        weights: &KdaWeights,
        normed: DevicePtr,
        verify_rows: usize,
        verify_state: &mut (dyn LayerState + 'static),
        prefill_rows: usize,
        prefill_state: &mut (dyn LayerState + 'static),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let rows = verify_rows.saturating_add(prefill_rows);
        ensure!(
            verify_rows > 0 && prefill_rows > 0,
            "GLM shared KDA lanes are empty"
        );
        ensure!(
            rows <= ctx.buffers.max_batch_tokens(),
            "GLM shared KDA rows exceed the model arena"
        );

        let row_width_bytes = KDA_WIDTH * size_of::<u16>();
        let verify_offset = verify_rows * row_width_bytes;

        // Every large dense projection is shared across the two row ranges.
        let projected = self.kda_project_inputs(weights, normed, rows, ctx, stream)?;
        let (query, key, value, forget, output_gate, beta) = (
            projected.query,
            projected.key,
            projected.value,
            projected.forget,
            projected.output_gate,
            projected.beta,
        );

        // Lane 0: exact speculative verification with its rollback images.
        let verify_state = verify_state
            .as_any()
            .downcast_ref::<KdaLayerState>()
            .context("GLM shared KDA verify received incompatible state")?;
        let snapshot_count = verify_rows.saturating_sub(1);
        ensure!(
            verify_state.intermediates.len() >= snapshot_count,
            "GLM shared KDA verify has insufficient rollback images"
        );
        let verify_table = std::iter::once(verify_state.current)
            .chain(
                verify_state
                    .intermediates
                    .iter()
                    .copied()
                    .take(snapshot_count),
            )
            .flat_map(|image| {
                [
                    image.recurrent.0,
                    image.q_conv.0,
                    image.k_conv.0,
                    image.v_conv.0,
                ]
            })
            .collect::<Vec<_>>();
        let state_images = self.upload_state_table(&verify_table, ctx, stream)?;
        let normalized = ctx.buffers.attn_output();
        ops::glm53_kda_verify(
            ctx.gpu,
            self.kernels.kda_verify_prepare,
            self.kernels.kda_verify_recurrent_tiled,
            self.kernels.kda_verify_norm,
            &ops::Glm53KdaVerifyArgs {
                query,
                key,
                value,
                query_conv_weight: weights.query_conv.weight,
                key_conv_weight: weights.key_conv.weight,
                value_conv_weight: weights.value_conv.weight,
                state_images,
                forget_projection: forget,
                dt_bias: weights.dt_bias.weight,
                a_log: weights.a_log.weight,
                beta_logit: beta,
                output_gate,
                norm_weight: weights.output_norm.weight,
                output: normalized,
                num_tokens: verify_rows as u32,
                num_sequences: 1,
                snapshot_count: snapshot_count as u32,
                num_heads: KDA_HEADS as u32,
                norm_epsilon: ctx.config.rms_norm_eps as f32,
            },
            stream,
        )?;

        // Lane 1: native prompt recurrence over offset slices of the shared
        // projections. The state-table upload is ordered after the verify
        // kernels on the same stream, so reusing the fixed table is safe.
        let prefill_state = prefill_state
            .as_any()
            .downcast_ref::<KdaLayerState>()
            .context("GLM shared KDA prefill received incompatible state")?;
        let prefill_table = [
            prefill_state.current.recurrent.0,
            prefill_state.current.q_conv.0,
            prefill_state.current.k_conv.0,
            prefill_state.current.v_conv.0,
        ];
        let state_table = self.upload_state_table(&prefill_table, ctx, stream)?;
        let (_, cu_i64) = self.upload_prefill_lengths(prefill_rows, ctx, stream)?;
        ops::glm53_kda_conv_silu_chunk(
            ctx.gpu,
            self.kernels.kda_conv,
            &ops::Glm53KdaConvChunkArgs {
                query: query.offset(verify_offset),
                key: key.offset(verify_offset),
                value: value.offset(verify_offset),
                query_weight: weights.query_conv.weight,
                key_weight: weights.key_conv.weight,
                value_weight: weights.value_conv.weight,
                cu_seqlens: cu_i64,
                sequence_state_ptrs: state_table,
                num_sequences: 1,
                num_heads: KDA_HEADS as u32,
            },
            stream,
        )?;

        let beta_ht = ctx.buffers.ssm_deinterleaved();
        ops::glm53_kda_beta_transpose(
            ctx.gpu,
            self.kernels.kda_beta_transpose,
            &ops::Glm53KdaBetaTransposeArgs {
                input: beta.offset(verify_rows * KDA_HEADS * size_of::<u16>()),
                output: beta_ht,
                total_tokens: prefill_rows as u32,
                num_heads: KDA_HEADS as u32,
            },
            stream,
        )?;
        let layout = ctx.buffers.glm_layout();
        let slot_ids = ctx.buffers.glm_workspace().offset(layout.state_slot_ids);
        ctx.gpu.copy_h2d_async(
            &(prefill_state.slot_idx as u32).to_le_bytes(),
            slot_ids,
            stream,
        )?;
        let state_bytes = KDA_HEADS * KDA_DIM * KDA_DIM * size_of::<f32>();
        let base_address = prefill_state
            .current
            .recurrent
            .0
            .checked_sub((prefill_state.slot_idx * state_bytes) as u64)
            .context("GLM shared KDA recurrent pool base underflow")?;
        let recurrent_tmp = beta_ht.offset(prefill_rows * KDA_HEADS * size_of::<u16>());
        let required = glm53_flash_kda_workspace_size(prefill_rows as u32, KDA_HEADS as u32, 1)?;
        ensure!(
            required <= layout.flash_kda_bytes,
            "GLM shared FlashKDA workspace exceeds its allocation"
        );
        glm53_flash_kda_prefill(
            &Glm53FlashKdaPrefillArgs {
                query: query.offset(verify_offset),
                key: key.offset(verify_offset),
                value: value.offset(verify_offset),
                forget: forget.offset(verify_offset),
                beta_ht,
                recurrent_state: DevicePtr(base_address),
                output: recurrent_tmp,
                workspace: ctx.buffers.glm_workspace().offset(layout.dynamic),
                a_log: weights.a_log.weight,
                dt_bias: weights.dt_bias.weight,
                cu_seqlens: cu_i64,
                state_slot_ids: slot_ids,
                total_tokens: prefill_rows as u32,
                heads: KDA_HEADS as u32,
                sequences: 1,
                state_capacity: prefill_state.slot_capacity as u32,
                query_scale: 1.0 / (KDA_DIM as f32).sqrt(),
                lower_bound: ctx.config.kda_gate_lower_bound,
            },
            stream,
        )?;
        ops::glm53_kda_gated_norm_chunk(
            ctx.gpu,
            self.kernels.kda_gated_norm,
            &ops::Glm53KdaGatedNormArgs {
                recurrent_output: recurrent_tmp,
                output_gate: output_gate.offset(verify_offset),
                norm_weight: weights.output_norm.weight,
                output: normalized.offset(verify_offset),
                token_heads: (prefill_rows * KDA_HEADS) as u32,
                norm_epsilon: ctx.config.rms_norm_eps as f32,
            },
            stream,
        )?;

        // Both normalized lane ranges are contiguous now, so the attention
        // output projection and its following TP reduction each run once.
        let output = ctx.buffers.residual();
        let hidden = ctx.config.hidden_size;
        self.project(
            normalized,
            &weights.output,
            output,
            rows,
            hidden,
            KDA_WIDTH,
            ctx,
            stream,
        )?;
        Ok(output)
    }
}
