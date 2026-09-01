// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::decode::{KDA_HEADS, KDA_WIDTH};
use super::super::types::{Glm5Layer, KdaWeights};
use crate::layer::{ForwardContext, KdaLayerState, LayerState};
use crate::layers::ops;
use crate::layers::ops::glm5_flash_kda::{
    Glm53FlashKdaPrefillArgs, glm53_flash_kda_prefill, glm53_flash_kda_workspace_size,
};

const KDA_DIM: usize = 128;

impl Glm5Layer {
    pub(in crate::layers::glm5) fn kda_prefill(
        &self,
        weights: &KdaWeights,
        normed: DevicePtr,
        rows: usize,
        states: &[&(dyn LayerState + '_)],
        cu_i64: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            !states.is_empty(),
            "GLM KDA prefill requires sequence state"
        );
        let states = states
            .iter()
            .map(|state| {
                state
                    .as_any()
                    .downcast_ref::<KdaLayerState>()
                    .context("GLM KDA prefill received incompatible state")
            })
            .collect::<Result<Vec<_>>>()?;
        let hidden = ctx.config.hidden_size;
        let projected = self.kda_project_inputs(weights, normed, rows, ctx, stream)?;
        let (query, key, value, forget, output_gate, beta) = (
            projected.query,
            projected.key,
            projected.value,
            projected.forget,
            projected.output_gate,
            projected.beta,
        );
        let state_values = states
            .iter()
            .flat_map(|state| {
                [
                    state.current.recurrent.0,
                    state.current.q_conv.0,
                    state.current.k_conv.0,
                    state.current.v_conv.0,
                ]
            })
            .collect::<Vec<_>>();
        let state_table = self.upload_state_table(&state_values, ctx, stream)?;
        ops::glm53_kda_conv_silu_chunk(
            ctx.gpu,
            self.kernels.kda_conv,
            &ops::Glm53KdaConvChunkArgs {
                query,
                key,
                value,
                query_weight: weights.query_conv.weight,
                key_weight: weights.key_conv.weight,
                value_weight: weights.value_conv.weight,
                cu_seqlens: cu_i64,
                sequence_state_ptrs: state_table,
                num_sequences: states.len() as u32,
                num_heads: KDA_HEADS as u32,
            },
            stream,
        )?;
        let beta_ht = ctx.buffers.ssm_deinterleaved();
        ops::glm53_kda_beta_transpose(
            ctx.gpu,
            self.kernels.kda_beta_transpose,
            &ops::Glm53KdaBetaTransposeArgs {
                input: beta,
                output: beta_ht,
                total_tokens: rows as u32,
                num_heads: KDA_HEADS as u32,
            },
            stream,
        )?;
        let layout = ctx.buffers.glm_layout();
        let slot_ids = ctx.buffers.glm_workspace().offset(layout.state_slot_ids);
        let slot_bytes = states
            .iter()
            .flat_map(|state| (state.slot_idx as u32).to_le_bytes())
            .collect::<Vec<_>>();
        ctx.gpu.copy_h2d_async(&slot_bytes, slot_ids, stream)?;
        let state_bytes = KDA_HEADS * KDA_DIM * KDA_DIM * 4;
        let first_state = states[0];
        let base_address = first_state
            .current
            .recurrent
            .0
            .checked_sub((first_state.slot_idx * state_bytes) as u64)
            .context("GLM KDA recurrent pool base underflow")?;
        let recurrent_output = ctx.buffers.attn_output();
        let required =
            glm53_flash_kda_workspace_size(rows as u32, KDA_HEADS as u32, states.len() as u32)?;
        ensure!(
            required <= layout.flash_kda_bytes,
            "GLM FlashKDA workspace requires {required} bytes, allocated {}",
            layout.flash_kda_bytes
        );
        glm53_flash_kda_prefill(
            &Glm53FlashKdaPrefillArgs {
                query,
                key,
                value,
                forget,
                beta_ht,
                recurrent_state: DevicePtr(base_address),
                output: recurrent_output,
                workspace: ctx.buffers.glm_workspace().offset(layout.dynamic),
                a_log: weights.a_log.weight,
                dt_bias: weights.dt_bias.weight,
                cu_seqlens: cu_i64,
                state_slot_ids: slot_ids,
                total_tokens: rows as u32,
                heads: KDA_HEADS as u32,
                sequences: states.len() as u32,
                state_capacity: first_state.slot_capacity as u32,
                query_scale: 1.0 / (KDA_DIM as f32).sqrt(),
                lower_bound: ctx.config.kda_gate_lower_bound,
            },
            stream,
        )?;
        let normalized = beta_ht.offset(rows * KDA_HEADS * 2);
        ops::glm53_kda_gated_norm_chunk(
            ctx.gpu,
            self.kernels.kda_gated_norm,
            &ops::Glm53KdaGatedNormArgs {
                recurrent_output,
                output_gate,
                norm_weight: weights.output_norm.weight,
                output: normalized,
                token_heads: (rows * KDA_HEADS) as u32,
                norm_epsilon: ctx.config.rms_norm_eps as f32,
            },
            stream,
        )?;
        let output = ctx.buffers.residual();
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
