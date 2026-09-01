// SPDX-License-Identifier: AGPL-3.0-only

//! GLM KDA decode path.

use anyhow::{Context, Result};
use spark_runtime::gpu::DevicePtr;

use super::decode::{KDA_HEADS, KDA_WIDTH};
use super::types::{Glm5Layer, KdaWeights};
use crate::layer::{ForwardContext, KdaLayerState, LayerState};
use crate::layers::ops;

impl Glm5Layer {
    pub(super) fn kda_forward(
        &self,
        weights: &KdaWeights,
        normed: DevicePtr,
        rows: usize,
        states: &[&(dyn LayerState + '_)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
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

        let mut table = Vec::with_capacity(rows * 4);
        for state in states {
            let state = state
                .as_any()
                .downcast_ref::<KdaLayerState>()
                .context("GLM KDA layer received incompatible state")?;
            table.extend([
                state.current.recurrent.0,
                state.current.q_conv.0,
                state.current.k_conv.0,
                state.current.v_conv.0,
            ]);
        }
        let table = self.upload_state_table(&table, ctx, stream)?;
        let recurrent_output = ctx.buffers.attn_output();
        ops::glm53_kda_fused_decode(
            ctx.gpu,
            self.kernels.kda_decode,
            &ops::Glm53KdaFusedDecodeArgs {
                query,
                key,
                value,
                query_conv_weight: weights.query_conv.weight,
                key_conv_weight: weights.key_conv.weight,
                value_conv_weight: weights.value_conv.weight,
                sequence_state_ptrs: table,
                forget_projection: forget,
                dt_bias: weights.dt_bias.weight,
                a_log: weights.a_log.weight,
                beta_logit: beta,
                output_gate,
                norm_weight: weights.output_norm.weight,
                output: recurrent_output,
                batch_heads: (rows * KDA_HEADS) as u32,
                num_heads: KDA_HEADS as u32,
                norm_epsilon: ctx.config.rms_norm_eps as f32,
            },
            stream,
        )?;
        let output = ctx.buffers.residual();
        self.project(
            recurrent_output,
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
