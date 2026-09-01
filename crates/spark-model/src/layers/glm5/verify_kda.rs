// SPDX-License-Identifier: AGPL-3.0-only

//! One-weight-sweep KDA target verification.

use anyhow::{Context, Result, ensure};
use spark_runtime::buffers::GLM53_VERIFY_MAX_ROWS;
use spark_runtime::gpu::DevicePtr;

use super::decode::{KDA_HEADS, KDA_WIDTH};
use super::types::{Glm5Layer, KdaWeights};
use crate::layer::{ForwardContext, KdaLayerState, LayerState};
use crate::layers::ops;

impl Glm5Layer {
    pub(super) fn kda_forward_verify_multi(
        &self,
        weights: &KdaWeights,
        normed: DevicePtr,
        rows_per_seq: usize,
        states: &mut [&mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let num_sequences = states.len();
        let rows = num_sequences.saturating_mul(rows_per_seq);
        ensure!(num_sequences > 0, "GLM KDA verify batch requires sequences");
        ensure!(
            rows_per_seq > 0 && rows <= spark_runtime::buffers::GLM53_VERIFY_MAX_BATCH_ROWS,
            "GLM KDA verify batch shape {num_sequences}x{rows_per_seq} is unsupported"
        );

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

        let snapshot_count = rows_per_seq.saturating_sub(1);
        let mut table = Vec::with_capacity(num_sequences * (snapshot_count + 1) * 4);
        for state in states.iter() {
            let state = state
                .as_any()
                .downcast_ref::<KdaLayerState>()
                .context("GLM KDA multi-verify received incompatible state")?;
            ensure!(
                state.intermediates.len() >= snapshot_count,
                "GLM KDA multi-verify needs {snapshot_count} rollback images but has {}",
                state.intermediates.len()
            );
            for image in std::iter::once(state.current)
                .chain(state.intermediates.iter().copied().take(snapshot_count))
            {
                table.extend([
                    image.recurrent.0,
                    image.q_conv.0,
                    image.k_conv.0,
                    image.v_conv.0,
                ]);
            }
        }
        let state_images = self.upload_state_table(&table, ctx, stream)?;
        let recurrent_output = ctx.buffers.attn_output();
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
                output: recurrent_output,
                num_tokens: rows_per_seq as u32,
                num_sequences: num_sequences as u32,
                snapshot_count: snapshot_count as u32,
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

    pub(super) fn kda_forward_verify(
        &self,
        weights: &KdaWeights,
        normed: DevicePtr,
        rows: usize,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            rows <= GLM53_VERIFY_MAX_ROWS,
            "GLM KDA verify width {rows} exceeds {GLM53_VERIFY_MAX_ROWS}"
        );
        let hidden = ctx.config.hidden_size;
        if ctx.profile {
            ctx.gpu.synchronize(stream)?;
        }
        let qkv_started = ctx.profile.then(std::time::Instant::now);
        let projected = self.kda_project_inputs(weights, normed, rows, ctx, stream)?;
        let (query, key, value, forget, output_gate, beta) = (
            projected.query,
            projected.key,
            projected.value,
            projected.forget,
            projected.output_gate,
            projected.beta,
        );
        let qkv_ms = if let Some(started) = qkv_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };

        // The merged projection now includes q/k/v, beta, and both low-rank
        // inputs; preserve the profile schema while reporting the full stage
        // under qkv_ms.
        let gates_ms = 0.0;

        let state = state
            .as_any()
            .downcast_ref::<KdaLayerState>()
            .context("GLM KDA verify received incompatible state")?;
        let snapshot_count = rows.saturating_sub(1);
        ensure!(
            state.intermediates.len() >= snapshot_count,
            "GLM KDA verify needs {snapshot_count} rollback images but has {}",
            state.intermediates.len()
        );
        let mut table = Vec::with_capacity((snapshot_count + 1) * 4);
        for image in std::iter::once(state.current)
            .chain(state.intermediates.iter().copied().take(snapshot_count))
        {
            table.extend([
                image.recurrent.0,
                image.q_conv.0,
                image.k_conv.0,
                image.v_conv.0,
            ]);
        }
        let state_images = self.upload_state_table(&table, ctx, stream)?;
        let recurrent_output = ctx.buffers.attn_output();
        let recurrent_started = ctx.profile.then(std::time::Instant::now);
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
                output: recurrent_output,
                num_tokens: rows as u32,
                num_sequences: 1,
                snapshot_count: snapshot_count as u32,
                num_heads: KDA_HEADS as u32,
                norm_epsilon: ctx.config.rms_norm_eps as f32,
            },
            stream,
        )?;
        let recurrent_ms = if let Some(started) = recurrent_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let output = ctx.buffers.residual();
        let output_started = ctx.profile.then(std::time::Instant::now);
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
        if let Some(started) = output_started {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "GLM_KDA_VERIFY_PROFILE layer={} rows={} qkv_ms={:.3} gates_ms={:.3} recurrent_ms={:.3} output_ms={:.3}",
                self.layer_idx,
                rows,
                qkv_ms,
                gates_ms,
                recurrent_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(output)
    }
}
