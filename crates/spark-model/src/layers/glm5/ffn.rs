// SPDX-License-Identifier: AGPL-3.0-only

//! GLM dense and EXL3 routed FFN execution.

use anyhow::{Result, ensure};
use spark_runtime::buffers::GLM53_EXL3_LOCK_BYTES;
use spark_runtime::gpu::DevicePtr;

use super::types::{Glm5Layer, GlmDenseFfnWeights, GlmExl3MoeWeights, GlmFfn};
use crate::layer::ForwardContext;
use crate::layers::ops::{
    self, GLM53_EXL3_FAT_CAP, GLM53_EXL3_ROWS, Glm53Exl3FatKernels, Glm53Exl3Workspace,
    glm53_exl3_concurrency,
};
use crate::weight_map::DenseWeight;

const GLM53_EXL3_ACTIVATION_LIMIT: f32 = 10.0;

impl Glm5Layer {
    #[allow(clippy::too_many_arguments)]
    fn dense_project(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        rows: usize,
        n: usize,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if rows == 1 {
            return ops::dense_gemv(
                ctx.gpu,
                self.kernels.dense_gemv,
                input,
                weight,
                output,
                n as u32,
                k as u32,
                stream,
            );
        }
        ops::cublas_bf16_proj_dense(
            input,
            weight.weight,
            output,
            rows as u32,
            n as u32,
            k as u32,
            stream,
        )
    }

    fn dense_ffn(
        &self,
        weights: &GlmDenseFfnWeights,
        input: DevicePtr,
        rows: usize,
        output: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hidden = ctx.config.hidden_size;
        let intermediate = weights.intermediate_size;
        let gate = ctx.buffers.expert_gate_out();
        let up = ctx.buffers.expert_up_out();
        self.dense_project(
            input,
            &weights.gate,
            gate,
            rows,
            intermediate,
            hidden,
            ctx,
            stream,
        )?;
        self.dense_project(
            input,
            &weights.up,
            up,
            rows,
            intermediate,
            hidden,
            ctx,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.kernels.silu_mul,
            gate,
            up,
            gate,
            (rows * intermediate) as u32,
            stream,
        )?;
        self.dense_project(
            gate,
            &weights.down,
            output,
            rows,
            hidden,
            intermediate,
            ctx,
            stream,
        )
    }

    fn exl3_workspace(&self, ctx: &ForwardContext) -> Result<Glm53Exl3Workspace> {
        let base = ctx.buffers.glm_workspace();
        ensure!(!base.is_null(), "GLM EXL3 workspace is not allocated");
        let layout = ctx.buffers.glm_layout();
        Ok(Glm53Exl3Workspace {
            hidden_fp16: base.offset(layout.exl3_hidden_fp16),
            output_fp32: base.offset(layout.exl3_output_fp32),
            temp_state_g: base.offset(layout.exl3_temp_state_g),
            temp_state_u: base.offset(layout.exl3_temp_state_u),
            temp_intermediate_g: base.offset(layout.exl3_temp_intermediate_g),
            temp_intermediate_u: base.offset(layout.exl3_temp_intermediate_u),
            expert_count: base.offset(layout.exl3_expert_count),
            fat_descriptors: base.offset(layout.exl3_fat_descriptors),
            token_sorted: base.offset(layout.exl3_token_sorted),
            weight_sorted: base.offset(layout.exl3_weight_sorted),
            locks: base.offset(layout.exl3_locks),
        })
    }

    fn exl3_window(
        &self,
        weights: &GlmExl3MoeWeights,
        input: DevicePtr,
        ids: DevicePtr,
        route_weights: DevicePtr,
        output: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hidden = ctx.config.hidden_size;
        let workspace = self.exl3_workspace(ctx)?;
        if ctx.profile {
            ctx.gpu.synchronize(stream)?;
        }
        let input_started = ctx.profile.then(std::time::Instant::now);
        ctx.gpu
            .memset_async(workspace.output_fp32, 0, rows * hidden * 4, stream)?;
        ctx.gpu
            .memset_async(workspace.locks, 0, GLM53_EXL3_LOCK_BYTES, stream)?;
        ops::glm53_exl3_bf16_to_fp16(
            ctx.gpu,
            self.kernels.exl3_bf16_to_fp16,
            input,
            workspace.hidden_fp16,
            (rows * hidden) as u32,
            stream,
        )?;
        let input_ms = if let Some(started) = input_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let routes_started = ctx.profile.then(std::time::Instant::now);
        let fat_enabled = rows > GLM53_EXL3_FAT_CAP
            && std::env::var("ATLAS_GLM_EXL3_FAT").ok().as_deref() != Some("0");
        let row_capacity = if fat_enabled {
            GLM53_EXL3_FAT_CAP as u32
        } else {
            rows.clamp(1, GLM53_EXL3_ROWS) as u32
        };
        ops::glm53_exl3_prepare_routes(
            ctx.gpu,
            self.kernels.exl3_prepare_routes,
            ids,
            route_weights,
            &workspace,
            rows as u32,
            ctx.config.num_experts_per_tok as u32,
            ctx.config.num_experts as u32,
            weights.local_expert_start as u32,
            weights.local_expert_end as u32,
            row_capacity,
            stream,
        )?;
        let routes_ms = if let Some(started) = routes_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let kernel_started = ctx.profile.then(std::time::Instant::now);
        ops::glm53_exl3_moe(
            ctx.gpu,
            self.kernels.exl3_moe,
            &weights.pointers,
            &workspace,
            hidden as u32,
            weights.intermediate_size as u32,
            ctx.config.num_experts as u32,
            ctx.config.num_experts_per_tok as u32,
            row_capacity,
            glm53_exl3_concurrency(rows),
            GLM53_EXL3_ACTIVATION_LIMIT,
            stream,
        )?;
        if fat_enabled {
            ops::glm53_exl3_fat(
                ctx.gpu,
                Glm53Exl3FatKernels {
                    gather: self.kernels.exl3_fat_gather,
                    gate_up: self.kernels.exl3_fat_gate_up,
                    activate: self.kernels.exl3_fat_activate,
                    down: self.kernels.exl3_fat_down,
                },
                &weights.pointers,
                &workspace,
                rows as u32,
                hidden as u32,
                weights.intermediate_size as u32,
                ctx.config.num_experts as u32,
                GLM53_EXL3_ACTIVATION_LIMIT,
                stream,
            )?;
        }
        let kernel_ms = if let Some(started) = kernel_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let output_started = ctx.profile.then(std::time::Instant::now);
        ops::glm53_exl3_fp32_to_bf16(
            ctx.gpu,
            self.kernels.exl3_fp32_to_bf16,
            workspace.output_fp32,
            output,
            (rows * hidden) as u32,
            stream,
        )?;
        if let Some(started) = output_started {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "GLM_EXL3_WINDOW_PROFILE layer={} rows={} input_ms={:.3} routes_ms={:.3} kernel_ms={:.3} output_ms={:.3}",
                self.layer_idx,
                rows,
                input_ms,
                routes_ms,
                kernel_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(())
    }

    fn exl3_ffn(
        &self,
        weights: &GlmExl3MoeWeights,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let hidden = ctx.config.hidden_size;
        let topk = ctx.config.num_experts_per_tok;
        ensure!(rows > 0, "GLM EXL3 requires at least one row");

        if ctx.profile {
            ctx.gpu.synchronize(stream)?;
        }
        let shared_started = ctx.profile.then(std::time::Instant::now);

        // Shared and routed experts are independent local TP2 partials. The
        // shared BF16 branch runs on one model-wide auxiliary stream while the
        // compute stream routes and executes EXL3. Event edges make both the
        // eager order and a multi-stream CUDA graph explicit.
        let shared_output = ctx.buffers.attn_output();
        let schedule = self.shared_expert_schedule.as_ref();
        let overlap_shared = schedule.enabled && !ctx.profile;
        if overlap_shared {
            ctx.gpu.record_event(schedule.input_ready, stream)?;
            ctx.gpu
                .stream_wait_event(schedule.stream, schedule.input_ready)?;
            self.dense_ffn(
                &weights.shared,
                input,
                rows,
                shared_output,
                ctx,
                schedule.stream,
            )?;
            ctx.gpu
                .record_event(schedule.output_ready, schedule.stream)?;
        } else {
            self.dense_ffn(&weights.shared, input, rows, shared_output, ctx, stream)?;
        }
        let shared_ms = if let Some(started) = shared_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };

        let router_started = ctx.profile.then(std::time::Instant::now);
        let logits = ctx.buffers.gate_logits_f32();
        ops::dense_gemm(
            ctx.gpu,
            self.kernels.dense_gemm_f32out,
            input,
            &weights.router,
            logits,
            rows as u32,
            ctx.config.num_experts as u32,
            hidden as u32,
            stream,
        )?;
        let router_ms = if let Some(started) = router_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let topk_started = ctx.profile.then(std::time::Instant::now);
        let ids = ctx.buffers.scratch();
        let route_weights = ids.offset(rows * topk * 4);
        ops::glm53_moe_topk_sigmoid_batched_f32(
            ctx.gpu,
            self.kernels.moe_topk_sigmoid_batched_f32,
            logits,
            weights.correction_bias.weight,
            ids,
            route_weights,
            ctx.config.num_experts as u32,
            topk as u32,
            ctx.config.norm_topk_prob,
            ctx.config.routed_scaling_factor as f32,
            rows as u32,
            stream,
        )?;
        let topk_ms = if let Some(started) = topk_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };

        let routed_started = ctx.profile.then(std::time::Instant::now);
        let routed = ctx.buffers.moe_output();
        for row in (0..rows).step_by(GLM53_EXL3_ROWS) {
            let window_rows = (rows - row).min(GLM53_EXL3_ROWS);
            self.exl3_window(
                weights,
                input.offset(row * hidden * 2),
                ids.offset(row * topk * 4),
                route_weights.offset(row * topk * 4),
                routed.offset(row * hidden * 2),
                window_rows,
                ctx,
                stream,
            )?;
        }
        let routed_ms = if let Some(started) = routed_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };

        let residual_started = ctx.profile.then(std::time::Instant::now);
        if overlap_shared {
            ctx.gpu.stream_wait_event(stream, schedule.output_ready)?;
        }
        ops::residual_add(
            ctx.gpu,
            self.kernels.residual_add,
            routed,
            shared_output,
            (rows * hidden) as u32,
            stream,
        )?;
        let residual_ms = if let Some(started) = residual_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let tp_started = ctx.profile.then(std::time::Instant::now);
        self.tp_sum(routed, rows, ctx, stream)?;
        if let Some(started) = tp_started {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "GLM_EXL3_PROFILE layer={} rows={} shared_ms={:.3} router_ms={:.3} topk_ms={:.3} routed_ms={:.3} residual_ms={:.3} tp_ms={:.3}",
                self.layer_idx,
                rows,
                shared_ms,
                router_ms,
                topk_ms,
                routed_ms,
                residual_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(routed)
    }

    pub(super) fn ffn_forward(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match &self.ffn {
            GlmFfn::Dense(weights) => {
                let output = ctx.buffers.moe_output();
                self.dense_ffn(weights, input, rows, output, ctx, stream)?;
                self.tp_sum(output, rows, ctx, stream)?;
                Ok(output)
            }
            GlmFfn::Exl3(weights) => self.exl3_ffn(weights, input, rows, ctx, stream),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_decode_and_prefill_rows() {
        let windows = |rows: usize| {
            (0..rows)
                .step_by(GLM53_EXL3_ROWS)
                .map(|row| (rows - row).min(GLM53_EXL3_ROWS))
                .collect::<Vec<_>>()
        };
        assert_eq!(windows(1), vec![1]);
        assert_eq!(windows(128), vec![128]);
        assert_eq!(windows(257), vec![257]);
        assert_eq!(windows(513), vec![513]);
        assert_eq!(windows(7168), vec![7168]);
        assert_eq!(windows(7169), vec![7168, 1]);
    }
}
