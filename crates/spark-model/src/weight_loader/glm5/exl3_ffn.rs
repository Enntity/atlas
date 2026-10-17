// SPDX-License-Identifier: AGPL-3.0-only

//! Borrowed native EXL3 SwiGLU execution for dense, shared and routed experts.

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::exl3::tp::ShardAxis;

use super::exl3_projection::{Exl3OutputSemantics, Glm5Exl3Projection};

/// Checkpoint activation policy; callers must select it explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Exl3FfnActivation {
    SiLu,
    /// Clamp SiLU(gate) above `limit` and up to [-limit,limit], matching
    /// EXL3's upstream standard SiLU implementation.
    ClampedSiLu {
        limit: f32,
    },
}

impl Exl3FfnActivation {
    /// Derive the text activation from the parsed checkpoint's sole policy.
    /// The GLM schema defines absence as unclamped; vision policy is separate.
    pub fn from_glm_config(config: &atlas_core::config::ModelConfig) -> Result<Self> {
        ensure!(
            config.model_type == "glm5_next",
            "EXL3 FFN activation requires glm5_next config"
        );
        let activation = match config.text_swiglu_limit {
            None => Self::SiLu,
            Some(limit) => Self::ClampedSiLu { limit },
        };
        activation.validate()?;
        Ok(activation)
    }

    pub(crate) fn validate(self) -> Result<()> {
        if let Self::ClampedSiLu { limit } = self {
            ensure!(
                limit.is_finite() && limit > 0.0,
                "EXL3 FFN requires a finite positive SiLU clamp"
            );
        }
        Ok(())
    }
}

/// Caller-owned BF16 activation buffers. Gate and up hold only local
/// intermediate channels. Gate is overwritten with the activation product.
#[derive(Debug, Clone, Copy)]
pub struct Exl3FfnBuffers {
    pub input: DevicePtr,
    pub gate: DevicePtr,
    pub up: DevicePtr,
    pub output: DevicePtr,
}

/// Borrows three projections: no extra weight upload or workspace allocation.
/// Serves one dense/shared/routed FFN; routing and expert-weighted accumulation
/// remain the caller's responsibility.
pub struct Glm5Exl3Ffn<'a> {
    gate: &'a Glm5Exl3Projection,
    up: &'a Glm5Exl3Projection,
    down: &'a Glm5Exl3Projection,
    activation: Exl3FfnActivation,
    silu: KernelHandle,
}

impl<'a> Glm5Exl3Ffn<'a> {
    pub fn new(
        gpu: &dyn GpuBackend,
        gate: &'a Glm5Exl3Projection,
        up: &'a Glm5Exl3Projection,
        down: &'a Glm5Exl3Projection,
        activation: Exl3FfnActivation,
    ) -> Result<Self> {
        activation.validate()?;
        for projection in [gate, up, down] {
            projection.validate_batch(0)?;
        }
        let (g, u, d) = (gate.shard().plan(), up.shard().plan(), down.shard().plan());
        ensure!(
            g.axis() == ShardAxis::Output
                && u.axis() == ShardAxis::Output
                && d.axis() == ShardAxis::Input,
            "EXL3 FFN requires output-parallel gate/up and input-parallel down"
        );
        ensure!(
            g.rank() == u.rank()
                && g.rank() == d.rank()
                && g.world() == u.world()
                && g.world() == d.world(),
            "EXL3 FFN projection TP topology mismatch"
        );
        ensure!(
            g.in_features() == u.in_features()
                && g.in_features() == d.out_features()
                && g.out_features() == u.out_features()
                && g.out_features() == d.in_features(),
            "EXL3 FFN local projection shape mismatch"
        );
        ensure!(
            g.output_ranges() == u.output_ranges() && g.output_ranges() == [d.input_range()],
            "EXL3 FFN intermediate-channel ownership mismatch"
        );
        ensure!(
            gate.stream() == up.stream() && gate.stream() == down.stream(),
            "EXL3 FFN projections must share one compute stream"
        );
        Ok(Self {
            gate,
            up,
            down,
            activation,
            silu: gpu.kernel("exl3", "exl3_silu_mul")?,
        })
    }

    /// Gate/up consume the replicated hidden state, activation remains local,
    /// and down performs the sole TP reduction. Preflight rejects topology,
    /// capacity or buffer errors before any projection launch.
    pub fn forward_bf16(
        &self,
        gpu: &dyn GpuBackend,
        buffers: Exl3FfnBuffers,
        rows: usize,
        tp_comm: Option<&dyn CommBackend>,
    ) -> Result<Exl3OutputSemantics> {
        let plan = self.down.shard().plan();
        if plan.world() > 1 {
            let comm = tp_comm.context("EXL3 FFN requires TP communicator")?;
            ensure!(
                comm.rank() == plan.rank() && comm.world_size() == plan.world(),
                "EXL3 FFN TP communicator mismatch"
            );
        }
        for projection in [self.gate, self.up, self.down] {
            projection.validate_batch(rows)?;
        }
        let count = rows
            .checked_mul(plan.in_features())
            .context("EXL3 FFN activation count overflow")?;
        let grid =
            u32::try_from(count.div_ceil(256)).context("EXL3 FFN activation grid overflow")?;
        if rows > 0 {
            let ptrs = [buffers.input, buffers.gate, buffers.up, buffers.output];
            ensure!(
                ptrs.iter().all(|p| !p.is_null()),
                "EXL3 FFN buffers must be non-null"
            );
            for (idx, ptr) in ptrs.iter().enumerate() {
                ensure!(
                    !ptrs[..idx].contains(ptr),
                    "EXL3 FFN activation buffers must be distinct"
                );
            }
        }
        self.gate
            .forward_bf16(gpu, buffers.input, buffers.gate, rows, None)?;
        self.up
            .forward_bf16(gpu, buffers.input, buffers.up, rows, None)?;
        if rows > 0 {
            let (limit, clamp) = match self.activation {
                Exl3FfnActivation::SiLu => (0.0, 0),
                Exl3FfnActivation::ClampedSiLu { limit } => (limit, 1),
            };
            KernelLaunch::new(gpu, self.silu)
                .grid([grid, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(buffers.gate)
                .arg_ptr(buffers.up)
                .arg_ptr(buffers.gate)
                .arg_u64(count as u64)
                .arg_f32(limit)
                .arg_u32(clamp)
                .launch(self.gate.stream())?;
        }
        self.down
            .forward_bf16(gpu, buffers.gate, buffers.output, rows, tp_comm)
    }
}

#[cfg(test)]
#[path = "exl3_ffn_tests.rs"]
mod tests;
