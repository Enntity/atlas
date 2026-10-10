// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 dense/shared/routed FFNs. Routing stays entirely on GPU;
//! TP ranks own every expert's local intermediate channels.

use crate::layer::ForwardContext;
use crate::weight_loader::Exl3FfnActivation;
use anyhow::{Context, Result, ensure};
use atlas_core::{config::ModelConfig, scope::ModelResource};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::exl3::{
    Exl3CheckpointRouter,
    tp::{Exl3DeviceShard, ShardAxis},
};

mod dense;
mod dense_policy;
mod diagnostics;
mod load;
mod moe;
mod profile;
mod scratch;
mod shared;
pub use scratch::{MoeScratch, MoeScratchBorrow, MoeScratchGeometry};
mod tables;
pub use dense::NativeDenseFfn;
pub use dense_policy::{NativeDenseActivation, NativeDenseFfnRole};
pub use load::{load_glm5_exl3_ffn, load_glm5_exl3_ffn_with_scratch};
pub use moe::NativeMoeFfn;

pub enum Glm5Exl3FfnLayer {
    Dense(Box<NativeDenseFfn>),
    Moe(Box<NativeMoeFfn>),
}
impl Glm5Exl3FfnLayer {
    pub fn layer_index(&self) -> Option<usize> {
        match self {
            Self::Dense(d) => d.layer,
            Self::Moe(m) => m.layer,
        }
    }
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }
    pub fn validate_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (hidden, rank, world) = match self {
            Self::Dense(d) => {
                d.validate(rows, stream, ctx.config.tp_rank, ctx.config.tp_world_size)?;
                (d.hidden, d.rank, d.world)
            }
            Self::Moe(m) => {
                m.validate(rows, stream, ctx.config.tp_rank, ctx.config.tp_world_size)?;
                (m.hidden, m.rank, m.world)
            }
        };
        let diagnostic = match self {
            Self::Dense(d) => d.capture.is_some(),
            Self::Moe(m) => m.capture.is_some(),
        };
        ensure!(
            !(diagnostic || ctx.profile) || !ctx.graph_capture,
            "EXL3 FFN diagnostics require eager execution"
        );
        ensure!(
            ctx.config.ep_world_size == 1 && ctx.config.ep_rank == 0 && !ctx.config.expert_tp,
            "Native EXL3 FFN requires pure TP"
        );
        if world > 1 {
            let comm = ctx
                .comm
                .context("Native EXL3 FFN requires TP communicator")?;
            ensure!(
                (comm.rank(), comm.world_size()) == (rank, world),
                "Native EXL3 FFN TP communicator mismatch"
            );
        }
        let bytes = rows
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(2))
            .context("EXL3 FFN output bytes overflow")?;
        ensure!(
            ctx.buffers.sizes().moe_output >= bytes,
            "EXL3 FFN output arena too small"
        );
        if rows > 0 {
            ensure!(!input.is_null(), "Native EXL3 FFN input missing");
        }
        Ok(())
    }
    pub fn forward_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.validate_rows(input, rows, ctx, stream)?;
        let (hidden, world) = match self {
            Self::Dense(d) => (d.hidden, d.world),
            Self::Moe(m) => (m.hidden, m.world),
        };
        let collective = if world > 1 { ctx.comm } else { None };
        let bytes = rows * hidden * 2; // validate_rows checked this product.
        let output = ctx.buffers.moe_output();
        if rows == 0 {
            return Ok(output);
        }
        ensure!(
            !input.is_null() && !output.is_null(),
            "Native EXL3 FFN input/output missing"
        );
        let (capture, invocation) = match self {
            Self::Dense(d) => (
                &d.capture,
                d.forward_local_profiled(ctx.gpu, input, output, rows, ctx.profile)?,
            ),
            Self::Moe(m) => (
                &m.capture,
                m.forward_local_profiled(ctx.gpu, input, output, rows, ctx.profile)?,
            ),
        };
        let timer = profile::FfnProfile::new(
            ctx.gpu,
            stream,
            ctx.profile,
            profile::ProfileMetadata {
                layer: self.layer_index(),
                rows,
                rank: ctx.config.tp_rank,
                world,
                routes: 0,
                role: "tp_sum",
                precision: None,
                thin_capacity: None,
            },
        );
        timer.measure("tp_sum", || {
            if let Some(comm) = collective {
                if ctx.graph_capture {
                    comm.all_reduce(output.0, bytes)?;
                } else {
                    comm.all_reduce_async(output.0, bytes, stream)?;
                }
            }
            Ok(())
        })?;
        diagnostics::capture(
            capture,
            invocation,
            ctx.gpu,
            "output_after_tp",
            &[(
                "output",
                diagnostics::DiagnosticDtype::Bf16,
                output,
                &[rows, hidden],
            )],
            stream,
        )?;
        Ok(output)
    }
}
impl<'a> ModelResource<dyn GpuBackend + 'a> for Glm5Exl3FfnLayer {
    fn label(&self) -> &'static str {
        "GLM native EXL3 FFN layer"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        match self {
            Self::Dense(d) => d.release(gpu),
            Self::Moe(m) => m.release(gpu),
        }
    }
}
fn retain_error(error: &mut Option<anyhow::Error>, result: Result<()>) {
    if let Err(e) = result
        && error.is_none()
    {
        *error = Some(e);
    }
}
fn finish(error: Option<anyhow::Error>) -> Result<()> {
    match error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
#[cfg(test)]
mod tests;

#[cfg(test)]
mod shared_tests;

#[cfg(test)]
mod scratch_tests;

#[cfg(test)]
mod profile_tests;
