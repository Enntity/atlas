// SPDX-License-Identifier: AGPL-3.0-only

//! Reusable native GLM EXL3 projections and explicit TP output ownership.

use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::exl3::{
    native::{Exl3InputPrecision, Exl3LinearNumerics, Exl3LinearWorkspace},
    tp::{Exl3DeviceShard, ShardAxis},
};

/// Output ownership after a successful projection and its required reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exl3OutputSemantics {
    /// Full output channels, replicated on each TP rank.
    FullChannels,
    /// Only this rank's output channels, in shard-plan segment order.
    LocalChannels,
}

#[cfg(test)]
#[path = "exl3_projection_tests.rs"]
mod tests;

/// Owns packed checkpoint weights and activation scratch, never a decoded
/// dense matrix. The caller must register this resource for model teardown.
pub struct Glm5Exl3Projection {
    shard: Exl3DeviceShard,
    workspace: Exl3LinearWorkspace,
    stream: u64,
    capacity: usize,
    active: bool,
}

impl Glm5Exl3Projection {
    /// Takes ownership of a validated rank-local checkpoint upload.
    pub fn new(
        gpu: &dyn GpuBackend,
        shard: Exl3DeviceShard,
        capacity: usize,
        stream: u64,
    ) -> Result<Self> {
        Self::new_with_input_precision(gpu, shard, capacity, stream, Exl3InputPrecision::F32)
    }

    /// Own a packed projection with an explicit input rounding contract.
    /// Kernel-resolution failure releases the uploaded shard; there is no fallback.
    pub fn new_with_input_precision(
        gpu: &dyn GpuBackend,
        shard: Exl3DeviceShard,
        capacity: usize,
        stream: u64,
        input_precision: Exl3InputPrecision,
    ) -> Result<Self> {
        let plan = shard.plan();
        let workspace = Exl3LinearWorkspace::new_with_input_precision(
            gpu,
            plan.in_features(),
            plan.out_features(),
            capacity,
            stream,
            input_precision,
        );
        Self::finish(gpu, shard, capacity, stream, workspace)
    }

    /// Own an explicitly configured cooperative projection. Failed preflight
    /// releases the transferred packed shard, with no scalar fallback.
    pub fn new_with_numerics(
        gpu: &dyn GpuBackend,
        shard: Exl3DeviceShard,
        capacity: usize,
        stream: u64,
        numerics: Exl3LinearNumerics,
    ) -> Result<Self> {
        let plan = shard.plan();
        let workspace = (|| {
            ensure!(
                plan.bits() == numerics.bits,
                "EXL3 projection bits differ from numerical profile"
            );
            Exl3LinearWorkspace::new_with_numerics(
                gpu,
                plan.in_features(),
                plan.out_features(),
                capacity,
                stream,
                numerics,
            )
        })();
        Self::finish(gpu, shard, capacity, stream, workspace)
    }

    fn finish(
        gpu: &dyn GpuBackend,
        mut shard: Exl3DeviceShard,
        capacity: usize,
        stream: u64,
        workspace: Result<Exl3LinearWorkspace>,
    ) -> Result<Self> {
        match workspace {
            Ok(workspace) => Ok(Self {
                shard,
                workspace,
                stream,
                capacity,
                active: true,
            }),
            Err(error) => {
                let cleanup = shard.release(gpu);
                Err(match cleanup {
                    Ok(()) => error,
                    Err(cleanup) => {
                        error.context(format!("EXL3 GLM shard cleanup failed: {cleanup:#}"))
                    }
                })
            }
        }
    }

    /// Apply local BF16 inputs and produce outputs with explicit ownership.
    /// The communicator must be the TP group used to construct the shard;
    /// supplying an EP group is a caller error. Row-parallel projections
    /// validate topology before launching and sum partials exactly once.
    pub fn forward_bf16(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        tp_comm: Option<&dyn CommBackend>,
    ) -> Result<Exl3OutputSemantics> {
        self.validate_batch(rows)?;
        let plan = self.shard.plan();
        let reduce = plan.axis() == ShardAxis::Input && plan.world() > 1;
        let collective = if reduce {
            let comm = tp_comm.context("EXL3 row-parallel projection requires TP communicator")?;
            ensure!(
                comm.rank() == plan.rank() && comm.world_size() == plan.world(),
                "EXL3 TP communicator does not match shard rank/world"
            );
            Some(comm)
        } else {
            None
        };
        let bytes = rows
            .checked_mul(plan.out_features())
            .and_then(|n| n.checked_mul(2))
            .context("EXL3 GLM output byte count overflow")?;
        self.launch_local_bf16(gpu, input, output, rows)?;
        if let Some(comm) = collective.filter(|_| rows > 0) {
            comm.all_reduce_async(output.0, bytes, self.stream)
                .context("EXL3 GLM TP output sum")?;
        }
        Ok(if plan.axis() == ShardAxis::Output && plan.world() > 1 {
            Exl3OutputSemantics::LocalChannels
        } else {
            Exl3OutputSemantics::FullChannels
        })
    }

    pub(super) fn validate_batch(&self, rows: usize) -> Result<()> {
        ensure!(self.active, "EXL3 GLM projection teardown has started");
        ensure!(
            rows <= self.capacity,
            "EXL3 GLM batch exceeds workspace capacity"
        );
        self.shard.buffers()?;
        Ok(())
    }

    /// Validate the geometry and execution topology supplied by an owning layer.
    pub fn validate_local_forward(
        &self,
        rows: usize,
        k: usize,
        n: usize,
        stream: u64,
        rank: usize,
        world: usize,
    ) -> Result<()> {
        self.validate_batch(rows)?;
        let plan = self.shard.plan();
        ensure!(
            (k, n) == (plan.in_features(), plan.out_features()),
            "EXL3 layer projection dimensions mismatch"
        );
        ensure!(stream == self.stream, "EXL3 layer compute stream mismatch");
        ensure!(
            (rank, world) == (plan.rank(), plan.world()),
            "EXL3 layer TP rank/world mismatch"
        );
        Ok(())
    }

    /// Submit only this rank's contribution. The owning layer must complete
    /// input-parallel outputs with its existing TP collective exactly once.
    pub fn launch_local_bf16(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        self.validate_batch(rows)?;
        self.workspace
            .launch_bf16(gpu, &self.shard, input, output, rows)
    }

    pub fn stream(&self) -> u64 {
        self.stream
    }

    /// Rank-local F32 logits, retaining the final H128/scale precision.
    pub fn launch_local_f32(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        self.validate_batch(rows)?;
        self.workspace
            .launch_f32(gpu, &self.shard, input, output, rows)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn shard(&self) -> &Exl3DeviceShard {
        &self.shard
    }
}

impl<'a> ModelResource<dyn GpuBackend + 'a> for Glm5Exl3Projection {
    fn label(&self) -> &'static str {
        "GLM EXL3 projection"
    }

    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        // Always attempt both releases. Their resources retain failed handles
        // so a later teardown retry can free them.
        self.active = false;
        let scratch = self.workspace.release(gpu);
        let weights = self.shard.release(gpu);
        match (scratch, weights) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(error), Err(other)) => Err(error.context(format!(
                "EXL3 GLM weights also failed to release: {other:#}"
            ))),
        }
    }
}
