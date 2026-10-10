// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint-format dispatch and validation at the existing KDA layer boundary.

use super::*;
use anyhow::Context;
use spark_runtime::weights::exl3::tp::ShardAxis;

impl Glm5KdaLayer {
    pub(super) fn validate_exl3_forward(
        &self,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let projections = [
            &self.weights.q_proj,
            &self.weights.k_proj,
            &self.weights.v_proj,
            &self.weights.o_proj,
        ];
        if !projections.iter().any(|p| p.is_exl3()) {
            return Ok(());
        }
        ensure!(
            projections.iter().all(|p| p.is_exl3()),
            "GLM KDA requires consistent projection formats"
        );
        ensure!(
            ctx.config.ep_world_size == 1 && !ctx.config.expert_tp,
            "EXL3 KDA requires pure TP"
        );
        ensure!(
            crate::layers::glm_sp::current().is_none(),
            "EXL3 KDA sequence-parallel forward is not implemented"
        );
        if ctx.config.tp_world_size > 1 {
            let comm = ctx.comm.context("EXL3 KDA requires TP communicator")?;
            ensure!(
                (comm.rank(), comm.world_size()) == (ctx.config.tp_rank, ctx.config.tp_world_size),
                "EXL3 KDA TP communicator mismatch"
            );
        }
        let width = self.heads * self.dim;
        for (index, weight) in projections.into_iter().enumerate() {
            let Glm5Projection::Exl3(weight) = weight else {
                unreachable!()
            };
            let (k, n, axis) = if index == 3 {
                (width, self.hidden_size, ShardAxis::Input)
            } else {
                (self.hidden_size, width, ShardAxis::Output)
            };
            ensure!(
                weight.shard().plan().axis() == axis,
                "EXL3 KDA projection TP axis mismatch"
            );
            weight.validate_local_forward(
                rows,
                k,
                n,
                stream,
                ctx.config.tp_rank,
                ctx.config.tp_world_size,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_exl3(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Glm5Projection::Exl3(weight) = weight else {
            return Ok(false);
        };
        weight.validate_local_forward(
            m as usize,
            k as usize,
            n as usize,
            stream,
            ctx.config.tp_rank,
            ctx.config.tp_world_size,
        )?;
        // KDA already owns the O projection reduction, including its fused
        // TP/mHC path. Leave this result rank-local to avoid summing twice.
        weight.launch_local_bf16(ctx.gpu, input, output, m as usize)?;
        Ok(true)
    }
}

impl<'a> atlas_core::scope::ModelResource<dyn GpuBackend + 'a> for Glm5KdaLayer {
    fn label(&self) -> &'static str {
        "GLM KDA native projection resources"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        let mut first = None;
        // Scratch only; the merged auxiliary weight belongs to the companions.
        if let Some(aux) = self.native_aux.as_mut()
            && let Err(error) = aux.release(gpu)
        {
            first = Some(error);
        }
        for weight in [
            &mut self.weights.q_proj,
            &mut self.weights.k_proj,
            &mut self.weights.v_proj,
            &mut self.weights.o_proj,
        ] {
            if let Glm5Projection::Exl3(weight) = weight
                && let Err(error) = weight.release(gpu)
                && first.is_none()
            {
                first = Some(error);
            }
        }
        if let Err(error) = self.ffn.release_exl3(gpu)
            && first.is_none()
        {
            first = Some(error);
        }
        first.map_or(Ok(()), Err)
    }
}
