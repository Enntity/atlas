// SPDX-License-Identifier: AGPL-3.0-only

//! Native projection dispatch at the existing GLM MLA layer boundary.

use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3AttentionLayer;
use crate::weight_loader::glm5::exl3_mla::Glm5Exl3MlaResources;
use crate::weight_map::DenseWeight;

impl Qwen3AttentionLayer {
    pub(super) fn exl3_hc_norm(
        &self,
        site: &super::HcSiteWeights,
        norm_weight: DevicePtr,
        normed: DevicePtr,
        rows: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(native) = &self.exl3_mla else {
            return Ok(false);
        };
        use crate::layers::glm5_exl3_mhc::{MhcResidualPrecision, NativeMhcBuffers};
        native
            .mhc
            .as_ref()
            .context("native EXL3 mHC kernel owner absent")?
            .pre_norm(
                ctx.gpu,
                site,
                norm_weight,
                NativeMhcBuffers {
                    highway: ctx.buffers.hc_streams(),
                    normed,
                    post: ctx.buffers.hc_post(),
                    comb: ctx.buffers.hc_comb(),
                },
                rows as usize,
                if self.block_idx == 0
                    && self
                        .hc
                        .as_ref()
                        .is_some_and(|hc| std::ptr::eq(site, &hc.attn))
                {
                    MhcResidualPrecision::Rounded
                } else {
                    MhcResidualPrecision::for_fused_rows(rows as usize)
                },
                stream,
            )?;
        Ok(true)
    }

    pub fn set_exl3_mla_resources(&mut self, resources: Glm5Exl3MlaResources) -> Result<()> {
        ensure!(
            resources.projections.len() == 5,
            "EXL3 MLA requires five projections"
        );
        ensure!(
            self.mla.is_some(),
            "EXL3 MLA requires installed MLA structure"
        );
        ensure!(
            self.exl3_mla.is_none(),
            "EXL3 MLA resources already installed"
        );
        ensure!(
            self.mla_q4.is_empty() && self.mla_mx.is_empty(),
            "EXL3 MLA cannot use converted projection caches"
        );
        self.exl3_mla = Some(resources);
        Ok(())
    }

    pub(crate) fn has_exl3_mla(&self) -> bool {
        self.exl3_mla.is_some()
    }

    pub(super) fn validate_exl3_mla_forward(
        &self,
        rows: usize,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(native) = &self.exl3_mla else {
            return Ok(());
        };
        ensure!(
            ctx.config.model_type == "glm5_next"
                && ctx.config.ep_world_size == 1
                && !ctx.config.expert_tp,
            "EXL3 MLA requires pure GLM TP"
        );
        ensure!(
            crate::layers::glm_sp::current().is_none(),
            "EXL3 MLA sequence parallelism unsupported"
        );
        if ctx.config.tp_world_size > 1 {
            let comm = ctx
                .comm
                .ok_or_else(|| anyhow::anyhow!("EXL3 MLA requires TP communicator"))?;
            ensure!(
                (comm.rank(), comm.world_size()) == (ctx.config.tp_rank, ctx.config.tp_world_size),
                "EXL3 MLA TP communicator mismatch"
            );
        }
        native.index_f32_params()?;
        native.index_exec()?.validate(
            rows,
            ctx.config.index_n_heads,
            ctx.config.index_head_dim,
            ctx.config.hidden_size,
            ctx.buffers.sizes().ssm_gates,
            stream,
        )?;
        for projection in &native.projections {
            let plan = projection.shard().plan();
            projection.validate_local_forward(
                rows,
                plan.in_features(),
                plan.out_features(),
                stream,
                ctx.config.tp_rank,
                ctx.config.tp_world_size,
            )?;
        }
        Ok(())
    }

    /// Projection references identify their logical field; the dense pointers
    /// are never interpreted as EXL3 storage or used as a dispatch discriminator.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_exl3_mla_projection(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        rows: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(native) = &self.exl3_mla else {
            return Ok(false);
        };
        let mla = self.mla.as_ref().expect("EXL3 MLA installed structure");
        let fields = [
            &mla.wq_a,
            &mla.wq_b,
            &mla.wkv_a,
            &mla.wo,
            &mla.glm_indexer.as_ref().expect("EXL3 MLA indexer").wq_b,
        ];
        let Some(slot) = fields.iter().position(|field| std::ptr::eq(*field, weight)) else {
            return Ok(false);
        };
        let projection = &native.projections[slot];
        let plan = projection.shard().plan();
        projection.validate_local_forward(
            rows as usize,
            k as usize,
            n as usize,
            stream,
            plan.rank(),
            plan.world(),
        )?;
        // Existing attention callers own O's TP reduction and the mHC seam.
        projection.launch_local_bf16(gpu, input, output, rows as usize)?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_dense_gemv(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.try_exl3_mla_projection(gpu, input, weight, output, 1, n, k, stream)? {
            return Ok(());
        }
        crate::layers::ops::dense_gemv(gpu, kernel, input, weight, output, n, k, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_dense_gemm(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        rows: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.try_exl3_mla_projection(gpu, input, weight, output, rows, n, k, stream)? {
            return Ok(());
        }
        crate::layers::ops::dense_gemm(gpu, kernel, input, weight, output, rows, n, k, stream)
    }

    pub(crate) fn release_exl3_mla(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let native = match self.exl3_mla.as_mut() {
            Some(resources) => resources.release(gpu),
            None => Ok(()),
        };
        let ffn = self.ffn.release_exl3(gpu);
        native.and(ffn)
    }
}
