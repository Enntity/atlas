// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash2-only grouped dynamic convolution.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;
use crate::layer::ForwardContext;
use crate::weight_loader::DflashDynamicConvWeights;

impl BlockDiffusionDraftHead {
    pub(super) fn dflash2_conv_prepare(
        &self,
        weights: &DflashDynamicConvWeights,
        input: DevicePtr,
        dynamic: DevicePtr,
        output: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rows = self.gamma as u32;
        let groups = self.dflash2_conv_groups as u32;
        crate::layers::ops::dense_gemm_bf16_pipelined(
            ctx.gpu,
            self.kernels.dense_gemm_pipelined,
            input,
            &weights.kernel_projection,
            dynamic,
            rows,
            groups * 4,
            self.hidden_size as u32,
            stream,
        )?;
        crate::layers::ops::dflash2_dynamic_conv2(
            ctx.gpu,
            self.kernels.dflash2_dynamic_conv,
            input,
            dynamic,
            weights.base_kernel.weight,
            output,
            rows,
            rows,
            self.hidden_size as u32,
            groups,
            0,
            stream,
        )
    }

    pub(super) fn dflash2_conv_finish(
        &self,
        weights: &DflashDynamicConvWeights,
        input: DevicePtr,
        dynamic: DevicePtr,
        output: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        crate::layers::ops::dflash2_dynamic_conv2(
            ctx.gpu,
            self.kernels.dflash2_dynamic_conv,
            input,
            dynamic,
            weights.base_kernel.weight,
            output,
            self.gamma as u32,
            self.gamma as u32,
            self.hidden_size as u32,
            self.dflash2_conv_groups as u32,
            1,
            stream,
        )
    }
}
