// SPDX-License-Identifier: AGPL-3.0-only

use super::FfnComponent;
use crate::layer::ForwardContext;
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

impl FfnComponent {
    /// Independent C4 FFN. Never aliases the temporal verifier dispatch.
    pub fn forward_c4(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            crate::model::glm_c4::enabled(&ctx.config.model_type),
            "GLM independent C4 FFN requires its explicit opt-in"
        );
        match self {
            Self::Moe(m) => m.forward_c4(input, ctx, stream),
            Self::Dense(d) if d.can_forward_km(4) => {
                d.forward_km(input, 4, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            // Native NVFP4 dense layers have no W4A16 weight placeholders.
            // Retain their existing four-row dense prefill implementation.
            Self::Dense(d) => {
                d.forward_prefill(input, 4, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::None => Ok(input),
        }
    }
}
