// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded stateless owner-batch FFN; does not grant a producer or serving authority.
use super::*;
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;
use crate::layers::FfnComponent;

impl FfnComponent {
    pub(crate) fn validate_owner_verify(
        &self,
        input: DevicePtr,
        shape: GlmOwnerBatchShape,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        super::forward_pair_validate::validate_rows(input, shape.rows(), ctx, stream)?;
        match self {
            Self::Moe(layer) => layer.validate_verify_resources(ctx),
            Self::Dense(layer) => {
                // Each dense owner still runs the established K5 kernel. The
                // wider output is assembled in-place, not a new dense GEMM.
                let bytes = ctx
                    .config
                    .intermediate_size
                    .checked_mul(10)
                    .ok_or_else(|| anyhow::anyhow!("owner dense scratch overflow"))?;
                anyhow::ensure!(
                    layer.can_forward_km(5)
                        && !layer.weights.up_proj.is_null()
                        && !layer.weights.down_proj.is_null()
                        && ctx.buffers.sizes().expert_gate_out >= bytes
                        && ctx.buffers.sizes().expert_up_out >= bytes,
                    "owner dense requires actual K5 kernels/weights and scratch"
                );
                Ok(())
            }
            Self::None => anyhow::bail!("owner FFN requires actual MoE or dense weights"),
        }
    }

    pub(crate) fn forward_owner_verify(
        &self,
        input: DevicePtr,
        shape: GlmOwnerBatchShape,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.validate_owner_verify(input, shape, ctx, stream)?;
        let output = ctx.buffers.moe_output();
        match self {
            Self::Moe(layer) => layer.forward_prefill_mode(
                input,
                shape.rows(),
                ctx,
                stream,
                false,
                super::forward_pair_verify::PrefillMode::OwnerVerify(shape),
            )?,
            Self::Dense(_) => {
                // K5 writes row0. Save each higher owner before the next call
                // overwrites it; every source remains in normalized rows[0,N).
                for owner in (0..shape.owners()).rev() {
                    let result = self.forward_k5(input.offset(owner * 40960), ctx, stream)?;
                    anyhow::ensure!(result == output, "owner dense K5 output arena changed");
                    if owner != 0 {
                        ctx.gpu.copy_d2d_async(
                            result,
                            output.offset(owner * 40960),
                            40960,
                            stream,
                        )?;
                    }
                }
            }
            Self::None => anyhow::bail!("owner FFN requires actual MoE or dense weights"),
        }
        Ok(output)
    }
}
