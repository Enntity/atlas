// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed two-K5 stateless FFN entry; no sequence or paired Model authority.

use super::*;
use crate::layer::glm_pair_verify::GlmPairShared;
use crate::layers::FfnComponent;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefillMode {
    Legacy,
    PairVerify(GlmPairShared),
}

impl PrefillMode {
    pub(super) fn is_pair(self) -> bool {
        matches!(self, Self::PairVerify(_))
    }
}

impl FfnComponent {
    /// Conservative common workspace/resources plus the actual frozen K5
    /// grouped-T control. Does not alter forward_k5 or its environment.
    pub(crate) fn validate_pair_k5(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.validate_pair_verify(input, ctx, stream)?;
        if let Self::Moe(layer) = self {
            layer.validate_pair_k5_control()?;
        }
        Ok(())
    }

    /// Read-only preflight for the layer/model before its first attention
    /// writer. Does not inspect row contents, allocate or reserve any ownership;
    /// the existing backend capture query is the only GPU operation.
    pub(crate) fn validate_pair_verify(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(layer) => layer.validate_pair_verify(input, ctx, stream),
            Self::Dense(layer) => {
                super::forward_pair_validate::validate_common(input, ctx, stream)?;
                let bytes = ctx
                    .config
                    .intermediate_size
                    .checked_mul(10)
                    .ok_or_else(|| anyhow::anyhow!("paired dense scratch overflow"))?;
                anyhow::ensure!(
                    layer.can_forward_km(5)
                        && !layer.weights.up_proj.is_null()
                        && !layer.weights.down_proj.is_null()
                        && ctx.buffers.sizes().expert_gate_out >= bytes
                        && ctx.buffers.sizes().expert_up_out >= bytes,
                    "paired dense requires existing K5 kernels/weights and scratch"
                );
                Ok(())
            }
            Self::None => anyhow::bail!("paired FFN requires actual MoE or dense weights"),
        }
    }

    /// Input is ten contiguous normalized rows, owner0 then owner1. The caller
    /// retains both owners' mHC state separately; this entry does not preserve it.
    /// Every FFN input access is confined to rows0..9; the saved norm tail at
    /// rows10..19 is neither used as scratch nor written by this entry.
    pub(crate) fn forward_pair_verify(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        shared: GlmPairShared,
    ) -> Result<DevicePtr> {
        self.validate_pair_verify(input, ctx, stream)?;
        match self {
            Self::Moe(layer) => layer.forward_pair_verify(input, ctx, stream, shared),
            Self::Dense(_) => {
                let output = ctx.buffers.moe_output();
                // Existing K5 writes row0. Complete owner1 and save it before
                // owner0 overwrites that scratch; no generic M10 dense kernel.
                for owner in [1, 0] {
                    let result = self.forward_k5(input.offset(owner * 40960), ctx, stream)?;
                    anyhow::ensure!(result == output, "paired dense changed K5 output arena");
                    if owner == 1 {
                        ctx.gpu
                            .copy_d2d_async(result, output.offset(40960), 40960, stream)?;
                    }
                }
                Ok(output)
            }
            Self::None => anyhow::bail!("paired FFN requires actual MoE or dense weights"),
        }
    }
}

impl MoeLayer {
    pub(crate) fn forward_pair_verify(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        shared: GlmPairShared,
    ) -> Result<DevicePtr> {
        self.validate_pair_verify(input, ctx, stream)?;
        self.forward_prefill_mode(
            input,
            10,
            ctx,
            stream,
            false,
            PrefillMode::PairVerify(shared),
        )?;
        Ok(ctx.buffers.moe_output())
    }

    /// Every previous entry retains the exact legacy policy, including K5's
    /// optional deferred shared blend. Pair mode is not selected by an env var.
    pub(super) fn forward_prefill_impl(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
        defer_shared_hc: bool,
    ) -> Result<()> {
        self.forward_prefill_mode(
            input,
            num_tokens,
            ctx,
            stream,
            defer_shared_hc,
            PrefillMode::Legacy,
        )
    }
}
