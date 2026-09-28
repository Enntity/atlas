// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit arithmetic policy for a shared temporal layer traversal.
use super::{ForwardContext, glm_owner_verify::GlmOwnerBatchShape, glm_pair_verify::GlmPairFfn};
use crate::layers::FfnComponent;
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

#[derive(Clone, Copy)]
pub(crate) enum GlmVerifyFfn {
    Pair(GlmPairFfn),
    Owners(GlmOwnerBatchShape),
}

impl GlmVerifyFfn {
    pub(crate) fn owners(self) -> usize {
        match self {
            Self::Pair(_) => 2,
            Self::Owners(shape) => shape.owners(),
        }
    }
    pub(crate) fn is_two_k5(self) -> bool {
        matches!(self, Self::Pair(GlmPairFfn::TwoK5))
    }
    pub(crate) fn validate(
        self,
        ffn: &FfnComponent,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Pair(GlmPairFfn::TwoK5) => ffn.validate_pair_k5(input, ctx, stream),
            Self::Pair(_) => ffn.validate_pair_verify(input, ctx, stream),
            Self::Owners(shape) => ffn.validate_owner_verify(input, shape, ctx, stream),
        }
    }
    pub(crate) fn forward(
        self,
        ffn: &FfnComponent,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        match self {
            Self::Pair(mode) => mode
                .joint_shared()
                .map(|shared| ffn.forward_pair_verify(input, ctx, stream, shared))
                .transpose(),
            Self::Owners(shape) => ffn
                .forward_owner_verify(input, shape, ctx, stream)
                .map(Some),
        }
    }
}
