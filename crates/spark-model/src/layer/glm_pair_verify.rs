// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed two-owner compute scratch. This does not allocate or grant a lease.

use super::glm_verify_scratch::GlmVerifyScratch;
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{ForwardContext, LayerState};

pub(crate) const ROW_BYTES: usize = 4096 * 2;
pub(crate) const OWNER_NORM_BYTES: usize = 5 * ROW_BYTES;
pub(crate) const OWNER_HIGHWAY_BYTES: usize = 5 * 4 * 4096 * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlmPairFfn {
    TwoK5,
    Joint,
    JointSharedM10,
}

/// Shared arithmetic width inside the already-joint routed FFN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GlmPairShared {
    TwoM5,
    M10,
}

impl GlmPairFfn {
    pub(crate) fn joint_shared(self) -> Option<GlmPairShared> {
        match self {
            Self::TwoK5 => None,
            Self::Joint => Some(GlmPairShared::TwoM5),
            Self::JointSharedM10 => Some(GlmPairShared::M10),
        }
    }
}
/// Borrowed real owner inputs, ordered by the model's selected physical owner group.
/// The model must validate actual slot/generation, state and KV ownership first.
pub struct GlmPairLayerInput<'a> {
    pub(crate) hidden: DevicePtr,
    pub(crate) state: &'a mut dyn LayerState,
    pub(crate) positions: &'a [usize; 5],
    pub(crate) block_table: &'a [u32],
}

/// Two-owner adapter over the original checked arena tails.
/// Norm saves remain rows10..20; mHC saves remain rows5..15.
pub struct GlmPairWorkspace<'a> {
    pub(crate) mode: GlmPairFfn,
    pub(crate) scratch: GlmVerifyScratch<'a>,
}

impl<'a> GlmPairWorkspace<'a> {
    pub(crate) fn new(ctx: &ForwardContext<'a>, mode: GlmPairFfn) -> Result<Self> {
        Ok(Self {
            mode,
            scratch: GlmVerifyScratch::new(ctx, 2)?,
        })
    }
    pub(crate) fn validate_context(&self, ctx: &ForwardContext, stream: u64) -> Result<()> {
        self.scratch.validate_context(ctx, stream)
    }
}
