// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit three/four-owner geometry; scratch is not transaction authority.
use super::{ForwardContext, glm_verify_scratch::GlmVerifyScratch};
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlmOwnerBatchShape(usize);

impl GlmOwnerBatchShape {
    pub fn new(owners: usize) -> Result<Self> {
        ensure!(
            (3..=4).contains(&owners),
            "GLM owner batch requires exactly3/4 owners"
        );
        Ok(Self(owners))
    }
    pub fn owners(self) -> usize {
        self.0
    }
    pub fn rows(self) -> usize {
        self.0 * 5
    }
}

pub struct GlmOwnerBatchWorkspace<'a> {
    shape: GlmOwnerBatchShape,
    pub(crate) scratch: GlmVerifyScratch<'a>,
}

impl<'a> GlmOwnerBatchWorkspace<'a> {
    pub(crate) fn new(ctx: &ForwardContext<'a>, shape: GlmOwnerBatchShape) -> Result<Self> {
        Ok(Self {
            shape,
            scratch: GlmVerifyScratch::new(ctx, shape.owners())?,
        })
    }
    pub(crate) fn shape(&self) -> GlmOwnerBatchShape {
        self.shape
    }
}

#[cfg(test)]
#[path = "glm_owner_workspace_tests.rs"]
mod tests;
