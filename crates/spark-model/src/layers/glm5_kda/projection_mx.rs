// SPDX-License-Identifier: AGPL-3.0-only

//! KDA q/k/v/o on the checkpoint's own MXFP8 bytes
//! (`ATLAS_GLM_FP4L_KDA_MXFP8_DECODE`, loaded by `weight_loader::glm5::kda_mx`).
//!
//! Every decode, multi-sequence decode and verify projection of a layer that
//! carries the twin goes through `ops::mxfp8_gemv_rows`: the
//! `mxfp8_gemv_tc{8,16,32}` tiers, 32 rows a launch, whose output row bits do
//! not depend on how many rows share the launch. So a 1-row decode, a k-row
//! DFlash verify and a multi-sequence batch agree bit for bit, the property
//! the canonical W4A16 tier gave the NVFP4 twins.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{Glm5KdaLayer, Glm5Projection};
use crate::layer::ForwardContext;

impl Glm5KdaLayer {
    /// `output[m, n] = input[m, k] · W^T` on `weight`'s MXFP8 twin; `false`,
    /// nothing launched, when it has none.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::glm5_kda) fn try_project_mxfp8(
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
        let Some(twin) = weight.mxfp8.as_ref() else {
            return Ok(false);
        };
        ensure!(
            self.mxfp8_gemv_k.iter().all(|k| k.0 != 0),
            "GLM KDA MXFP8 decode: the mxfp8_gemv tiers are missing from the kernel target"
        );
        let (m, n, k) = (m as usize, n as usize, k as usize);
        crate::layers::ops::mxfp8_gemv_rows(
            ctx.gpu,
            &self.mxfp8_gemv_k,
            twin,
            input,
            output,
            m,
            n,
            k,
            n,
            stream,
        )?;
        Ok(true)
    }

    /// Whether this layer's q/k/v/o read MXFP8 twins.
    pub(in crate::layers::glm5_kda) fn has_mxfp8_projections(&self) -> bool {
        self.weights.q_proj.mxfp8.is_some()
    }
}
