// SPDX-License-Identifier: AGPL-3.0-only

//! GPU IORouter for rank-local, byte-preserving EXL3 uploads.

use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;

use crate::gpu::{DevicePtr, GpuBackend};
use crate::weights::exl3::Exl3LinearView;

use super::TpShardPlan;

/// Owns this rank's packed weights and original FP16 scales. Register with
/// model teardown; borrowed buffer handles are invalid after release.
pub struct Exl3DeviceShard {
    plan: TpShardPlan,
    buffers: [DevicePtr; 3],
}

impl Exl3DeviceShard {
    /// Upload directly from the checkpoint views. Allocates only local
    /// buffers, with no full-trellis GPU copy or host repacking temporary.
    pub fn upload(
        gpu: &dyn GpuBackend,
        view: &Exl3LinearView<'_>,
        plan: &TpShardPlan,
    ) -> Result<Self> {
        plan.validate_source(view)?;
        let mut result = Self {
            plan: plan.clone(),
            buffers: [DevicePtr::NULL; 3],
        };
        let upload = (|| -> Result<()> {
            for (slot, bytes) in [
                plan.trellis_byte_size(),
                plan.in_features() * 2,
                plan.out_features() * 2,
            ]
            .into_iter()
            .enumerate()
            {
                result.buffers[slot] = gpu.alloc(bytes).context("EXL3 shard allocation")?;
            }
            plan.visit_trellis(view, |offset, bytes| {
                gpu.copy_h2d(bytes, result.buffers[0].offset(offset))
            })?;
            let input = plan.input_range();
            gpu.copy_h2d(
                &view.input_scale().data()[input.start * 2..input.end * 2],
                result.buffers[1],
            )?;
            plan.visit_output_scale(view, |offset, bytes| {
                gpu.copy_h2d(bytes, result.buffers[2].offset(offset))
            })?;
            Ok(())
        })();
        if let Err(error) = upload {
            let cleanup = result.release(gpu);
            return Err(match cleanup {
                Ok(()) => error,
                Err(cleanup) => {
                    error.context(format!("EXL3 shard cleanup also failed: {cleanup:#}"))
                }
            });
        }
        Ok(result)
    }

    pub fn plan(&self) -> &TpShardPlan {
        &self.plan
    }

    /// Buffer order: packed I16 trellis, FP16 input scales, FP16 output scales.
    /// The mul1 multiplier is a kernel constant, validated before upload.
    pub fn buffers(&self) -> Result<[DevicePtr; 3]> {
        ensure!(
            self.buffers.iter().all(|p| !p.is_null()),
            "EXL3 shard buffers have been released"
        );
        Ok(self.buffers)
    }
}

impl<'a> ModelResource<dyn GpuBackend + 'a> for Exl3DeviceShard {
    fn label(&self) -> &'static str {
        "EXL3 TP shard"
    }

    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        let mut error = None;
        for buffer in self.buffers.iter_mut().rev() {
            if buffer.is_null() {
                continue;
            }
            match gpu.free(*buffer) {
                Ok(()) => *buffer = DevicePtr::NULL,
                Err(e) => {
                    if error.is_none() {
                        error = Some(e);
                    }
                }
            }
        }
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
