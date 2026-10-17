// SPDX-License-Identifier: AGPL-3.0-only

//! Dense companion checkpoint/GPU I/O boundary for native EXL3 models.

use super::Exl3CheckpointRouter;
use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;
pub use safetensors::Dtype as Exl3CheckpointDtype;
use safetensors::{Dtype, tensor::TensorView};
use std::ops::Range;

impl Exl3CheckpointRouter {
    /// Borrow an unquantized companion with explicit checkpoint dtype/shape.
    /// F16 conversion is an upload decision, never a checkpoint reinterpretation.
    pub fn dense_view(
        &self,
        name: &str,
        expected_dtype: Dtype,
        shape: &[usize],
    ) -> Result<TensorView<'_>> {
        ensure!(
            matches!(expected_dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32),
            "EXL3 dense view requires F16, BF16 or F32 dtype"
        );
        ensure!(
            !shape.is_empty() && shape.iter().all(|&d| d > 0),
            "EXL3 dense view requires explicit positive dimensions"
        );
        let view = self.tensor_view(name)?;
        ensure!(
            view.dtype() == expected_dtype && view.shape() == shape,
            "EXL3 dense {name} expected {expected_dtype:?} {shape:?}, got {:?} {:?}",
            view.dtype(),
            view.shape()
        );
        Ok(view)
    }
}

/// Explicit device representation for dense companions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exl3DenseDtype {
    Bf16,
    F32,
}
impl Exl3DenseDtype {
    fn bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Owns a row-selected dense tensor. Failed frees retain the pointer for retry.
pub struct Exl3DenseDeviceTensor {
    ptr: DevicePtr,
    shape: Vec<usize>,
    dtype: Exl3DenseDtype,
}
impl Exl3DenseDeviceTensor {
    /// Select first-axis rows in original order. All later axes are preserved.
    /// BF16/F32 sources copy unchanged; F16 converts to BF16 in bounded host
    /// chunks before upload. The GPU never receives incorrectly typed F16.
    pub fn upload(
        gpu: &dyn GpuBackend,
        view: &TensorView<'_>,
        dtype: Exl3DenseDtype,
        rows: &[Range<usize>],
    ) -> Result<Self> {
        ensure!(
            matches!(
                (view.dtype(), dtype),
                (Dtype::F16 | Dtype::BF16, Exl3DenseDtype::Bf16)
                    | (Dtype::F16 | Dtype::F32, Exl3DenseDtype::F32)
            ),
            "EXL3 unsupported dense source/device dtype conversion"
        );
        let shape = view.shape();
        ensure!(
            !shape.is_empty() && shape.iter().all(|&d| d > 0),
            "EXL3 dense requires positive shape"
        );
        ensure!(
            !rows.is_empty(),
            "EXL3 dense requires explicit row selection"
        );
        let inner = shape[1..].iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d).context("EXL3 dense shape overflow")
        })?;
        let mut local_rows = 0usize;
        let mut last_end = 0;
        for row in rows {
            ensure!(
                row.start < row.end && row.end <= shape[0] && row.start >= last_end,
                "EXL3 dense row ranges must be nonempty, ordered, disjoint and in bounds"
            );
            local_rows = local_rows
                .checked_add(row.len())
                .context("EXL3 dense selected rows overflow")?;
            last_end = row.end;
        }
        let source_bytes = inner
            .checked_mul(view.dtype().bitsize() / 8)
            .context("EXL3 dense row bytes overflow")?;
        let source_len = shape[0]
            .checked_mul(source_bytes)
            .context("EXL3 dense source size overflow")?;
        ensure!(
            view.data().len() == source_len,
            "EXL3 dense source byte size mismatch"
        );
        let bytes = local_rows
            .checked_mul(inner)
            .and_then(|n| n.checked_mul(dtype.bytes()))
            .context("EXL3 dense allocation size overflow")?;
        let mut local_shape = shape.to_vec();
        local_shape[0] = local_rows;
        let mut tensor = Self {
            ptr: gpu.alloc(bytes).context("EXL3 dense allocation")?,
            shape: local_shape,
            dtype,
        };
        let result = (|| -> Result<()> {
            let mut offset = 0;
            for row in rows {
                let source = &view.data()[row.start * source_bytes..row.end * source_bytes];
                if view.dtype() == Dtype::F16 {
                    // 64 KiB source chunks bound conversion memory independently
                    // of checkpoint matrix size, while preserving FP16 rounding.
                    for chunk in source.chunks(65536) {
                        let converted = match dtype {
                            Exl3DenseDtype::Bf16 => crate::weights::f16_to_bf16_bytes(chunk),
                            Exl3DenseDtype::F32 => crate::weights::f16_to_f32_bytes(chunk),
                        };
                        gpu.copy_h2d(&converted, tensor.ptr.offset(offset))?;
                        offset += converted.len();
                    }
                } else {
                    gpu.copy_h2d(source, tensor.ptr.offset(offset))?;
                    offset += source.len();
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            return Err(match tensor.release(gpu) {
                Ok(()) => error,
                Err(cleanup) => error.context(format!("EXL3 dense cleanup failed: {cleanup:#}")),
            });
        }
        Ok(tensor)
    }
    pub fn ptr(&self) -> Result<DevicePtr> {
        ensure!(!self.ptr.is_null(), "EXL3 dense tensor has been released");
        Ok(self.ptr)
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn dtype(&self) -> Exl3DenseDtype {
        self.dtype
    }
}
impl<'a> ModelResource<dyn GpuBackend + 'a> for Exl3DenseDeviceTensor {
    fn label(&self) -> &'static str {
        "EXL3 dense companion"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        if !self.ptr.is_null() {
            gpu.free(self.ptr)?;
            self.ptr = DevicePtr::NULL;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "dense_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "dense_concat_tests.rs"]
mod concat_tests;

#[path = "dense_concat.rs"]
mod concat;
