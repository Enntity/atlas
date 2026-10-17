// SPDX-License-Identifier: AGPL-3.0-only
//! Owned row concatenation; source tensors remain independently owned.
use super::*;

impl Exl3DenseDeviceTensor {
    pub fn concat_rows(gpu: &dyn GpuBackend, parts: &[&Self]) -> Result<Self> {
        ensure!(
            !parts.is_empty(),
            "EXL3 concatenation requires explicit matrices"
        );
        let first = parts[0];
        ensure!(
            first.shape.len() == 2 && first.shape[1] > 0,
            "EXL3 concat requires positive matrix columns"
        );
        let mut rows = 0usize;
        let mut pointers = Vec::new();
        for part in parts {
            ensure!(
                part.shape.len() == 2
                    && part.shape[0] > 0
                    && part.shape[1] == first.shape[1]
                    && part.dtype == first.dtype,
                "EXL3 concat matrix shape/dtype differs"
            );
            rows = rows
                .checked_add(part.shape[0])
                .context("EXL3 concat rows overflow")?;
            pointers.push(part.ptr()?);
        }
        let width = match first.dtype {
            Exl3DenseDtype::Bf16 => 2,
            Exl3DenseDtype::F32 => 4,
        };
        let row_bytes = first.shape[1]
            .checked_mul(width)
            .context("EXL3 concat row bytes overflow")?;
        let bytes = rows
            .checked_mul(row_bytes)
            .context("EXL3 concat bytes overflow")?;
        let mut tensor = Self {
            ptr: gpu.alloc(bytes)?,
            shape: vec![rows, first.shape[1]],
            dtype: first.dtype,
        };
        let result = (|| -> Result<()> {
            let mut offset = 0;
            for (part, source) in parts.iter().zip(pointers) {
                let size = part.shape[0] * row_bytes;
                gpu.copy_d2d(source, tensor.ptr.offset(offset), size)?;
                offset += size;
            }
            Ok(())
        })();
        if let Err(error) = result {
            return Err(match tensor.release(gpu) {
                Ok(()) => error,
                Err(cleanup) => error.context(format!("EXL3 concat rollback failed: {cleanup:#}")),
            });
        }
        Ok(tensor)
    }
}
