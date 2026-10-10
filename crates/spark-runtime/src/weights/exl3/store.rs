// SPDX-License-Identifier: AGPL-3.0-only

//! Native checkpoint attachment and ordinary companion upload I/O boundary.
use super::{
    Exl3CheckpointRouter,
    dense::{Exl3DenseDeviceTensor, Exl3DenseDtype},
};
use crate::{
    gpu::GpuBackend,
    weights::{WeightDtype, WeightStore, WeightTensor},
};
use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;

impl WeightStore {
    pub fn exl3_checkpoint(&self) -> Option<&Exl3CheckpointRouter> {
        self.exl3_checkpoint.as_ref()
    }

    /// Upload only explicitly selected ordinary companions. Packed EXL3 groups
    /// stay mmap-backed until their rank-local projection owners upload them.
    pub fn from_exl3(
        checkpoint: Exl3CheckpointRouter,
        selected: &[(String, Exl3DenseDtype)],
        gpu: &dyn GpuBackend,
        reserve_bytes: usize,
    ) -> Result<Self> {
        let mut store = Self::empty();
        let result = (|| -> Result<()> {
            for (name, dtype) in selected {
                ensure!(!store.contains(name), "EXL3 duplicate companion {name}");
                let view = checkpoint.tensor_view(name)?;
                let elements = view.shape().iter().try_fold(1usize, |n, &d| {
                    n.checked_mul(d).context("EXL3 companion shape overflow")
                })?;
                let bytes = elements
                    .checked_mul(match dtype {
                        Exl3DenseDtype::Bf16 => 2,
                        Exl3DenseDtype::F32 => 4,
                    })
                    .context("EXL3 companion byte count overflow")?;
                ensure!(
                    gpu.free_memory()?
                        >= bytes
                            .checked_add(reserve_bytes)
                            .context("EXL3 reserve overflow")?,
                    "EXL3 insufficient device memory for companion {name}"
                );
                let rows = *view
                    .shape()
                    .first()
                    .context("EXL3 scalar companion requires explicit layer loader")?;
                let selection = 0..rows;
                let tensor = Exl3DenseDeviceTensor::upload(
                    gpu,
                    &view,
                    *dtype,
                    std::slice::from_ref(&selection),
                )?;
                store.weights.insert(
                    name.clone(),
                    WeightTensor {
                        ptr: tensor.ptr()?,
                        shape: tensor.shape().to_vec(),
                        dtype: match dtype {
                            Exl3DenseDtype::Bf16 => WeightDtype::BF16,
                            Exl3DenseDtype::F32 => WeightDtype::FP32,
                        },
                    },
                );
                // WeightStore now owns the allocation; this value has no Drop.
            }
            checkpoint.evict_cached_pages()?;
            Ok(())
        })();
        if let Err(error) = result {
            return Err(match store.release(gpu) {
                Ok(()) => error,
                Err(cleanup) => error.context(format!("EXL3 companion cleanup: {cleanup:#}")),
            });
        }
        store.exl3_checkpoint = Some(checkpoint);
        Ok(store)
    }

    /// Register auxiliary allocations transferred out of a native layer loader.
    pub fn adopt_exl3_companions(&mut self, tensors: Vec<Exl3DenseDeviceTensor>) {
        self.exl3_companions.extend(tensors);
    }
}
