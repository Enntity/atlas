// SPDX-License-Identifier: AGPL-3.0-only

//! Exact F32 companions for the native GLM indexer; never legacy BF16 weights.

use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::exl3::dense::{
    Exl3CheckpointDtype as Dtype, Exl3DenseDeviceTensor, Exl3DenseDtype,
};
use spark_runtime::weights::{WeightDtype, WeightStore};

/// Typed pointers consumed only by F32-parameter native indexer kernels.
#[derive(Clone, Copy)]
pub struct Glm5Exl3IndexF32Params {
    pub ape: DevicePtr,
    pub norm_weight: DevicePtr,
    pub norm_bias: DevicePtr,
}

/// Norms own their original F16-to-F32 uploads; APE borrows the WeightStore.
pub struct Glm5Exl3IndexParams {
    ape: DevicePtr,
    norm_weight: Exl3DenseDeviceTensor,
    norm_bias: Exl3DenseDeviceTensor,
}

impl Glm5Exl3IndexParams {
    #[allow(clippy::single_range_in_vec_init)]
    pub fn load(
        store: &WeightStore,
        prefix: &str,
        head_dim: usize,
        pool_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            head_dim == 128 && pool_size > 0,
            "native EXL3 index parameters require H128 and an explicit pool"
        );
        let checkpoint = store
            .exl3_checkpoint()
            .context("native index parameters require EXL3 checkpoint")?;
        let ape_name = format!("{prefix}.index_kpool_compress_ape");
        let ape_view = checkpoint.dense_view(&ape_name, Dtype::F32, &[pool_size, head_dim])?;
        ensure!(
            finite(ape_view.data(), Dtype::F32),
            "native index APE must be finite"
        );
        let ape = store.get(&ape_name)?;
        ensure!(
            ape.dtype == WeightDtype::FP32
                && ape.shape == [pool_size, head_dim]
                && !ape.ptr.is_null(),
            "native index APE requires a live F32 WeightStore owner with exact shape"
        );
        let names = [
            format!("{prefix}.k_norm.weight"),
            format!("{prefix}.k_norm.bias"),
        ];
        let mut views = Vec::with_capacity(2);
        for name in &names {
            let dtype = checkpoint.tensor_dtype(name)?;
            ensure!(
                matches!(dtype, Dtype::F16 | Dtype::F32),
                "native index norm {name} requires original F16 or F32"
            );
            let view = checkpoint.dense_view(name, dtype, &[head_dim])?;
            ensure!(
                finite(view.data(), dtype),
                "native index norm {name} must be finite"
            );
            views.push(view);
        }
        let mut norm_weight =
            Exl3DenseDeviceTensor::upload(gpu, &views[0], Exl3DenseDtype::F32, &[0..head_dim])?;
        let norm_bias = match Exl3DenseDeviceTensor::upload(
            gpu,
            &views[1],
            Exl3DenseDtype::F32,
            &[0..head_dim],
        ) {
            Ok(tensor) => tensor,
            Err(error) => {
                return Err(match norm_weight.release(gpu) {
                    Ok(()) => error,
                    Err(cleanup) => {
                        error.context(format!("native index norm cleanup failed: {cleanup:#}"))
                    }
                });
            }
        };
        Ok(Self {
            ape: ape.ptr,
            norm_weight,
            norm_bias,
        })
    }

    pub fn f32_params(&self) -> Result<Glm5Exl3IndexF32Params> {
        Ok(Glm5Exl3IndexF32Params {
            ape: self.ape,
            norm_weight: self.norm_weight.ptr()?,
            norm_bias: self.norm_bias.ptr()?,
        })
    }
}

fn finite(bytes: &[u8], dtype: Dtype) -> bool {
    match dtype {
        Dtype::F16 => bytes
            .chunks_exact(2)
            .all(|b| half::f16::from_le_bytes([b[0], b[1]]).is_finite()),
        Dtype::F32 => bytes
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite()),
        _ => false,
    }
}

/// Ordinary index projections keep the checkpoint F16 -> model BF16 contract.
pub(crate) fn validate_bf16_index_matrix(
    store: &WeightStore,
    name: &str,
    shape: &[usize],
) -> Result<()> {
    let checkpoint = store
        .exl3_checkpoint()
        .context("native index matrix requires EXL3 checkpoint")?;
    checkpoint.dense_view(name, Dtype::F16, shape)?;
    let matrix = store.get(name)?;
    ensure!(
        matrix.dtype == WeightDtype::BF16 && matrix.shape == shape && !matrix.ptr.is_null(),
        "native index matrix {name} requires a live BF16 owner with exact shape"
    );
    Ok(())
}

impl<'a> ModelResource<dyn GpuBackend + 'a> for Glm5Exl3IndexParams {
    fn label(&self) -> &'static str {
        "GLM native EXL3 F32 index parameters"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        let weight = self.norm_weight.release(gpu);
        let bias = self.norm_bias.release(gpu);
        weight.and(bias)
    }
}

#[cfg(test)]
#[path = "exl3_index_tests.rs"]
mod tests;
