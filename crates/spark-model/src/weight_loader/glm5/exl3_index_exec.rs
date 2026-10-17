// SPDX-License-Identifier: AGPL-3.0-only

//! Typed native index quantization dispatch and bounded per-layer scale scratch.

use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub struct Glm5Exl3IndexExec {
    capacity: u32,
    heads: u32,
    hidden: u32,
    stream: u64,
    scales: DevicePtr,
    query_quant: KernelHandle,
    weight_gemv: KernelHandle,
    weight_scale: KernelHandle,
    pub norm: KernelHandle,
    pub finalize: KernelHandle,
    pub logits: KernelHandle,
    pub logits_dynamic: KernelHandle,
}

impl Glm5Exl3IndexExec {
    pub fn new(
        gpu: &dyn GpuBackend,
        capacity: usize,
        heads: usize,
        hidden: usize,
        stream: u64,
    ) -> Result<Self> {
        ensure!(
            capacity > 0 && heads > 0 && hidden > 0,
            "native index dimensions must be explicit and positive"
        );
        let bytes = capacity
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(4))
            .context("native index scale allocation overflow")?;
        let capacity = u32::try_from(capacity)?;
        let heads = u32::try_from(heads)?;
        let hidden = u32::try_from(hidden)?;
        ensure!(
            capacity.checked_mul(heads).is_some(),
            "native index flattened query rows overflow"
        );
        let kernel = |name| gpu.kernel("glm_indexer_quant", name);
        let query_quant = kernel("glm_index_query_quant_bf16")?;
        let weight_gemv = kernel("glm_index_weight_gemv_f32")?;
        let weight_scale = kernel("glm_index_weight_scale_f32")?;
        let norm = kernel("glm_index_layernorm_f32_params_bf16")?;
        let finalize = kernel("glm_index_kpool_finalize_quant_bf16")?;
        let logits = kernel("glm_index_logits_quant_bf16")?;
        let logits_dynamic = kernel("glm_index_logits_quant_bf16_dynamic")?;
        Ok(Self {
            capacity,
            heads,
            hidden,
            stream,
            scales: gpu.alloc(bytes)?,
            query_quant,
            weight_gemv,
            weight_scale,
            norm,
            finalize,
            logits,
            logits_dynamic,
        })
    }

    pub fn validate(
        &self,
        rows: usize,
        heads: usize,
        dim: usize,
        hidden: usize,
        weight_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(!self.scales.is_null(), "native index scratch released");
        ensure!(
            rows > 0
                && rows <= self.capacity as usize
                && heads == self.heads as usize
                && dim == 128
                && hidden == self.hidden as usize
                && stream == self.stream,
            "native index query shape/capacity/stream mismatch"
        );
        let needed = rows
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(4))
            .context("native index weight arena overflow")?;
        ensure!(
            weight_bytes >= needed,
            "native F32 index head weights exceed arena capacity"
        );
        Ok(())
    }

    /// Query input/output alias intentionally: H128 consumes each row before stores.
    pub fn query_and_weights(
        &self,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        matrix: DevicePtr,
        query: DevicePtr,
        weights: DevicePtr,
        rows: u32,
    ) -> Result<()> {
        self.validate(
            rows as usize,
            self.heads as usize,
            128,
            self.hidden as usize,
            rows as usize * self.heads as usize * 4,
            self.stream,
        )?;
        ensure!(
            !normed.is_null() && !matrix.is_null() && !query.is_null() && !weights.is_null(),
            "native index query dispatch requires live inputs and outputs"
        );
        KernelLaunch::new(gpu, self.weight_gemv)
            .grid([self.heads.div_ceil(128), rows, 1])
            .block([128, 1, 1])
            .arg_ptr(normed)
            .arg_ptr(matrix)
            .arg_ptr(weights)
            .arg_u32(rows)
            .arg_u32(self.heads)
            .arg_u32(self.hidden)
            .launch(self.stream)?;
        KernelLaunch::new(gpu, self.query_quant)
            .grid([rows * self.heads, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(query)
            .arg_ptr(query)
            .arg_ptr(self.scales)
            .arg_u32(rows * self.heads)
            .launch(self.stream)?;
        let count = u64::from(rows) * u64::from(self.heads);
        // Match upstream's separately evaluated head and dimension normalization.
        let scale = ((128.0f64).powf(-0.5) * (self.heads as f64).powf(-0.5)) as f32;
        KernelLaunch::new(gpu, self.weight_scale)
            .grid([(count as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(weights)
            .arg_ptr(self.scales)
            .arg_u64(count)
            .arg_f32(scale)
            .launch(self.stream)
    }
}

impl<'a> ModelResource<dyn GpuBackend + 'a> for Glm5Exl3IndexExec {
    fn label(&self) -> &'static str {
        "GLM native index quantization scratch"
    }
    fn release(&mut self, gpu: &(dyn GpuBackend + 'a)) -> Result<()> {
        if !self.scales.is_null() {
            gpu.free(self.scales)?;
            self.scales = DevicePtr::NULL;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "exl3_index_exec_tests.rs"]
mod tests;
