// SPDX-License-Identifier: AGPL-3.0-only
//! Optional target-only BF16 prefill cache; packed verification weights remain.
use super::*;
use anyhow::ensure;
use atlas_core::config::ModelConfig;
impl DenseFfnLayer {
    /// Called only by the deferred target factory pass, before arena/KV sizing.
    pub(crate) fn cache_glm_prefill_bf16(
        &mut self,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        crate::factory::glm_dense_cache::validate(config)?;
        ensure!(
            self.prefill_bf16_weights.is_none()
                && self.bf16_weights.is_none()
                && self.fp8_weights.is_none()
                && self.q2_weights.is_none()
                && self.lora.is_none(),
            "dense BF16 prefill cache conflicts with existing weights/adapters"
        );
        ensure!(
            self.dequant_nvfp4_bf16_k.0 != 0,
            "dense BF16 dequant kernel missing"
        );
        let sources = [
            &self.weights.gate_proj,
            &self.weights.up_proj,
            &self.weights.down_proj,
        ];
        for w in sources {
            ensure!(
                !w.weight.is_null()
                    && !w.weight_scale.is_null()
                    && !w.has_per_row_scale2()
                    && w.weight_scale_2.is_finite()
                    && w.weight_scale_2 > 0.,
                "invalid dense BF16 source weight/scales"
            );
        }
        let mut ptrs = Vec::with_capacity(3);
        let stream = gpu.default_stream();
        let result = (|| -> Result<()> {
            for (w, (n, k)) in
                sources
                    .into_iter()
                    .zip([(12288, 4096), (12288, 4096), (4096, 12288)])
            {
                let ptr = gpu.alloc(n as usize * k as usize * 2)?;
                ptrs.push(ptr);
                ops::dequant_nvfp4_to_bf16(
                    gpu,
                    self.dequant_nvfp4_bf16_k,
                    w.weight,
                    w.weight_scale,
                    ptr,
                    w.weight_scale_2,
                    n,
                    k,
                    stream,
                )?;
            }
            gpu.synchronize(stream)?;
            Ok(())
        })();
        if let Err(error) = result {
            // Synchronize before releasing buffers referenced by queued work.
            let _ = gpu.synchronize(stream);
            for ptr in ptrs {
                let _ = gpu.free(ptr);
            }
            return Err(error);
        }
        self.prefill_bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: DenseWeight { weight: ptrs[0] },
            up_proj: DenseWeight { weight: ptrs[1] },
            down_proj: DenseWeight { weight: ptrs[2] },
        });
        Ok(())
    }

    pub(super) fn try_glm_prefill_bf16(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(w) = &self.prefill_bf16_weights else {
            return Ok(false);
        };
        // All existing native speculative verification widths retain W4A16,
        // including fallbacks that happen to enter forward_prefill_inner.
        if rows <= 8 {
            return Ok(false);
        }
        crate::factory::glm_dense_cache::validate(ctx.config)?;
        ensure!(
            ctx.dispatch.cublas_gemm,
            "dense BF16 prefill cache requires cuBLAS GEMM"
        );
        ensure!(
            self.activation == FfnActivation::SiLU && self.lora.is_none(),
            "dense BF16 prefill supports unadapted SiLU only"
        );
        let m = u32::try_from(rows)?;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let gate = ctx.buffers.expert_gate_out();
        let up = ctx.buffers.expert_up_out();
        ops::cublas_bf16_proj_dense(input, w.gate_proj.weight, gate, m, inter, h, stream)?;
        ops::cublas_bf16_proj_dense(input, w.up_proj.weight, up, m, inter, h, stream)?;
        ops::silu_mul(ctx.gpu, self.act_mul, gate, up, gate, m * inter, stream)?;
        ops::cublas_bf16_proj_dense(
            gate,
            w.down_proj.weight,
            ctx.buffers.moe_output(),
            m,
            h,
            inter,
            stream,
        )?;
        Ok(true)
    }
}
#[cfg(test)]
#[path = "dense_ffn_prefill_bf16_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "dense_ffn_prefill_bf16_native.rs"]
mod native;
