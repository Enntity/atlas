// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter NVFP4: load-time quant + small-M `w4a16_gemv_batch4` dispatch.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{BlockDiffusionDraftHead, DflashQuantization};
use crate::layers::ops;
use crate::weight_loader::dflash_preshrink::take_twin;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight, quantize_to_nvfp4};

impl BlockDiffusionDraftHead {
    pub(super) fn try_install_nvfp4(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if std::env::var("ATLAS_NO_DFLASH_DRAFTER_NVFP4").is_ok() {
            return Ok(());
        }
        if self.kernels.w4a16_gemv_batch4.0 == 0 {
            tracing::warn!("ATLAS_DFLASH_DRAFTER_NVFP4=1 but w4a16_gemv_batch4 missing");
            return Ok(());
        }
        let absmax = match gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax") {
            Ok(k) => k,
            Err(e) => {
                tracing::warn!("NVFP4 absmax kernel missing: {e}");
                return Ok(());
            }
        };
        let quant = match gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4") {
            Ok(k) => k,
            Err(e) => {
                tracing::warn!("NVFP4 quant kernel missing: {e}");
                return Ok(());
            }
        };
        let stream = 0u64;
        let h = self.hidden_size;
        let q_dim = self.num_q_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let inter = self.intermediate_size;
        tracing::info!(
            "DFlash NVFP4: quantizing {} layers × 7 GEMMs for w4a16_gemv_batch4",
            self.layers.len()
        );
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.q_proj_nvfp4 = Some(match take_twin(i, "q_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(
                    &layer.q_proj,
                    q_dim,
                    h,
                    gpu,
                    absmax,
                    quant,
                    stream,
                )?,
            });
            layer.k_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.k_proj,
                kv_dim,
                h,
                gpu,
                absmax,
                quant,
                stream,
            )?);
            layer.v_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.v_proj,
                kv_dim,
                h,
                gpu,
                absmax,
                quant,
                stream,
            )?);
            layer.o_proj_nvfp4 = Some(match take_twin(i, "o_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(
                    &layer.o_proj,
                    h,
                    q_dim,
                    gpu,
                    absmax,
                    quant,
                    stream,
                )?,
            });
            layer.gate_proj_nvfp4 = Some(match take_twin(i, "gate_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(
                    &layer.gate_proj,
                    inter,
                    h,
                    gpu,
                    absmax,
                    quant,
                    stream,
                )?,
            });
            layer.up_proj_nvfp4 = Some(match take_twin(i, "up_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(
                    &layer.up_proj,
                    inter,
                    h,
                    gpu,
                    absmax,
                    quant,
                    stream,
                )?,
            });
            layer.down_proj_nvfp4 = Some(match take_twin(i, "down_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(
                    &layer.down_proj,
                    h,
                    inter,
                    gpu,
                    absmax,
                    quant,
                    stream,
                )?,
            });
        }
        if std::env::var("ATLAS_DFLASH_CTX_NVFP4").as_deref() == Ok("1")
            && let Some(fused_kv) = self.fused_kv_weight
        {
            let fc_in = self.target_layer_ids.len() * self.target_hidden_size;
            let fused_n = self.num_layers * 2 * kv_dim;
            let fused = DenseWeight { weight: fused_kv };
            self.ctx_q4 = Some([
                quantize_to_nvfp4(&self.fc, h, fc_in, gpu, absmax, quant, stream)?,
                quantize_to_nvfp4(&fused, fused_n, h, gpu, absmax, quant, stream)?,
            ]);
        }
        self.quant = DflashQuantization::Nvfp4Weights;
        tracing::info!(
            "DFlash NVFP4: ready (quant = Nvfp4Weights, context twins {})",
            self.ctx_q4.is_some()
        );
        Ok(())
    }

    /// Quantize the five large drafter projections (when `layers`) and the
    /// head to MXFP8 twins (`ATLAS_DFLASH_MXFP8=1`). BF16 originals stay:
    /// they serve rows the tensor-core GEMVs do not (wider than 32).
    pub(super) fn install_mxfp8(&mut self, gpu: &dyn GpuBackend, layers: bool) -> Result<()> {
        let quant = self.kernels.mxfp8_quantize;
        anyhow::ensure!(
            quant.0 != 0 && self.kernels.mxfp8_gemv.iter().all(|k| k.0 != 0),
            "ATLAS_DFLASH_MXFP8=1 but the mxfp8_gemv kernels are missing"
        );
        let h = self.hidden_size;
        let q_dim = self.num_q_heads * self.head_dim;
        let inter = self.intermediate_size;
        let stream = gpu.default_stream();
        let quantize = |w: &DenseWeight, n: usize, k: usize| -> Result<super::Mxfp8Weight> {
            let data = gpu.alloc(n * k)?;
            let scales = gpu.alloc(n * k / ops::MXFP8_BLOCK)?;
            ops::mxfp8_quantize(gpu, quant, w.weight, data, scales, n, k, stream)?;
            Ok(super::Mxfp8Weight { data, scales })
        };
        for layer in self.layers.iter_mut().filter(|_| layers) {
            layer.q_proj_mx = Some(quantize(&layer.q_proj, q_dim, h)?);
            layer.o_proj_mx = Some(quantize(&layer.o_proj, h, q_dim)?);
            layer.gate_proj_mx = Some(quantize(&layer.gate_proj, inter, h)?);
            layer.up_proj_mx = Some(quantize(&layer.up_proj, inter, h)?);
            layer.down_proj_mx = Some(quantize(&layer.down_proj, h, inter)?);
        }
        let head = DenseWeight {
            weight: self.lm_head_shared,
        };
        if std::env::var("ATLAS_DFLASH_NVFP4_HEAD").as_deref() == Ok("1")
            && self.lm_head_nvfp4.is_none()
            && self.lm_head_shared.0 != 0
        {
            self.lm_head_q4 = Some(crate::weight_map::quantize_to_nvfp4(
                &head,
                self.vocab_size,
                h,
                gpu,
                gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
                gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
                stream,
            )?);
        } else if std::env::var("ATLAS_DFLASH_MXFP8_HEAD").as_deref() == Ok("1")
            && self.lm_head_nvfp4.is_none()
            && self.lm_head_shared.0 != 0
        {
            self.lm_head_mx = Some(quantize(&head, self.vocab_size, h)?);
        }
        gpu.synchronize(stream)?;
        tracing::info!(
            "DFlash MXFP8: {} layers x 5 projections, lm_head {} (NVFP4 head {})",
            if layers { self.layers.len() } else { 0 },
            self.lm_head_mx.is_some(),
            self.lm_head_q4.is_some()
        );
        Ok(())
    }

    /// Drafter logits `out[rows, vocab] = input[rows, H] · lm_headᵀ`: the NVFP4
    /// twin on the tensor-core GEMV tier when present (up to 32 rows), else
    /// the MXFP8 twin / BF16 head.
    pub(super) fn project_head(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        out: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let (vocab, h) = (self.vocab_size as u32, self.hidden_size as u32);
        // Wider batches (many owners' blocks) run the tier in 32-row pieces:
        // the NVFP4 head read twice still beats one BF16 head read.
        let pieces = || (0..rows).step_by(32).map(move |r0| (r0, (rows - r0).min(32)));
        let tier = crate::layers::w4a16_gemv_tiers::tc_kernel;
        if let Some(q4) = self.lm_head_q4.as_ref()
            && pieces().all(|(_, n)| tier(n).0 != 0)
        {
            for (r0, n) in pieces() {
                ops::w4a16_gemv_batchm(
                    gpu,
                    tier(n),
                    input.offset(r0 as usize * h as usize * 2),
                    q4,
                    out.offset(r0 as usize * vocab as usize * 2),
                    n,
                    vocab,
                    h,
                    stream,
                )?;
            }
            return Ok(());
        }
        let head = DenseWeight {
            weight: self.lm_head_shared,
        };
        self.kernels
            .project(gpu, input, &head, self.lm_head_mx.as_ref(), out, rows, vocab, h, stream)
    }

    pub(super) fn drafter_gemm(
        &self,
        gpu: &dyn GpuBackend,
        w_bf16: &DenseWeight,
        w_fp8: &Option<Fp8DenseWeight>,
        w_nvfp4: &Option<QuantizedWeight>,
        w_mx: Option<&super::Mxfp8Weight>,
        src: DevicePtr,
        dst: DevicePtr,
        n_out: u32,
        k_in: u32,
        stream: u64,
    ) -> Result<()> {
        let g = self.gamma as u32;
        if matches!(self.quant, DflashQuantization::Nvfp4Weights)
            && let Some(w) = w_nvfp4
        {
            // The target's tensor-core tier serves the whole γ block; the
            // scalar batch4 kernel only blocks of up to four rows.
            let tc = crate::layers::w4a16_gemv_tiers::tc_kernel(g);
            let kernel = if tc.0 != 0 {
                tc
            } else if g <= 4 {
                self.kernels.w4a16_gemv_batch4
            } else {
                spark_runtime::gpu::KernelHandle(0)
            };
            if kernel.0 != 0 {
                return ops::w4a16_gemv_batchm(gpu, kernel, src, w, dst, g, n_out, k_in, stream);
            }
        }
        if matches!(self.quant, DflashQuantization::Fp8Weights)
            && let Some(fp8) = w_fp8
        {
            return ops::fp8_gemm_n128_row_scaled(
                gpu,
                self.kernels.fp8_gemm_n128_row_scaled,
                src,
                fp8,
                dst,
                g,
                n_out,
                k_in,
                stream,
            );
        }
        self.kernels
            .project(gpu, src, w_bf16, w_mx, dst, g, n_out, k_in, stream)
    }
}
