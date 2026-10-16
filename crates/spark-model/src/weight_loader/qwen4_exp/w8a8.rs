// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_W8A8` at load (`layers::ops::qwen4exp_w8a8`): each
//! FP4-Local GDN and QSA layer that carries the MXFP8 decode twins gets its
//! W8A8 prefill operands over the twins' bytes, and the exact BF16 copies of
//! those projections, which nothing reads any more, are freed for the KV
//! cache. Per TP2 rank: 57.7 MB a GDN layer (qkvz, out_proj) and 49.8 MB a
//! QSA layer (q, k, v, o), ~2.67 GB over 36 + 12 layers, against ~1.5 MB of
//! retiled scales. The twins exist only with `ATLAS_QWEN4EXP_MXFP8_DECODE` on
//! (its `=0` decodes, and so prefills, the BF16 copies, which then stay).

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use crate::layer::TransformerLayer;
use crate::layers::ops::{W8a8Swap, mx_prepare_ms, qwen4exp_w8a8};
use crate::layers::{Qwen3AttentionLayer, Qwen3SsmLayer};

/// What the installs did over the model, logged once.
#[derive(Debug, Default)]
pub(super) struct W8a8Load {
    layers: usize,
    allocated: usize,
    freed: usize,
}

impl W8a8Load {
    /// Install W8A8 on `layer` of an MXFP8 checkpoint when it is enabled and
    /// the layer has the twins, and free the BF16 copies it replaces.
    pub(super) fn install(
        &mut self,
        layer: &mut Box<dyn TransformerLayer>,
        mxfp8_backbone: bool,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if !(mxfp8_backbone && qwen4exp_w8a8::enabled()) {
            return Ok(());
        }
        let Some(any) = layer.as_any_mut() else {
            return Ok(());
        };
        let m_max = config.max_batch_tokens;
        let swap = if let Some(l) = any.downcast_mut::<Qwen3SsmLayer>() {
            if !l.has_mxfp8_twins() {
                return Ok(());
            }
            l.install_w8a8_prefill(gpu, m_max, stream)
        } else if let Some(l) = any.downcast_mut::<Qwen3AttentionLayer>() {
            if l.dense_mx.is_empty() {
                return Ok(());
            }
            l.install_w8a8_prefill(gpu, m_max, stream)
        } else {
            return Ok(());
        }
        .context(
            "ATLAS_QWEN4EXP_PREFILL_W8A8 install (set ATLAS_QWEN4EXP_PREFILL_W8A8=0 to prefill \
             in BF16)",
        )?;
        self.free(gpu, swap)
    }

    fn free(&mut self, gpu: &dyn GpuBackend, swap: W8a8Swap) -> Result<()> {
        for (ptr, bytes) in swap.bf16 {
            gpu.free(ptr)?;
            self.freed += bytes;
        }
        self.allocated += swap.allocated;
        self.layers += 1;
        Ok(())
    }

    /// One line: what W8A8 replaced, or why it did not.
    pub(super) fn log(&self, mxfp8_backbone: bool) {
        if !mxfp8_backbone {
            return;
        }
        if self.layers == 0 {
            tracing::info!(
                "ATLAS_QWEN4EXP_PREFILL_W8A8: off ({}); prefill reads the exact BF16 copies",
                if !qwen4exp_w8a8::requested() {
                    "=0"
                } else if !qwen4exp_w8a8::enabled() {
                    "ATLAS_QWEN4EXP_PREFILL_BF16_PROJ prefills at decode precision"
                } else {
                    "no MXFP8 decode twins: ATLAS_QWEN4EXP_MXFP8_DECODE=0"
                }
            );
            return;
        }
        tracing::info!(
            "ATLAS_QWEN4EXP_PREFILL_W8A8: {} GDN/QSA layers prefill MXFP8 W8A8 on the decode \
             twins' bytes; freed {:.2} GB of BF16 copies, {:.1} MB of retiled scales added, \
             GEMM tuning {:.1} s",
            self.layers,
            self.freed as f64 / 1e9,
            self.allocated as f64 / 1e6,
            mx_prepare_ms() / 1e3,
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "w8a8_gpu_tests.rs"]
mod gpu_tests;
