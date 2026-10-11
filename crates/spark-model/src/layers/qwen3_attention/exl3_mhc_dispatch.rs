// SPDX-License-Identifier: AGPL-3.0-only

//! Select native EXL3 residual arithmetic at the existing mHC boundaries.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{HcWeights, Qwen3AttentionLayer};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_post_site(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        hc: &HcWeights,
        block_output: DevicePtr,
        residual: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        output: DevicePtr,
        rows: u32,
        hidden: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(native) = &self.exl3_mla {
            ensure!(
                hc.hc_mult == 4 && hc.attn.lowrank.is_none() && hc.ffn.lowrank.is_none(),
                "native EXL3 post requires full four-stream mixing"
            );
            if let Some(trace) = &native.trace {
                trace.post(gpu, block_output, rows as usize, stream)?;
            }
            return native
                .mhc
                .as_ref()
                .context("native EXL3 mHC kernel owner absent")?
                .post(
                    gpu,
                    block_output,
                    residual,
                    post,
                    comb,
                    output,
                    rows as usize,
                    stream,
                );
        }
        ops::hc_post_site(
            gpu,
            kernel,
            hc,
            block_output,
            residual,
            post,
            comb,
            output,
            rows,
            hidden,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_contract(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        residual: DevicePtr,
        output: DevicePtr,
        rows: u32,
        hidden: u32,
        streams: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(native) = &self.exl3_mla {
            ensure!(
                streams == 4,
                "native EXL3 contraction requires four streams"
            );
            return native
                .mhc
                .as_ref()
                .context("native EXL3 mHC kernel owner absent")?
                .contract(gpu, residual, output, rows as usize, stream);
        }
        ops::hc_contract(gpu, kernel, residual, output, rows, hidden, streams, stream)
    }
}
