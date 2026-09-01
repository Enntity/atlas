// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    pub(in crate::model::trait_impl) fn copy_logits_to_host_dispatch(
        &self,
        logits_ptr: DevicePtr,
        dst: &mut [u8],
    ) -> Result<()> {
        self.gpu.copy_d2h(logits_ptr, dst)
    }

    pub(in crate::model::trait_impl) fn logits_ptr_is_fp32_dispatch(
        &self,
        logits_ptr: DevicePtr,
    ) -> bool {
        self.use_fp32_logits && logits_ptr.0 == self.logits_fp32_buf.0
    }

    pub(in crate::model::trait_impl) fn logits_buffer_ptr_dispatch(&self) -> DevicePtr {
        self.buffers.logits()
    }

    pub(in crate::model::trait_impl) fn argmax_on_device_dispatch(
        &self,
        logits_ptr: DevicePtr,
        _stream: u64,
    ) -> Result<u32> {
        let stream = self.gpu.default_stream();
        let out_ptr = self.buffers.scratch();
        let kernel = if self.use_fp32_logits && logits_ptr.0 == self.logits_fp32_buf.0 {
            self.argmax_logits_kernel
        } else {
            self.argmax_kernel
        };
        ops::argmax_bf16(
            self.gpu.as_ref(),
            kernel,
            logits_ptr,
            out_ptr,
            self.config.vocab_size as u32,
            stream,
        )?;
        let mut buffer = [0u8; 4];
        self.gpu.copy_d2h(out_ptr, &mut buffer)?;
        Ok(u32::from_le_bytes(buffer))
    }

    pub(in crate::model::trait_impl) fn argmax_batch_dispatch(
        &self,
        logits_ptr: DevicePtr,
        n: usize,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let stream = self.gpu.default_stream();
        let vocab = self.config.vocab_size;
        let out_ptr = self.buffers.scratch();
        fn batch_enabled() -> bool {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ENABLED
                .get_or_init(|| std::env::var("ATLAS_NO_ARGMAX_BATCH").ok().as_deref() != Some("1"))
        }
        if self.argmax_batch_kernel.0 != 0 && batch_enabled() {
            ops::argmax_bf16_batch(
                self.gpu.as_ref(),
                self.argmax_batch_kernel,
                logits_ptr,
                out_ptr,
                vocab as u32,
                n as u32,
                vocab as u32,
                stream,
            )?;
        } else {
            for index in 0..n {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_ptr.offset(index * vocab * 2),
                    out_ptr.offset(index * 4),
                    vocab as u32,
                    stream,
                )?;
            }
        }
        let mut buffer = vec![0u8; n * 4];
        self.gpu.copy_d2h(out_ptr, &mut buffer)?;
        Ok(buffer
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect())
    }

    pub(in crate::model::trait_impl) fn hidden_after_norm_dispatch(&self) -> DevicePtr {
        self.buffers.norm_output()
    }
}
