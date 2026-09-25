// SPDX-License-Identifier: AGPL-3.0-only
//! Exact TP2 vocabulary split for the greedy GLM verify head.
//!
//! Each rank projects only its contiguous half of the BF16 head with the same
//! batched GEMV (per-column arithmetic unchanged), reduces it with
//! `argmax_bf16_value`, and swaps one `(f32, u32)` pair per row with its peer;
//! a device merge applies the full-vocabulary tie rule. The other half of the
//! logits buffer is left stale, so this serves only argmax-consuming verify.
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

use super::types::TransformerModel;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Local pairs, then peer pairs, above the verify argmax words in scratch.
const PAIRS_OFFSET: usize = 16384;
const PAIRS_BYTES: usize = 1024;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_VERIFY_VOCAB_SPLIT").as_deref() == Ok("1"))
}

impl TransformerModel {
    /// Project `normed` [rows, H] and write global argmax IDs to `out` [rows].
    /// Returns `Ok(false)` without launching when the split does not apply.
    pub(super) fn glm_split_head_argmax(
        &self,
        normed: DevicePtr,
        rows: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(false);
        };
        if !enabled()
            || self.config.model_type != "glm5_next"
            || comm.world_size() != 2
            || !comm.supports_peer_exchange_async()
            || self.lm_head_fp8.is_some()
            || self.lm_head_nvfp4.is_some()
            || self.overlays.is_some()
            || self.logit_softcap_kernel.0 != 0
            || self.dense_gemv_batchm_kernel.0 == 0
            || rows == 0
            || rows * 8 > PAIRS_BYTES
        {
            return Ok(false);
        }
        static KERNELS: OnceLock<(KernelHandle, KernelHandle)> = OnceLock::new();
        let (value_k, merge_k) = match KERNELS.get() {
            Some(k) => *k,
            None => *KERNELS.get_or_init(|| {
                (
                    self.gpu.kernel("argmax", "argmax_bf16_value").unwrap_or(KernelHandle(0)),
                    self.gpu.kernel("argmax", "argmax_pair_merge").unwrap_or(KernelHandle(0)),
                )
            }),
        };
        if value_k.0 == 0 || merge_k.0 == 0 {
            return Ok(false);
        }
        let vocab = self.config.vocab_size;
        let h = self.config.hidden_size;
        ensure!(vocab % 2 == 0, "GLM vocab split needs an even vocabulary ({vocab})");
        let shard = vocab / 2;
        let start = comm.rank() * shard;
        let logits = self.buffers.logits();
        let shard_weight = DenseWeight {
            weight: self.lm_head_weight.weight.offset(start * h * 2),
        };
        // batchm is bit-identical per row at every M, so wide verifies chunk.
        let max_m = ops::DENSE_GEMV_BATCHM_MAX_M as usize;
        for first in (0..rows).step_by(max_m) {
            let m = (rows - first).min(max_m);
            ops::dense_gemv_batchm(
                self.gpu.as_ref(),
                self.dense_gemv_batchm_kernel,
                normed.offset(first * h * 2),
                &shard_weight,
                logits.offset((first * vocab + start) * 2),
                m as u32,
                shard as u32,
                h as u32,
                vocab as u32,
                stream,
            )?;
        }
        let local = self.buffers.scratch().offset(PAIRS_OFFSET);
        let peer = local.offset(PAIRS_BYTES);
        for r in 0..rows {
            ops::argmax_bf16_value(
                self.gpu.as_ref(),
                value_k,
                logits.offset((r * vocab + start) * 2),
                local.offset(r * 8),
                shard as u32,
                stream,
            )?;
        }
        comm.peer_exchange_async(local.0, peer.0, rows * 8, stream)?;
        KernelLaunch::new(self.gpu.as_ref(), merge_k)
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(local)
            .arg_ptr(peer)
            .arg_ptr(out)
            .arg_u32(rows as u32)
            .arg_u32(shard as u32)
            .arg_u32(comm.rank() as u32)
            .launch(stream)?;
        Ok(true)
    }
}
