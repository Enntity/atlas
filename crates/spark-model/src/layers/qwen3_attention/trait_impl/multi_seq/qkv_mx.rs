// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence / verify Q, K, V on a checkpoint's MXFP8 twins
//! (`qwen3_attention/dense_mx.rs`): one launch per projection for every row,
//! written straight into each row's `qkv_buf` slot. The tiers are
//! row-invariant, so row `i` is serial decode's bytes (`attention_forward`
//! runs the same twin at one row) whatever the width.

use anyhow::{Result, ensure};

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// The multi-sequence / verify Q, K and V of every row through the MXFP8
    /// twins, straight into `qkv_buf`'s per-row slots, then the gated Q's
    /// deinterleave (deferred past the q LoRA fold when an adapter is
    /// resident, as `ms_qkv_seq_rows` does). `Ok(false)` without twins.
    pub(super) fn ms_qkv_mx(&self, c: &MultiSeqCtx<'_>) -> Result<bool> {
        if self.dense_mx.is_empty() {
            return Ok(false);
        }
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let (stride, kv) = (per_seq_qkv / bf16, nkv * hd);
        let q_n = if self.gated { q_proj_dim } else { q_dim };
        let k_off = q_proj_bytes;
        let v_off = k_off + kv as usize * bf16;
        for (w, out, rows_n) in [
            (&self.attn.q_proj, qkv_buf, q_n),
            (&self.attn.k_proj, qkv_buf.offset(k_off), kv),
            (&self.attn.v_proj, qkv_buf.offset(v_off), kv),
        ] {
            let ran = self.mx_rows(fwd.gpu, normed, w, out, n, rows_n, h as u32, stride, stream)?;
            ensure!(
                ran,
                "MXFP8 attention twins installed without one for every Q/K/V"
            );
        }
        if self.gated && !self.q_lora_active() {
            ops::deinterleave_qg(
                fwd.gpu,
                self.deinterleave_qg_k,
                qkv_buf,
                n as u32,
                nq,
                hd,
                stride as u32,
                stream,
            )?;
        }
        Ok(true)
    }
}
