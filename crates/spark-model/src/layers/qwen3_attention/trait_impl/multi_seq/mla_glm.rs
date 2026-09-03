// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 zero-RoPE MLA decode for two to five independent rows.
//!
//! The generic absorbed-MLA implementation is deliberately per-sequence to
//! accommodate several architectures. GLM-5 has a simpler fixed shape: no
//! RoPE arm, no output LoRA, and BF16 Q/KV/O projections. This first guarded
//! step batches only the four stateless projections.
//! Absorption, cache mutation, paged attention, and value extraction remain on
//! the byte-for-byte proven per-sequence path.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use super::mla_gemv::MlaDims;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::layers::qwen3_attention::types::MlaWeights;
use crate::weight_map::{DenseWeight, QuantizedWeight};

fn enabled() -> bool {
    std::env::var("ATLAS_GLM_MLA_MULTI_SEQ").ok().as_deref() == Some("1")
}

impl Qwen3AttentionLayer {
    pub(super) fn glm_mla_multi_seq_eligible(&self, c: &MultiSeqCtx<'_>, mla: &MlaWeights) -> bool {
        enabled() && (2..=5).contains(&c.n) && mla.rope == 0 && mla.o_lora_rank == 0
    }

    #[allow(clippy::too_many_arguments)]
    fn glm_mla_project(
        &self,
        c: &MultiSeqCtx<'_>,
        input: DevicePtr,
        nvfp4: Option<&QuantizedWeight>,
        dense: &DenseWeight,
        output: DevicePtr,
        n_out: u32,
        k: u32,
    ) -> Result<()> {
        if let Some(weight) = nvfp4 {
            match c.n {
                2 => ops::w4a16_gemv_batch2(
                    c.fwd.gpu,
                    self.w4a16_gemv_batch2_k,
                    input,
                    weight,
                    output,
                    n_out,
                    k,
                    c.stream,
                ),
                3 => ops::w4a16_gemv_batch3(
                    c.fwd.gpu,
                    self.w4a16_gemv_batch3_k,
                    input,
                    weight,
                    output,
                    n_out,
                    k,
                    c.stream,
                ),
                4 => {
                    let kernel = self.w4a16_batchm.kernel(4);
                    ensure!(
                        kernel.0 != 0,
                        "GLM MLA batch4 projection kernel is unavailable"
                    );
                    ops::w4a16_gemv_batchm(
                        c.fwd.gpu, kernel, input, weight, output, 4, n_out, k, c.stream,
                    )
                }
                5 => {
                    let kernel = self.w4a16_batchm.kernel(5);
                    ensure!(
                        kernel.0 != 0,
                        "GLM MLA batch5 projection kernel is unavailable"
                    );
                    ops::w4a16_gemv_batchm(
                        c.fwd.gpu, kernel, input, weight, output, 5, n_out, k, c.stream,
                    )
                }
                n => anyhow::bail!("GLM MLA multi-sequence projection requires N=2..=5, got {n}"),
            }
        } else {
            let kernel = if c.n == 5 && self.dense_gemv_batch5_k.0 != 0 {
                self.dense_gemv_batch5_k
            } else {
                self.dense_gemv_batchm_k
            };
            ops::dense_gemv_batchm(
                c.fwd.gpu, kernel, input, dense, output, c.n as u32, n_out, k, n_out, c.stream,
            )
        }
    }

    pub(super) fn ms_glm_mla_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        mla: &MlaWeights,
        o_out: DevicePtr,
    ) -> Result<DevicePtr> {
        ensure!(
            self.glm_mla_multi_seq_eligible(c, mla),
            "GLM MLA batched path called for an unsupported MLA shape"
        );

        let gpu = c.fwd.gpu;
        let buffers = c.fwd.buffers;
        let stream = c.stream;
        let rows = c.n as u32;
        let h = c.h as u32;
        let nq = c.nq;
        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let nope = mla.nope as u32;
        let v_dim = mla.v_dim as u32;
        let q_dim = nq * c.hd;
        let cache_dim = kv_lora;
        let bf16 = c.bf16;

        // Q down/up projections and latent normalization, all row-major.
        let q_latent = buffers.ssm_ba();
        self.glm_mla_project(
            c,
            c.normed,
            mla.wq_a_nvfp4.as_ref(),
            &mla.wq_a,
            q_latent,
            q_lora,
            h,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            rows,
            q_lora,
            c.eps,
            stream,
        )?;
        let q_full = buffers.ssm_deinterleaved();
        self.glm_mla_project(
            c,
            q_latent,
            mla.wq_b_nvfp4.as_ref(),
            &mla.wq_b,
            q_full,
            q_dim,
            q_lora,
        )?;

        // KV latent projection and norm share the same row batch.
        let kv_latent = buffers.expert_gate_out();
        self.glm_mla_project(
            c,
            c.normed,
            mla.wkv_a_nvfp4.as_ref(),
            &mla.wkv_a,
            kv_latent,
            kv_lora,
            h,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            rows,
            kv_lora,
            c.eps,
            stream,
        )?;

        let dims = MlaDims {
            h,
            nq,
            hd: c.hd,
            q_dim,
            q_lora,
            kv_lora,
            mla_nope: nope,
            mla_v_dim: v_dim,
            mla_rope: 0,
            mla_cache_dim: cache_dim,
            eps: c.eps,
            bs: c.bs as usize,
            inv_sqrt_d: self.effective_attn_scale(c.hd),
            o_lora_rank: 0,
        };

        // Preserve the established sequence-private absorbed-attention chain.
        // Its transient buffers are consumed before the next row reuses them;
        // only v_extracted keeps one dedicated output row for the final batched
        // O projection.
        let q_absorbed = buffers.expert_up_out();
        let k_entries = buffers.qkv_output();
        let cache_row = cache_dim as usize * bf16;
        let v_entries = k_entries.offset(cache_row);
        let attn_out = buffers.attn_output();
        let v_extracted = buffers.ssm_qkvz();
        let q_full_row = q_dim as usize * bf16;
        let kv_row = kv_lora as usize * bf16;
        let v_row = (nq * v_dim) as usize * bf16;
        for i in 0..c.n {
            let meta_i = AttnMetadataDev {
                positions: meta.positions.offset(i * 4),
                positions_h: meta.positions_h.offset(i * 4),
                positions_w: meta.positions_w.offset(i * 4),
                slot: meta.slot.offset(i * 8),
                seq_len: meta.seq_len.offset(i * 4),
                block_table: meta
                    .block_table
                    .offset(i * meta.max_blocks_per_seq as usize * 4),
                max_blocks_per_seq: meta.max_blocks_per_seq,
                num_seqs: 1,
                seq_slot: DevicePtr::NULL,
                moe_row_adapter: DevicePtr::NULL,
            };

            self.ms_mla_q_absorb(
                c,
                mla,
                &dims,
                q_full.offset(i * q_full_row),
                q_absorbed,
                stream,
            )?;
            ops::mla_cache_assemble(
                gpu,
                self.mla_cache_assemble_k,
                kv_latent.offset(i * kv_row),
                DevicePtr::NULL,
                k_entries,
                v_entries,
                kv_lora,
                0,
                cache_dim,
                stream,
            )?;
            self.write_kv_cache(
                gpu,
                k_entries,
                v_entries,
                kv_cache,
                meta_i.slot,
                1,
                1,
                cache_dim,
                c.bs,
                cache_dim,
                cache_dim,
                stream,
                c.fwd.graph_capture,
            )?;
            ops::paged_decode_attn_bf16(
                gpu,
                self.paged_decode_mla_k,
                q_absorbed,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta_i.block_table,
                meta_i.seq_len,
                meta_i.max_blocks_per_seq,
                1,
                nq,
                1,
                cache_dim,
                c.bs,
                dims.inv_sqrt_d,
                nq * cache_dim,
                0,
                stream,
            )?;
            self.ms_mla_v_extract(
                c,
                mla,
                &dims,
                attn_out,
                v_extracted.offset(i * v_row),
                stream,
            )?;
        }

        // Read the large O matrix once after all rows have been extracted.
        self.glm_mla_project(
            c,
            v_extracted,
            mla.wo_nvfp4.as_ref(),
            &mla.wo,
            o_out,
            h,
            nq * v_dim,
        )?;
        Ok(o_out)
    }
}

#[cfg(test)]
mod tests {
    use super::enabled;

    #[test]
    fn glm_mla_multiseq_is_gated() {
        if std::env::var_os("ATLAS_GLM_MLA_MULTI_SEQ").is_none() {
            assert!(!enabled());
        }
    }
}
