// SPDX-License-Identifier: AGPL-3.0-only

//! Exact dense stepping stone for chunked GLM-5 prefill.
//!
//! GLM-5 selects at most `index_topk` raw history tokens. At or below that
//! length the selection is the complete causal history, so dense attention is
//! exact. Unlike the old generic MLA branch, this path reads every preceding
//! chunk from the compressed paged cache. Longer sequences deliberately fail
//! closed until the native k-pool selector supplies sparse token indices.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use super::paged_mla::MlaPrefillArgs;
use crate::layer::ForwardContext;
use crate::layers::ops;

fn dense_selection_is_exact(sequence_end: usize, index_topk: usize) -> bool {
    index_topk > 0 && sequence_end <= index_topk
}

impl Qwen3AttentionLayer {
    /// Chunked zero-RoPE MLA using the 512-wide absorbed cache.
    pub(super) fn prefill_attention_paged_glm_dense(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
        seq_len_start: usize,
    ) -> Result<DevicePtr> {
        let MlaPrefillArgs {
            normed,
            num_tokens,
            n,
            h,
            nq,
            nkv: _,
            hd,
            kv_dim: _,
            eps,
            bf16,
            bs,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("GLM paged prefill called without MLA weights");
        ensure!(
            mla.glm_indexer.is_some() && mla.rope == 0,
            "GLM paged prefill requires the zero-RoPE semantic-index MLA shape"
        );
        ensure!(
            self.kv_dtype == KvCacheDtype::Bf16,
            "GLM chunked prefill currently requires BF16 KV cache; got {:?}",
            self.kv_dtype
        );
        let sequence_end = seq_len_start
            .checked_add(num_tokens)
            .ok_or_else(|| anyhow::anyhow!("GLM prefill sequence length overflow"))?;
        let use_dense = dense_selection_is_exact(sequence_end, ctx.config.index_topk);

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let nope = mla.nope as u32;
        let v_dim = mla.v_dim as u32;
        ensure!(
            kv_lora == 512 && nope == hd && v_dim == hd,
            "unsupported GLM MLA geometry: kv_lora={kv_lora}, nope={nope}, v={v_dim}, hd={hd}"
        );

        // Q down/up projections. q_full is [N, nq, nope].
        let q_latent = ctx.buffers.ssm_ba();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wq_a,
            q_latent,
            n,
            q_lora,
            h,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            n,
            q_lora,
            eps,
            stream,
        )?;
        self.glm_index_prefill_cache_update(normed, n, kv_cache, ctx, stream)?;
        let sparse_indices = if use_dense {
            None
        } else {
            Some(self.glm_index_prefill_select(
                q_latent,
                normed,
                n,
                seq_len_start,
                kv_cache,
                ctx,
                stream,
            )?)
        };
        let q_full = ctx.buffers.qkv_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            q_latent,
            &mla.wq_b,
            q_full,
            n,
            nq * hd,
            q_lora,
            stream,
        )?;

        // Absorb W_UK into Q. This is the same representation used by Atlas's
        // established GLM decode path: [N, nq, kv_lora].
        let q_absorbed = ctx.buffers.ssm_deinterleaved();
        ops::grouped_gemm_mla(
            ctx.gpu,
            self.grouped_gemm_mla_k,
            q_full,
            mla.w_uk_t.weight,
            q_absorbed,
            n,
            nq,
            nope,
            kv_lora,
            nq * hd,
            nq * kv_lora,
            stream,
        )?;

        // Project the shared KV latent. With zero RoPE both physical cache
        // sides contain this identical vector; retaining the conventional two
        // pools keeps this correctness milestone compatible with the existing
        // paged-attention kernels.
        let kv_latent = ctx.buffers.expert_gate_out();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wkv_a,
            kv_latent,
            n,
            kv_lora,
            h,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            n,
            kv_lora,
            eps,
            stream,
        )?;
        let k_entries = ctx.buffers.ssm_qkvz();
        let v_entries = k_entries.offset(num_tokens * kv_lora as usize * bf16);
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            DevicePtr::NULL,
            k_entries,
            v_entries,
            n,
            kv_lora,
            0,
            kv_lora,
            stream,
        )?;
        let meta = ctx
            .attn_metadata
            .expect("GLM paged prefill requires metadata");
        self.write_kv_cache(
            ctx.gpu,
            k_entries,
            v_entries,
            kv_cache,
            meta.slot,
            n,
            1,
            kv_lora,
            bs,
            kv_lora,
            kv_lora,
            stream,
            ctx.graph_capture,
        )?;

        let attn_latent = ctx.buffers.attn_output();
        if let Some((indices, index_width)) = sparse_indices {
            ops::glm_sparse_mla_prefill(
                ctx.gpu,
                self.glm_sparse_attn_k,
                q_absorbed,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                indices,
                attn_latent,
                meta.block_table,
                n,
                nq,
                kv_lora,
                index_width,
                bs,
                self.effective_attn_scale(hd),
                stream,
            )?;
        } else {
            ensure!(
                self.prefill_attn_paged_512_k.0 != 0,
                "GLM paged prefill kernel inferspark_prefill_paged_512 is unavailable"
            );
            ops::prefill_attention_paged_512(
                ctx.gpu,
                self.prefill_attn_paged_512_k,
                q_absorbed,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_latent,
                meta.block_table,
                n,
                sequence_end as u32,
                seq_len_start as u32,
                nq,
                1,
                kv_lora,
                bs,
                0,
                self.effective_attn_scale(hd),
                stream,
            )?;
        }

        // Convert the latent attention result back to each head's value
        // width, then apply the row-parallel output projection.
        let v_extracted = ctx.buffers.qkv_output();
        ops::grouped_gemm_mla(
            ctx.gpu,
            self.grouped_gemm_mla_k,
            attn_latent,
            mla.w_uv.weight,
            v_extracted,
            n,
            nq,
            kv_lora,
            v_dim,
            nq * kv_lora,
            nq * v_dim,
            stream,
        )?;
        let o_out = ctx.buffers.norm_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            v_extracted,
            &mla.wo,
            o_out,
            n,
            h,
            nq * v_dim,
            stream,
        )?;
        Ok(o_out)
    }
}

#[cfg(test)]
mod tests {
    use super::dense_selection_is_exact;

    #[test]
    fn dense_reference_stops_at_the_semantic_topk_boundary() {
        assert!(dense_selection_is_exact(2048, 2048));
        assert!(!dense_selection_is_exact(2049, 2048));
        assert!(!dense_selection_is_exact(1, 0));
    }
}
