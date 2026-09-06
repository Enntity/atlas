// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 semantic-index maintenance and selection for single-token decode.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache, SparseIndexCacheDtype};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Append this token to the paged four-token index staging tail, finalize a
    /// pool on phase three, and return sparse token IDs once history exceeds
    /// the checkpoint's exact-attention threshold.
    pub(in crate::layers::qwen3_attention) fn glm_index_decode_update_and_select(
        &self,
        normed: DevicePtr,
        q_latent: DevicePtr,
        pos: u32,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, u32)>> {
        let mla = self.mla.as_ref().expect("GLM decode index without MLA");
        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("GLM decode index without checkpoint weights");
        let spec = kv_cache
            .sparse_index_config()
            .ok_or_else(|| anyhow::anyhow!("GLM semantic-index cache is not attached"))?;
        ensure!(
            spec.dtype == SparseIndexCacheDtype::Bf16
                && kv_cache.dtype_for_layer(self.attn_layer_idx) == KvCacheDtype::Bf16,
            "GLM sparse decode correctness path currently requires BF16 index and KV caches"
        );
        ensure!(
            self.glm_index_layernorm_k.0 != 0
                && self.glm_index_tail_write_k.0 != 0
                && self.glm_index_kpool_finalize_k.0 != 0,
            "GLM decode index-maintenance kernels are unavailable"
        );

        let dim = spec.head_dim as u32;
        let h = ctx.config.hidden_size as u32;
        let keys = ctx.buffers.ssm_qkvz();
        let gates = keys.offset(spec.head_dim * 2);
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &indexer.wk,
            keys,
            dim,
            h,
            stream,
        )?;
        ops::glm_index_layernorm(
            ctx.gpu,
            self.glm_index_layernorm_k,
            keys,
            indexer.k_norm_weight.weight,
            indexer.k_norm_bias.weight,
            1,
            dim,
            1e-6,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &indexer.kpool_gate,
            gates,
            dim,
            h,
            stream,
        )?;
        let meta = ctx
            .attn_metadata
            .expect("GLM decode index requires slot metadata");
        ops::glm_index_tail_write(
            ctx.gpu,
            self.glm_index_tail_write_k,
            keys,
            gates,
            kv_cache.sparse_index_tail_pool_ptr(self.attn_layer_idx),
            meta.slot,
            1,
            kv_cache.block_size() as u32,
            spec.tokens_per_pool as u32,
            dim,
            kv_cache.sparse_index_tail_block_stride_bytes(self.attn_layer_idx) as u64,
            stream,
        )?;
        ops::glm_index_kpool_finalize(
            ctx.gpu,
            self.glm_index_kpool_finalize_k,
            kv_cache.sparse_index_tail_pool_ptr(self.attn_layer_idx),
            indexer.kpool_ape.weight,
            kv_cache.sparse_index_pool_ptr(self.attn_layer_idx),
            meta.slot,
            1,
            kv_cache.block_size() as u32,
            spec.tokens_per_pool as u32,
            dim,
            kv_cache.sparse_index_tail_block_stride_bytes(self.attn_layer_idx) as u64,
            kv_cache.sparse_index_block_stride_bytes(self.attn_layer_idx) as u64,
            stream,
        )?;

        let seq_len = pos + 1;
        let topk = ctx.config.index_topk as u32;
        if seq_len <= topk {
            return Ok(None);
        }
        ensure!(
            self.glm_index_logits_k.0 != 0 && self.glm_index_topk_expand_k.0 != 0,
            "GLM sparse decode selection kernels are unavailable"
        );
        let index_heads = ctx.config.index_n_heads as u32;
        let index_dim = ctx.config.index_head_dim as u32;
        let pool_size = spec.tokens_per_pool as u32;
        let logits_stride = seq_len.div_ceil(pool_size);
        let output_width = topk + pool_size - 1;
        let query = ctx.buffers.ssm_deinterleaved();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            q_latent,
            &indexer.wq_b,
            query,
            index_heads * index_dim,
            mla.q_lora_rank as u32,
            stream,
        )?;
        let weights = ctx.buffers.ssm_gates();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &indexer.weights_proj,
            weights,
            index_heads,
            h,
            stream,
        )?;
        let logits = ctx.buffers.expert_down_out();
        ops::glm_index_logits(
            ctx.gpu,
            self.glm_index_logits_decode_k,
            query,
            weights,
            kv_cache.sparse_index_pool_ptr(self.attn_layer_idx),
            logits,
            meta.block_table,
            1,
            pos,
            logits_stride,
            index_heads,
            index_dim,
            pool_size,
            kv_cache.block_size() as u32,
            kv_cache.sparse_index_block_stride_bytes(self.attn_layer_idx) as u64,
            1,
            8,
            stream,
        )?;
        // Cache writeback has completed in stream order, so the QKV arena can
        // safely hold the compact token IDs until attention consumes them.
        let selected = ctx.buffers.qkv_output();
        ops::glm_index_topk_expand(
            ctx.gpu,
            self.glm_index_topk_expand_k,
            logits,
            selected,
            1,
            pos,
            logits_stride,
            topk,
            pool_size,
            output_width,
            stream,
        )?;
        Ok(Some((selected, output_width)))
    }
}
