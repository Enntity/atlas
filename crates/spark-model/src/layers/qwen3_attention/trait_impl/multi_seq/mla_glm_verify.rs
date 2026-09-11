// SPDX-License-Identifier: AGPL-3.0-only
//! Eager causal semantic indexing for repaired K3 verification.
use super::*;
use crate::layer::ForwardContext;
use crate::layers::qwen3_attention::glm_long_context;
use anyhow::ensure;
use spark_runtime::kv_cache::{KvCacheDtype, SparseIndexCacheDtype};

impl Qwen3AttentionLayer {
    pub(super) fn validate_glm_long_verify(
        &self,
        c: &MultiSeqCtx<'_>,
        cache: &PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<bool> {
        if !glm_long_context::enabled(&c.fwd.config.model_type) {
            return Ok(false);
        }
        ensure!(
            crate::speculative::glm_repair_policy::enabled()
                && c.n == 3
                && c.seq_lens.len() == 3
                && c.fwd.config.tp_world_size == 2
                && c.fwd.config.ep_world_size == 2
                && !c.fwd.graph_capture
                && !c.fwd.gpu.stream_is_capturing(c.stream),
            "GLM long MTP attention requires repaired eager K3 TP2/EP2"
        );
        let mla = self.mla.as_ref().expect("MLA dispatch owns weights");
        ensure!(
            mla.glm_indexer.is_some()
                && mla.rope == 0
                && mla.o_lora_rank == 0
                && mla.kv_lora_rank == 512
                && mla.q_lora_rank == 1536
                && c.h == 4096
                && c.hd == 256
                && matches!(c.nq, 32 | 64)
                && c.fwd.config.index_topk == 2048
                && c.fwd.config.index_kpool == 4
                && c.fwd.config.index_head_dim == 128
                && c.fwd.config.index_n_heads == 32,
            "GLM long MTP requires indexed NoPE512 checkpoint geometry"
        );
        let index = cache
            .sparse_index_config()
            .ok_or_else(|| anyhow::anyhow!("GLM long MTP needs semantic index storage"))?;
        ensure!(
            self.kv_dtype == KvCacheDtype::Bf16
                && cache.dtype_for_layer(self.attn_layer_idx) == KvCacheDtype::Bf16
                && index.dtype == SparseIndexCacheDtype::Bf16
                && index.head_dim == 128
                && index.tokens_per_pool == 4,
            "GLM long MTP requires BF16 K/V and semantic index"
        );
        ensure!(
            meta.num_seqs == 3
                && !meta.slot.is_null()
                && !meta.seq_len.is_null()
                && !meta.block_table.is_null(),
            "GLM K3 metadata incomplete"
        );
        let shape = ops::GlmDynamicShape::new(meta.max_blocks_per_seq, c.bs)?;
        shape.validate_positions(
            c.seq_lens.iter().copied(),
            3,
            c.fwd.config.max_position_embeddings,
        )?;
        ensure!(
            c.seq_lens
                .windows(2)
                .all(|p| p[0].checked_add(1) == Some(p[1])),
            "GLM K3 positions must be causal and consecutive"
        );
        shape.validate_arenas(
            c.fwd.buffers.sizes().expert_down_out,
            c.fwd.buffers.sizes().qkv_output,
        )?;
        ensure!(
            self.glm_index_layernorm_k.0 != 0
                && self.glm_index_tail_write_k.0 != 0
                && self.glm_index_kpool_finalize_k.0 != 0
                && self.glm_index_logits_decode_k.0 != 0
                && self.glm_index_topk_expand_k.0 != 0
                && self.glm_sparse_attn_decode_k.0 != 0,
            "GLM long MTP semantic kernels unavailable"
        );
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_long_verify_attention(
        &self,
        c: &MultiSeqCtx<'_>,
        cache: &PagedKvCache,
        meta: AttnMetadataDev,
        normed: DevicePtr,
        q_latent: DevicePtr,
        query: DevicePtr,
        output: DevicePtr,
        position: usize,
        d: &MlaDims,
        stream: u64,
    ) -> Result<bool> {
        if !glm_long_context::enabled(&c.fwd.config.model_type) {
            return Ok(false);
        }
        let row_ctx = ForwardContext {
            attn_metadata: Some(meta),
            midchunk_capture: None,
            ..*c.fwd
        };
        // q_latent survives the zero-RoPE chain. Index scratch overwrites only
        // consumed Q expansion / K/V assembly; normed and absorbed Q stay live.
        let selected = self.glm_index_decode_update_and_select(
            normed,
            q_latent,
            position as u32,
            cache,
            &row_ctx,
            stream,
        )?;
        let Some((ids, width)) = selected else {
            return Ok(false);
        };
        ops::glm_sparse_mla_prefill(
            c.fwd.gpu,
            self.glm_sparse_attn_decode_k,
            query,
            cache.k_pool_ptr(self.attn_layer_idx),
            cache.v_pool_ptr(self.attn_layer_idx),
            ids,
            output,
            meta.block_table,
            1,
            d.nq,
            d.mla_cache_dim,
            width,
            c.bs,
            1,
            d.inv_sqrt_d,
            stream,
        )?;
        Ok(true)
    }
}
