// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 semantic-index cache population shared by first and later chunks.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{PagedKvCache, SparseIndexCacheDtype};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

fn profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_GLM_INDEX_PROFILE").ok().as_deref() == Some("1"))
}

pub(super) fn profile_start(
    ctx: &ForwardContext,
    stream: u64,
) -> Result<Option<std::time::Instant>> {
    if !profile_enabled() {
        return Ok(None);
    }
    ctx.gpu.synchronize(stream)?;
    Ok(Some(std::time::Instant::now()))
}

pub(super) fn profile_lap(
    ctx: &ForwardContext,
    stream: u64,
    timer: &mut Option<std::time::Instant>,
) -> Result<u128> {
    let Some(started) = timer.take() else {
        return Ok(0);
    };
    ctx.gpu.synchronize(stream)?;
    let elapsed = started.elapsed().as_micros();
    *timer = Some(std::time::Instant::now());
    Ok(elapsed)
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_index_prefill_cache_update(
        &self,
        normed: spark_runtime::gpu::DevicePtr,
        n: u32,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mla = self
            .mla
            .as_ref()
            .expect("GLM index cache update called without MLA weights");
        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("GLM index cache update called without indexer weights");
        let spec = kv_cache
            .sparse_index_config()
            .ok_or_else(|| anyhow::anyhow!("GLM semantic-index cache is not attached"))?;
        ensure!(
            spec.dtype == SparseIndexCacheDtype::Bf16
                && spec.tokens_per_pool == ctx.config.index_kpool
                && spec.head_dim == ctx.config.index_head_dim,
            "GLM semantic-index cache geometry does not match the checkpoint"
        );
        ensure!(
            self.glm_index_layernorm_k.0 != 0
                && self.glm_index_tail_write_k.0 != 0
                && self.glm_index_kpool_finalize_k.0 != 0,
            "GLM semantic-index kernels are unavailable"
        );

        let mut profile = profile_start(ctx, stream)?;

        let rows = n;
        let h = ctx.config.hidden_size as u32;
        let dim = spec.head_dim as u32;
        let keys = ctx.buffers.ssm_qkvz();
        let gates = keys.offset(n as usize * spec.head_dim * 2);
        self.mla_prefill_dense(normed, &indexer.wk, keys, rows, dim, h, ctx, stream)?;
        ops::glm_index_layernorm(
            ctx.gpu,
            self.glm_index_layernorm_k,
            keys,
            indexer.k_norm_weight.weight,
            indexer.k_norm_bias.weight,
            rows,
            dim,
            1e-6,
            stream,
        )?;
        let key_projection_us = profile_lap(ctx, stream, &mut profile)?;
        self.mla_prefill_dense(
            normed,
            &indexer.kpool_gate,
            gates,
            rows,
            dim,
            h,
            ctx,
            stream,
        )?;
        let gate_projection_us = profile_lap(ctx, stream, &mut profile)?;
        let meta = ctx
            .attn_metadata
            .expect("GLM index cache update requires slot metadata");
        ops::glm_index_tail_write(
            ctx.gpu,
            self.glm_index_tail_write_k,
            keys,
            gates,
            kv_cache.sparse_index_tail_pool_ptr(self.attn_layer_idx),
            kv_cache.sparse_index_tail_map_ptr(),
            meta.slot,
            rows,
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
            kv_cache.sparse_index_tail_map_ptr(),
            indexer.kpool_ape.weight,
            kv_cache.sparse_index_pool_ptr(self.attn_layer_idx),
            meta.slot,
            rows,
            kv_cache.block_size() as u32,
            spec.tokens_per_pool as u32,
            dim,
            kv_cache.sparse_index_tail_block_stride_bytes(self.attn_layer_idx) as u64,
            kv_cache.sparse_index_block_stride_bytes(self.attn_layer_idx) as u64,
            stream,
        )?;
        let cache_write_us = profile_lap(ctx, stream, &mut profile)?;
        if profile.is_some() {
            tracing::info!(
                "ATLAS_GLM_INDEX_PROFILE phase=cache layer={} rows={} key_projection_us={} gate_projection_us={} cache_write_us={}",
                self.attn_layer_idx,
                rows,
                key_projection_us,
                gate_projection_us,
                cache_write_us,
            );
        }
        Ok(())
    }

    /// Project semantic queries and select token-granular sparse history for
    /// every row in this prefill chunk. Logits are processed in bounded row
    /// tiles using the existing MoE activation arena, so scratch does not grow
    /// with the configured model context.
    pub(super) fn glm_index_prefill_select(
        &self,
        q_latent: DevicePtr,
        normed: DevicePtr,
        n: u32,
        seq_len_start: usize,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, u32)> {
        let mla = self.mla.as_ref().expect("GLM index selection without MLA");
        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("GLM index selection without indexer weights");
        ensure!(
            self.glm_index_logits_k.0 != 0
                && self.glm_index_topk_expand_k.0 != 0
                && self.glm_sparse_attn_k.0 != 0,
            "GLM sparse selection/attention kernels are unavailable"
        );
        let index_heads = ctx.config.index_n_heads as u32;
        let index_dim = ctx.config.index_head_dim as u32;
        let pool_size = ctx.config.index_kpool as u32;
        let topk = ctx.config.index_topk as u32;
        let output_width = topk + pool_size - 1;
        let sequence_end = seq_len_start + n as usize;
        let logits_stride = sequence_end.div_ceil(pool_size as usize) as u32;

        let mut profile = profile_start(ctx, stream)?;
        let index_query = ctx.buffers.ssm_deinterleaved();
        self.mla_prefill_dense(
            q_latent,
            &indexer.wq_b,
            index_query,
            n,
            index_heads * index_dim,
            mla.q_lora_rank as u32,
            ctx,
            stream,
        )?;
        // BF16 is sufficient for the first functional selector. A later
        // measured refinement will retain this projection's FP32 accumulator,
        // matching upstream's near-tie ranking treatment.
        let weights = ctx.buffers.ssm_gates();
        self.mla_prefill_dense(
            normed,
            &indexer.weights_proj,
            weights,
            n,
            index_heads,
            ctx.config.hidden_size as u32,
            ctx,
            stream,
        )?;

        let logits = ctx.buffers.expert_up_out();
        let tile_rows = super::super::glm_index_capacity::tile_rows(
            n as usize,
            logits_stride as usize,
            ctx.buffers.sizes().expert_up_out,
        )?;
        let selected = ctx.buffers.expert_down_out();
        let query_row_bytes = index_heads as usize * index_dim as usize * 2;
        let weights_row_bytes = index_heads as usize * 2;
        let output_row_bytes = output_width as usize * std::mem::size_of::<i32>();
        // Projection launches above are intentionally included in this first
        // lap. They produce the semantic query and per-head weights consumed
        // by every history tile.
        let projection_us = profile_lap(ctx, stream, &mut profile)?;
        let mut logits_us = 0u128;
        let mut topk_us = 0u128;
        let mut tiles = 0usize;
        let mut row_start = 0usize;
        while row_start < n as usize {
            let rows = tile_rows.min(n as usize - row_start) as u32;
            ops::glm_index_logits(
                ctx.gpu,
                self.glm_index_logits_k,
                index_query.offset(row_start * query_row_bytes),
                weights.offset(row_start * weights_row_bytes),
                kv_cache.sparse_index_pool_ptr(self.attn_layer_idx),
                logits,
                ctx.attn_metadata
                    .expect("GLM index selection requires block table")
                    .block_table,
                rows,
                (seq_len_start + row_start) as u32,
                logits_stride,
                index_heads,
                index_dim,
                pool_size,
                kv_cache.block_size() as u32,
                kv_cache.sparse_index_block_stride_bytes(self.attn_layer_idx) as u64,
                self.glm_index_logits_rows_per_cta,
                self.glm_index_logits_pools_per_cta,
                stream,
            )?;
            logits_us += profile_lap(ctx, stream, &mut profile)?;
            ops::glm_index_topk_expand(
                ctx.gpu,
                self.glm_index_topk_expand_k,
                logits,
                selected.offset(row_start * output_row_bytes),
                rows,
                (seq_len_start + row_start) as u32,
                logits_stride,
                topk,
                pool_size,
                output_width,
                stream,
            )?;
            topk_us += profile_lap(ctx, stream, &mut profile)?;
            tiles += 1;
            row_start += rows as usize;
        }
        if profile.is_some() {
            tracing::info!(
                "ATLAS_GLM_INDEX_PROFILE phase=select layer={} rows={} seq_end={} pools={} tile_rows={} tiles={} projection_us={} logits_us={} topk_us={}",
                self.attn_layer_idx,
                n,
                sequence_end,
                logits_stride,
                tile_rows,
                tiles,
                projection_us,
                logits_us,
                topk_us,
            );
        }
        Ok((selected, output_width))
    }
}
