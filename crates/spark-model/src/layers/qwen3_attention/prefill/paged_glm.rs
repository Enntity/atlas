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

use super::super::{MlaWeights, Qwen3AttentionLayer};
use super::paged_mla::MlaPrefillArgs;
use crate::layer::ForwardContext;
use crate::layers::ops;

#[path = "paged_glm_projection.rs"]
mod projection;

fn dense_selection_is_exact(sequence_end: usize, index_topk: usize) -> bool {
    index_topk > 0 && sequence_end <= index_topk
}

/// One owner of a (possibly multi-sequence) GLM chunk attention: `rows`
/// rows at `row0` of the joint inputs, continuing its sequence at
/// `seq_len_start` with its own single-sequence metadata.
#[derive(Clone, Copy)]
pub(in crate::layers::qwen3_attention) struct GlmChunkOwner {
    pub row0: usize,
    pub rows: usize,
    pub seq_len_start: usize,
    pub meta: crate::layer::AttnMetadataDev,
}

/// One sequence's prefill chunk of `rows` rows from `seq_len_start` as chunk
/// owners at rows `[0, rows)` of `meta`. Several pieces are consecutive
/// same-sequence owners: the joint cache write lands every row first, and
/// piece k's causal extent is entry 1 + k of the chunk's seq_len buffer
/// (Model::upload_chunk_seq_lens).
pub(in crate::layers::qwen3_attention) fn glm_chunk_pieces(
    meta: crate::layer::AttnMetadataDev,
    seq_len_start: usize,
    rows: usize,
    index_topk: usize,
) -> Vec<GlmChunkOwner> {
    let pieces = crate::layer::prefill_attention_pieces(seq_len_start, rows, index_topk);
    if pieces.len() == 1 {
        return vec![GlmChunkOwner { row0: 0, rows, seq_len_start, meta }];
    }
    pieces
        .iter()
        .enumerate()
        .map(|(k, &(row0, rows))| GlmChunkOwner {
            row0,
            rows,
            seq_len_start: seq_len_start + row0,
            meta: crate::layer::AttnMetadataDev {
                positions: meta.positions.offset(row0 * 4),
                positions_h: meta.positions_h.offset(row0 * 4),
                positions_w: meta.positions_w.offset(row0 * 4),
                slot: meta.slot.offset(row0 * 8),
                seq_len: meta.seq_len.offset((1 + k) * 4),
                ..meta
            },
        })
        .collect()
}

/// An `fp8_g128` owner's latents dequantized to BF16 in the arena scratch,
/// addressed through an identity block table of `blocks` entries.
#[derive(Clone, Copy)]
struct Bf16LatentView {
    latents: DevicePtr,
    identity_table: DevicePtr,
    blocks: usize,
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
        let meta = ctx
            .attn_metadata
            .expect("GLM paged prefill requires metadata");
        let owners = glm_chunk_pieces(meta, seq_len_start, args.num_tokens, ctx.config.index_topk);
        self.glm_chunk_attention(&owners, kv_cache, ctx, args)
    }

    /// GLM MLA over the causal chunks of one or more sequences whose rows are
    /// stacked in `args.normed` (`ctx.attn_metadata` covers every row).
    /// Row-wise projections (q_a, kv_a, W_uv, o) and the KV cache write run
    /// once over all rows; the semantic index, q_b + W_uk absorb and sparse
    /// attention run per owner in the pinned native-sparse operand buffers.
    /// With several owners each owner's attention rows are parked in the
    /// (idle until the LM head) logits arena until the joint W_uv.
    pub(super) fn glm_chunk_attention(
        &self,
        owners: &[GlmChunkOwner],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
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
            seq_len_start: _,
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
            matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128),
            "GLM chunked prefill requires a BF16 or fp8_g128 KV cache; got {:?}",
            self.kv_dtype
        );
        ensure!(
            !owners.is_empty()
                && owners.iter().map(|o| o.rows).sum::<usize>() == num_tokens
                && owners.windows(2).all(|w| w[0].row0 + w[0].rows == w[1].row0),
            "GLM chunk owners must tile the stacked rows"
        );
        let accelerated = projection::enabled(&ctx.config.model_type)?;
        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let nope = mla.nope as u32;
        let v_dim = mla.v_dim as u32;
        ensure!(
            kv_lora == 512 && nope == hd && v_dim == hd,
            "unsupported GLM MLA geometry: kv_lora={kv_lora}, nope={nope}, v={v_dim}, hd={hd}"
        );

        // Joint: q_a (+ norm), kv_a (+ norm) and the cache write.
        let q_latent = ctx.buffers.ssm_ba();
        self.paged_glm_projection(normed, &mla.wq_a, q_latent, n, q_lora, h, ctx, stream, accelerated)?;
        ops::rms_norm(ctx.gpu, self.rms_norm_w_k, q_latent, &mla.q_a_norm, q_latent, n, q_lora, eps, stream)?;
        let kv_latent = ctx.buffers.expert_gate_out();
        self.paged_glm_projection(normed, &mla.wkv_a, kv_latent, n, kv_lora, h, ctx, stream, accelerated)?;
        ops::rms_norm(ctx.gpu, self.rms_norm_w_k, kv_latent, &mla.kv_a_norm, kv_latent, n, kv_lora, eps, stream)?;
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
        let joint = ctx
            .attn_metadata
            .expect("GLM paged prefill requires metadata");
        self.write_kv_cache(
            ctx.gpu,
            k_entries,
            v_entries,
            kv_cache,
            joint.slot,
            n,
            1,
            kv_lora,
            bs,
            kv_lora,
            kv_lora,
            stream,
            ctx.graph_capture,
        )?;

        // Per owner: semantic index, q_b + absorb, attention.
        let attn_latent = ctx.buffers.attn_output();
        let latent_row = nq as usize * kv_lora as usize * bf16;
        // Owners share the attention scratch, so each one's latent result is
        // parked for one joint W_uv + o_proj. When the chunk outgrows the
        // park (8K sub-chunked prefill) each owner projects its own rows.
        let o_out = ctx.buffers.norm_output();
        let per_owner = owners.len() > 1 && ctx.buffers.sizes().logits < num_tokens * latent_row;
        let parked = (owners.len() > 1 && !per_owner).then(|| ctx.buffers.logits());
        for o in owners {
            let on = o.rows as u32;
            let octx = ForwardContext {
                attn_metadata: Some(o.meta),
                midchunk_capture: None,
                ..*ctx
            };
            let sequence_end = o
                .seq_len_start
                .checked_add(o.rows)
                .ok_or_else(|| anyhow::anyhow!("GLM prefill sequence length overflow"))?;
            let use_dense = dense_selection_is_exact(sequence_end, ctx.config.index_topk);
            let o_normed = normed.offset(o.row0 * h as usize * bf16);
            let o_latent = q_latent.offset(o.row0 * q_lora as usize * bf16);
            self.glm_index_prefill_cache_update(o_normed, on, kv_cache, &octx, stream)?;
            let sparse_indices = if use_dense {
                None
            } else {
                Some(self.glm_index_prefill_select(
                    o_latent,
                    o_normed,
                    on,
                    o.seq_len_start,
                    kv_cache,
                    &octx,
                    stream,
                )?)
            };
            // The BF16 dense and native kernels read an fp8_g128 owner through
            // a dequantized view; long owners and verify rows read FP8 directly.
            let view = self.glm_owner_bf16_view(
                kv_cache,
                o.meta.block_table,
                sequence_end,
                use_dense || on >= 2048,
                bs,
                ctx,
                stream,
            )?;
            ensure!(
                !use_dense || self.kv_dtype == KvCacheDtype::Bf16 || view.is_some(),
                "GLM dense prefill has no BF16 view of the fp8_g128 cache"
            );
            let (k_cache, v_cache, block_table, cache_dtype) = match view {
                Some(v) => (v.latents, v.latents, v.identity_table, KvCacheDtype::Bf16),
                None => (
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    o.meta.block_table,
                    self.kv_dtype,
                ),
            };
            let q_full = ctx.buffers.qkv_output();
            self.paged_glm_projection(
                o_latent, &mla.wq_b, q_full, on, nq * hd, q_lora, &octx, stream, accelerated,
            )?;
            let q_absorbed = ctx.buffers.ssm_deinterleaved();
            ops::glm_paged_grouped_gemm_mla(
                ctx.gpu,
                self.grouped_gemm_mla_k,
                &ctx.config.model_type,
                q_full,
                mla.w_uk_t.weight,
                q_absorbed,
                on,
                nq,
                nope,
                kv_lora,
                nq * hd,
                nq * kv_lora,
                stream,
            )?;
            if let Some((indices, index_width)) = sparse_indices {
                let mut profile = super::glm_index::profile_start(&octx, stream)?;
                let sparse_args = ops::GlmSparsePrefillTc {
                    config: ctx.config,
                    dtype: cache_dtype,
                    // mla_cache_assemble_batched above writes the same normalized
                    // NoPE latent to both conventional paged cache sides.
                    identical_kv_latent: true,
                    query: q_absorbed,
                    k_cache,
                    v_cache,
                    indices,
                    output: attn_latent,
                    block_table,
                    rows: on,
                    heads: nq,
                    head_dim: kv_lora,
                    index_width,
                    block_size: bs,
                    scale: self.effective_attn_scale(hd),
                };
                let (physical_blocks, table_blocks, block_bytes) = match view {
                    Some(v) => (v.blocks, v.blocks, 16 * 512 * 2),
                    None => (
                        kv_cache.num_blocks(),
                        o.meta.max_blocks_per_seq as usize,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx),
                    ),
                };
                let native = cache_dtype == KvCacheDtype::Bf16
                    && ops::try_glm_sparse_native(
                        &octx,
                        &sparse_args,
                        o.seq_len_start,
                        physical_blocks,
                        table_blocks,
                        block_bytes,
                        // This caller is ordinary continued prefill. Repaired K3
                        // verification has a separate multi-sequence attention path.
                        false,
                        stream,
                    )?;
                let accelerated =
                    native || ops::try_glm_sparse_prefill_tc(ctx.gpu, &sparse_args, stream)?;
                if !accelerated {
                    ensure!(
                        cache_dtype == KvCacheDtype::Bf16,
                        "fp8_g128 GLM sparse attention requires ATLAS_GLM_SPARSE_PREFILL_TC=1"
                    );
                    ops::glm_sparse_mla_prefill(
                        ctx.gpu,
                        self.glm_sparse_attn_k,
                        q_absorbed,
                        k_cache,
                        v_cache,
                        indices,
                        attn_latent,
                        block_table,
                        on,
                        nq,
                        kv_lora,
                        index_width,
                        bs,
                        self.glm_sparse_attn_heads_per_cta,
                        self.effective_attn_scale(hd),
                        stream,
                    )?;
                }
                let sparse_attention_us = super::glm_index::profile_lap(&octx, stream, &mut profile)?;
                if profile.is_some() {
                    tracing::info!(
                        "ATLAS_GLM_INDEX_PROFILE phase=attention layer={} rows={} seq_end={} selected={} sparse_attention_us={}",
                        self.attn_layer_idx,
                        on,
                        sequence_end,
                        index_width,
                        sparse_attention_us,
                    );
                }
            } else {
                ensure!(
                    self.prefill_attn_paged_512_k.0 != 0,
                    "GLM paged prefill kernel inferspark_prefill_paged_512 is unavailable"
                );
                ops::prefill_attention_paged_512(
                    ctx.gpu,
                    self.prefill_attn_paged_512_k,
                    q_absorbed,
                    k_cache,
                    v_cache,
                    attn_latent,
                    block_table,
                    on,
                    sequence_end as u32,
                    o.seq_len_start as u32,
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
            if per_owner {
                self.paged_glm_output(
                    mla,
                    attn_latent,
                    o_out.offset(o.row0 * h as usize * bf16),
                    [o.rows as u32, nq, h],
                    &octx,
                    stream,
                    accelerated,
                )?;
            } else if let Some(park) = parked {
                ctx.gpu.copy_d2d_async(
                    attn_latent,
                    park.offset(o.row0 * latent_row),
                    o.rows * latent_row,
                    stream,
                )?;
            }
        }

        if !per_owner {
            self.paged_glm_output(mla, parked.unwrap_or(attn_latent), o_out, [n, nq, h], ctx, stream, accelerated)?;
        }
        Ok(o_out)
    }

    /// W_uv then the row-parallel o_proj for `rows` latent attention rows
    /// of `nq` local heads into `h` hidden columns.
    #[allow(clippy::too_many_arguments)]
    fn paged_glm_output(
        &self,
        mla: &MlaWeights,
        latent: DevicePtr,
        out: DevicePtr,
        [rows, nq, h]: [u32; 3],
        ctx: &ForwardContext,
        stream: u64,
        accelerated: bool,
    ) -> Result<()> {
        let (kv_lora, v_dim) = (mla.kv_lora_rank as u32, mla.v_dim as u32);
        let v_extracted = ctx.buffers.qkv_output();
        ops::glm_paged_grouped_gemm_mla(
            ctx.gpu,
            self.grouped_gemm_mla_k,
            &ctx.config.model_type,
            latent,
            mla.w_uv.weight,
            v_extracted,
            rows,
            nq,
            kv_lora,
            v_dim,
            nq * kv_lora,
            nq * v_dim,
            stream,
        )?;
        self.paged_glm_projection(v_extracted, &mla.wo, out, rows, h, nq * v_dim, ctx, stream, accelerated)
    }

    /// When `wanted` and the cache is `fp8_g128`, dequantize the owner's
    /// tokens `[0, end)` into the BF16 view — unless they outgrow it, in
    /// which case the caller reads the FP8 cache directly.
    #[allow(clippy::too_many_arguments)]
    fn glm_owner_bf16_view(
        &self,
        kv_cache: &PagedKvCache,
        block_table: DevicePtr,
        end: usize,
        wanted: bool,
        block_size: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<Bf16LatentView>> {
        if self.kv_dtype != KvCacheDtype::Fp8G128 || !wanted {
            return Ok(None);
        }
        let (latents, identity_table, capacity) = ctx
            .buffers
            .glm_latent_scratch()
            .ok_or_else(|| anyhow::anyhow!("fp8_g128 GLM cache has no BF16 view scratch"))?;
        if end > capacity {
            return Ok(None);
        }
        ops::glm_latent_dequant_fp8g128(
            ctx.gpu,
            self.glm_latent_dequant_k,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            block_table,
            latents,
            end as u32,
            block_size,
            stream,
        )?;
        Ok(Some(Bf16LatentView {
            latents,
            identity_table,
            blocks: capacity / 16,
        }))
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
