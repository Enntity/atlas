// SPDX-License-Identifier: AGPL-3.0-only

//! BF16 KV-only row execution shared by the existing prompt primer and a future
//! explicit accepted-prefix repair. No scheduler, cursor, or wire policy here.

use super::kv_rows_plan::{DeviceSpan, KvRowsPlan, use_cublas};
use super::*;
use anyhow::{Context, ensure};

#[cfg(test)]
#[path = "kv_rows_execution_tests.rs"]
mod tests;

/// Actual dense and semantic-index owners share one checked extent source.
pub(super) fn cache_spans(cache: &PagedKvCache) -> Result<Vec<DeviceSpan>> {
    let mut spans = Vec::with_capacity(5);
    let mut push = |ptr, stride: usize| -> Result<()> {
        if stride == 0 {
            return Ok(());
        }
        let bytes = cache
            .num_blocks()
            .checked_mul(stride)
            .context("GLM cache span overflow")?;
        let span = DeviceSpan { ptr, bytes };
        ensure!(!ptr.is_null(), "GLM cache owner missing");
        span.end()?;
        spans.push(span);
        Ok(())
    };
    push(
        cache.k_cache_ptr(0, 0),
        cache.k_block_stride_bytes_for_layer(0),
    )?;
    push(
        cache.v_cache_ptr(0, 0),
        cache.v_block_stride_bytes_for_layer(0),
    )?;
    if let Some(index) = cache.sparse_index_config() {
        let values = cache.sparse_index_block_stride_bytes(0);
        let tail = cache.sparse_index_tail_block_stride_bytes(0);
        let scales = index
            .block_bytes(cache.block_size())?
            .checked_sub(values)
            .and_then(|bytes| bytes.checked_sub(tail))
            .context("GLM index span mismatch")?;
        push(cache.sparse_index_pool_ptr(0), values)?;
        push(cache.sparse_index_scale_pool_ptr(0), scales)?;
        push(cache.sparse_index_tail_pool_ptr(0), tail)?;
    }
    Ok(spans)
}

impl Glm5MtpHead {
    pub(super) fn prefill_kv_batched(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let result = self.prefill_kv_batched_inner(prompt_tokens, hiddens, state, ctx, stream);
        if result.is_err() {
            state.hidden_trace.prompt.fail();
        }
        result
    }
    fn prefill_kv_batched_inner(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        if state.seq_len != 0 || prompt_tokens.len() < 2 {
            return Ok(0);
        }
        let started = std::time::Instant::now();
        let tokens = &prompt_tokens[1..];
        // DraftProposer::prefill_drafter guarantees prompt-sized owned rows;
        // the model caller checks capture capacity/generation before calling.
        let bytes = tokens
            .len()
            .checked_mul(ctx.config.hidden_size)
            .and_then(|n| n.checked_mul(2))
            .context("GLM primer hidden extent overflow")?;
        let source = DeviceSpan {
            ptr: hiddens,
            bytes,
        };
        {
            let mut cache = self.kv_cache.lock();
            self.validate_kv_inputs(tokens, source, ctx, &cache)?;
            self.validate_kv_blocks(&cache, &state.block_table)?;
            let needed = tokens.len().div_ceil(cache.block_size());
            ensure!(
                needed.saturating_sub(state.block_table.len()) <= cache.num_free_blocks(),
                "GLM primer cache capacity exhausted"
            );
            while state.block_table.len() < needed {
                state.block_table.push(cache.alloc_block()?);
            }
        }
        state
            .hidden_trace
            .prompt
            .primer_before(tokens, source, ctx, stream)?;
        if let Err(error) = self.write_kv_rows(tokens, source, 0, &state.block_table, ctx, stream) {
            state.hidden_trace.prompt.fail();
            return Err(error);
        }
        if state.hidden_trace.prompt.active() {
            let cache = self.kv_cache.lock();
            state
                .hidden_trace
                .prompt
                .primer_after(&cache, &state.block_table, ctx, stream)?;
        }
        state.seq_len = tokens.len();
        tracing::info!(
            "GLM MTP batched KV prefill: {} rows in {:.1} ms",
            tokens.len(),
            started.elapsed().as_secs_f64() * 1e3
        );
        Ok(tokens.len())
    }

    pub(super) fn validate_kv_blocks(&self, cache: &PagedKvCache, blocks: &[u32]) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for &block in blocks {
            ensure!(
                (block as usize) < cache.num_blocks() && seen.insert(block),
                "GLM KV block is out of range or repeated"
            );
            ensure!(
                cache.ref_count(block) == 1,
                "GLM KV write requires exclusive allocated blocks"
            );
        }
        Ok(())
    }

    pub(super) fn validate_kv_inputs(
        &self,
        tokens: &[u32],
        source: DeviceSpan,
        ctx: &ForwardContext,
        cache: &PagedKvCache,
    ) -> Result<Vec<DeviceSpan>> {
        let config = cache.config();
        ensure!(
            ctx.config.model_type == "glm5_next"
                && ctx.config.qk_rope_head_dim == 0
                && ctx.config.kv_lora_rank == 512
                && config.num_layers == 1
                && config.num_kv_heads == 1
                && config.head_dim == 512
                && config.dtype == KvCacheDtype::Bf16
                && config.layer_dtypes.iter().all(|&d| d == KvCacheDtype::Bf16)
                && config.layer_dims.is_empty(),
            "GLM KV writer requires its BF16 NoPE512 cache"
        );
        ensure!(!ctx.graph_capture, "GLM KV writer is eager only");
        ensure!(
            self.module.body.supports_mla_kv_only(),
            "GLM KV writer requires a KV-only MLA body"
        );
        let h = ctx.config.hidden_size;
        let row_bytes = h.checked_mul(2).context("GLM KV row overflow")?;
        let n = tokens.len().min(ctx.buffers.max_batch_tokens());
        let hidden_bytes = n.checked_mul(row_bytes).context("GLM KV chunk overflow")?;
        let latent_bytes = n
            .checked_mul(config.head_dim)
            .and_then(|v| v.checked_mul(2))
            .context("GLM KV latent overflow")?;
        let sizes = ctx.buffers.sizes();
        let buffers = [
            (
                ctx.buffers.ssm_deinterleaved(),
                sizes.ssm_deinterleaved,
                hidden_bytes,
            ),
            (ctx.buffers.attn_output(), sizes.attn_output, hidden_bytes),
            (ctx.buffers.residual(), sizes.residual, hidden_bytes),
            (
                ctx.buffers.ssm_qkvz(),
                sizes.ssm_qkvz,
                hidden_bytes
                    .checked_mul(2)
                    .context("GLM KV EH extent overflow")?,
            ),
            (
                ctx.buffers.hidden_states(),
                sizes.hidden_states,
                hidden_bytes,
            ),
            (ctx.buffers.norm_output(), sizes.norm_output, hidden_bytes),
            (
                ctx.buffers.expert_gate_out(),
                sizes.expert_gate_out,
                latent_bytes,
            ),
            (
                ctx.buffers.expert_up_out(),
                sizes.expert_up_out,
                latent_bytes,
            ),
            (
                ctx.buffers.expert_down_out(),
                sizes.expert_down_out,
                latent_bytes,
            ),
            (ctx.buffers.scratch(), sizes.scratch, MTP_META_OFFSET),
        ];
        let mut forbidden = Vec::with_capacity(buffers.len() + 2);
        for (ptr, bytes, required) in buffers {
            ensure!(
                ptr.0 != 0 && ptr.0.is_multiple_of(2) && bytes >= required,
                "GLM KV scratch buffer {:?} is absent, unaligned, or undersized: {bytes} < {required}",
                ptr
            );
            let span = DeviceSpan { ptr, bytes };
            span.end()?;
            forbidden.push(span);
        }
        forbidden.extend(cache_spans(cache)?);
        KvRowsPlan::inputs(
            tokens,
            source,
            h,
            ctx.config.vocab_size,
            ctx.buffers.max_batch_tokens(),
            ctx.buffers.scratch_bytes(),
            &forbidden,
        )?;
        let embedding_bytes = ctx
            .config
            .vocab_size
            .checked_mul(row_bytes)
            .context("GLM KV embedding span overflow")?;
        for (ptr, bytes) in [
            (self.embed_tokens.weight, embedding_bytes),
            (self.module.enorm.weight, row_bytes),
            (self.module.hnorm.weight, row_bytes),
            (
                self.module.eh_proj.weight,
                h.checked_mul(row_bytes)
                    .and_then(|v| v.checked_mul(2))
                    .context("GLM KV EH weight span overflow")?,
            ),
        ] {
            ensure!(
                ptr.0 != 0 && ptr.0.is_multiple_of(2),
                "GLM KV weight is absent or unaligned"
            );
            DeviceSpan { ptr, bytes }.end()?;
        }
        let dense_needed = !ctx.dispatch.cublas_gemm
            || ctx.buffers.max_batch_tokens() == 1
            || tokens.len() % ctx.buffers.max_batch_tokens() == 1;
        ensure!(
            self.rms_norm_k.0 != 0
                && self.bf16_concat_k.0 != 0
                && (!dense_needed || self.dense_gemm_k.0 != 0),
            "GLM KV writer is missing a projection kernel"
        );
        Ok(forbidden)
    }

    /// Sources are already shifted and owned outside mutable scratch. Blocks
    /// are already allocated exclusively. This writes rows but never changes
    /// seq_len/last_num_drafted or allocates blocks; callers own commit policy.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_kv_rows(
        &self,
        tokens: &[u32],
        source: DeviceSpan,
        row_base: usize,
        blocks: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut cache = self.kv_cache.lock();
        let forbidden = self.validate_kv_inputs(tokens, source, ctx, &cache)?;
        self.validate_kv_blocks(&cache, blocks)?;
        let plan = KvRowsPlan::new(
            tokens,
            source,
            ctx.config.hidden_size,
            ctx.config.vocab_size,
            row_base,
            cache.block_size(),
            blocks,
            cache.num_blocks(),
            ctx.buffers.max_batch_tokens(),
            ctx.buffers.scratch_bytes(),
            &forbidden,
        )?;
        if super::kv_rows_oracle::enabled(tokens.len()) {
            self.verify_kv_rows(tokens, &plan, source, blocks, &mut cache, ctx, stream)
        } else {
            self.execute_kv_rows(&plan, source, &mut cache, ctx, stream)
        }
    }

    pub(super) fn execute_kv_rows(
        &self,
        plan: &KvRowsPlan,
        source: DeviceSpan,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let rows_total = plan.rows;
        let chunk_rows = plan.chunk_rows;
        let mut done = 0usize;
        while done < rows_total {
            let n = (rows_total - done).min(chunk_rows);
            let embed = ctx.buffers.ssm_deinterleaved();
            plan.copy_embeddings(ctx.gpu, self.embed_tokens.weight, embed, done, n, stream)?;

            let normed_embed = ctx.buffers.attn_output();
            let normed_hidden = ctx.buffers.residual();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                embed,
                &self.module.enorm,
                normed_embed,
                n as u32,
                h as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                source.ptr.offset(done * h * bf16),
                &self.module.hnorm,
                normed_hidden,
                n as u32,
                h as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;

            let eh_input = ctx.buffers.ssm_qkvz();
            for row in 0..n {
                ops::bf16_concat(
                    ctx.gpu,
                    self.bf16_concat_k,
                    normed_embed.offset(row * h * bf16),
                    normed_hidden.offset(row * h * bf16),
                    eh_input.offset(row * 2 * h * bf16),
                    h as u32,
                    stream,
                )?;
            }
            let h_in = ctx.buffers.hidden_states();
            if use_cublas(ctx.dispatch.cublas_gemm, n) {
                ops::cublas_bf16_proj_dense(
                    eh_input,
                    self.module.eh_proj.weight,
                    h_in,
                    n as u32,
                    h as u32,
                    (2 * h) as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    eh_input,
                    &self.module.eh_proj,
                    h_in,
                    n as u32,
                    h as u32,
                    (2 * h) as u32,
                    stream,
                )?;
            }

            let slots_dev = ctx.buffers.scratch().offset(MTP_META_OFFSET);
            plan.upload_slots(ctx.gpu, slots_dev, done, n, stream)?;

            let mtp_ctx = ForwardContext {
                ssm_batch: None,
                buffers: ctx.buffers,
                gpu: ctx.gpu,
                config: ctx.config,
                dispatch: ctx.dispatch,
                derived: ctx.derived,
                levers: ctx.levers,
                stats: ctx.stats,
                attn_metadata: None,
                profile: ctx.profile,
                comm: None,
                graph_capture: false,
                gdn_exact_replay: false,
                token_ids: ctx.token_ids,
                routed_lora_layers: None,
                midchunk_capture: None,
                moe_lora_route: crate::layer::MoeLoraRoute::Skip,
            };
            anyhow::ensure!(
                self.module
                    .body
                    .prefill_mla_kv_only(h_in, n, kv_cache, slots_dev, &mtp_ctx, stream,)?,
                "GLM MTP appended layer does not support MLA KV-only prefill"
            );
            // The plan owns pageable slot storage; keep it alive until
            // its async upload and the dependent cache write have completed.
            ctx.gpu.synchronize(stream)?;
            done += n;
        }
        Ok(())
    }
}
