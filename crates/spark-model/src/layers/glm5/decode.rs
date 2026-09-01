// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::types::{DsaWeights, Glm5Layer, GlmAttentionWeights};
use crate::layer::{ForwardContext, GlmSparseMlaLayerState, LayerState};
use crate::layers::ops;

mod trait_impl;

pub(super) const KDA_HEADS: usize = 32;
pub(super) const KDA_WIDTH: usize = KDA_HEADS * 128;
pub(super) const DSA_HEADS: usize = 32;
pub(super) const DSA_QUERY_WIDTH: usize = DSA_HEADS * 256;
pub(super) const INDEX_QUERY_WIDTH: usize = 4096;
pub(super) const INDEX_DIM: usize = 128;
pub(super) const INDEX_HEADS: usize = 32;
pub(super) const DSA_TOP_POOLS: u32 = 512;

impl Glm5Layer {
    fn dsa_metadata(
        &self,
        rows: usize,
        states: &[&(dyn LayerState + '_)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
        let workspace = ctx.buffers.glm_workspace();
        let layout = ctx.buffers.glm_layout();
        let metadata = ctx
            .attn_metadata
            .context("GLM DSA decode requires staged positions")?;
        ensure!(
            metadata.num_seqs as usize >= rows,
            "GLM DSA position batch is too small"
        );
        let mut table = Vec::with_capacity(rows * 5);
        for state in states {
            let state = state
                .as_any()
                .downcast_ref::<GlmSparseMlaLayerState>()
                .context("GLM DSA layer received incompatible state")?;
            table.extend([
                state.current.latent_cache.0,
                state.current.pooled_keys.0,
                state.current.tail_keys.0,
                state.current.tail_gates.0,
                state.current.tail_metadata.0,
            ]);
        }
        let table = self.upload_state_table(&table, ctx, stream)?;
        let cu_device = workspace.offset(layout.cu_seqlens_i32);
        let valid = workspace.offset(layout.valid);
        if !ctx.graph_capture {
            let cu = (0..=rows as i32)
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>();
            ctx.gpu.copy_h2d_async(&cu, cu_device, stream)?;
            ctx.gpu.memset_async(valid, 1, rows, stream)?;
        }
        Ok((table, cu_device, metadata.positions))
    }

    #[allow(clippy::too_many_arguments)]
    fn dsa_forward(
        &self,
        weights: &DsaWeights,
        normed: DevicePtr,
        rows: usize,
        pool_limit: u32,
        states: &[&(dyn LayerState + '_)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let hidden = ctx.config.hidden_size;
        let q_resid = ctx.buffers.ssm_ba();
        self.project(
            normed,
            &weights.query_a,
            q_resid,
            rows,
            1536,
            hidden,
            ctx,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.kernels.rms_norm,
            q_resid,
            &weights.query_a_norm,
            q_resid,
            rows as u32,
            1536,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        let query = ctx.buffers.qkv_output();
        self.project(
            q_resid,
            &weights.query_b,
            query,
            rows,
            DSA_QUERY_WIDTH,
            1536,
            ctx,
            stream,
        )?;
        let absorbed_query = ctx.buffers.ssm_deinterleaved();
        ops::glm53_dsa_absorb_query_bf16(
            ctx.gpu,
            self.kernels.dsa_absorb_query,
            &ops::Glm53DsaBf16ProjectionArgs {
                input: query,
                kv_b_weight: weights.kv_b.weight,
                output: absorbed_query,
                num_tokens: rows as u32,
                num_heads: DSA_HEADS as u32,
            },
            stream,
        )?;

        let needs_sparse_ranking = pool_limit > DSA_TOP_POOLS;
        let index_query = query;
        if needs_sparse_ranking {
            self.project(
                q_resid,
                &weights.index_query,
                index_query,
                rows,
                INDEX_QUERY_WIDTH,
                1536,
                ctx,
                stream,
            )?;
        }
        let latent = q_resid;
        self.project(
            normed,
            &weights.kv_a,
            latent,
            rows,
            512,
            hidden,
            ctx,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.kernels.rms_norm,
            latent,
            &weights.kv_a_norm,
            latent,
            rows as u32,
            512,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;

        let index_key = query.offset(rows * INDEX_QUERY_WIDTH * 2);
        let index_gates = index_key.offset(rows * INDEX_DIM * 2);
        let index_weights = index_gates.offset(rows * INDEX_DIM * 2);
        self.project(
            normed,
            &weights.index_key,
            index_key,
            rows,
            INDEX_DIM,
            hidden,
            ctx,
            stream,
        )?;
        ops::glm53_dsa_index_layernorm(
            ctx.gpu,
            self.kernels.dsa_index_norm,
            &ops::Glm53DsaIndexNormArgs {
                input: index_key,
                weight: weights.index_key_norm.weight,
                bias: weights.index_key_bias.weight,
                output: index_key,
                rows: rows as u32,
                epsilon: 1.0e-6,
            },
            stream,
        )?;
        self.project(
            normed,
            &weights.index_gates,
            index_gates,
            rows,
            INDEX_DIM,
            hidden,
            ctx,
            stream,
        )?;
        if needs_sparse_ranking {
            self.project(
                normed,
                &weights.index_head_weights,
                index_weights,
                rows,
                INDEX_HEADS,
                hidden,
                ctx,
                stream,
            )?;
        }

        let (state_table, cu_seqlens, positions) = self.dsa_metadata(rows, states, ctx, stream)?;
        let layout = ctx.buffers.glm_layout();
        let workspace = ctx.buffers.glm_workspace();
        let valid = workspace.offset(layout.valid);
        ops::glm53_dsa_latent_append(
            ctx.gpu,
            self.kernels.dsa_latent_append,
            &ops::Glm53DsaLatentAppendArgs {
                latent,
                cu_seqlens,
                positions,
                valid,
                sequence_state_ptrs: state_table,
                total_tokens: rows as u32,
                num_sequences: rows as u32,
                latent_capacity: layout.latent_capacity as u32,
            },
            stream,
        )?;
        ops::glm53_dsa_pool_append(
            ctx.gpu,
            self.kernels.dsa_pool,
            &ops::Glm53DsaPoolAppendArgs {
                keys: index_key,
                gates: index_gates,
                ape: weights.index_ape.weight,
                cu_seqlens,
                positions,
                valid,
                sequence_state_ptrs: state_table,
                num_sequences: rows as u32,
            },
            stream,
        )?;
        let scores = workspace.offset(layout.dsa_scores);
        let selected = workspace.offset(layout.dsa_selected);
        if needs_sparse_ranking {
            ops::glm53_dsa_score(
                ctx.gpu,
                self.kernels.dsa_score,
                &ops::Glm53DsaScoreArgs {
                    query: index_query,
                    head_weights: index_weights,
                    query_valid: valid,
                    sequence_state_ptrs: state_table,
                    scores,
                    max_pools: pool_limit,
                    num_sequences: rows as u32,
                },
                stream,
            )?;
        }
        ops::glm53_dsa_topk_expand_decode(
            ctx.gpu,
            self.kernels.dsa_topk,
            &ops::Glm53DsaTopkArgs {
                scores,
                query_valid: valid,
                sequence_state_ptrs: state_table,
                output: selected,
                max_pools: pool_limit,
                num_sequences: rows as u32,
            },
            stream,
        )?;
        let latent_output = ctx.buffers.attn_output();
        ops::glm53_dsa_sparse_mla_decode(
            ctx.gpu,
            self.kernels.dsa_sparse_mla,
            &ops::Glm53DsaSparseMlaArgs {
                absorbed_query,
                selected_indices: selected,
                sequence_state_ptrs: state_table,
                output: latent_output,
                num_heads: DSA_HEADS as u32,
                num_sequences: rows as u32,
                attention_scale: 0.0625,
            },
            stream,
        )?;
        let expanded = query;
        ops::glm53_dsa_expand_value_bf16(
            ctx.gpu,
            self.kernels.dsa_expand_value,
            &ops::Glm53DsaBf16ProjectionArgs {
                input: latent_output,
                kv_b_weight: weights.kv_b.weight,
                output: expanded,
                num_tokens: rows as u32,
                num_heads: DSA_HEADS as u32,
            },
            stream,
        )?;
        let output = ctx.buffers.residual();
        self.project(
            expanded,
            &weights.output,
            output,
            rows,
            hidden,
            DSA_QUERY_WIDTH,
            ctx,
            stream,
        )?;
        Ok(output)
    }

    fn forward_rows(
        &self,
        hidden: DevicePtr,
        rows: usize,
        states: &[&(dyn LayerState + '_)],
        seq_lens: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            rows > 0 && states.len() == rows,
            "GLM decode state batch mismatch"
        );
        ensure!(seq_lens.len() == rows, "GLM decode length batch mismatch");
        let h = ctx.config.hidden_size;
        if ctx.profile {
            ctx.gpu.synchronize(stream)?;
        }
        let attention_started = ctx.profile.then(std::time::Instant::now);
        if self.layer_idx == 0 {
            ops::hc_expand(
                ctx.gpu,
                self.kernels.hc_expand,
                hidden,
                ctx.buffers.hc_streams(),
                rows as u32,
                h as u32,
                ctx.config.hc_mult as u32,
                stream,
            )?;
        }
        let normed = ctx.buffers.norm_output();
        self.hc_pre_norm(
            &self.hc_attention,
            &self.input_norm,
            hidden,
            normed,
            rows,
            ctx,
            stream,
        )?;
        let attention_output = match &self.attention {
            GlmAttentionWeights::Kda(weights) => {
                self.kda_forward(weights, normed, rows, states, ctx, stream)?
            }
            GlmAttentionWeights::Dsa(weights) => {
                let next_tokens = seq_lens.iter().copied().max().unwrap_or(0) + 1;
                let pool_limit = dsa_pool_limit(
                    next_tokens,
                    ctx.config.index_kpool,
                    ctx.buffers.glm_layout().dsa_max_pools,
                );
                self.dsa_forward(weights, normed, rows, pool_limit, states, ctx, stream)?
            }
        };
        self.tp_sum(attention_output, rows, ctx, stream)?;
        self.hc_post(attention_output, rows, ctx, stream)?;

        let attention_us = if let Some(started) = attention_started {
            ctx.gpu.synchronize(stream)?;
            started.elapsed().as_micros() as u64
        } else {
            0
        };
        let ffn_started = ctx.profile.then(std::time::Instant::now);

        self.hc_pre_norm(
            &self.hc_ffn,
            &self.post_attention_norm,
            hidden,
            normed,
            rows,
            ctx,
            stream,
        )?;
        let ffn_output = self.ffn_forward(normed, rows, ctx, stream)?;
        self.hc_post(ffn_output, rows, ctx, stream)?;
        self.contract_hc_for_output_or_dflash(hidden, rows, ctx, stream)?;
        if let Some(started) = ffn_started {
            ctx.gpu.synchronize(stream)?;
            let ffn_us = started.elapsed().as_micros() as u64;
            let attention_kind = match &self.attention {
                GlmAttentionWeights::Kda(_) => "kda",
                GlmAttentionWeights::Dsa(_) => "dsa",
            };
            let ffn_kind = match &self.ffn {
                super::types::GlmFfn::Dense(_) => "dense",
                super::types::GlmFfn::Exl3(_) => "exl3",
            };
            tracing::info!(
                "GLM_PROFILE layer={} rows={} attention={} attention_ms={:.3} ffn={} ffn_ms={:.3}",
                self.layer_idx,
                rows,
                attention_kind,
                attention_us as f64 / 1000.0,
                ffn_kind,
                ffn_us as f64 / 1000.0,
            );
        }
        Ok(())
    }
}

pub(super) fn dsa_pool_limit(tokens: usize, pool_size: usize, capacity: usize) -> u32 {
    (tokens / pool_size.max(1)).max(1).min(capacity.max(1)) as u32
}

/// Stable sparse-index launch bucket for CUDA-graphed speculative verify.
///
/// Up to 512 complete pools, top-k is the full visible set and the scoring
/// kernel is absent, so one 512-pool graph covers the entire common <=2K-token
/// context range. Above that boundary, power-of-two grids bound cache churn;
/// the score kernel masks entries beyond each sequence's metadata pool count.
pub(crate) fn dsa_verify_pool_bucket(tokens: usize, pool_size: usize, capacity: usize) -> u32 {
    let capacity = capacity.max(1);
    let visible = (tokens / pool_size.max(1)).max(1).min(capacity);
    if visible <= DSA_TOP_POOLS as usize {
        return (DSA_TOP_POOLS as usize).min(capacity) as u32;
    }
    visible.next_power_of_two().min(capacity) as u32
}

#[cfg(test)]
mod tests {
    use super::{dsa_pool_limit, dsa_verify_pool_bucket};

    #[test]
    fn pool_grid_tracks_visible_complete_pools() {
        assert_eq!(dsa_pool_limit(1, 4, 8192), 1);
        assert_eq!(dsa_pool_limit(23, 4, 8192), 5);
        assert_eq!(dsa_pool_limit(498, 4, 8192), 124);
        assert_eq!(dsa_pool_limit(40_000, 4, 8192), 8192);
        assert_eq!(dsa_pool_limit(23, 0, 0), 1);
    }

    #[test]
    fn verify_pool_grid_is_stable_until_sparse_ranking_begins() {
        assert_eq!(dsa_verify_pool_bucket(1, 4, 8192), 512);
        assert_eq!(dsa_verify_pool_bucket(2047, 4, 8192), 512);
        assert_eq!(dsa_verify_pool_bucket(2048, 4, 8192), 512);
        assert_eq!(dsa_verify_pool_bucket(2051, 4, 8192), 512);
        assert_eq!(dsa_verify_pool_bucket(2052, 4, 8192), 1024);
        assert_eq!(dsa_verify_pool_bucket(4099, 4, 8192), 1024);
        assert_eq!(dsa_verify_pool_bucket(4100, 4, 8192), 2048);
        assert_eq!(dsa_verify_pool_bucket(40_000, 4, 8192), 8192);
        assert_eq!(dsa_verify_pool_bucket(23, 4, 128), 128);
    }
}
