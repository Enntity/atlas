// SPDX-License-Identifier: AGPL-3.0-only

//! Single-request GLM prefill with batched projections and native state updates.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::decode::{
    DSA_HEADS, DSA_QUERY_WIDTH, DSA_TOP_POOLS, INDEX_DIM, INDEX_HEADS, INDEX_QUERY_WIDTH,
};
use super::types::{DsaWeights, Glm5Layer};
use crate::layer::{ForwardContext, GlmSparseMlaLayerState, LayerState};
use crate::layers::ops;

const DSA_LATENT: usize = 512;

mod kda;

impl Glm5Layer {
    pub(super) fn dsa_prefill(
        &self,
        weights: &DsaWeights,
        normed: DevicePtr,
        rows: usize,
        pool_limit: u32,
        states: &[&(dyn LayerState + '_)],
        cu: DevicePtr,
        positions: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            !states.is_empty(),
            "GLM DSA prefill requires sequence state"
        );
        let states = states
            .iter()
            .map(|state| {
                state
                    .as_any()
                    .downcast_ref::<GlmSparseMlaLayerState>()
                    .context("GLM DSA prefill received incompatible state")
            })
            .collect::<Result<Vec<_>>>()?;
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
        let absorbed = ctx.buffers.ssm_deinterleaved();
        ops::glm53_dsa_absorb_query_bf16(
            ctx.gpu,
            self.kernels.dsa_absorb_query,
            &ops::Glm53DsaBf16ProjectionArgs {
                input: query,
                kv_b_weight: weights.kv_b.weight,
                output: absorbed,
                num_tokens: rows as u32,
                num_heads: DSA_HEADS as u32,
            },
            stream,
        )?;

        let needs_sparse_ranking = pool_limit > DSA_TOP_POOLS;
        let index_query = query;
        let legacy_sparse_prefill = std::env::var("ATLAS_GLM_DSA_PREFILL_LEGACY")
            .ok()
            .as_deref()
            == Some("1");
        let tensor_core_prefill = rows > 64
            && !legacy_sparse_prefill
            && std::env::var("ATLAS_GLM_DSA_PREFILL_TC").ok().as_deref() != Some("0");
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
            DSA_LATENT,
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
            DSA_LATENT as u32,
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

        let state_values = states
            .iter()
            .flat_map(|state| {
                [
                    state.current.latent_cache.0,
                    state.current.pooled_keys.0,
                    state.current.tail_keys.0,
                    state.current.tail_gates.0,
                    state.current.tail_metadata.0,
                ]
            })
            .collect::<Vec<_>>();
        let state_table = self.upload_state_table(&state_values, ctx, stream)?;
        let layout = ctx.buffers.glm_layout();
        let workspace = ctx.buffers.glm_workspace();
        let valid = workspace.offset(layout.valid);
        ctx.gpu.memset_async(valid, 1, rows, stream)?;
        let scores = workspace.offset(layout.dsa_scores);
        let selected = workspace.offset(layout.dsa_selected);
        let latent_output = ctx.buffers.attn_output();
        // Persist the full chunk once, then score every query against the
        // position-derived visible prefix. This preserves the token loop's
        // exact causal boundary while replacing ~5*rows tiny launches with
        // five prompt-batched launches per DSA layer.
        ops::glm53_dsa_latent_append(
            ctx.gpu,
            self.kernels.dsa_latent_append,
            &ops::Glm53DsaLatentAppendArgs {
                latent,
                cu_seqlens: cu,
                positions,
                valid,
                sequence_state_ptrs: state_table,
                total_tokens: rows as u32,
                num_sequences: states.len() as u32,
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
                cu_seqlens: cu,
                positions,
                valid,
                sequence_state_ptrs: state_table,
                num_sequences: states.len() as u32,
            },
            stream,
        )?;
        if needs_sparse_ranking {
            ops::glm53_dsa_score_prefill(
                ctx.gpu,
                self.kernels.dsa_score_prefill,
                &ops::Glm53DsaScorePrefillArgs {
                    query: index_query,
                    head_weights: index_weights,
                    positions,
                    valid,
                    sequence_state_ptrs: state_table,
                    cu_seqlens: cu,
                    scores,
                    max_pools: pool_limit,
                    total_tokens: rows as u32,
                    num_sequences: states.len() as u32,
                },
                stream,
            )?;
        }
        if needs_sparse_ranking || legacy_sparse_prefill || tensor_core_prefill {
            ops::glm53_dsa_topk_prefill(
                ctx.gpu,
                self.kernels.dsa_topk_prefill,
                &ops::Glm53DsaTopkPrefillArgs {
                    scores,
                    positions,
                    valid,
                    sequence_state_ptrs: state_table,
                    cu_seqlens: cu,
                    output: selected,
                    max_pools: pool_limit,
                    total_tokens: rows as u32,
                    num_sequences: states.len() as u32,
                },
                stream,
            )?;
            if tensor_core_prefill {
                // The scalar 8-query tile wins before sparse pruning starts;
                // it shares each causal latent across nearby queries without
                // materializing an attention matrix. The tensor-core kernels
                // skip those rows and own only the >2,048-position suffix.
                ops::glm53_dsa_causal_mla_prefill(
                    ctx.gpu,
                    self.kernels.dsa_causal_mla_prefill,
                    &ops::Glm53DsaCausalMlaPrefillArgs {
                        absorbed_query: absorbed,
                        positions,
                        valid,
                        sequence_state_ptrs: state_table,
                        cu_seqlens: cu,
                        output: latent_output,
                        num_heads: DSA_HEADS as u32,
                        total_tokens: rows as u32,
                        num_sequences: states.len() as u32,
                        attention_scale: 0.0625,
                    },
                    stream,
                )?;
                ops::glm53_dsa_prefill_tc(
                    ctx.gpu,
                    self.kernels.dsa_prefill_tc_scores,
                    self.kernels.dsa_prefill_tc_values,
                    &ops::Glm53DsaPrefillTcArgs {
                        query: absorbed,
                        selected_indices: selected,
                        sequence_state_ptrs: state_table,
                        cu_seqlens: cu,
                        scores: workspace.offset(layout.dsa_attention_scores),
                        weights: workspace.offset(layout.dsa_attention_weights),
                        output: latent_output,
                        total_tokens: rows as u32,
                        num_sequences: states.len() as u32,
                        attention_scale: 0.0625,
                    },
                    stream,
                )?;
            } else {
                let args = ops::Glm53DsaSparseMlaPrefillArgs {
                    absorbed_query: absorbed,
                    selected_indices: selected,
                    sequence_state_ptrs: state_table,
                    cu_seqlens: cu,
                    output: latent_output,
                    num_heads: DSA_HEADS as u32,
                    total_tokens: rows as u32,
                    num_sequences: states.len() as u32,
                    attention_scale: 0.0625,
                };
                if legacy_sparse_prefill {
                    ops::glm53_dsa_sparse_mla_prefill(
                        ctx.gpu,
                        self.kernels.dsa_sparse_mla_prefill,
                        &args,
                        stream,
                    )?;
                } else {
                    ops::glm53_dsa_sparse_mla_prefill_warp(
                        ctx.gpu,
                        self.kernels.dsa_sparse_mla_prefill_warp,
                        &args,
                        stream,
                    )?;
                }
            }
        } else {
            ops::glm53_dsa_causal_mla_prefill(
                ctx.gpu,
                self.kernels.dsa_causal_mla_prefill,
                &ops::Glm53DsaCausalMlaPrefillArgs {
                    absorbed_query: absorbed,
                    positions,
                    valid,
                    sequence_state_ptrs: state_table,
                    cu_seqlens: cu,
                    output: latent_output,
                    num_heads: DSA_HEADS as u32,
                    total_tokens: rows as u32,
                    num_sequences: states.len() as u32,
                    attention_scale: 0.0625,
                },
                stream,
            )?;
        }
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
}
