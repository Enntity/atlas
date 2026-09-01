// SPDX-License-Identifier: AGPL-3.0-only

//! Batched-projection, causally ordered sparse-MLA target verification.

use anyhow::{Context, Result, ensure};
use spark_runtime::buffers::GLM53_VERIFY_MAX_ROWS;
use spark_runtime::gpu::DevicePtr;

use super::decode::{
    DSA_HEADS, DSA_QUERY_WIDTH, DSA_TOP_POOLS, INDEX_DIM, INDEX_HEADS, INDEX_QUERY_WIDTH,
};
use super::types::{DsaWeights, Glm5Layer};
use crate::layer::{ForwardContext, GlmSparseMlaLayerState, LayerState};
use crate::layers::ops;

impl Glm5Layer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dsa_forward_verify(
        &self,
        weights: &DsaWeights,
        normed: DevicePtr,
        rows: usize,
        pool_limit: u32,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            rows <= GLM53_VERIFY_MAX_ROWS,
            "GLM DSA verify width {rows} exceeds {GLM53_VERIFY_MAX_ROWS}"
        );
        let metadata = ctx
            .attn_metadata
            .context("GLM DSA verify requires staged positions")?;
        ensure!(
            metadata.num_seqs as usize >= rows,
            "GLM DSA verify position batch is too small"
        );
        let state = state
            .as_any()
            .downcast_ref::<GlmSparseMlaLayerState>()
            .context("GLM DSA verify received incompatible state")?;
        let current = state.current;
        let snapshot_count = rows.saturating_sub(1);
        ensure!(
            state.intermediates.len() >= snapshot_count,
            "GLM DSA verify needs {snapshot_count} rollback images but has {}",
            state.intermediates.len()
        );
        let snapshots = state.intermediates[..snapshot_count].to_vec();
        let state_table = self.upload_state_table(
            &[
                current.latent_cache.0,
                current.pooled_keys.0,
                current.tail_keys.0,
                current.tail_gates.0,
                current.tail_metadata.0,
            ],
            ctx,
            stream,
        )?;

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

        let index_key = query.offset(rows * INDEX_QUERY_WIDTH * size_of::<u16>());
        let index_gates = index_key.offset(rows * INDEX_DIM * size_of::<u16>());
        let index_weights = index_gates.offset(rows * INDEX_DIM * size_of::<u16>());
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

        let workspace = ctx.buffers.glm_workspace();
        let layout = ctx.buffers.glm_layout();
        let cu_seqlens = workspace.offset(layout.cu_seqlens_i32);
        let valid = workspace.offset(layout.valid);
        if !ctx.graph_capture {
            let cu = [0i32, 1i32]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>();
            ctx.gpu.copy_h2d_async(&cu, cu_seqlens, stream)?;
            ctx.gpu.memset_async(valid, 1, 1, stream)?;
        }
        let scores = workspace.offset(layout.dsa_scores);
        let selected = workspace.offset(layout.dsa_selected);
        let latent_output = ctx.buffers.attn_output();
        for row in 0..rows {
            let latent_row = latent.offset(row * 512 * size_of::<u16>());
            let key_row = index_key.offset(row * INDEX_DIM * size_of::<u16>());
            let gates_row = index_gates.offset(row * INDEX_DIM * size_of::<u16>());
            let absorbed_row = absorbed_query.offset(row * DSA_HEADS * 512 * size_of::<u16>());
            let output_row = latent_output.offset(row * DSA_HEADS * 512 * size_of::<u16>());
            ops::glm53_dsa_latent_append(
                ctx.gpu,
                self.kernels.dsa_latent_append,
                &ops::Glm53DsaLatentAppendArgs {
                    latent: latent_row,
                    cu_seqlens,
                    positions: metadata.positions.offset(row * size_of::<u32>()),
                    valid,
                    sequence_state_ptrs: state_table,
                    total_tokens: 1,
                    num_sequences: 1,
                    latent_capacity: layout.latent_capacity as u32,
                },
                stream,
            )?;
            ops::glm53_dsa_pool_append(
                ctx.gpu,
                self.kernels.dsa_pool,
                &ops::Glm53DsaPoolAppendArgs {
                    keys: key_row,
                    gates: gates_row,
                    ape: weights.index_ape.weight,
                    cu_seqlens,
                    positions: metadata.positions.offset(row * size_of::<u32>()),
                    valid,
                    sequence_state_ptrs: state_table,
                    num_sequences: 1,
                },
                stream,
            )?;
            if needs_sparse_ranking {
                ops::glm53_dsa_score(
                    ctx.gpu,
                    self.kernels.dsa_score,
                    &ops::Glm53DsaScoreArgs {
                        query: index_query.offset(row * INDEX_QUERY_WIDTH * size_of::<u16>()),
                        head_weights: index_weights.offset(row * INDEX_HEADS * size_of::<u16>()),
                        query_valid: valid,
                        sequence_state_ptrs: state_table,
                        scores,
                        max_pools: pool_limit,
                        num_sequences: 1,
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
                    num_sequences: 1,
                },
                stream,
            )?;
            ops::glm53_dsa_sparse_mla_decode(
                ctx.gpu,
                self.kernels.dsa_sparse_mla,
                &ops::Glm53DsaSparseMlaArgs {
                    absorbed_query: absorbed_row,
                    selected_indices: selected,
                    sequence_state_ptrs: state_table,
                    output: output_row,
                    num_heads: DSA_HEADS as u32,
                    num_sequences: 1,
                    attention_scale: 0.0625,
                },
                stream,
            )?;
            if let Some(&snapshot) = snapshots.get(row) {
                super::verify_state::copy_dsa_tail(current, snapshot, ctx, stream)?;
            }
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
