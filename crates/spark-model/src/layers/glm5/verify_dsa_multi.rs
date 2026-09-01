// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence GLM sparse-MLA verification.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::decode::{
    DSA_HEADS, DSA_QUERY_WIDTH, DSA_TOP_POOLS, INDEX_DIM, INDEX_HEADS, INDEX_QUERY_WIDTH,
};
use super::types::{DsaWeights, Glm5Layer};
use crate::layer::{ForwardContext, GlmSparseMlaLayerState, LayerState};
use crate::layers::ops;

impl Glm5Layer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dsa_forward_verify_multi(
        &self,
        weights: &DsaWeights,
        normed: DevicePtr,
        rows_per_seq: usize,
        pool_limits: &[u32],
        states: &mut [&mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let num_sequences = states.len();
        let rows = num_sequences.saturating_mul(rows_per_seq);
        ensure!(
            num_sequences > 0 && pool_limits.len() == num_sequences,
            "GLM DSA multi-verify state/limit mismatch"
        );
        ensure!(
            rows_per_seq > 0 && rows <= spark_runtime::buffers::GLM53_VERIFY_MAX_BATCH_ROWS,
            "GLM DSA verify batch shape {num_sequences}x{rows_per_seq} is unsupported"
        );
        let metadata = ctx
            .attn_metadata
            .context("GLM DSA multi-verify requires staged positions")?;
        ensure!(
            metadata.num_seqs as usize >= rows,
            "GLM DSA multi-verify position batch is too small"
        );

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

        let needs_sparse_ranking = pool_limits.iter().any(|&limit| limit > DSA_TOP_POOLS);
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
        let valid = workspace.offset(layout.valid);
        if !ctx.graph_capture {
            ctx.gpu.memset_async(valid, 1, num_sequences, stream)?;
        }
        let scores = workspace.offset(layout.dsa_scores);
        let selected = workspace.offset(layout.dsa_selected);
        let latent_output = ctx.buffers.attn_output();
        let snapshot_count = rows_per_seq.saturating_sub(1);

        // One layer-wide pointer image, staged once. The old loop uploaded
        // the same fixed workspace address once per sequence and relied on
        // stream ordering to swap its contents between tiny kernels. Besides
        // N H2D operations per DSA layer/step, that shape could never be
        // captured: a graph replay would see only the last sequence's table.
        let mut state_values =
            Vec::with_capacity(num_sequences * (5 + snapshot_count.saturating_mul(3)));
        for state in states.iter() {
            let state = state
                .as_any()
                .downcast_ref::<GlmSparseMlaLayerState>()
                .context("GLM DSA multi-verify received incompatible state")?;
            ensure!(
                state.intermediates.len() >= snapshot_count,
                "GLM DSA multi-verify needs {snapshot_count} rollback images but has {}",
                state.intermediates.len()
            );
            let current = state.current;
            state_values.extend([
                current.latent_cache.0,
                current.pooled_keys.0,
                current.tail_keys.0,
                current.tail_gates.0,
                current.tail_metadata.0,
            ]);
        }
        // Snapshot destinations follow all current images. The CUDA snapshot
        // kernel addresses them as [depth, sequence, key/gate/metadata], so
        // one N-wide launch replaces 3*N copy-engine submissions per depth.
        for depth in 0..snapshot_count {
            for state in states.iter() {
                let state = state
                    .as_any()
                    .downcast_ref::<GlmSparseMlaLayerState>()
                    .context("GLM DSA multi-verify received incompatible state")?;
                let snapshot = state.intermediates[depth];
                state_values.extend([
                    snapshot.tail_keys.0,
                    snapshot.tail_gates.0,
                    snapshot.tail_metadata.0,
                ]);
            }
        }
        let state_tables = self.upload_state_table(&state_values, ctx, stream)?;
        let max_pools = pool_limits.iter().copied().max().unwrap_or(1);
        let (rows_per_sequence, num_sequences_u32) = (rows_per_seq as u32, num_sequences as u32);
        // Latents are append-only and visibility is controlled by the causal
        // pool/tail metadata below, so all K rows can be copied in one launch.
        ops::glm53_dsa_verify_multi_latent_append(
            ctx.gpu,
            self.kernels.dsa_verify_multi_latent_append,
            &ops::Glm53DsaVerifyMultiLatentArgs {
                latent,
                positions: metadata.positions,
                sequence_state_ptrs: state_tables,
                rows_per_sequence,
                num_sequences: num_sequences_u32,
                latent_capacity: layout.latent_capacity as u32,
            },
            stream,
        )?;
        for token in 0..rows_per_seq {
            let token_depth = token as u32;
            ops::glm53_dsa_verify_multi_pool_append(
                ctx.gpu,
                self.kernels.dsa_verify_multi_pool,
                &ops::Glm53DsaVerifyMultiPoolArgs {
                    keys: index_key,
                    gates: index_gates,
                    ape: weights.index_ape.weight,
                    positions: metadata.positions,
                    sequence_state_ptrs: state_tables,
                    rows_per_sequence,
                    num_sequences: num_sequences_u32,
                    token_depth,
                },
                stream,
            )?;
            if needs_sparse_ranking {
                ops::glm53_dsa_verify_multi_score(
                    ctx.gpu,
                    self.kernels.dsa_verify_multi_score,
                    &ops::Glm53DsaVerifyMultiScoreArgs {
                        query: index_query,
                        head_weights: index_weights,
                        sequence_state_ptrs: state_tables,
                        scores,
                        max_pools,
                        rows_per_sequence,
                        num_sequences: num_sequences_u32,
                        token_depth,
                    },
                    stream,
                )?;
            }
            if needs_sparse_ranking {
                ops::glm53_dsa_topk_expand_decode(
                    ctx.gpu,
                    self.kernels.dsa_topk,
                    &ops::Glm53DsaTopkArgs {
                        scores,
                        query_valid: valid,
                        sequence_state_ptrs: state_tables,
                        output: selected,
                        max_pools,
                        num_sequences: num_sequences_u32,
                    },
                    stream,
                )?;
            }
            ops::glm53_dsa_verify_multi_sparse_mla(
                ctx.gpu,
                self.kernels.dsa_verify_multi_sparse_mla,
                &ops::Glm53DsaVerifyMultiSparseArgs {
                    absorbed_query,
                    selected_indices: selected,
                    sequence_state_ptrs: state_tables,
                    output: latent_output,
                    num_heads: DSA_HEADS as u32,
                    rows_per_sequence,
                    num_sequences: num_sequences_u32,
                    token_depth,
                    attention_scale: 0.0625,
                    direct_selection: !needs_sparse_ranking,
                },
                stream,
            )?;
            // Pool append also snapshots its completed tail/metadata for all
            // non-final depths, eliminating one extra launch per draft row.
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
