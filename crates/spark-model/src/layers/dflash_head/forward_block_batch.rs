// SPDX-License-Identifier: AGPL-3.0-only

//! Native one-to-four-stream DFlash2 block forward for the fixed GLM appliance.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{BlockDiffusionDraftHead, DflashQuantization};
use crate::layer::ForwardContext;

impl BlockDiffusionDraftHead {
    pub(super) fn forward_block_batch(
        &self,
        last_tokens: &[u32],
        positions: &[usize],
        paged: &[(DevicePtr, u32)],
        gamma: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        use crate::layers::ops;

        let n = last_tokens.len();
        ensure!(
            (1..=4).contains(&n) && positions.len() == n && paged.len() == n,
            "DFlash2 batch must contain one to four matched sequences"
        );
        ensure!(
            matches!(self.quant, DflashQuantization::Bf16),
            "fixed GLM DFlash batching requires BF16 drafter weights"
        );
        ensure!(
            (2..=self.gamma).contains(&gamma),
            "DFlash2 batch rows must fit the configured gamma"
        );
        let rows = n * gamma;
        let rows_u32 = rows as u32;
        let g = gamma as u32;
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let bf16 = size_of::<u16>();
        let gpu = ctx.gpu;
        let inverse_sqrt_d = 1.0f32 / (self.head_dim as f32).sqrt();

        let positions_host = positions
            .iter()
            .flat_map(|&position| (0..gamma).map(move |row| (position + row) as i32))
            .collect::<Vec<_>>();
        let position_bytes = positions_host
            .iter()
            .flat_map(|position| position.to_le_bytes())
            .collect::<Vec<_>>();
        gpu.copy_h2d(&position_bytes, self.scratch.position_ids)?;
        let token_ids = last_tokens
            .iter()
            .flat_map(|&token| {
                std::iter::once(token as i32)
                    .chain(std::iter::repeat_n(self.mask_token_id as i32, gamma - 1))
            })
            .collect::<Vec<_>>();
        let token_bytes = token_ids
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        gpu.copy_h2d(&token_bytes, self.scratch.draft_tokens_dev)?;

        // Every request-dependent value is staged before graph capture. The
        // captured region may consume only stable device addresses; a blocking
        // H2D inside capture is unsupported and would invalidate the stream.
        for (sequence, &(block_table, context_count)) in paged.iter().enumerate() {
            let row_base = sequence * gamma;
            ops::fill_slots_from_block_table(
                gpu,
                self.kernels.fill_slots,
                self.scratch
                    .slot_mapping_dev
                    .offset(row_base * size_of::<i64>()),
                block_table,
                context_count,
                g,
                16,
                stream,
            )?;
            let mut indirect = [0u8; 16];
            indirect[0..4].copy_from_slice(&(context_count + g).to_ne_bytes());
            indirect[4..8].copy_from_slice(&context_count.to_ne_bytes());
            indirect[8..12].copy_from_slice(&(positions[sequence] as u32).to_ne_bytes());
            indirect[12..16].copy_from_slice(&last_tokens[sequence].to_ne_bytes());
            gpu.copy_h2d(
                &indirect,
                self.scratch
                    .option_b_indirect_args_dev
                    .offset(sequence * 16),
            )?;
        }

        let run_body = || -> Result<()> {
            ops::batched_embed(
                gpu,
                self.kernels.batched_embed,
                self.scratch.draft_tokens_dev,
                self.embed_tokens_shared,
                self.scratch.stream_buf,
                rows_u32,
                h,
                stream,
            )?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                self.forward_block_batch_pre(layer, rows_u32, g, h, q_dim, kv_dim, ctx, stream)?;
                let (k_pool, v_pool) = {
                    let cache = self.kv_cache.lock();
                    (cache.k_pool_ptr(layer_idx), cache.v_pool_ptr(layer_idx))
                };
                ops::reshape_and_cache(
                    gpu,
                    self.kernels.reshape_cache_bf16,
                    self.scratch.k_buf,
                    self.scratch.v_buf,
                    k_pool,
                    v_pool,
                    self.scratch.slot_mapping_dev,
                    rows_u32,
                    self.num_kv_heads as u32,
                    self.head_dim as u32,
                    16,
                    kv_dim,
                    kv_dim,
                    0,
                    stream,
                )?;
                for (sequence, &(block_table, _)) in paged.iter().enumerate() {
                    let row_base = sequence * gamma;
                    ops::prefill_attention_paged_dflash_bf16_indirect(
                        gpu,
                        self.kernels.prefill_attn_dflash_bf16_indirect,
                        self.scratch.q_buf.offset(row_base * q_dim as usize * bf16),
                        k_pool,
                        v_pool,
                        self.scratch
                            .attn_out
                            .offset(row_base * q_dim as usize * bf16),
                        block_table,
                        g,
                        self.scratch
                            .option_b_indirect_args_dev
                            .offset(sequence * 16),
                        self.num_q_heads as u32,
                        self.num_kv_heads as u32,
                        self.head_dim as u32,
                        16,
                        self.window_size.unwrap_or(0) as u32,
                        inverse_sqrt_d,
                        stream,
                    )?;
                }
                self.forward_block_batch_post(layer, rows_u32, g, h, q_dim, inter, ctx, stream)?;
            }

            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.stream_buf,
                &self.norm,
                self.scratch.norm_buf,
                rows_u32,
                h,
                self.rms_norm_eps,
                stream,
            )?;
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                self.scratch.norm_buf,
                &crate::weight_map::DenseWeight {
                    weight: self.lm_head_shared,
                },
                self.scratch.logits,
                rows_u32,
                self.vocab_size as u32,
                h,
                stream,
            )?;
            if let Some(selector) = self.dflash2_selector.as_ref() {
                ops::dense_gemm_bf16_pipelined(
                    gpu,
                    self.kernels.dense_gemm_pipelined,
                    self.scratch.norm_buf,
                    &selector.hidden_projection,
                    self.scratch.selector_hidden,
                    rows_u32,
                    self.dflash2_selector_rank as u32,
                    h,
                    stream,
                )?;
                for sequence in 0..n {
                    let row_base = sequence * gamma;
                    let first_draft_row = row_base + 1;
                    ops::dflash2_select_path16(
                        gpu,
                        self.kernels.dflash2_select_path,
                        self.scratch
                            .logits
                            .offset(first_draft_row * self.vocab_size * bf16),
                        self.scratch
                            .selector_hidden
                            .offset(first_draft_row * self.dflash2_selector_rank * bf16),
                        selector.predecessor_codebook.weight,
                        selector.successor_codebook.weight,
                        self.scratch
                            .draft_tokens_dev
                            .offset(first_draft_row * size_of::<u32>()),
                        self.scratch
                            .selector_candidate_ids
                            .offset(sequence * (gamma - 1) * 16 * size_of::<u32>()),
                        self.scratch
                            .selector_edge_scores
                            .offset(sequence * (gamma - 1) * 16 * 16 * size_of::<f32>()),
                        g - 1,
                        self.vocab_size as u32,
                        self.dflash2_selector_rank as u32,
                        self.scratch
                            .option_b_indirect_args_dev
                            .offset(sequence * 16 + 12),
                        stream,
                    )?;
                }
            } else {
                for row in 0..rows {
                    ops::argmax_bf16(
                        gpu,
                        self.kernels.argmax,
                        self.scratch.logits.offset(row * self.vocab_size * bf16),
                        self.scratch.draft_tokens_dev.offset(row * size_of::<u32>()),
                        self.vocab_size as u32,
                        stream,
                    )?;
                }
            }
            Ok(())
        };

        let graph_eligible = !self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
            && std::env::var("ATLAS_DFLASH_PROPOSE_NO_GRAPH").is_err();
        let warmup_target = std::env::var("ATLAS_DFLASH_PROPOSE_WARMUP_N")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2);
        if graph_eligible {
            let graph_key = (n, gamma);
            let mut graphs = self.propose_batch_graphs.lock();
            if let Some(&graph) = graphs.get(&graph_key) {
                if graph.0 != 0 {
                    gpu.launch_graph(graph, stream)?;
                } else {
                    run_body()?;
                }
            } else {
                let mut warmups = self.propose_batch_warmups.lock();
                let warmed = warmups.entry(graph_key).or_default();
                if *warmed < warmup_target {
                    *warmed += 1;
                    run_body()?;
                } else {
                    gpu.begin_capture(stream)?;
                    if let Err(error) = run_body() {
                        gpu.abort_capture_if_active(stream);
                        return Err(error).context("capturing the full DFlash batch graph");
                    }
                    let graph = match gpu.end_capture(stream) {
                        Ok(graph) => graph,
                        Err(error) => {
                            gpu.abort_capture_if_active(stream);
                            return Err(error)
                                .context("ending the full DFlash batch graph capture");
                        }
                    };
                    if graph.0 != 0 {
                        gpu.launch_graph(graph, stream)?;
                        tracing::info!("DFlash full batch graph captured for N={n} gamma={gamma}");
                    } else {
                        run_body()?;
                    }
                    graphs.insert(graph_key, graph);
                }
            }
        } else {
            run_body()?;
        }

        let pinned_ptr = self
            .scratch
            .draft_tokens_host_pinned
            .load(std::sync::atomic::Ordering::Relaxed);
        ensure!(
            !pinned_ptr.is_null(),
            "DFlash batch draft-token staging buffer is null"
        );
        // SAFETY: from_weights allocates `4 * gamma * sizeof(u32)` bytes and
        // this path admits at most four sequences. The proposer is driven by
        // one scheduler lane, so no second proposal aliases the staging span.
        let bytes = unsafe { std::slice::from_raw_parts_mut(pinned_ptr, rows * size_of::<u32>()) };
        gpu.copy_d2h_async(self.scratch.draft_tokens_dev, bytes, stream)?;
        gpu.record_event(self.scratch.draft_tokens_event, stream)?;
        gpu.event_synchronize(self.scratch.draft_tokens_event)?;
        let all_tokens = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 draft")))
            .collect::<Vec<_>>();
        Ok((0..n)
            .map(|sequence| {
                let start = sequence * gamma;
                all_tokens[start + 1..start + gamma].to_vec()
            })
            .collect())
    }
}
