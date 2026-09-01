// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed two-Spark GLM DFlash verifier across concurrent sequences.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::block_mgmt::ensure_blocks_through_decode;
use super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;
use crate::traits::{DflashPrefillResult, Model, SequenceState};

fn take_multi_profile_once() -> bool {
    static PENDING: std::sync::OnceLock<std::sync::atomic::AtomicBool> = std::sync::OnceLock::new();
    let pending = PENDING.get_or_init(|| {
        let enabled = std::env::var("ATLAS_GLM_MULTI_PROFILE_ONCE")
            .is_ok_and(|value| value == "1" || value == "true");
        std::sync::atomic::AtomicBool::new(enabled)
    });
    pending.swap(false, std::sync::atomic::Ordering::Relaxed)
}

impl TransformerModel {
    pub(in crate::model) fn decode_verify_dflash_batched_dispatch(
        &self,
        tokens: &[u32],
        k: usize,
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_dflash_batched_inner(tokens, k, seqs, stream)
    }

    /// Co-dispatch one active DFlash target block and one arbitrary-width
    /// prompt slice. Stateful attention remains lane-private; each layer's
    /// stateless FFN/MoE runs once over the concatenated rows.
    pub(in crate::model) fn decode_verify_dflash_with_prefill_dispatch(
        &self,
        target_tokens: &[u32],
        target_seq: &mut SequenceState,
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_total_len: usize,
        _stream: u64,
    ) -> Result<DflashPrefillResult> {
        let k = target_tokens.len();
        ensure!(
            (2..=spark_runtime::buffers::GLM53_VERIFY_MAX_ROWS).contains(&k),
            "GLM DFlash target width {k} is unsupported"
        );
        ensure!(
            !prefill_tokens.is_empty(),
            "GLM fused prompt slice is empty"
        );
        ensure!(
            prefill_seq.seq_len.saturating_add(prefill_tokens.len()) <= prefill_total_len,
            "GLM DFlash/prefill slice exceeds the prompt"
        );
        let position_start = prefill_seq.seq_len;
        let p = prefill_tokens.len();
        let rows = k + p;
        ensure!(
            rows <= self.buffers.max_batch_tokens(),
            "GLM DFlash/prefill rows exceed the model arena"
        );
        self.ssm_pool.require_verify_rollback_supported()?;

        let stream = self.gpu.default_stream();
        let hidden = self.buffers.hidden_states();
        let h = self.config.hidden_size;
        let mut tokens = Vec::with_capacity(rows);
        tokens.extend_from_slice(target_tokens);
        tokens.extend_from_slice(prefill_tokens);
        for (row, &token) in tokens.iter().enumerate() {
            self.embed(token, hidden.offset(row * h * size_of::<u16>()), stream)?;
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        for (seq, width) in [(&mut *target_seq, k), (&mut *prefill_seq, p)] {
            for token in 0..width {
                let pos = seq.seq_len + token;
                let blocks_needed = pos / bs + 1;
                ensure_blocks_through_decode(
                    seq,
                    blocks_needed - 1,
                    &mut kv_cache,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                    self.levers.kv_poison,
                )?;
            }
        }

        let workspace = self.buffers.glm_workspace();
        let layout = self.buffers.glm_layout();
        let positions_dev = workspace.offset(layout.positions);
        let positions = (0..k)
            .map(|row| (target_seq.seq_len + row) as u32)
            .chain((0..p).map(|row| (prefill_seq.seq_len + row) as u32))
            .collect::<Vec<_>>();
        let position_bytes = positions
            .iter()
            .flat_map(|position| position.to_le_bytes())
            .collect::<Vec<_>>();
        self.gpu
            .copy_h2d_async(&position_bytes, positions_dev, stream)?;
        let metadata = AttnMetadataDev {
            positions: positions_dev,
            positions_h: positions_dev,
            positions_w: positions_dev,
            slot: DevicePtr::NULL,
            seq_len: DevicePtr::NULL,
            block_table: DevicePtr::NULL,
            max_blocks_per_seq: self.max_blocks_per_seq,
            num_seqs: rows as u32,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };
        let verify_seq_len = target_seq.seq_len;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            layer.decode_verify_glm_with_prefill(
                hidden,
                k,
                verify_seq_len,
                target_seq.layer_states[layer_idx].as_mut(),
                p,
                position_start,
                prefill_seq.layer_states[layer_idx].as_mut(),
                &ctx,
                stream,
            )?;
            self.try_dflash_capture_all(layer_idx, k, stream)?;
            self.try_dflash_prefill_capture_from_row_layer(
                prefill_seq,
                layer_idx,
                position_start,
                p,
                k,
                stream,
            )?;
        }

        let normed = self.buffers.norm_output();
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            rows as u32,
            h as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;
        self.lm_head_batched(normed, rows as u32, self.buffers.logits(), stream)?;
        let vocab = self.config.vocab_size;
        let argmax_out = self.buffers.scratch();
        if self.argmax_batch_kernel.0 != 0 {
            ops::argmax_bf16_batch(
                self.gpu.as_ref(),
                self.argmax_batch_kernel,
                self.buffers.logits(),
                argmax_out,
                vocab as u32,
                rows as u32,
                vocab as u32,
                stream,
            )?;
        } else {
            for row in 0..k {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    self.buffers.logits().offset(row * vocab * size_of::<u16>()),
                    argmax_out.offset(row * size_of::<u32>()),
                    vocab as u32,
                    stream,
                )?;
            }
        }
        let mut bytes = vec![0u8; size_of_val(target_tokens)];
        self.gpu.copy_d2h(argmax_out, &mut bytes)?;
        let target_argmax = bytes
            .chunks_exact(size_of::<u32>())
            .map(|value| u32::from_le_bytes(value.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();

        target_seq.tokens.extend_from_slice(target_tokens);
        target_seq.seq_len += k;
        prefill_seq.tokens.extend_from_slice(prefill_tokens);
        prefill_seq.seq_len += p;

        prefill_seq.prompt_len = prefill_total_len;
        prefill_seq.kv_valid_tokens = prefill_seq.seq_len;
        self.update_dflash_ctx_len_after_prefill(prefill_seq, position_start, p)?;

        let prefill_logits = self
            .buffers
            .logits()
            .offset((rows - 1) * self.config.vocab_size * size_of::<u16>());
        Ok(DflashPrefillResult {
            target_argmax,
            prefill_logits,
        })
    }

    fn decode_verify_dflash_batched_inner(
        &self,
        tokens: &[u32],
        k: usize,
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let n = seqs.len();
        ensure!(
            self.config.model_type == "glm5_next",
            "batched DFlash is GLM-only"
        );
        ensure!(
            (2..=spark_runtime::buffers::GLM53_VERIFY_MAX_SEQS).contains(&n),
            "GLM DFlash batch width {n} is unsupported"
        );
        ensure!(
            (2..=spark_runtime::buffers::GLM53_VERIFY_MAX_ROWS).contains(&k),
            "GLM DFlash target width {k} is unsupported"
        );
        let rows = n * k;
        ensure!(tokens.len() == rows, "GLM DFlash token matrix is not n*k");
        ensure!(
            rows <= self.buffers.max_batch_tokens(),
            "GLM DFlash rows exceed the model arena"
        );
        self.ssm_pool.require_verify_rollback_supported()?;

        let stream = self.gpu.default_stream();
        let hidden = self.buffers.hidden_states();
        let h = self.config.hidden_size;
        for (row, &token) in tokens.iter().enumerate() {
            self.embed(token, hidden.offset(row * h * size_of::<u16>()), stream)?;
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let seq_lens = seqs.iter().map(|seq| seq.seq_len).collect::<Vec<_>>();
        for seq in seqs.iter_mut() {
            for token in 0..k {
                let pos = seq.seq_len + token;
                let blocks_needed = pos / bs + 1;
                ensure_blocks_through_decode(
                    seq,
                    blocks_needed - 1,
                    &mut kv_cache,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                    self.levers.kv_poison,
                )?;
            }
        }

        let workspace = self.buffers.glm_workspace();
        let layout = self.buffers.glm_layout();
        let positions_dev = workspace.offset(layout.positions);
        let positions = seqs
            .iter()
            .flat_map(|seq| (0..k).map(move |token| (seq.seq_len + token) as u32))
            .collect::<Vec<_>>();
        let position_bytes = positions
            .iter()
            .flat_map(|position| position.to_le_bytes())
            .collect::<Vec<_>>();
        self.gpu
            .copy_h2d_async(&position_bytes, positions_dev, stream)?;
        let metadata = AttnMetadataDev {
            positions: positions_dev,
            positions_h: positions_dev,
            positions_w: positions_dev,
            slot: DevicePtr::NULL,
            seq_len: DevicePtr::NULL,
            block_table: DevicePtr::NULL,
            max_blocks_per_seq: self.max_blocks_per_seq,
            num_seqs: rows as u32,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let pool_bucket = seq_lens
            .iter()
            .map(|&seq_len| {
                crate::layers::glm5::dsa_verify_pool_bucket(
                    seq_len.saturating_add(k),
                    self.config.index_kpool,
                    layout.dsa_max_pools,
                )
            })
            .max()
            .unwrap_or(1);
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        let ep_graphs =
            std::env::var("ATLAS_EP_GRAPHS").is_ok_and(|value| value == "1" || value == "true");
        let force_eager = std::env::var("ATLAS_NO_DFLASH_MULTI_GRAPH").is_ok()
            || std::env::var("ATLAS_DFLASH_DEBUG_NO_GRAPH").ok().as_deref() == Some("1");
        let profile_this_step = take_multi_profile_once();
        let profile_enabled = self.profile || profile_this_step;
        let use_graphs = (self.comm.is_none() || ep_graphs)
            && !profile_enabled
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager
            && self.lora.is_none();
        if use_graphs {
            self.stage_glm_verify_multi_graph_metadata(seqs, k, stream)?;
        }
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(metadata),
            profile: profile_enabled,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        let graph_key = (n, k, pool_bucket);
        let mut graph_cache = use_graphs.then(|| self.dflash_verify_batched_graphs.lock());
        let cached_graph = graph_cache
            .as_ref()
            .and_then(|cache| cache.get(&graph_key).copied());
        if let Some(graph) = cached_graph {
            self.gpu.launch_graph(graph, stream)?;
        }
        if cached_graph.is_none() {
            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mut states: Vec<&mut (dyn LayerState + 'static)> = seqs
                    .iter_mut()
                    .map(|seq| seq.layer_states[layer_idx].as_mut())
                    .collect();
                layer.decode_verify_glm_multi(hidden, k, &seq_lens, &mut states, &ctx, stream)?;
                drop(states);
                self.try_dflash_capture_all(layer_idx, rows, stream)?;
            }

            let normed = self.buffers.norm_output();
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                rows as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;
            self.lm_head_batched(normed, rows as u32, self.buffers.logits(), stream)?;
            let vocab = self.config.vocab_size;
            let argmax_out = self.buffers.scratch();
            if self.argmax_batch_kernel.0 != 0 {
                ops::argmax_bf16_batch(
                    self.gpu.as_ref(),
                    self.argmax_batch_kernel,
                    self.buffers.logits(),
                    argmax_out,
                    vocab as u32,
                    rows as u32,
                    vocab as u32,
                    stream,
                )?;
            } else {
                for row in 0..rows {
                    ops::argmax_bf16(
                        self.gpu.as_ref(),
                        self.argmax_kernel,
                        self.buffers.logits().offset(row * vocab * size_of::<u16>()),
                        argmax_out.offset(row * size_of::<u32>()),
                        vocab as u32,
                        stream,
                    )?;
                }
            }
            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                ensure!(
                    graph.0 != 0,
                    "GLM concurrent DFlash capture returned no graph"
                );
                tracing::info!(
                    "Captured GLM concurrent DFlash graph n={n} K={k} pools={pool_bucket}"
                );
                graph_cache
                    .as_mut()
                    .expect("graph cache exists while capture is enabled")
                    .insert(graph_key, graph);
                self.gpu.launch_graph(graph, stream)?;
            }
        }
        let argmax_out = self.buffers.scratch();
        let mut bytes = vec![0u8; rows * size_of::<u32>()];
        self.gpu.copy_d2h(argmax_out, &mut bytes)?;
        let verdicts = bytes
            .chunks_exact(4)
            .map(|value| u32::from_le_bytes(value.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();

        for (sequence, seq) in seqs.iter_mut().enumerate() {
            let start = sequence * k;
            seq.tokens.extend_from_slice(&tokens[start..start + k]);
            seq.seq_len += k;
        }
        Ok(verdicts)
    }
}
