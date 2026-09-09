// SPDX-License-Identifier: AGPL-3.0-only

//! K=γ (DFlash) verify path.
//! H2D POD byte slices follow the safety contract documented in `verify_c.rs`.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use atlas_core::config::LayerType;
use std::time::Instant;

use super::super::block_mgmt::ensure_blocks_through_decode;
use super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

#[path = "verify_d_oracle.rs"]
mod oracle;

impl TransformerModel {
    pub(super) fn decode_verify_graphed_kgamma_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        if self.lightning_dspark_identity.policy().is_some()
            && std::env::var("ATLAS_LIGHTNING_VERIFY_SERIAL_M1").as_deref() == Ok("1")
        {
            return self.decode_verify_serial_m1_dispatch(tokens, seq, _stream);
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;

        // Item #2 (STree-style in-place K=γ verify): `h_state` IS canonical
        // — the verify kernel reads/writes it directly and the commit
        // (`commit_accepted_prefix`) rewinds it in place on reject. No
        // scratch/canonical split — dual-buffer pre-verify copy eliminated.
        // Modeled on verify_b.rs (K=2 in-place).

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        let paired = self.paired_allocate_target(seq, &mut kv_cache, k, stream)?;
        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──

        // 1a. Embed K tokens
        for t in 0..k {
            self.embed(tokens[t], hidden.offset(t * h * fp32), stream)?;
        }

        // 1b. Allocate KV blocks for all K positions
        let bs = kv_cache.block_size();
        for t in 0..if paired { 0 } else { k } {
            let pos = seq.seq_len + t;
            let blocks_needed = (pos / bs) + 1;
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

        // 1c. Upload K-entry attention metadata. Layout in scratch (after
        // mtp metadata reservation): positions[K*4] | slots[K*8] | seq_lens[K*4]
        // | block_table[K*max_blocks*4]. Need K*16 + K*max_blocks*4 bytes per
        // call — at K=17 max_blocks=512 that's ~36 KB which fits comfortably
        // in the scratch arena (offset 32768).
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        let positions: Vec<u32> = (0..k).map(|t| (seq.seq_len + t) as u32).collect();
        // SAFETY: `positions` is built one line above by `(0..k).map(..)
        // .collect()`, so `positions.len() == k` exactly (collect on a
        // `Range` yields one element per step) — `k * 4 == size_of_val(&
        // positions[..])`. Every element is written by the collect, so no
        // uninitialised spare capacity is read. `u32` is POD.
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let mut slots = vec![0i64; k];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = self.paired_physical_block(seq, block_idx, paired)?;
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        // 256-byte gap mirrors K=4 layout for ABI compatibility with
        // attention kernels that index meta_base + fixed offsets.
        // SAFETY: `slots` is `vec![0i64; k]`, so its LEN (not merely its
        // capacity) is `k` and every element is zero-initialised before the
        // `for t in 0..k` loop overwrites it — `k * 8 == size_of_val(&
        // slots[..])`, with no read past `len` into spare capacity.
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let seq_lens: Vec<i32> = (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect();
        // SAFETY: `seq_lens` is `(0..k).map(..).collect()` on the line above,
        // so `seq_lens.len() == k` and `k * 4 == size_of_val(&seq_lens[..])`;
        // all `k` elements are initialised by the collect. `i32` is POD.
        let sl_bytes = unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        let mb = max_blocks as usize;
        let needed = k * mb;
        let mut bt_buf = vec![0i32; needed];
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        // SAFETY: `bt_buf` is `vec![0i32; needed]` on the line above, so its
        // LEN is `needed` and `needed * 4 == size_of_val(&bt_buf[..])` — the
        // read stops at `len`, never in the `Vec`'s spare capacity. The
        // zero-init at construction covers the tail the `for row in 0..k`
        // fill leaves untouched when `block_table.len() < mb`.
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        // Request-scoped LoRA routing (graphed γ-verify) — see verify_b.rs. One
        // sequence → one adapter; [K]-all-equal buffer at the +128 gap, uploaded
        // pre-`begin_capture`. γ spec depth MUST stay ≤ 32 or +128+K*4 would
        // overrun slot@+256. `DevicePtr(0)` (no pool) → installed-pair path.
        debug_assert!(k <= 32, "γ verify seq_slot +128 gap holds K ≤ 32");
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, k, meta_base.offset(128), stream)?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
            seq_slot,
            moe_row_adapter: spark_runtime::gpu::DevicePtr::NULL,
        };

        // Phase 6.2.c — HSS host I/O is illegal under CUDA graph capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_DFLASH_DEBUG_NO_GRAPH=1 forces eager (no graph capture) so
        // CUDA_LAUNCH_BLOCKING=1 reports the exact failing kernel — used
        // to localize K=γ illegal-address crashes downstream of SSM.
        // Product Lightning serves froze this switch at admission: the
        // admitted policy rejects any presence of the variable, so the
        // product path never consults the environment here. Generic and
        // diagnostic serves keep the legacy read.
        let force_eager = if self.lightning_dspark_identity.policy().is_some() {
            false
        } else {
            std::env::var("ATLAS_DFLASH_DEBUG_NO_GRAPH").ok().as_deref() == Some("1")
        };
        // ATLAS_LORA_EAGER: LoRA graph-vs-eager debugging hatch (see decode_a).
        let lora_eager = self.lora.is_some() && self.levers.lora_eager;
        // Host-maintained per-layer state must not freeze during graph replay.
        let layer_veto = self.layers.iter().any(|l| l.decode_graph_unsupported());
        // Exact GLM K=5 on two ranks is pointer- and shape-static. Both ranks
        // enter the same layer collectives in lockstep, so recent CUDA/NCCL
        // stacks can capture this forward. Keep distributed capture opt-in;
        // all other models and topologies retain the eager default.
        static GLM_TP_VERIFY_GRAPH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let glm_tp_graphs = self.config.model_type == "glm5_next"
            && k == 5
            && self.config.tp_world_size == 2
            && *GLM_TP_VERIFY_GRAPH.get_or_init(|| {
                std::env::var("ATLAS_GLM_TP_VERIFY_GRAPH")
                    .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            });
        let use_graphs = (self.comm.is_none() || glm_tp_graphs)
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager
            && !super::verify_layer_trace::enabled()
            && !lora_eager
            && !layer_veto;
        // The optional M16 oracle must never perform D2H inside capture.
        crate::layers::moe::validate_m16_gate_up_graphs(&self.config.model_type, use_graphs)?;
        crate::layers::moe::validate_shared_fp8_cache_graphs(&self.config.model_type, use_graphs)?;
        crate::layers::moe::validate_m5_projection_graphs(&self.config.model_type, k, use_graphs)?;
        let verify_profile = std::env::var("ATLAS_GLM_VERIFY_PROFILE").ok().as_deref() == Some("1")
            && self.config.model_type == "glm5_next"
            && !use_graphs;

        // PLE's host half (n-gram hash + NVMe fault-in + slot upload) for the
        // WHOLE draft window, hoisted before capture/replay exactly as
        // `decode_a` hoists the single decode token. Without it the verify
        // forward has no staging to consume and falls back to a D2H readback,
        // which invalidates a recording graph (901) — the "PLE: no
        // host_token_ids ... capture-unsupported" refusal. #753 item B.
        for (li, l) in self.layers.iter().enumerate() {
            l.verify_prestage(
                tokens,
                seq.layer_states[li].as_mut(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(metadata),
            profile: verify_profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: Some(tokens),
            routed_lora_layers: None, // #30: decode/verify never routes prefill.
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(), // route-aware: base(Skip) decodes; adapter refuses
        };

        // ── Phase 2: CUDA graph capture / replay ──

        let mut graph_cache = if use_graphs {
            Some(self.verify_kgamma_graph.lock())
        } else {
            None
        };

        let cache_key = (seq.slot_idx, k);
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&cache_key).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            self.gpu.launch_graph(graph, stream)?;
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            let mut attn_us = 0u128;
            let mut kda_us = 0u128;
            let mut attn_layers = 0usize;
            let mut kda_layers = 0usize;
            // Product Lightning serves carry force_eager=false from the
            // frozen admission, so this timing hatch only arms on
            // diagnostic/generic serves (it requires eager anyway).
            let time_layers = force_eager
                && std::env::var("ATLAS_DFLASH_LAYER_TIMING").ok().as_deref() == Some("1");
            let mut t_attn = 0u128;
            let mut t_moe = 0u128;
            let mut t_lin = 0u128;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);
                let layer_started = if verify_profile {
                    self.gpu.synchronize(stream)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                if time_layers {
                    self.gpu.synchronize(stream)?;
                }
                let t0 = Instant::now();

                if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // HSS path: decode_multi_seq's paged-decode kernel
                        // reads K/V from HBM only, missing the long-context
                        // history on disk. Fall back to decode_batched
                        // (sequential single-token decodes via the HSS
                        // orchestrator). See verify_b.rs for full rationale.
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else {
                        // k ROWS of ONE sequence, not k sequences: per-sequence
                        // aux state (the QSA indexer) must advance once per row
                        // against this sequence's own state. See
                        // `decode_multi_seq_rows`.
                        let mut seq_state_arr: [&mut (dyn LayerState + 'static); 1] =
                            [seq.layer_states[layer_idx].as_mut()];
                        let row_owner = vec![0usize; k];
                        layer.decode_multi_seq_rows(
                            hidden,
                            residual,
                            k,
                            &mut seq_state_arr,
                            &row_owner,
                            &mut kv_cache,
                            &seq_lens_vec,
                            &block_tables_vec,
                            &ctx,
                            stream,
                        )?;
                    }
                } else {
                    layer.decode_batched(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
                self.trace_lightning_hidden_rows("k4", seq.seq_len, layer_idx, hidden, k, stream)?;
                // DFlash intermediate hidden capture: snapshot each capture
                // layer's output at position k-1 (last verify token) into
                // dflash_hidden_save[slot] while hidden_states still holds
                // this layer's activation — mirrors verify_b.rs for K=2.
                // Must be inside the graph capture region so the per-layer
                // intermediate (not the final-layer-only post-loop value) is
                // recorded. Under ATLAS_DFLASH_EAGLE_FIX=1 OR
                // ATLAS_DFLASH_UNIFIED_CTX=1, capture ALL k verify rows so
                // the scheduler can append rows 0..=num_accepted to ctx
                // after the accept walk (EAGLE order). UNIFIED_CTX requires
                // the same full capture: commit_ctx copies scratch rows
                // 0..=num_accepted — with only the k-1 capture, row 0 holds
                // the WRONG token's hidden and rows 1.. are stale garbage
                // (2026-07-09 accept-collapse root cause: EAGLE_FIX=0 under
                // UNIFIED=1 starved this capture and poisoned drafter ctx).
                // Always capture every verify row. commit_ctx copies
                // 0..=num_accepted; capturing only k-1 poisons the next
                // propose (2026-07-09 accept-collapse). Opt out with
                // ATLAS_DFLASH_CAPTURE_LAST_ONLY=1 for ablation.
                // Ablation only: product Lightning serves never arm this
                // (the admitted policy freezes the diagnostic surface).
                let capture_last_only = self.lightning_dspark_identity.policy().is_none()
                    && std::env::var("ATLAS_DFLASH_CAPTURE_LAST_ONLY")
                        .ok()
                        .as_deref()
                        == Some("1");
                if capture_last_only {
                    self.try_dflash_capture(layer_idx, k - 1, stream)?;
                } else {
                    self.try_dflash_capture_all(layer_idx, k, stream)?;
                }
                if let Some(started) = layer_started {
                    self.gpu.synchronize(stream)?;
                    let elapsed = started.elapsed().as_micros();
                    if layer_type == LayerType::FullAttention {
                        attn_us += elapsed;
                        attn_layers += 1;
                    } else {
                        kda_us += elapsed;
                        kda_layers += 1;
                    }
                }
                if time_layers {
                    self.gpu.synchronize(stream)?;
                    let dt = t0.elapsed().as_micros();
                    match layer_type {
                        LayerType::FullAttention | LayerType::SlidingAttention => t_attn += dt,
                        LayerType::Moe => t_moe += dt,
                        LayerType::LinearAttention => t_lin += dt,
                    }
                }
            }

            if verify_profile {
                tracing::info!(
                    "GLM Kgamma layer profile K={k}: kda={:.2}ms({}L) mla={:.2}ms({}L)",
                    kda_us as f64 / 1000.0,
                    kda_layers,
                    attn_us as f64 / 1000.0,
                    attn_layers,
                );
            }
            if time_layers {
                tracing::info!(
                    "DFLASH LAYER_TIMING K={k}: attn={:.1}ms moe={:.1}ms mamba={:.1}ms",
                    t_attn as f64 / 1000.0,
                    t_moe as f64 / 1000.0,
                    t_lin as f64 / 1000.0
                );
            }

            // Final norm [K, H]
            let normed = self.buffers.norm_output();
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                k as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            // LM head for K tokens
            self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;

            // Argmax inside graph (fixed scratch addresses — graph-safe)
            let vocab = self.config.vocab_size;
            let argmax_out = self.buffers.scratch();
            for t in 0..k {
                let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                let out_t = argmax_out.offset(t * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_t,
                    out_t,
                    vocab as u32,
                    stream,
                )?;
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for K=γ verify (slot={} K={})",
                        seq.slot_idx,
                        k
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut out = Vec::with_capacity(k);
        for t in 0..k {
            let off = t * 4;
            out.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        self.check_glm_k5_bf16_head(k, &out, stream)?;

        // See decode_verify_graphed for rationale on `seq_len += k` fix.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok(out)
    }
}
