// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 Flash model-specific MTP draft proposer.
//!
//! GLM's predictor is a complete appended decoder layer, not the compact
//! Qwen-shaped head represented by [`crate::layers::MtpHead`]. Every draft is
//! still verified by the distributed target model. The default proposer runs
//! independently on rank 0 with a private one-layer MLA cache; the opt-in
//! dual-Spark path mirrors that exact body and splits only its vocabulary
//! projection.

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;
use crate::model::glm_cache_plan::{GlmCachePlan, GlmMlaShape};
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::glm5::Glm5MtpModule;
use crate::weight_map::{DenseWeight, QuantizedWeight};

#[path = "glm5_mtp/kv_rows_plan.rs"]
mod kv_rows_plan;

#[path = "glm5_mtp/kv_rows.rs"]
mod kv_rows;

#[path = "glm5_mtp/kv_rows_oracle.rs"]
mod kv_rows_oracle;

#[path = "glm5_mtp/repair_state.rs"]
mod repair_state;

#[path = "glm5_mtp/repair.rs"]
mod repair;

#[path = "glm5_mtp/paired.rs"]
mod paired;

#[path = "glm5_mtp/new.rs"]
mod new;

#[path = "glm5_mtp/hidden_trace.rs"]
pub(crate) mod hidden_trace;

pub(crate) fn distributed_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_GLM_MTP_DISTRIBUTED").ok().as_deref() == Some("1"))
}

fn all_gather_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_GLM_MTP_ALL_GATHER").ok().as_deref() == Some("1"))
}

fn distributed_argmax_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ATLAS_GLM_MTP_DISTRIBUTED_ARGMAX")
            .ok()
            .as_deref()
            == Some("1")
    })
}

fn fused_eh_norm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("ATLAS_GLM_MTP_FUSED_EH_NORM").ok().as_deref() == Some("1"))
}

fn fused_eh_norm_check_once() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    *ENABLED.get_or_init(|| {
        std::env::var("ATLAS_GLM_MTP_FUSED_EH_CHECK")
            .ok()
            .as_deref()
            == Some("1")
    }) && !CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed)
}

fn mtp_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_GLM_MTP_PROFILE").ok().as_deref() == Some("1"))
}

fn mtp_bf16_drafts() -> usize {
    static COUNT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("ATLAS_GLM_MTP_BF16_DRAFTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .min(4)
    })
}

pub struct Glm5MtpProposerState {
    paired: Option<paired::Lease>,
    hidden_trace: hidden_trace::HiddenTrace,
    repair: repair_state::RepairPhase,
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub last_num_drafted: usize,
    pub body_state: Box<dyn LayerState>,
}

impl ProposerState for Glm5MtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Glm5MtpProposerState {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_verified(
        &mut self,
        generation: u64,
        capture_generation: u64,
        base: usize,
        tokens: &[u32],
        accepted: usize,
        position: usize,
        hidden_rows: usize,
    ) -> Result<()> {
        self.repair.record(
            generation,
            capture_generation,
            base,
            tokens,
            accepted,
            position,
            self.seq_len,
            hidden_rows,
        )
    }
}

pub struct Glm5MtpHead {
    paired: Option<Mutex<paired::Pool>>,
    hidden_trace_enabled: bool,
    module: Glm5MtpModule,
    embed_tokens: DenseWeight,
    lm_head: DenseWeight,
    lm_head_nvfp4: Option<QuantizedWeight>,
    mtp_vocab_size: u32,
    kv_cache: Mutex<PagedKvCache>,
    rms_norm_k: KernelHandle,
    fused_eh_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    bf16_concat_k: KernelHandle,
    argmax_k: KernelHandle,
    argmax_value_k: KernelHandle,
}

impl Glm5MtpHead {
    fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<Glm5MtpProposerState> {
        if self.paired.is_some() {
            return self.alloc_paired_state(gpu);
        }
        Ok(Glm5MtpProposerState {
            paired: None,
            hidden_trace: hidden_trace::HiddenTrace::new(self.hidden_trace_enabled),
            repair: repair_state::RepairPhase::Capture,
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            body_state: self.module.body.alloc_state(gpu)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_body_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        mut trace: Option<&mut hidden_trace::StepTrace>,
    ) -> Result<DevicePtr> {
        self.validate_paired_live(state, ctx.gpu)?;
        let h = ctx.config.hidden_size;
        let h_u32 = h as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;
        let profile = mtp_profile_enabled();
        let mut started = profile.then(std::time::Instant::now);

        // Upstream GLM: eh_proj(cat(enorm(embed(token)), hnorm(target_hidden))).
        let eh_input = ctx.buffers.ssm_qkvz();
        if fused_eh_norm_enabled() && self.fused_eh_norm_k.0 != 0 {
            ops::glm_mtp_eh_norm(
                ctx.gpu,
                self.fused_eh_norm_k,
                self.embed_tokens.weight,
                token,
                target_hidden,
                &self.module.enorm,
                &self.module.hnorm,
                eh_input,
                h_u32,
                eps,
                stream,
            )?;
            // One-shot on-device A/B for bring-up. Preserve the fused result
            // in disposable attention scratch, run the established copy + two
            // RMSNorm oracle, then require every BF16 output bit to match.
            if fused_eh_norm_check_once() {
                let bytes = 2 * row_bytes;
                let fused = ctx.buffers.attn_output();
                ctx.gpu.copy_d2d_async(eh_input, fused, bytes, stream)?;
                let embed = ctx.buffers.ssm_deinterleaved();
                ctx.gpu.copy_d2d_async(
                    self.embed_tokens.weight.offset(token as usize * row_bytes),
                    embed,
                    row_bytes,
                    stream,
                )?;
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    embed,
                    &self.module.enorm,
                    eh_input,
                    1,
                    h_u32,
                    eps,
                    stream,
                )?;
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    target_hidden,
                    &self.module.hnorm,
                    eh_input.offset(row_bytes),
                    1,
                    h_u32,
                    eps,
                    stream,
                )?;
                let mut got = vec![0u8; bytes];
                let mut want = vec![0u8; bytes];
                ctx.gpu.copy_d2h(fused, &mut got)?;
                ctx.gpu.copy_d2h(eh_input, &mut want)?;
                let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                anyhow::ensure!(
                    mismatches == 0,
                    "GLM MTP fused eh norms differ from oracle in {mismatches}/{bytes} bytes"
                );
                tracing::info!(
                    "GLM MTP fused embedding/hidden norms: exact oracle match ({bytes} bytes)"
                );
            }
        } else {
            let embed = ctx.buffers.ssm_deinterleaved();
            ctx.gpu.copy_d2d_async(
                self.embed_tokens.weight.offset(token as usize * row_bytes),
                embed,
                row_bytes,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                embed,
                &self.module.enorm,
                eh_input,
                1,
                h_u32,
                eps,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                target_hidden,
                &self.module.hnorm,
                eh_input.offset(row_bytes),
                1,
                h_u32,
                eps,
                stream,
            )?;
        }
        if profile {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "GLM MTP PROFILE input_norm={}us",
                started
                    .take()
                    .expect("MTP profile timer")
                    .elapsed()
                    .as_micros()
            );
            started = Some(std::time::Instant::now());
        }
        let h_in = ctx.buffers.hidden_states();
        if let Some(ref eh_proj) = self.module.eh_proj_nvfp4 {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                eh_input,
                eh_proj,
                h_in,
                h_u32,
                (2 * h) as u32,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                eh_input,
                &self.module.eh_proj,
                h_in,
                h_u32,
                (2 * h) as u32,
                stream,
            )?;
        }
        if let Some(trace) = trace.as_mut() {
            trace.post_eh(h_in, ctx, stream)?;
        }
        if profile {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "GLM MTP PROFILE eh_proj={}us",
                started
                    .take()
                    .expect("MTP profile timer")
                    .elapsed()
                    .as_micros()
            );
            started = Some(std::time::Instant::now());
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = state.seq_len / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let block_idx = state.block_table[state.seq_len / bs];
        let slot = block_idx as i64 * bs as i64 + (state.seq_len % bs) as i64;
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            slot,
            (state.seq_len + 1) as i32,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;
        let mtp_meta = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: state.block_table.len() as u32,
            num_seqs: 1,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr::NULL,
        };
        let mtp_ctx = ForwardContext {
            ssm_batch: None,
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(mtp_meta),
            profile: ctx.profile || profile,
            // Both ranks execute the checkpoint-native full proposer body.
            // Only the vocabulary projection below is distributed.
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let mut disk_ids = Vec::new();
        let mut disk_last = vec![0u32; 1];
        // The MTP decoder layer is constructed with `residual=None` upstream.
        // Atlas represents that as a zeroed residual row before its fused
        // add+norm path initializes the residual stream from `h_in`.
        let residual = ctx.buffers.residual();
        if let Some(trace) = trace.as_mut() {
            trace.kv_before(&kv_cache, state, ctx, stream)?;
        }
        ctx.gpu.memset_async(residual, 0, row_bytes, stream)?;
        self.module.body.decode(
            h_in,
            residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_ids,
            &mut disk_last,
            &mtp_ctx,
            stream,
        )?;
        if let Some(trace) = trace.as_mut() {
            trace.kv_after(&kv_cache, state, ctx, stream)?;
        }
        if profile {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "GLM MTP PROFILE body_total={}us",
                started
                    .take()
                    .expect("MTP profile timer")
                    .elapsed()
                    .as_micros()
            );
        }
        drop(kv_cache);

        state.seq_len += 1;
        Ok(ctx.buffers.hidden_states())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        draft_index: usize,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let mut trace = state.hidden_trace.input(
            token,
            position,
            draft_index,
            target_hidden,
            state.seq_len,
            ctx,
            stream,
        )?;
        let h_out = self.forward_body_one(
            token,
            target_hidden,
            position,
            state,
            ctx,
            stream,
            trace.as_mut(),
        )?;
        let profile = mtp_profile_enabled();
        let mut started = profile.then(std::time::Instant::now);
        let h_u32 = ctx.config.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;

        let final_hidden = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            h_out,
            &self.module.norm,
            final_hidden,
            1,
            h_u32,
            eps,
            stream,
        )?;
        if let Some(trace) = &mut trace {
            trace.final_hidden(final_hidden, state.seq_len, ctx, stream)?;
        }
        if profile {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "GLM MTP PROFILE final_norm={}us",
                started
                    .take()
                    .expect("MTP profile timer")
                    .elapsed()
                    .as_micros()
            );
            started = Some(std::time::Instant::now());
        }
        let vocab = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        let (vocab_start, projected_vocab) = if distributed_enabled() {
            let comm = ctx
                .comm
                .ok_or_else(|| anyhow::anyhow!("distributed GLM MTP requires a communicator"))?;
            anyhow::ensure!(
                comm.world_size() == 2 && vocab as usize % comm.world_size() == 0,
                "distributed GLM MTP requires an even TP2 vocabulary (got {vocab})"
            );
            let local = vocab as usize / comm.world_size();
            (comm.rank() * local, local as u32)
        } else {
            (0, vocab)
        };
        let local_logits = logits.offset(vocab_start * 2);
        // Preserve the configured leading draft decisions with the target's
        // BF16 tied head. Later drafts are always verified before emission and
        // can use the cheaper NVFP4 projection; the launcher keeps p1 in BF16.
        if draft_index >= mtp_bf16_drafts()
            && let Some(ref head) = self.lm_head_nvfp4
        {
            let local_head = QuantizedWeight {
                weight: head.weight.offset(vocab_start * ctx.config.hidden_size / 2),
                weight_scale: head
                    .weight_scale
                    .offset(vocab_start * ctx.config.hidden_size / 16),
                weight_scale_2: head.weight_scale_2,
                input_scale: head.input_scale,
                weight_scale_2_vec: if head.weight_scale_2_vec.is_null() {
                    head.weight_scale_2_vec
                } else {
                    head.weight_scale_2_vec.offset(vocab_start * 4)
                },
            };
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                final_hidden,
                &local_head,
                local_logits,
                projected_vocab,
                h_u32,
                stream,
            )?;
        } else {
            let local_head = DenseWeight {
                weight: self
                    .lm_head
                    .weight
                    .offset(vocab_start * ctx.config.hidden_size * 2),
            };
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                final_hidden,
                &local_head,
                local_logits,
                projected_vocab,
                h_u32,
                stream,
            )?;
        }
        if profile {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "GLM MTP PROFILE lm_head={}us draft={draft_index}",
                started
                    .take()
                    .expect("MTP profile timer")
                    .elapsed()
                    .as_micros()
            );
            started = Some(std::time::Instant::now());
        }
        let local_argmax =
            distributed_enabled() && distributed_argmax_enabled() && grammar_bitmask.is_none();
        if distributed_enabled() && !local_argmax {
            let comm = ctx.comm.expect("distributed communicator checked above");
            let local = projected_vocab as usize;
            if all_gather_enabled() {
                // Each rank projected directly into its rank-ordered slice of
                // the full logits buffer. NCCL's in-place form gathers both
                // slices with one launch. The communicator's legacy stream is
                // this same default compute stream, so GEMV -> gather -> global
                // argmax remain FIFO without events or a host synchronization.
                comm.all_gather(local_logits.0, logits.0, local * 2)?;
            } else {
                // Synchronous two-broadcast oracle retained for same-image A/B
                // and rollback. Both paths produce the same BF16 logits layout.
                ctx.gpu.synchronize(stream)?;
                for root in 0..comm.world_size() {
                    comm.broadcast(logits.offset(root * local * 2).0, local * 2, root)?;
                }
            }
        }
        if profile {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "GLM MTP PROFILE vocab_collective={}us",
                started
                    .take()
                    .expect("MTP profile timer")
                    .elapsed()
                    .as_micros()
            );
            started = Some(std::time::Instant::now());
        }
        let out = ctx.buffers.scratch();
        let mut trace_pairs = None;
        let draft = if let Some(mask) = grammar_bitmask {
            grammar_argmax(ctx.gpu, logits, vocab as usize, mask)?
        } else if local_argmax {
            // Preserve full-vocabulary argmax semantics without materializing
            // peer logits. Each rank reduces its contiguous half with the same
            // first-strict-max tree, then exchanges one `(f32,u32)` pair. Rank
            // zero wins an equal-value tie because it owns lower token IDs.
            let local_pair = out.offset(32);
            ops::argmax_bf16_value(
                ctx.gpu,
                self.argmax_value_k,
                local_logits,
                local_pair,
                projected_vocab,
                stream,
            )?;
            let comm = ctx.comm.expect("distributed communicator checked above");
            comm.all_gather(local_pair.0, out.0, 8)?;
            let mut pairs = [0u8; 16];
            ctx.gpu.copy_d2h(out, &mut pairs)?;
            if trace.is_some() {
                trace_pairs = Some(pairs);
            }
            let v0 = f32::from_le_bytes(pairs[0..4].try_into().expect("rank-0 max bytes"));
            let i0 = u32::from_le_bytes(pairs[4..8].try_into().expect("rank-0 index bytes"));
            let v1 = f32::from_le_bytes(pairs[8..12].try_into().expect("rank-1 max bytes"));
            let i1 = u32::from_le_bytes(pairs[12..16].try_into().expect("rank-1 index bytes"));
            if v1 > v0 { i1 + projected_vocab } else { i0 }
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out, vocab, stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h(out, &mut bytes)?;
            u32::from_le_bytes(bytes)
        };
        if profile {
            tracing::info!(
                "GLM MTP PROFILE argmax_readback={}us",
                started
                    .take()
                    .expect("MTP profile timer")
                    .elapsed()
                    .as_micros()
            );
        }
        if let Some(trace) = trace {
            trace.emit(
                draft,
                trace_pairs,
                self.module.eh_proj_nvfp4.is_some(),
                draft_index >= mtp_bf16_drafts() && self.lm_head_nvfp4.is_some(),
            );
        }
        Ok(draft)
    }
}

fn grammar_argmax(
    gpu: &dyn GpuBackend,
    logits: DevicePtr,
    vocab: usize,
    mask: &[i32],
) -> Result<u32> {
    let mut raw = vec![0u8; vocab * 2];
    gpu.copy_d2h(logits, &mut raw)?;
    let mut best = (0u32, f32::NEG_INFINITY);
    for token in 0..vocab {
        if token / 32 >= mask.len() || mask[token / 32] & (1i32 << (token % 32)) == 0 {
            continue;
        }
        let bf16 = u16::from_le_bytes([raw[2 * token], raw[2 * token + 1]]);
        let value = f32::from_bits((bf16 as u32) << 16);
        if value > best.1 {
            best = (token as u32, value);
        }
    }
    Ok(best.0)
}

impl DraftProposer for Glm5MtpHead {
    fn glm_pair_repair(&self) -> Option<&dyn crate::speculative::glm_repair::GlmPairRepair> {
        Some(self)
    }
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .map(|s| s.seq_len)
            .unwrap_or(0)
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        anyhow::ensure!(
            self.paired.is_none()
                && !state
                    .as_any()
                    .downcast_ref::<Glm5MtpProposerState>()
                    .is_some_and(|state| state.paired.is_some()),
            "paired owners require the owned primer entry"
        );
        let batched = std::env::var("ATLAS_GLM_MTP_BATCHED_PREFILL")
            .ok()
            .as_deref()
            == Some("1");
        let serial = std::env::var("ATLAS_GLM_MTP_SERIAL_PREFILL")
            .ok()
            .as_deref()
            == Some("1");
        if !batched
            && let Some(s) = state.as_any_mut().downcast_mut::<Glm5MtpProposerState>()
            && s.hidden_trace.prompt.active()
        {
            s.hidden_trace.prompt.fail();
            anyhow::bail!("GLM prompt diagnostic requires actual batched KV primer");
        }
        if !batched && !serial {
            return Ok(0);
        }
        let state = match state.as_any_mut().downcast_mut::<Glm5MtpProposerState>() {
            Some(s) => s,
            None => return Ok(0),
        };
        if batched {
            return self.prefill_kv_batched(prompt_tokens, hiddens, state, ctx, stream);
        }

        // Correctness fallback: mirror the shifted-pair contract one row at a
        // time through the full decode body. This remains an opt-in diagnostic
        // oracle for the batched KV-only implementation.
        if state.seq_len != 0 || prompt_tokens.len() < 2 {
            return Ok(0);
        }

        let started = std::time::Instant::now();
        let h = ctx.config.hidden_size;
        let rows = prompt_tokens.len() - 1;
        for i in 0..rows {
            self.forward_body_one(
                prompt_tokens[i + 1],
                hiddens.offset(i * h * 2),
                i + 1,
                state,
                ctx,
                stream,
                None,
            )?;
        }
        ctx.gpu.synchronize(stream)?;
        tracing::info!(
            "GLM MTP serial drafter prefill probe: {rows} rows in {:.1} ms",
            started.elapsed().as_secs_f64() * 1e3,
        );
        Ok(rows)
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("invalid GLM-5 MTP proposer state"))?;
        self.authorize_paired_propose(state, target_hidden, ctx)?;
        if self.paired.is_some() || crate::speculative::glm_repair_policy::enabled() {
            let repair_state::RepairPhase::Proposed(plan) = state.repair else {
                anyhow::bail!("GLM repair proposal was not prepared");
            };
            anyhow::ensure!(
                num_drafts == 4
                    && grammar_bitmask.is_none()
                    && plan.position() == position
                    && state.seq_len.checked_add(4) == Some(plan.speculative_cache_end()),
                "GLM repair proposal metadata changed after prepare"
            );
        }
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        let mut hidden = target_hidden;
        for i in 0..num_drafts {
            let draft = self.forward_one(
                token,
                hidden,
                position + i,
                i,
                state,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            drafts.push(draft);
            token = draft;
            // GLM recycles the MTP layer's post-shared-norm hidden.
            hidden = ctx.buffers.norm_output();
        }
        state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("invalid GLM-5 MTP proposer state"))?;
        anyhow::ensure!(
            self.paired.is_none() && state.paired.is_none(),
            "paired verdict consumption is not available in Gate1"
        );
        if crate::speculative::glm_repair_policy::enabled() {
            return state.repair.acknowledge(num_accepted);
        }
        state.seq_len = state
            .seq_len
            .saturating_sub(state.last_num_drafted.saturating_sub(num_accepted));
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Glm5MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("invalid GLM-5 MTP proposer state"))?;
        if self.paired.is_some() {
            return self.free_paired_state(_gpu, state);
        }
        anyhow::ensure!(
            state.paired.is_none(),
            "legacy head cannot free a paired lease"
        );
        if !state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&state.block_table);
            state.block_table.clear();
        }
        state.seq_len = 0;
        state.repair = repair_state::RepairPhase::Capture;
        state.last_num_drafted = 0;
        state.hidden_trace.reset();
        Ok(())
    }
}
