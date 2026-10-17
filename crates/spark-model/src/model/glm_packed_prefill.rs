// SPDX-License-Identifier: AGPL-3.0-only

//! Packed multi-prompt prefill (`ATLAS_GLM_PACKED_PREFILL=1`, default off).
//!
//! # Why
//!
//! A prefill pass at a few dozen rows is bound by the routed-expert weights
//! it reads: GLM-5.3-Flash holds ~83 GB of experts per rank under EP2, and 50
//! tokens x top-8 over 288 experts touch ~75% of them, ~270 ms of a ~350 ms
//! pass. Eight short prompts prefilled one after another read most of the
//! experts eight times. Packed into one traversal (400 rows touch ~100%) they
//! read them once.
//!
//! # What runs
//!
//! New prompts of at most `ATLAS_GLM_PACKED_PREFILL_MAX_TOKENS` tokens
//! (default 512) with no restorable prefix-cache state are laid out back to
//! back in the arena (`layer::glm_packed`). Each prompt gets exactly the
//! metadata, block reservation, embedding, DFlash capture and last-row LM
//! head a cold single-prompt prefill gives it; the layers run their row-local
//! work once over all rows and each prompt's recurrence and attention over
//! its own rows. The prompts skip the prefix-cache lookup and insert
//! (a prompt that has restorable state takes the ordinary path instead).
//!
//! # Exactness
//!
//! With the switch on a prompt runs this path whether it is alone (C1) or
//! packed with others (C4/C8), and every row-count dependent choice inside a
//! packed pass is made from the prompt's own row count or pinned
//! (`packed_scope`), so its output does not depend on its company.
//! `ATLAS_GLM_PACKED_PREFILL_CHECK=1` logs one hash line per prompt to prove
//! it on hardware (`PACKED_CHECK`).
//!
//! # EP
//!
//! One worker command, `0xFFFFFFE8`: preamble slot of the first prompt, then
//! N, slots[N], lengths[N] and every prompt's tokens in one bulk broadcast.
//! The head decides the packing; the worker follows it.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::layer::glm_packed::GlmPackedOwner;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::traits::{Model, SequenceState};

pub(super) const EP_CMD_GLM_PACKED_PREFILL: u32 = 0xFFFF_FFE8;

/// Largest prompt the packing takes by default.
pub const DEFAULT_MAX_TOKENS: usize = 512;
/// The FlashKDA recurrence starts at 2048 rows and borrows the packed
/// buffer; a packed prompt stays below it.
pub const MAX_TOKENS_CAP: usize = 2047;
/// Most prompts in one packed pass.
pub const MAX_PROMPTS: usize = 16;

/// `ATLAS_GLM_PACKED_PREFILL=1`. Read once per process.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PACKED_PREFILL").as_deref() == Ok("1"))
}

/// `ATLAS_GLM_PACKED_PREFILL_MAX_TOKENS` (default 512, at most 2047).
pub fn max_tokens() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        parse_max_tokens(
            std::env::var("ATLAS_GLM_PACKED_PREFILL_MAX_TOKENS")
                .ok()
                .as_deref(),
        )
    })
}

pub(crate) fn parse_max_tokens(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .clamp(2, MAX_TOKENS_CAP)
}

/// `ATLAS_GLM_PACKED_PREFILL_CHECK=1`: per-prompt hash lines.
fn check_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PACKED_PREFILL_CHECK").as_deref() == Ok("1"))
}

/// Arena rows of each prompt: back to back from row 0.
pub(crate) fn row_offsets(lens: &[usize]) -> Vec<usize> {
    lens.iter()
        .scan(0usize, |next, &len| {
            let at = *next;
            *next += len;
            Some(at)
        })
        .collect()
}

/// Bytes of one prompt's metadata block (positions [x3 under MRoPE], then
/// slots), 256-aligned like the fused chunk's.
pub(crate) fn meta_block_bytes(rows: usize, mrope: bool) -> usize {
    let streams = if mrope { 3 } else { 1 };
    let slot_offset = (rows * 4 * streams + 7) & !7;
    (slot_offset + rows * 8).div_ceil(256) * 256
}

/// Scratch layout of a packed pass: the MoE top-k staging at 0 (as for one
/// chunk of `total` rows), the joint positions/slots, then one metadata block
/// per prompt. Returns `(joint_at, owners_at, end)`.
pub(crate) fn scratch_layout(
    lens: &[usize],
    top_k: usize,
    mrope: bool,
) -> (usize, Vec<usize>, usize) {
    let total: usize = lens.iter().sum();
    let align = |b: usize| b.div_ceil(256) * 256;
    let joint_at = align(total * top_k * 4 * 2);
    let mut at = joint_at + meta_block_bytes(total, mrope);
    let owners = lens
        .iter()
        .map(|&rows| {
            let here = at;
            at += meta_block_bytes(rows, mrope);
            here
        })
        .collect();
    (joint_at, owners, at)
}

impl TransformerModel {
    /// Whether this model serves packed prefill at all.
    pub(super) fn glm_packed_supported(&self) -> bool {
        enabled()
            && self.config.model_type == "glm5_next"
            // The shared MTP prompt capture reads chunk rows from 0.
            && self.mtp_prefill_hidden.is_null()
            // The FP32 logits scratch holds one row.
            && !self.use_fp32_logits
            // The row-local fused mHC mixer the pins rely on (`glm_packed_scope`).
            && crate::layers::ops::hc_bf16_for(&self.config.model_type)
    }

    /// Whether `prompt` (for adapter `adapter_id`) may be packed: short, text
    /// only, and with no restorable prefix-cache state (such a prompt takes
    /// the ordinary path, which restores it).
    pub(super) fn glm_packed_admits(&self, prompt: &[u32], adapter_id: u64) -> bool {
        self.glm_packed_supported()
            && (2..=max_tokens()).contains(&prompt.len())
            && !self.tokens_have_vision_pad(prompt)
            && !self.glm_packed_restorable(prompt, adapter_id)
    }

    /// A resident snapshot at a whole-block depth the radix holds for
    /// `prompt`: what a cold lookup could restore.
    fn glm_packed_restorable(&self, prompt: &[u32], adapter_id: u64) -> bool {
        if !self.prefix_cache.is_active() {
            return false;
        }
        let bs = self.kv_cache.lock().block_size();
        let matched = self
            .prefix_cache
            .peek_matched_tokens(prompt, bs, adapter_id);
        (1..=matched / bs).any(|b| {
            self.prefix_cache
                .snapshot_at(prompt, b * bs, adapter_id)
                .is_some()
        })
    }

    /// `(rows, prompts)` one packed pass may hold.
    pub(super) fn glm_packed_capacity(&self) -> (usize, usize) {
        let v = self.config.vocab_size * 2;
        let logits_rows = self.buffers.sizes().expert_gate_out / v.max(1);
        let rows = self.buffers.max_batch_tokens();
        (rows, MAX_PROMPTS.min(logits_rows))
    }

    /// Head: announce the packing and run it. Returns each prompt's
    /// last-row logits.
    pub(super) fn prefill_packed_impl(
        &self,
        prompts: &[&[u32]],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<DevicePtr>> {
        self.glm_packed_validate(prompts, seqs)?;
        if self.multi_rank_protocol_active() {
            let slots: Vec<u32> = seqs.iter().map(|s| s.slot_idx as u32).collect();
            let lens: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
            let tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
            self.ep_broadcast_seq_and_cmd(
                slots[0],
                EP_CMD_GLM_PACKED_PREFILL,
                self.ep_protocol_v2,
            )?;
            self.ep_broadcast_u32(prompts.len() as u32)?;
            self.ep_broadcast_tokens(&slots)?;
            self.ep_broadcast_tokens(&lens)?;
            self.ep_broadcast_tokens(&tokens)?;
        }
        self.glm_packed_compute(prompts, seqs, self.gpu.default_stream())
    }

    /// Worker side of `0xFFFFFFE8`: the same pass, then each prompt's SSM
    /// normalization (the head's scheduler normalizes after the prefill).
    pub(super) fn glm_packed_receive(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        ensure!(
            (1..=MAX_PROMPTS).contains(&n) && n <= slots.len(),
            "GLM packed prefill of {n} prompts"
        );
        let ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let lens = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        ensure!(
            total <= self.buffers.max_batch_tokens(),
            "GLM packed prefill of {total} rows"
        );
        let tokens = self.ep_broadcast_tokens(&vec![0u32; total])?;
        let mut refs: Vec<(usize, &mut SequenceState)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.as_mut().map(|s| (i, s)))
            .collect();
        let mut seqs = Vec::with_capacity(n);
        for &id in &ids {
            let at = refs
                .iter()
                .position(|(i, _)| *i == id as usize)
                .with_context(|| format!("GLM packed prefill slot {id} unallocated or repeated"))?;
            seqs.push(refs.swap_remove(at).1);
        }
        let prompts: Vec<&[u32]> =
            row_offsets(&lens.iter().map(|&l| l as usize).collect::<Vec<_>>())
                .into_iter()
                .zip(&lens)
                .map(|(at, &len)| &tokens[at..at + len as usize])
                .collect();
        self.glm_packed_validate(&prompts, &seqs)?;
        self.sync_secondary()?;
        let stream = self.gpu.default_stream();
        self.glm_packed_compute(&prompts, &mut seqs, stream)?;
        for seq in seqs.iter() {
            if let Err(e) = self.normalize_ssm_states(seq, stream) {
                tracing::warn!("Worker SSM state normalization failed: {e:#}");
            }
        }
        Ok(true)
    }

    /// `ATLAS_GLM_PACKED_PREFILL_CHECK`: the exact bytes at `ptr`, hashed
    /// (synchronizes `stream`).
    fn packed_check_hash(&self, ptr: DevicePtr, bytes: usize, stream: u64) -> Result<u64> {
        crate::det_trace::device_hash(self.gpu.as_ref(), stream, ptr, bytes, 32 << 20)
            .context("PACKED_CHECK: device read failed")
    }

    fn glm_packed_validate(&self, prompts: &[&[u32]], seqs: &[&mut SequenceState]) -> Result<()> {
        let (row_cap, prompt_cap) = self.glm_packed_capacity();
        let total: usize = prompts.iter().map(|p| p.len()).sum();
        ensure!(
            !prompts.is_empty() && prompts.len() == seqs.len() && prompts.len() <= prompt_cap,
            "GLM packed prefill of {} prompts (cap {prompt_cap})",
            prompts.len()
        );
        ensure!(
            total <= row_cap,
            "GLM packed prefill of {total} rows (cap {row_cap})"
        );
        for (p, s) in prompts.iter().zip(seqs) {
            ensure!(
                (2..=MAX_TOKENS_CAP).contains(&p.len()),
                "GLM packed prefill prompt of {} tokens",
                p.len()
            );
            ensure!(
                s.seq_len == 0 && s.tokens.is_empty() && s.block_table.is_empty(),
                "GLM packed prefill needs a fresh sequence (slot {})",
                s.slot_idx
            );
        }
        let top_k = self.config.num_experts_per_tok;
        let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let (_, _, end) = scratch_layout(&lens, top_k, self.config.mrope_interleaved);
        ensure!(
            end <= self.buffers.scratch_bytes(),
            "GLM packed prefill metadata ({end} bytes) exceeds scratch"
        );
        Ok(())
    }

    /// Both ranks: one traversal over every prompt's rows.
    fn glm_packed_compute(
        &self,
        prompts: &[&[u32]],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let rows0 = row_offsets(&lens);
        let total: usize = lens.iter().sum();
        let hidden = self.buffers.hidden_states();

        // Every packed prompt runs the installed adapter (`-1`), as a chunk of
        // one does (`prefill_chunk_entry`).
        self.stamp_overlay_route(-1);
        // ── Arena and embeddings: a cold chunk's, at each prompt's rows ──
        if self.comm.is_some() {
            self.warm_zero_arena(stream)?;
        } else {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }
        for ((prompt, seq), &row0) in prompts.iter().zip(seqs.iter()).zip(&rows0) {
            self.prefill_b_embed_chunk_at(
                seq,
                prompt,
                0,
                prompt.len(),
                hidden.offset(row0 * h * bf16),
                stream,
            )?;
        }
        let joint_tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
        self.gpu.copy_h2d_async(
            // SAFETY: POD u32 vector viewed as its own bytes.
            unsafe { std::slice::from_raw_parts(joint_tokens.as_ptr() as *const u8, total * 4) },
            self.buffers.token_ids(),
            stream,
        )?;

        // ── Per prompt: a cold lookup's bookkeeping and its blocks ──
        let mut kv_cache = self.kv_cache.lock();
        for (prompt, seq) in prompts.iter().zip(seqs.iter_mut()) {
            seq.prompt_len = prompt.len();
            seq.prefix_ref_tokens.clear();
            seq.cached_prefix_tokens = 0;
            seq.cached_prefix_blocks = 0;
            seq.marconi_skip_to = 0;
            seq.prefix_lookup_skip = false;
            seq.prefix_lookup_applied = true;
            seq.marconi_exact_snap = None;
            let key = self.pc_key(seq, prompt).map(|k| k.into_owned());
            self.dflash_warm_begin(seq, key.as_deref(), 0, prompt.len(), stream)?;
            self.reserve_prefill_blocks(seq, prompt.len(), &mut kv_cache, stream)?;
            seq.kv_valid_tokens = prompt.len();
        }
        self.buffers.note_rows(total);

        // ── Metadata: each prompt's own, then the joint positions/slots ──
        let mrope = self.config.mrope_interleaved;
        let (joint_at, owners_at, _) =
            scratch_layout(&lens, self.config.num_experts_per_tok, mrope);
        let scratch = self.buffers.scratch();
        let mut metas: Vec<AttnMetadataDev> = Vec::with_capacity(seqs.len());
        for (k, (prompt, seq)) in prompts.iter().zip(seqs.iter_mut()).enumerate() {
            let rows = prompt.len();
            let base = scratch.offset(owners_at[k]);
            let layout = self.prefill_b_upload_meta_at(
                prompt,
                seq,
                0,
                rows,
                0,
                rows,
                0,
                &kv_cache,
                base,
                meta_block_bytes(rows, mrope),
                stream,
            )?;
            ensure!(
                layout.needs_paged,
                "GLM packed prefill expects paged metadata"
            );
            self.prefill_b_upload_paged(
                seq,
                rows,
                0,
                rows,
                base,
                layout.slot_offset,
                &kv_cache,
                stream,
            )?;
            // The metadata pack reuses one pinned staging buffer.
            self.gpu.synchronize(stream)?;
            let page = seq
                .chunked_prefill_meta
                .as_ref()
                .context("GLM packed prefill: paged metadata missing")?;
            let pos = |s: usize| {
                if layout.use_mrope {
                    base.offset(s * layout.pos_stream_bytes)
                } else {
                    base
                }
            };
            metas.push(AttnMetadataDev {
                positions: pos(0),
                positions_h: pos(1),
                positions_w: pos(2),
                slot: base.offset(layout.slot_offset),
                seq_len: page.seq_len,
                block_table: page.block_table,
                max_blocks_per_seq: seq.block_table.len() as u32,
                num_seqs: 1,
                seq_slot: DevicePtr::NULL,
                moe_row_adapter: DevicePtr::NULL,
            });
        }
        let joint = scratch.offset(joint_at);
        let stream_bytes = total * 4;
        let streams = if mrope { 3 } else { 1 };
        let joint_slots = joint.offset((stream_bytes * streams + 7) & !7);
        for (k, m) in metas.iter().enumerate() {
            let (row0, rows) = (rows0[k], lens[k]);
            let src = [m.positions, m.positions_h, m.positions_w];
            for (s, &src) in src.iter().enumerate().take(streams) {
                self.gpu.copy_d2d_async(
                    src,
                    joint.offset(s * stream_bytes + row0 * 4),
                    rows * 4,
                    stream,
                )?;
            }
            self.gpu
                .copy_d2d_async(m.slot, joint_slots.offset(row0 * 8), rows * 8, stream)?;
        }
        let joint_pos = |s: usize| {
            if mrope {
                joint.offset(s * stream_bytes)
            } else {
                joint
            }
        };
        // Force the H2D/D2D metadata before the layers (as a chunk does).
        self.gpu.synchronize(stream)?;

        let seq_slot =
            self.upload_seq_slot_uniform(-1, total, self.buffers.lora_seq_slot(), stream)?;
        let attn_metadata = AttnMetadataDev {
            positions: joint_pos(0),
            positions_h: joint_pos(1),
            positions_w: joint_pos(2),
            slot: joint_slots,
            seq_slot,
            ..metas[0]
        };
        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(attn_metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: Some(self.buffers.token_ids()),
            host_token_ids: None,
            routed_lora_layers: self.routed_slot_layers(-1),
            midchunk_capture: None,
            moe_lora_route: self.moe_lora_route(-1),
        };

        // ── Layers ──
        let check = check_enabled();
        let mut layer_hash = vec![0u64; seqs.len()];
        let hc_row =
            self.config.hc_mult * h * crate::layers::ops::hc_elem_bytes(&self.config.model_type);
        {
            let _pin = crate::layers::glm_packed_scope::enter();
            for (i, layer) in self.layers.iter().enumerate() {
                let mut owners: Vec<GlmPackedOwner<'_>> = seqs
                    .iter_mut()
                    .zip(rows0.iter().zip(&lens))
                    .zip(&metas)
                    .map(|((seq, (&row0, &rows)), &meta)| GlmPackedOwner {
                        state: seq.layer_states[i].as_mut() as &mut (dyn LayerState + 'static),
                        row0,
                        rows,
                        meta,
                    })
                    .collect();
                layer
                    .prefill_glm_packed(hidden, total, &mut owners, &mut kv_cache, &ctx, stream)
                    .map_err(|e| anyhow::anyhow!("GLM packed prefill layer {i} failed: {e}"))?;
                drop(owners);
                for (k, seq) in seqs.iter_mut().enumerate() {
                    self.try_dflash_prefill_capture_layer_at(seq, i, 0, lens[k], rows0[k], stream)?;
                    if check {
                        let hash = self.packed_check_hash(
                            self.buffers.hc_streams().offset(rows0[k] * hc_row),
                            lens[k] * hc_row,
                            stream,
                        )?;
                        layer_hash[k] = atlas_tier::hash::mix64(layer_hash[k], hash);
                    }
                }
            }
        }

        // ── Per prompt: tokens, last-row norm + LM head, DFlash context ──
        let v = self.config.vocab_size;
        let normed = self.buffers.norm_output();
        // Each prompt's row through the same one-row `lm_head` a single-chunk
        // prefill runs, its logits then moved to their own row of the idle
        // MoE scratch (the logits buffer is the head's own working space).
        let park = self.buffers.expert_gate_out();
        let n = seqs.len();
        let mut logits = vec![DevicePtr::NULL; n];
        for k in 0..n {
            let last = hidden.offset((rows0[k] + lens[k] - 1) * h * bf16);
            let row_normed = normed.offset(k * h * bf16);
            self.final_norm_rows(last, row_normed, 1, stream)?;
            let out = self.lm_head(row_normed, stream)?;
            logits[k] = park.offset(k * v * 2);
            self.gpu.copy_d2d_async(out, logits[k], v * 2, stream)?;
        }
        let bs = kv_cache.block_size();
        drop(kv_cache);
        for (k, (prompt, seq)) in prompts.iter().zip(seqs.iter_mut()).enumerate() {
            seq.tokens.extend_from_slice(prompt);
            seq.seq_len = prompt.len();
            seq.last_decode_ckpt_block = seq.tokens.len() / bs;
            self.update_dflash_ctx_len_after_prefill(seq, 0, prompt.len())?;
            if check {
                let final_hash = self.packed_check_hash(
                    hidden.offset(rows0[k] * h * bf16),
                    lens[k] * h * bf16,
                    stream,
                )?;
                let logits_hash = self.packed_check_hash(logits[k], v * 2, stream)?;
                tracing::info!(
                    "PACKED_CHECK rank={} prompt={:016x} tokens={} packed={n}/{total} layers={:016x} final={final_hash:016x} logits={logits_hash:016x}",
                    self.config.ep_rank,
                    crate::det_trace::hash_bytes(
                        // SAFETY: POD u32 slice viewed as its own bytes.
                        unsafe {
                            std::slice::from_raw_parts(
                                prompt.as_ptr() as *const u8,
                                prompt.len() * 4,
                            )
                        }
                    ),
                    prompt.len(),
                    layer_hash[k],
                );
            }
        }
        Ok(logits)
    }
}

#[cfg(test)]
#[path = "glm_packed_prefill_tests.rs"]
mod tests;
