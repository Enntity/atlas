// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched repaired long-context K3 verify (see `layer::glm_long_owner`).
//!
//! Wire protocol, both commands v2-addressed:
//!
//! ```text
//! E9  (seq_id 0)     N, slots[N], tokens[3N]      -> both ranks run one traversal
//! EA  (seq_id slot)  owner, tokens[3], verdict    -> per owner, in order
//! ```
//!
//! After E9 both ranks hold every owner's final rows in the stage. Before each
//! owner's verdict EA restores that owner's rows to arena rows [0, 3), so the
//! unchanged single-owner tail (verdict, repair record, commit, the owner's own
//! distributed propose) observes exactly what a single-owner verify leaves.

use super::TransformerModel;
use super::block_mgmt::ensure_blocks_through_decode;
use crate::layer::glm_long_owner::{self as owner, GlmLongOwner, GlmLongStage, MAX_OWNERS, ROWS};
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

pub(super) const EP_CMD_GLM_LONG_VERIFY: u32 = 0xFFFF_FFE9;
pub(super) const EP_CMD_GLM_LONG_TAIL: u32 = 0xFFFF_FFEA;

/// Byte offsets of one verify metadata block (same layout as `verify_d`).
const META_SEQ_SLOT: usize = 128;
const META_SLOTS: usize = 256;
const META_SEQ_LENS: usize = 512;
const META_BLOCK_TABLE: usize = 768;
/// verify_d's metadata base, after the MTP metadata reservation.
const META_BASE: usize = 32768;

fn align(bytes: usize) -> usize {
    bytes.div_ceil(256) * 256
}

impl TransformerModel {
    pub(super) fn glm_long_stage_rows(&self) -> owner::RowBytes {
        owner::RowBytes::new(
            self.config.hidden_size,
            self.config.hc_mult,
            self.config.vocab_size,
        )
    }

    pub(super) fn can_batch_glm_long_verify_impl(&self, owners: usize) -> bool {
        self.glm_long_stage.is_some()
            && (1..=MAX_OWNERS).contains(&owners)
            && self.config.model_type == "glm5_next"
            && self.config.tp_world_size == 2
            && self.config.ep_world_size == 2
            && self.config.hc_mult == 4
            && crate::speculative::glm_repair_policy::enabled()
            && crate::speculative::glm_repair_policy::long_context_enabled()
            && self.lora.is_none()
            && self.paired_handoff().is_none()
            && self.multi_rank_protocol_active()
            && self.ep_protocol_v2
    }

    /// Stage spans of the final verify rows the per-owner tail consumes.
    fn glm_long_final_spans(&self, stage: &GlmLongStage) -> [(DevicePtr, DevicePtr, usize); 4] {
        let b = &self.buffers;
        let r = stage.rows;
        [
            (b.hidden_states(), stage.hidden, r.hidden),
            (b.norm_output(), stage.norm, r.hidden),
            (b.hc_streams(), stage.highway, r.highway),
            (b.logits(), stage.logits, r.logits),
        ]
    }

    /// Upload one verify metadata block for `rows` = `(position, sequence)`.
    fn glm_long_upload_meta(
        &self,
        base: DevicePtr,
        rows: &[(usize, &SequenceState)],
        block_size: usize,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        let k = rows.len();
        let mb = self.max_blocks_per_seq as usize;
        let positions: Vec<u32> = rows.iter().map(|&(p, _)| p as u32).collect();
        let seq_lens: Vec<i32> = rows.iter().map(|&(p, _)| p as i32 + 1).collect();
        let mut slots = Vec::with_capacity(k);
        let mut table = vec![0i32; k * mb];
        for (row, &(pos, seq)) in rows.iter().enumerate() {
            let block = seq
                .physical_block_for(pos / block_size)
                .context("GLM long owner verify block missing")?;
            slots.push(block as i64 * block_size as i64 + (pos % block_size) as i64);
            for (j, &b) in seq.block_table.iter().enumerate().take(mb) {
                table[row * mb + j] = b as i32;
            }
        }
        fn bytes<T>(v: &[T]) -> &[u8] {
            // SAFETY: POD integer vectors; the byte view covers exactly `len`.
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
        }
        self.gpu.copy_h2d_async(bytes(&positions), base, stream)?;
        self.gpu
            .copy_h2d_async(bytes(&slots), base.offset(META_SLOTS), stream)?;
        self.gpu
            .copy_h2d_async(bytes(&seq_lens), base.offset(META_SEQ_LENS), stream)?;
        self.gpu
            .copy_h2d_async(bytes(&table), base.offset(META_BLOCK_TABLE), stream)?;
        let seq_slot = self.upload_seq_slot_uniform(
            rows[0].1.adapter_slot,
            k,
            base.offset(META_SEQ_SLOT),
            stream,
        )?;
        Ok(AttnMetadataDev {
            positions: base,
            positions_h: base,
            positions_w: base,
            slot: base.offset(META_SLOTS),
            seq_len: base.offset(META_SEQ_LENS),
            block_table: base.offset(META_BLOCK_TABLE),
            max_blocks_per_seq: self.max_blocks_per_seq,
            num_seqs: k as u32,
            seq_slot,
            moe_row_adapter: DevicePtr::NULL,
        })
    }

    /// One target traversal for every owner, identical on both ranks. On Ok
    /// each owner advanced by three rows (the caller's verdict rewinds), the
    /// stage holds every owner's final rows, and the argmax IDs are returned.
    pub(super) fn glm_long_owner_compute(
        &self,
        tokens: &[[u32; ROWS]],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<[u32; ROWS]>> {
        let n = seqs.len();
        ensure!(
            self.can_batch_glm_long_verify_impl(n) && tokens.len() == n,
            "GLM long owner verify refused for {n} owners"
        );
        let stage = self
            .glm_long_stage
            .context("GLM long owner stage missing")?;
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let rows = n * ROWS;
        let hidden = self.buffers.hidden_states();
        let mut kv_cache = self.kv_cache.lock();
        ensure!(
            kv_cache.config().cache_blocks_per_seq.is_none(),
            "GLM long owner verify requires resident KV"
        );
        let bs = kv_cache.block_size();
        let mb = self.max_blocks_per_seq as usize;
        let joint_bytes = align(META_BLOCK_TABLE + rows * mb * 4);
        let owner_bytes = align(META_BLOCK_TABLE + ROWS * mb * 4);
        ensure!(
            META_BASE + joint_bytes + n * owner_bytes <= self.buffers.sizes().scratch,
            "GLM long owner verify metadata exceeds scratch"
        );

        let oracle = oracle_enabled();
        let snapshot = if oracle {
            Some(self.glm_long_snapshot(seqs, true, None)?)
        } else {
            None
        };
        for (o, t) in tokens.iter().enumerate() {
            for (r, &token) in t.iter().enumerate() {
                self.embed(token, hidden.offset((o * ROWS + r) * h * 2), stream)?;
            }
        }
        for seq in seqs.iter_mut() {
            for t in 0..ROWS {
                ensure_blocks_through_decode(
                    seq,
                    (seq.seq_len + t) / bs,
                    &mut kv_cache,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                    self.levers.kv_poison,
                )?;
            }
        }
        if std::env::var("ATLAS_GLM_LONG_BATCH_ALIAS_CHECK").as_deref() == Ok("1") {
            let mut owner_of = std::collections::HashMap::new();
            for (o, seq) in seqs.iter().enumerate() {
                let used = (seq.seq_len + ROWS).div_ceil(bs);
                for (idx, &block) in seq.block_table.iter().take(used).enumerate() {
                    if let Some((prev, prev_idx)) = owner_of.insert(block, (o, idx)) {
                        tracing::error!(
                            "GLM long owner KV alias: physical block {block} is owner {prev} slot {} \
                             block {prev_idx} and owner {o} slot {} block {idx}",
                            seqs[prev].slot_idx,
                            seq.slot_idx
                        );
                    }
                }
            }
        }
        let scratch = self.buffers.scratch();
        let mut joint_rows = Vec::with_capacity(rows);
        let mut metas = Vec::with_capacity(n);
        for (o, seq) in seqs.iter().enumerate() {
            let own: Vec<(usize, &SequenceState)> =
                (0..ROWS).map(|t| (seq.seq_len + t, &**seq)).collect();
            let base = scratch.offset(META_BASE + joint_bytes + o * owner_bytes);
            metas.push(self.glm_long_upload_meta(base, &own, bs, stream)?);
            joint_rows.extend(own);
        }
        let joint =
            self.glm_long_upload_meta(scratch.offset(META_BASE), &joint_rows, bs, stream)?;
        drop(joint_rows);

        for (seq, t) in seqs.iter_mut().zip(tokens) {
            for (li, layer) in self.layers.iter().enumerate() {
                layer.verify_prestage(
                    t,
                    seq.layer_states[li].as_mut(),
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
        }
        let flat: Vec<u32> = tokens.iter().flatten().copied().collect();
        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(joint),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: Some(&flat),
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };
        let verify_profile = std::env::var("ATLAS_GLM_VERIFY_PROFILE").as_deref() == Ok("1");
        let (mut kda_us, mut mla_us) = (0u128, 0u128);
        for (li, layer) in self.layers.iter().enumerate() {
            let started = if verify_profile {
                self.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            let attention =
                self.config.layer_type(li) == atlas_core::config::LayerType::FullAttention;
            if serial_diagnostic(attention) {
                self.glm_long_serial_layer(li, seqs, &metas, &stage, &mut kv_cache, &ctx, stream)?;
            } else {
                let mut owners: Vec<GlmLongOwner<'_>> = seqs
                    .iter_mut()
                    .zip(&metas)
                    .map(|(seq, &meta)| GlmLongOwner {
                        positions: [seq.seq_len, seq.seq_len + 1, seq.seq_len + 2],
                        state: seq.layer_states[li].as_mut(),
                        meta,
                    })
                    .collect();
                layer.decode_glm_long_owners(&mut owners, &mut kv_cache, &stage, &ctx, stream)?;
            }
            if let Some(started) = started {
                self.gpu.synchronize(stream)?;
                let us = started.elapsed().as_micros();
                if self.config.layer_type(li) == atlas_core::config::LayerType::FullAttention {
                    mla_us += us;
                } else {
                    kda_us += us;
                }
            }
        }
        if verify_profile {
            tracing::info!(
                "GLM long owner verify profile owners={n}: kda={:.2}ms mla={:.2}ms",
                kda_us as f64 / 1000.0,
                mla_us as f64 / 1000.0
            );
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
        let argmax = self.buffers.scratch();
        if !self.glm_split_head_argmax(normed, rows, argmax, stream)? {
            self.lm_head_batched(normed, rows as u32, self.buffers.logits(), stream)?;
            let vocab = self.config.vocab_size;
            for r in 0..rows {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    self.buffers.logits().offset(r * vocab * 2),
                    argmax.offset(r * 4),
                    vocab as u32,
                    stream,
                )?;
            }
        }
        stage.copy(
            self.gpu.as_ref(),
            &self.glm_long_final_spans(&stage),
            0,
            0,
            rows,
            true,
            stream,
        )?;
        let mut buf = vec![0u8; rows * 4];
        self.gpu.copy_d2h(argmax, &mut buf)?;
        let ids: Vec<u32> = buf
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        if let Some(snapshot) = snapshot {
            drop(kv_cache);
            return self.glm_long_oracle(tokens, seqs, &stage, snapshot, &ids);
        }
        for (seq, t) in seqs.iter_mut().zip(tokens) {
            seq.tokens.extend_from_slice(t);
            seq.seq_len += ROWS;
        }
        Ok(ids.chunks_exact(ROWS).map(|c| [c[0], c[1], c[2]]).collect())
    }

    /// Diagnostic (`ATLAS_GLM_LONG_BATCH_SERIAL`): run one layer through its
    /// ordinary single-owner verify, owner by owner, with each owner's hidden
    /// and highway rows moved to arena rows [0, 3) and back.
    #[allow(clippy::too_many_arguments)]
    fn glm_long_serial_layer(
        &self,
        li: usize,
        seqs: &mut [&mut SequenceState],
        metas: &[AttnMetadataDev],
        stage: &GlmLongStage,
        kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let b = &self.buffers;
        let spans = [
            (b.hidden_states(), stage.hidden, stage.rows.hidden),
            (b.hc_streams(), stage.highway, stage.rows.highway),
        ];
        let rows = seqs.len() * ROWS;
        let gpu = self.gpu.as_ref();
        stage.copy(gpu, &spans, 0, 0, rows, true, stream)?;
        let layer = &self.layers[li];
        let attention = self.config.layer_type(li) == atlas_core::config::LayerType::FullAttention;
        for (o, seq) in seqs.iter_mut().enumerate() {
            stage.copy(gpu, &spans, 0, o * ROWS, ROWS, false, stream)?;
            let owner_ctx = ForwardContext {
                attn_metadata: Some(metas[o]),
                midchunk_capture: None,
                ..*ctx
            };
            if attention {
                let positions: Vec<usize> = (0..ROWS).map(|t| seq.seq_len + t).collect();
                let tables = vec![seq.block_table.clone(); ROWS];
                let mut states: [&mut (dyn crate::layer::LayerState + 'static); 1] =
                    [seq.layer_states[li].as_mut()];
                layer.decode_multi_seq_rows(
                    b.hidden_states(),
                    b.residual(),
                    ROWS,
                    &mut states,
                    &[0; ROWS],
                    kv_cache,
                    &positions,
                    &tables,
                    &owner_ctx,
                    stream,
                )?;
            } else {
                let seq = &mut **seq;
                layer.decode_batched(
                    b.hidden_states(),
                    b.residual(),
                    ROWS,
                    seq.layer_states[li].as_mut(),
                    kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    &owner_ctx,
                    stream,
                )?;
            }
            stage.copy(gpu, &spans, 0, o * ROWS, ROWS, true, stream)?;
        }
        stage.copy(gpu, &spans, 0, 0, rows, false, stream)
    }

    /// Oracle helper: copy every owner's KDA recurrent/conv state into a fresh
    /// buffer (`save`), or back from `buffer` (restore).
    fn glm_long_snapshot(
        &self,
        seqs: &[&mut SequenceState],
        save: bool,
        buffer: Option<DevicePtr>,
    ) -> Result<DevicePtr> {
        let conv = self.config.ssm_conv_state_bytes();
        let h = self.ssm_pool.h_stored_bytes;
        let spans: Vec<(DevicePtr, usize)> = seqs
            .iter()
            .flat_map(|seq| seq.layer_states.iter())
            .filter_map(|s| s.as_any().downcast_ref::<crate::layer::SsmLayerState>())
            .flat_map(|s| [(s.h_state, h), (s.conv_state, conv)])
            .collect();
        let total: usize = spans.iter().map(|&(_, b)| b).sum();
        let base = match buffer {
            Some(b) => b,
            None => self.gpu.alloc(total)?,
        };
        let stream = self.gpu.default_stream();
        let mut at = 0usize;
        for (ptr, bytes) in spans {
            let (src, dst) = if save {
                (ptr, base.offset(at))
            } else {
                (base.offset(at), ptr)
            };
            self.gpu.copy_d2d_async(src, dst, bytes, stream)?;
            at += bytes;
        }
        Ok(base)
    }

    /// `ATLAS_GLM_LONG_BATCH_ORACLE=1`: compare the batched verify against the
    /// ordinary serial verify of every owner from the same state, then continue
    /// with the SERIAL results. Logs argmax agreement, max |logit delta| and the
    /// serial top-2 margin per row.
    fn glm_long_oracle(
        &self,
        tokens: &[[u32; ROWS]],
        seqs: &mut [&mut SequenceState],
        stage: &GlmLongStage,
        snapshot: DevicePtr,
        batched_ids: &[u32],
    ) -> Result<Vec<[u32; ROWS]>> {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static ROWS_SEEN: AtomicU64 = AtomicU64::new(0);
        static MISMATCH: AtomicU64 = AtomicU64::new(0);
        let vocab = self.config.vocab_size;
        let row_bytes = vocab * 2;
        let n = seqs.len();
        let mut batched = vec![0u8; n * ROWS * row_bytes];
        self.gpu.copy_d2h(self.buffers.logits(), &mut batched)?;
        self.glm_long_snapshot(seqs, false, Some(snapshot))?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        self.gpu.free(snapshot)?;
        let bf = |b: &[u8], i: usize| {
            f32::from_bits((u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16)
        };
        let mut out = Vec::with_capacity(n);
        for (o, seq) in seqs.iter_mut().enumerate() {
            let ids = self.decode_verify_graphed_kgamma(&tokens[o], seq, 0)?;
            let mut serial = vec![0u8; ROWS * row_bytes];
            self.gpu.copy_d2h(self.buffers.logits(), &mut serial)?;
            stage.copy(
                self.gpu.as_ref(),
                &self.glm_long_final_spans(stage),
                0,
                o * ROWS,
                ROWS,
                true,
                self.gpu.default_stream(),
            )?;
            for r in 0..ROWS {
                let b = &batched[(o * ROWS + r) * row_bytes..][..row_bytes];
                let s = &serial[r * row_bytes..][..row_bytes];
                let (mut diff, mut top, mut second) = (0f32, f32::MIN, f32::MIN);
                for i in 0..vocab {
                    let (x, y) = (bf(b, i), bf(s, i));
                    diff = diff.max((x - y).abs());
                    if y > top {
                        second = top;
                        top = y;
                    } else if y > second {
                        second = y;
                    }
                }
                ROWS_SEEN.fetch_add(1, Relaxed);
                let batched_id = batched_ids[o * ROWS + r];
                if batched_id != ids[r] {
                    MISMATCH.fetch_add(1, Relaxed);
                    tracing::warn!(
                        "GLM long oracle MISMATCH owner={o} row={r} pos={} batched={batched_id} \
                         serial={} max_dlogit={diff:.4} serial_margin={:.4}",
                        seq.seq_len - ROWS + r,
                        ids[r],
                        top - second
                    );
                } else if diff > 0.5 {
                    tracing::info!(
                        "GLM long oracle owner={o} row={r} max_dlogit={diff:.4} margin={:.4}",
                        top - second
                    );
                }
            }
            out.push([ids[0], ids[1], ids[2]]);
        }
        let seen = ROWS_SEEN.load(Relaxed);
        if seen % 600 < (n * ROWS) as u64 {
            tracing::info!(
                "GLM long oracle summary: rows={seen} argmax_mismatch={}",
                MISMATCH.load(Relaxed)
            );
        }
        Ok(out)
    }

    /// Put owner `owner`'s final verify rows back at arena rows [0, 3).
    pub(super) fn glm_long_restore_owner(&self, owner: usize) -> Result<()> {
        ensure!(owner < MAX_OWNERS, "GLM long owner index {owner}");
        let stage = self
            .glm_long_stage
            .context("GLM long owner stage missing")?;
        stage.copy(
            self.gpu.as_ref(),
            &self.glm_long_final_spans(&stage),
            0,
            owner * ROWS,
            ROWS,
            false,
            self.gpu.default_stream(),
        )
    }

    /// Head: announce and run the batched traversal.
    pub(super) fn decode_verify_glm_long_owners_impl(
        &self,
        tokens: &[[u32; ROWS]],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<[u32; ROWS]>> {
        ensure!(
            self.can_batch_glm_long_verify_impl(seqs.len()),
            "GLM long owner verify is not available"
        );
        let slots: Vec<u32> = seqs.iter().map(|s| s.slot_idx as u32).collect();
        let flat: Vec<u32> = tokens.iter().flatten().copied().collect();
        self.ep_broadcast_seq_and_cmd(0, EP_CMD_GLM_LONG_VERIFY, true)?;
        self.ep_broadcast_u32(slots.len() as u32)?;
        self.ep_broadcast_tokens(&slots)?;
        self.ep_broadcast_tokens(&flat)?;
        self.glm_long_owner_compute(tokens, seqs)
    }

    /// Head: announce owner `owner`'s tail and restore its rows. The caller
    /// then runs the ordinary verdict tail, beginning with its verdict word.
    pub(super) fn begin_glm_long_owner_tail_impl(
        &self,
        slot: u32,
        owner: usize,
        tokens: &[u32; ROWS],
    ) -> Result<()> {
        self.ep_broadcast_seq_and_cmd(slot, EP_CMD_GLM_LONG_TAIL, true)?;
        self.ep_broadcast_u32(owner as u32)?;
        self.ep_broadcast_tokens(tokens)?;
        self.glm_long_restore_owner(owner)
    }

    /// Worker side of E9.
    pub(super) fn glm_long_receive_verify(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        ensure!(
            (1..=MAX_OWNERS).contains(&n) && n <= slots.len(),
            "GLM long owner verify width {n}"
        );
        let ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let flat = self.ep_broadcast_tokens(&vec![0u32; n * ROWS])?;
        let mut seen = [false; 64];
        for &id in &ids {
            let id = id as usize;
            ensure!(
                id < slots.len() && id < seen.len() && !seen[id],
                "GLM long owner verify slot {id} invalid or repeated"
            );
            seen[id] = true;
        }
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
                .with_context(|| format!("GLM long owner verify slot {id} unallocated"))?;
            seqs.push(refs.swap_remove(at).1);
        }
        let tokens: Vec<[u32; ROWS]> = flat
            .chunks_exact(ROWS)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        self.sync_secondary()?;
        self.glm_long_owner_compute(&tokens, &mut seqs)?;
        Ok(true)
    }

    /// Worker side of EA: restore this owner's rows, then the F5 verdict tail.
    pub(super) fn glm_long_receive_tail(&self, seq: &mut SequenceState) -> Result<()> {
        let owner = self.ep_broadcast_u32(0)? as usize;
        let tokens = self.ep_broadcast_tokens(&[0u32; ROWS])?;
        self.glm_long_restore_owner(owner)?;
        let accepted = self.ep_broadcast_u32(0)? as usize;
        ensure!(accepted < ROWS, "GLM long owner verdict {accepted} for K3");
        let base = seq
            .seq_len
            .checked_sub(ROWS)
            .context("GLM long owner verify base underflow")?;
        self.ep_worker_apply_verdict(seq, base, &tokens, accepted)
    }
}

/// `ATLAS_GLM_LONG_BATCH_SERIAL=kda|mla|all`: bisection aid that keeps the
/// named layer kind on its single-owner verify inside the batched step.
fn serial_diagnostic(attention: bool) -> bool {
    static MODE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let mode =
        MODE.get_or_init(|| std::env::var("ATLAS_GLM_LONG_BATCH_SERIAL").unwrap_or_default());
    match mode.as_str() {
        "all" => true,
        "mla" => attention,
        "kda" => !attention,
        _ => false,
    }
}

fn oracle_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_LONG_BATCH_ORACLE").as_deref() == Ok("1"))
}
