// SPDX-License-Identifier: AGPL-3.0-only

//! Model side of the qwen4_exp mid-chunk prefix-cache checkpoint
//! (`ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1`, default off; the layer side and
//! the why are in `layers::qwen4exp_ckpt`).
//!
//! The last chunk of a prefix-caching prompt runs as ONE pass (no tail
//! split), so its numerics are the caching-off run's, and the checkpoint the
//! split existed for is captured inside the pass at `cp`: the last 64-token
//! chunk boundary of the pass at or below the tail cut (so `cp` sits up to 63
//! tokens below where the split put it; the next turn replays those). Plan:
//! reserve a snapshot slot and the per-GDN-layer destinations
//! ([`MidCapturePlan`] with `ckpt` set, which the GDN conv split and the
//! layers read through `ForwardContext::midchunk_capture`), arm the per-pass
//! layer hooks. Finalize: require every GDN layer's state and the PLE carry
//! captured, rebuild the PLE history and QSA keys at `cp`, attach them as
//! the slot's aux, and index it as the intermediate checkpoint of
//! `tokens[..cp]` -- the same registration the tail split's first pass made
//! (`register_intermediate_checkpoint`). Anything missing frees the slot.
//!
//! Every rank plans from the same tokens, pass and config; the restore
//! depth of the next turn is agreed across ranks as before
//! (`pc_policy::agree_restore`), so a rank that could not capture (no free
//! slot) only makes the pair restore shallower.

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;
use std::cell::Cell;

use super::super::super::types::TransformerModel;
use super::midchunk_capture::MidCapturePlan;
use crate::layers::qwen4exp_ckpt as ckpt;
use crate::traits::SequenceState;

thread_local! {
    /// `(ptr, bytes)` of this thread's PLE conv-carry staging buffer.
    static PLE_STAGE: Cell<(u64, usize)> = const { Cell::new((0, 0)) };
}

/// Where `cp` lands for a pass over `[start, start + count)` and a tail cut
/// `cut`: the last 64-token chunk boundary of the pass at or below the cut,
/// when that is a block boundary strictly inside the pass.
pub(super) fn ckpt_row(start: usize, count: usize, cut: usize, bs: usize) -> Option<usize> {
    if bs == 0 || !(start < cut && cut < start + count) {
        return None;
    }
    let cp = start + (cut - start) / ckpt::CHUNK * ckpt::CHUNK;
    (cp > start && cp.is_multiple_of(bs)).then_some(cp)
}

impl TransformerModel {
    /// The last chunk runs unsplit: the checkpoint is captured in-pass.
    pub(super) fn qwen4exp_ckpt_takes_tail(&self) -> bool {
        ckpt::requested() && self.config.model_type == "qwen4_exp"
    }

    /// The pass's in-pass capture plan: the qwen4_exp checkpoint, else the
    /// generic mid-chunk tail capture (`midchunk_capture`).
    pub(super) fn plan_pass_capture(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        span: [usize; 2],
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<Option<MidCapturePlan>> {
        if let Some(p) =
            self.prepare_qwen4exp_ckpt(tokens, seq, kv_cache, span, is_last_chunk, stream)?
        {
            return Ok(Some(p));
        }
        let [start, count] = span;
        Ok(self.prepare_midchunk_capture(tokens, seq, kv_cache, start, count, stream))
    }

    /// Plan the in-pass checkpoint of the last chunk's pass over
    /// `[proc_start, proc_start + proc_count)`, or `None`.
    pub(super) fn prepare_qwen4exp_ckpt(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        span: [usize; 2],
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<Option<MidCapturePlan>> {
        let [proc_start, proc_count] = span;
        // A pass that failed after planning left its hooks armed: disarm.
        let _ = ckpt::end();
        if !is_last_chunk
            || !self.qwen4exp_ckpt_takes_tail()
            || !self.ssm_snapshots.is_enabled()
            || !self.prefix_cache.is_active()
            || self.tokens_have_vision_pad(tokens)
        {
            return Ok(None);
        }
        let bs = kv_cache.block_size();
        let cut = super::pc_policy::tail_cut(tokens.len(), bs);
        let ratio = self.config.indexer_compress_ratio.max(1);
        let Some(cp) = ckpt_row(proc_start, proc_count, cut, bs).filter(|cp| cp % ratio == 0)
        else {
            tracing::info!(
                "qwen4_exp mid-chunk checkpoint: no 64-token boundary in [{proc_start}, {cut}]"
            );
            return Ok(None);
        };
        let Some(slot) = self.reserve_snapshot_slot(seq.session_hash, kv_cache) else {
            tracing::warn!("qwen4_exp mid-chunk checkpoint: no snapshot slot for token {cp}");
            return Ok(None);
        };
        let n = self.ssm_snapshots.num_ssm_layers();
        let h_dsts = (0..n)
            .map(|l| self.ssm_snapshots.tail_h_dst(l, slot))
            .collect();
        let conv_dsts = (0..n)
            .map(|l| self.ssm_snapshots.tail_conv_dst(l, slot))
            .collect();
        let ple_dst = self.ple_stage(stream)?;
        ckpt::begin(cp - proc_start, ple_dst);
        Ok(Some(MidCapturePlan {
            cap_local: cp - proc_start,
            snap_slot: slot,
            tb: cp,
            h_dsts,
            conv_dsts,
            h_bytes: self.ssm_snapshots.h_bytes(),
            conv_bytes: self.ssm_snapshots.conv_bytes(),
            bs,
            cap_local_early: None,
            snap_slot_early: None,
            tb_early: None,
            h_dsts_early: Vec::new(),
            conv_dsts_early: Vec::new(),
            ckpt: true,
        }))
    }

    /// After the pass: attach the aux at `cp` and index the checkpoint, or
    /// free the slot when anything was not captured.
    pub(super) fn finalize_qwen4exp_ckpt(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        plan: &MidCapturePlan,
        stream: u64,
    ) -> Result<()> {
        let (h_done, ple_done) = ckpt::end();
        let has_ple = self.config.ple_layer_ids.iter().any(|&id| id > 0);
        let n = self.ssm_snapshots.num_ssm_layers();
        // Never index blocks past the contiguous fully-written KV (as
        // `prefill_b_save_checkpoint`).
        let kv_ok = seq.kv_valid_tokens / plan.bs >= plan.tb / plan.bs;
        if h_done != n || (has_ple && !ple_done) || !kv_ok {
            tracing::warn!(
                "qwen4_exp mid-chunk checkpoint at token {}: captured {h_done}/{n} GDN states, \
                 PLE {ple_done}, KV valid {kv_ok}; not registered",
                plan.tb
            );
            self.ssm_snapshots.free(plan.snap_slot);
            return Ok(());
        }
        let mut aux = self.ssm_snapshots.take_aux(plan.snap_slot);
        self.collect_aux_states_into(seq, stream, &mut aux)?;
        for (i, blob) in aux.iter_mut() {
            let rebuilt = match self.config.layer_type(*i as usize) {
                LayerType::FullAttention | LayerType::SlidingAttention => {
                    let row = self.config.indexer_head_dim * 2;
                    qsa_blob_at(blob, plan.tb, self.config.indexer_compress_ratio, row)
                }
                _ => self.ple_blob_at(blob, tokens, plan.tb, stream)?,
            };
            match rebuilt {
                Some(b) => *blob = b,
                None => {
                    tracing::warn!(
                        "qwen4_exp mid-chunk checkpoint at token {}: layer {i} aux cannot be \
                         rebuilt; not registered",
                        plan.tb
                    );
                    self.ssm_snapshots.free(plan.snap_slot);
                    return Ok(());
                }
            }
        }
        if !aux.is_empty() {
            self.ssm_snapshots.set_aux(plan.snap_slot, aux);
        }
        if self.register_intermediate_checkpoint(
            tokens,
            seq,
            kv_cache,
            plan.tb,
            plan.snap_slot,
            false,
        ) {
            tracing::info!(
                "qwen4_exp mid-chunk SSM checkpoint saved at token {} (snapshot_id {})",
                plan.tb,
                plan.snap_slot
            );
        }
        Ok(())
    }

    /// The PLE aux blob `[n u32][history][conv]` at `cp`: the last `n` token
    /// ids below `cp`, and the conv carry captured at `cp`.
    fn ple_blob_at(
        &self,
        blob: &[u8],
        tokens: &[u32],
        cp: usize,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        if blob.len() < 4 {
            return Ok(None);
        }
        let n = u32::from_le_bytes(blob[..4].try_into().expect("4 bytes")) as usize;
        let conv_bytes = blob.len().saturating_sub(4 + n * 4);
        let (stage, size) = PLE_STAGE.with(Cell::get);
        if cp < n || conv_bytes == 0 || conv_bytes > size {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(blob.len());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        for t in &tokens[cp - n..cp] {
            out.extend_from_slice(&t.to_le_bytes());
        }
        let off = out.len();
        out.resize(off + conv_bytes, 0);
        self.gpu
            .copy_d2h_on_stream(DevicePtr(stage), &mut out[off..], stream)?;
        Ok(Some(out))
    }

    /// The PLE conv-carry staging buffer (grown once to the carry's size).
    fn ple_stage(&self, stream: u64) -> Result<DevicePtr> {
        // `[(k - 1) * dilation, hc_mult * hidden]` FP32; the dilation is the
        // n-gram neighbour count (`weight_loader::qwen4_exp::ple`).
        let c = self.config.hc_mult * self.config.hidden_size;
        let steps = self.config.ple_conv_kernel_size.saturating_sub(1)
            * self.config.emb_neighbor_num.max(1);
        let bytes = steps.max(1) * c * 4;
        let (ptr, size) = PLE_STAGE.with(Cell::get);
        if size >= bytes {
            return Ok(DevicePtr(ptr));
        }
        if ptr != 0 {
            self.gpu.synchronize(stream)?;
            self.gpu.free(DevicePtr(ptr))?;
        }
        let p = self.gpu.alloc(bytes)?;
        PLE_STAGE.with(|s| s.set((p.0, bytes)));
        Ok(p)
    }
}

/// The QSA aux blob `[ingested u64][pooled u64][pooled keys][raw tail]` at
/// `cp` (a block boundary): `cp` ingested, `cp / ratio` blocks pooled -- the
/// first rows of the pass-end blob's pooled keys, which a block's key depends
/// on nothing else to form -- and no raw tail. `None` when the pass-end blob
/// does not reach `cp`.
pub(super) fn qsa_blob_at(blob: &[u8], cp: usize, ratio: usize, row: usize) -> Option<Vec<u8>> {
    if blob.len() < 16 || ratio == 0 || !cp.is_multiple_of(ratio) {
        return None;
    }
    let ingested = u64::from_le_bytes(blob[..8].try_into().ok()?) as usize;
    let pooled = u64::from_le_bytes(blob[8..16].try_into().ok()?) as usize;
    let keep = cp / ratio;
    if ingested < cp || pooled < keep || blob.len() < 16 + pooled * row {
        return None;
    }
    let mut out = Vec::with_capacity(16 + keep * row);
    out.extend_from_slice(&(cp as u64).to_le_bytes());
    out.extend_from_slice(&(keep as u64).to_le_bytes());
    out.extend_from_slice(&blob[16..16 + keep * row]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{ckpt_row, qsa_blob_at};

    #[test]
    fn the_row_is_the_last_chunk_boundary_at_or_below_the_cut() {
        // Cold 16046-token prompt: cut 16016 (bs 16) -> 16000.
        assert_eq!(ckpt_row(0, 16046, 16016, 16), Some(16000));
        // Last chunk of a 28K prompt starting at 16384.
        assert_eq!(ckpt_row(16384, 11616, 27984, 16), Some(27968));
        // A warm pass from a block boundary that is not 64-aligned.
        assert_eq!(ckpt_row(1008, 900, 1888, 16), Some(1840));
        // No full chunk below the cut, or the cut outside the pass.
        assert_eq!(ckpt_row(1000, 100, 1050, 16), None);
        assert_eq!(ckpt_row(0, 500, 600, 16), None);
        assert_eq!(ckpt_row(0, 500, 0, 16), None);
        for start in (0..512).step_by(16) {
            for cut in start + 1..start + 700 {
                if let Some(cp) = ckpt_row(start, 1000, cut, 16) {
                    assert!(cp > start && cp <= cut && cut - cp < 64 && (cp - start) % 64 == 0);
                    assert_eq!(cp % 16, 0);
                }
            }
        }
    }

    #[test]
    fn the_qsa_blob_keeps_the_pooled_prefix() {
        let (ratio, row) = (4usize, 6usize);
        let mut blob = Vec::new();
        blob.extend_from_slice(&(1003u64).to_le_bytes());
        blob.extend_from_slice(&(250u64).to_le_bytes());
        let keys: Vec<u8> = (0..250 * row).map(|i| (i % 251) as u8).collect();
        blob.extend_from_slice(&keys);
        blob.extend_from_slice(&[9u8; 3 * 6]); // raw tail of 3 rows
        let got = qsa_blob_at(&blob, 960, ratio, row).unwrap();
        assert_eq!(u64::from_le_bytes(got[..8].try_into().unwrap()), 960);
        assert_eq!(u64::from_le_bytes(got[8..16].try_into().unwrap()), 240);
        assert_eq!(&got[16..], &keys[..240 * row]);
        assert!(
            qsa_blob_at(&blob, 962, ratio, row).is_none(),
            "not a block boundary"
        );
        assert!(
            qsa_blob_at(&blob, 1004, ratio, row).is_none(),
            "past ingested"
        );
    }
}
