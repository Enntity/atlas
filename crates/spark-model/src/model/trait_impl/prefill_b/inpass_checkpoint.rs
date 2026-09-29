// SPDX-License-Identifier: AGPL-3.0-only

//! In-pass tail checkpoint (`ATLAS_GLM_KDA_INPASS_CKPT=1`, default off).
//!
//! The tail split (`tail_split`) runs a prompt's last chunk as two full
//! passes, `[start, cut)` + `[cut, len)`, so the prefill checkpoint at `cut`
//! is a pass end. With this flag, and when every recurrent layer can split
//! its own recurrence (`TransformerLayer::captures_ssm_state_in_pass`, the
//! GLM KDA layer), the chunk runs as one pass instead: the recurrent layers
//! split only their convolution and recurrence at `cut` and copy the state
//! there into a snapshot slot reserved before the pass, and the slot is then
//! registered exactly as the split's `prefill_b_save_checkpoint` registers
//! its snapshot. Projections, attention, MoE and dense layers see the whole
//! chunk once.
//!
//! The split is decided exactly as before (same `cut`, a pure function of
//! tokens, configuration and environment), and whether the recurrences split
//! depends only on the pass's row range, never on the snapshot pool: without
//! a free slot the pass still splits and copies nothing. So every rank runs
//! the same kernels. Aux-carrying models are excluded (their aux state would
//! be the pass end's, not the cut's).

use anyhow::{Result, bail};
use atlas_core::config::LayerType;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use super::midchunk_capture::MidCapturePlan;
use crate::traits::SequenceState;

/// How a last chunk takes its tail checkpoint at `cut`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) enum TailCheckpoint {
    /// Two passes, `[start, cut)` + `[cut, len)`.
    Split(usize),
    /// One pass; the recurrent layers capture their state at `cut`.
    InPass(usize),
}

/// `ATLAS_GLM_KDA_INPASS_CKPT`: unset or `0` off, `1` on, anything else an
/// error.
pub(in crate::model) fn parse_inpass_flag(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_GLM_KDA_INPASS_CKPT must be 0 or 1, got {other:?}"),
    }
}

fn inpass_requested() -> Result<bool> {
    static FLAG: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    FLAG.get_or_init(|| {
        parse_inpass_flag(std::env::var("ATLAS_GLM_KDA_INPASS_CKPT").ok().as_deref())
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// Whether `cut` falls strictly inside the pass `[proc_start, proc_start +
/// proc_count)`, i.e. the pass computes the rows on both sides of it.
pub(in crate::model) fn pass_spans(cut: usize, proc_start: usize, proc_count: usize) -> bool {
    proc_start < cut && cut < proc_start + proc_count
}

impl TransformerModel {
    /// The tail checkpoint of the chunk `[chunk_start, len)` of `tokens`, if
    /// the chunk takes one (`prefill_tail_split`).
    pub(in crate::model) fn prefill_tail_checkpoint(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        is_last_chunk: bool,
    ) -> Result<Option<TailCheckpoint>> {
        let Some(cut) = self.prefill_tail_split(tokens, chunk_start, is_last_chunk) else {
            return Ok(None);
        };
        Ok(Some(
            if inpass_requested()? && self.inpass_capture_supported() {
                TailCheckpoint::InPass(cut)
            } else {
                TailCheckpoint::Split(cut)
            },
        ))
    }

    /// Every recurrent layer captures its state in pass, and no layer keeps
    /// aux state. A function of the loaded model alone.
    fn inpass_capture_supported(&self) -> bool {
        !self.requires_aux_state()
            && self.layers.iter().enumerate().all(|(i, layer)| {
                self.config.layer_type(i) != LayerType::LinearAttention
                    || layer.captures_ssm_state_in_pass()
            })
    }

    /// Plan the in-pass capture at `cut` for the pass `[proc_start, proc_start
    /// + proc_count)`: `None` when the pass does not span `cut` (the split
    /// would have left that side cached, saving nothing). Reserves the
    /// snapshot slot, reclaiming from the cache on exhaustion; without one
    /// the plan still splits the recurrences.
    pub(in crate::model) fn prepare_inpass_checkpoint(
        &self,
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        cut: usize,
        proc_start: usize,
        proc_count: usize,
    ) -> Option<MidCapturePlan> {
        if !pass_spans(cut, proc_start, proc_count) {
            return None;
        }
        let slot = self.reserve_snapshot_slot(seq.session_hash, kv_cache);
        let n = self.ssm_snapshots.num_ssm_layers();
        let (h_dsts, conv_dsts) = match slot {
            Some(s) => (
                (0..n)
                    .map(|l| self.ssm_snapshots.tail_h_dst(l, s))
                    .collect(),
                (0..n)
                    .map(|l| self.ssm_snapshots.tail_conv_dst(l, s))
                    .collect(),
            ),
            None => {
                tracing::warn!(
                    "SSM snapshot pool exhausted and no evictable cached entries — \
                     in-pass checkpoint at token {cut} dropped"
                );
                (Vec::new(), Vec::new())
            }
        };
        Some(MidCapturePlan {
            cap_local: cut - proc_start,
            snap_slot: slot,
            checkpoint: true,
            tb: cut,
            h_dsts,
            conv_dsts,
            h_bytes: self.ssm_snapshots.h_bytes(),
            conv_bytes: self.ssm_snapshots.conv_bytes(),
            bs: kv_cache.block_size(),
            cap_local_early: None,
            snap_slot_early: None,
            tb_early: None,
            h_dsts_early: Vec::new(),
            conv_dsts_early: Vec::new(),
        })
    }

    /// After the pass: register the captured slot as the prefill checkpoint
    /// at the cut, under the split's own checks (the blocks below the cut
    /// hold fully written K/V). Without `completed` (the pass failed) the
    /// slot is only returned.
    pub(in crate::model) fn finish_inpass_checkpoint(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        plan: &MidCapturePlan,
        completed: bool,
    ) -> Result<()> {
        let Some(slot) = plan.snap_slot else {
            return Ok(());
        };
        let end_block = plan.tb / plan.bs;
        if !completed || seq.kv_valid_tokens / plan.bs < end_block {
            self.ssm_snapshots.free(slot);
            return Ok(());
        }
        self.prefill_b_register_checkpoint(tokens, seq, kv_cache, plan.tb, slot)
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_inpass_flag, pass_spans};

    #[test]
    fn the_flag_is_explicit_and_strict() {
        assert!(!parse_inpass_flag(None).unwrap());
        assert!(!parse_inpass_flag(Some("0")).unwrap());
        assert!(parse_inpass_flag(Some("1")).unwrap());
        assert!(parse_inpass_flag(Some("true")).is_err());
    }

    #[test]
    fn only_a_pass_computing_both_sides_of_the_cut_captures() {
        // Cold last chunk [8_192, 16_000) cut at 15_872.
        assert!(pass_spans(15_872, 8_192, 7_808));
        // Warm pass resuming exactly at the cut: the split's first pass was
        // wholly cached and saved nothing; the restored state is the cut's.
        assert!(!pass_spans(15_872, 15_872, 128));
        // Warm pass resuming past the cut.
        assert!(!pass_spans(15_872, 15_936, 64));
        // A pass ending at the cut.
        assert!(!pass_spans(15_872, 8_192, 7_680));
    }
}
