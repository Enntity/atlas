// SPDX-License-Identifier: AGPL-3.0-only

//! In-pass prefill checkpoint (`ATLAS_GLM_KDA_INPASS_CKPT`).
//!
//! When a prefill pass carries `ForwardContext::midchunk_capture`, this layer
//! runs its convolution and its recurrence as two calls each, over rows
//! `[0, cut)` and `[cut, tokens)`, and between the two calls copies the live
//! conv and recurrent state into the pass's reserved snapshot slot. Both
//! kernels carry their state across calls in place and keep their per-row
//! arithmetic: the convolution is a sliding window over inputs (fixed tap
//! order), the recurrences walk tokens in order over FP32 state. So the two
//! calls compute what one call over the pass computes, with the state@cut
//! read off in between. Every other stage of this layer, and every other
//! layer, still runs over the whole pass.

use super::*;

/// Row segments `(row0, rows)` of one prefill recurrence over `tokens` rows:
/// two when `cut` lies strictly inside, else one.
pub(super) fn segments(tokens: usize, cut: Option<usize>) -> Vec<(usize, usize)> {
    match cut {
        Some(cut) if cut > 0 && cut < tokens => vec![(0, cut), (cut, tokens - cut)],
        _ => vec![(0, tokens)],
    }
}

impl Glm5KdaLayer {
    /// The pass's capture point, for a prefill recurrence of `tokens` rows.
    pub(super) fn inpass_cut(
        &self,
        tokens: usize,
        decode: bool,
        ctx: &ForwardContext,
    ) -> Option<usize> {
        let cap = ctx.midchunk_capture.as_ref()?;
        (!decode && cap.cap_local > 0 && cap.cap_local < tokens).then_some(cap.cap_local)
    }

    /// Copy this layer's live recurrent (`h == true`) or conv state into the
    /// pass's snapshot slot: the bytes `SsmSnapshotPool::save` copies from
    /// the sequence's pool slot, which this layer state points into. No-op
    /// when the pass reserved no slot: the pass still splits at the cut, so
    /// its arithmetic never depends on the snapshot pool.
    pub(super) fn inpass_copy(
        &self,
        state: &SsmLayerState,
        h: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(cap) = ctx.midchunk_capture.as_ref() else {
            return Ok(());
        };
        let (dsts, src, bytes) = if h {
            (cap.h_dsts, state.h_state, cap.h_bytes)
        } else {
            (cap.conv_dsts, state.conv_state, cap.conv_bytes)
        };
        match dsts.get(self.ssm_ordinal) {
            Some(&dst) => ctx.gpu.copy_d2d_async(src, dst, bytes, stream),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::segments;

    #[test]
    fn a_cut_strictly_inside_splits_into_two_covering_segments() {
        assert_eq!(segments(8_000, Some(7_872)), vec![(0, 7_872), (7_872, 128)]);
    }

    #[test]
    fn no_cut_or_a_cut_on_an_edge_keeps_one_call() {
        for cut in [None, Some(0), Some(8_000), Some(9_000)] {
            assert_eq!(segments(8_000, cut), vec![(0, 8_000)]);
        }
    }
}
