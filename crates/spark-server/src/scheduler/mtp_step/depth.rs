// SPDX-License-Identifier: AGPL-3.0-only

//! Per-step draft-depth adjustments on the MTP verify dispatch: the n=1
//! single-depth adaptation, the lone-DFlash width, and the serial ladder cap.

use super::*;

/// `ladder_nd` after the n=1 single-depth adaptation.
pub(super) fn single_depth_ladder(
    active: &[ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    ladder_nd: usize,
    dflash_verify_raw_argmax: bool,
) -> usize {
    // At n=1, GLM can choose between the measured K=3 and K=5 kernels from
    // this request's own acceptance history.  This composes after the global
    // concurrency ladder and is a carried per-run lever rather than global
    // process state (`ATLAS_MTP_SINGLE_DEPTH_ADAPT=1`).
    if active.len() == 1 && !dflash_verify_raw_argmax && sched.levers.mtp_single_depth_adapt {
        active[0]
            .mtp_acct
            .depth_drafts(ladder_nd, sched.levers.mtp_single_depth_adapt)
    } else {
        ladder_nd
    }
}

/// Trim a serial DFlash verify (drafts taken with their confidences `conf`)
/// to its cost-aware width and log the width.
pub(super) fn lone_dflash_width(
    a: &ActiveSeq,
    conf: &[f32],
    drafts: &mut Vec<u32>,
    dflash_verify_raw_argmax: bool,
) {
    // A lone DFlash verify pays single-owner cost for each row it adds.
    if dflash_verify_raw_argmax
        && let Some(width) = super::dflash_width::choose(
            std::iter::once((&a.spec_adapt.survival, conf)),
            drafts.len(),
        )
    {
        drafts.truncate(width);
    }
    if dflash_verify_raw_argmax {
        super::dflash_width::log_verify(1, drafts.len());
    }
}

/// Cap a grammarless serial verify (drafts with confidences `conf`) at the
/// step's ladder depth, or a long copy at [`lone_copy_cap`].
pub(super) fn ladder_truncate(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &mut Vec<u32>,
    conf: &[f32],
    ladder_nd: usize,
    max_seq_len: usize,
) {
    // The n=1 path does not pass through the batched partition's depth
    // truncation above.  Honor the same ladder decision here so a K5
    // proposal can immediately step down to the cheaper K3 verifier.
    let cap = lone_copy_cap(
        ladder_nd,
        spark_model::speculative::long_copy::drafts(),
        model.lone_verify_draft_capacity(a.seq.slot_idx),
        super::super::dflash_conf_width::is_copy(conf),
        a.seq.seq_len,
        max_seq_len,
    );
    if a.grammar_state.is_none() && drafts.len() > cap {
        drafts.truncate(cap);
    }
}

/// Drafts a lone owner verifies at most: the ladder's `ladder_nd`, or for a
/// copy (`copy_drafts`) up to `long` (0 = long copies off) within the
/// drafts its lone verify may hold now (`slot_cap`,
/// `Model::lone_verify_draft_capacity`: rows lent by idle slots, so a
/// request admitted since the proposal cuts it back) and below the context
/// ceiling (`mtp_gate::ceiling`, which gated the step at `ladder_nd` only).
pub(super) fn lone_copy_cap(
    ladder_nd: usize,
    long: usize,
    slot_cap: usize,
    copy: bool,
    seq_len: usize,
    max_seq_len: usize,
) -> usize {
    let long = long.min(slot_cap);
    if copy
        && long > ladder_nd
        && super::super::mtp_gate::ceiling::verify_rows_fit(seq_len, long, max_seq_len)
    {
        long
    } else {
        ladder_nd
    }
}

#[cfg(test)]
mod tests {
    use super::lone_copy_cap;

    #[test]
    fn only_a_long_copy_verifies_past_the_ladder_within_its_slot_and_the_ceiling() {
        // (ladder, long, slot capacity, copy, seq_len, max_seq_len) -> cap
        let cap = |long, slot, copy, seq_len| lone_copy_cap(7, long, slot, copy, seq_len, 1000);
        // Long copies off, or DFlash2's block: the ladder, as shipped.
        assert_eq!(cap(0, 31, true, 10), 7);
        assert_eq!(cap(31, 31, false, 10), 7);
        // A copy: the long maximum, within the slot's verify pools.
        assert_eq!(cap(31, 31, true, 10), 31);
        assert_eq!(cap(31, 15, true, 10), 15);
        assert_eq!(cap(15, usize::MAX, true, 10), 15);
        // Never below the ladder, and not where its rows would reach the
        // context ceiling (the step was gated at the ladder's width).
        assert_eq!(cap(31, 5, true, 10), 7);
        assert_eq!(cap(31, 31, true, 1000 - 32), 31);
        assert_eq!(cap(31, 31, true, 1000 - 31), 7);
        assert_eq!(lone_copy_cap(7, 31, 31, true, 10, 0), 31);
    }
}
