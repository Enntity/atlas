// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence row counts of the batched MTP verify (`verify_e`).
//!
//! A sequence verifies `drafts + 1` rows: 2..=4 under the MTP ladder. On the
//! qwen4_exp exact lane (`ATLAS_QWEN4EXP_BATCH_FAST=1` +
//! `ATLAS_QWEN4EXP_EXACT_VERIFY=1`) a sequence holding NO drafts may ride the
//! same forward at one row, a DECODE ROW: the scheduler puts every active
//! sequence in one forward a step instead of a batched verify plus a
//! batched bootstrap decode (`spark-server` `mtp_step/one_forward.rs`).
//!
//! Why one row is the serial decode row, bit for bit, on that lane: every op
//! of the verify forward computes each row with serial decode's arithmetic
//! (`qwen4exp_exact_verify.rs`'s table; the GDN chain is the per-token exact
//! arm, which runs token 0 first and needs nothing after it), and the
//! batch-fast lane makes each row independent of the rows beside it
//! (`qwen4exp_batch_fast.rs`). Row 0 of a verify window never reads the
//! window's later rows (causal attention, sequential GDN chain, per-row MoE
//! and head), so a one-row window is row 0 of any wider one. Its commit is a
//! full accept of one row: the live GDN state the chain leaves IS the state
//! after the token (`commit_accepted_prefix(1, 1)`), the QSA and PLE carries
//! advance by exactly that row, and there is no drafter row to trim.
//!
//! Every other lane keeps 2..=4: their WY verify kernels start at K=2, and
//! their verify rows are not decode's arithmetic.

/// Whether a batched MTP verify (not DFlash) admits `ks`: every count in
/// 2..=4, or 1..=4 with `decode_rows`; a batch of decode rows only is a
/// batched decode (`decode_batch`), never a verify.
pub(in crate::model) fn mtp_verify_rows_ok(ks: &[usize], decode_rows: bool) -> bool {
    let min = if decode_rows { 1 } else { 2 };
    ks.iter().all(|k| (min..=4).contains(k)) && ks.iter().any(|&k| k >= 2)
}

/// Whether the worker trims its drafter after a verified window of `k`
/// rows. A decode row (`k == 1`) drafted nothing, as the head's verdict
/// skips the trim for it (`k4_apply_verdict`).
pub(in crate::model) fn verdict_trims_drafter(k: usize) -> bool {
    k > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_rows_only_on_the_lane_that_verifies_them() {
        assert!(mtp_verify_rows_ok(&[4, 3, 2], false));
        assert!(!mtp_verify_rows_ok(&[4, 1], false));
        assert!(mtp_verify_rows_ok(&[4, 2, 1], true));
        assert!(!mtp_verify_rows_ok(&[5, 1], true), "the ladder caps at 4");
        assert!(!mtp_verify_rows_ok(&[0, 2], true));
    }

    #[test]
    fn decode_rows_alone_are_not_a_verify() {
        assert!(!mtp_verify_rows_ok(&[1, 1, 1], true));
        assert!(!mtp_verify_rows_ok(&[], true));
    }

    #[test]
    fn only_a_window_with_drafts_trims_the_drafter() {
        assert!(!verdict_trims_drafter(1));
        assert!(verdict_trims_drafter(2) && verdict_trims_drafter(4));
    }
}
