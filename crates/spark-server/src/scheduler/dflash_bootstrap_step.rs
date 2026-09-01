// SPDX-License-Identifier: AGPL-3.0-only

//! Batched first DFlash proposal for fresh concurrent sequences.
//!
//! A fresh DFlash sequence has no pending drafts. The legacy bootstrap loop
//! therefore proposed and verified each sequence completely before visiting
//! the next one: at C=4 that meant four target weight sweeps before all four
//! requests could join the steady-state batch. DFlash does not consume the
//! generic target-hidden stash, so its native `propose_batch` can initialize
//! every fresh proposer state in one pass. Phase B then verifies those equal-
//! width proposals with the existing causal sequence-major batch forward.

use super::*;

fn proposals_match_batch(proposals: &[Vec<u32>], n: usize, min_width: usize) -> bool {
    proposals.len() == n
        && proposals
            .first()
            .is_some_and(|first| first.len() >= min_width)
        && proposals
            .windows(2)
            .all(|pair| pair[0].len() == pair[1].len())
}

/// Initialize DFlash proposals for the batchable subset of `idxs`.
///
/// Returns indices consumed by this path. Successful indices carry pending
/// drafts and are added to Phase B by the caller. A failed in-flight batch is
/// also consumed after marking its sequences finished: proposer state may
/// have advanced partially, so replaying the serial bootstrap is unsafe.
pub(super) fn step_dflash_bootstrap_batched(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    idxs: &[usize],
    num_drafts: usize,
) -> Vec<usize> {
    if idxs.len() < 2
        || num_drafts < 4
        || bootstrap_batch_disabled()
        || !spark_model::speculative::mtp_multi_seq_mode()
        || sched.levers.dflash_seam_serial
    {
        return Vec::new();
    }

    let static_candidates = idxs
        .iter()
        .copied()
        .filter(|&index| !active[index].finished && active[index].grammar_state.is_none())
        .collect::<Vec<_>>();
    if static_candidates.len() < 2 {
        return Vec::new();
    }

    // This is the same once-per-step policy decision made by the serial
    // bootstrap. A second call in its fallback is harmless: an allowed
    // re-probe is already unsuspended, while a denied sequence stays denied.
    let candidates = static_candidates
        .into_iter()
        .filter(|&index| crate::scheduler::adaptive_spec::spec_allowed(&mut active[index], sched))
        .collect::<Vec<_>>();
    let n = candidates.len();
    if n < 2
        || model.mtp_propose_batch_max() < n
        || !model.can_batch_verify_dflash(n, num_drafts + 1)
    {
        return Vec::new();
    }

    let mut refs = Vec::with_capacity(n);
    let mut iterator = active.iter_mut();
    let mut consumed_through = 0usize;
    for &index in &candidates {
        let sequence = iterator
            .nth(index - consumed_through)
            .expect("DFlash bootstrap index is active");
        consumed_through = index + 1;
        refs.push(sequence);
    }
    let tokens = refs
        .iter()
        .map(|active| active.last_token)
        .collect::<Vec<_>>();
    let positions = refs
        .iter()
        .map(|active| active.seq.seq_len)
        .collect::<Vec<_>>();
    // DFlash ignores target hidden rows; zero is the established steady-state
    // convention in `step_verify_dflash_batched`.
    let stash = vec![0usize; n];
    let started = std::time::Instant::now();
    let result = {
        let mut seqs = refs
            .iter_mut()
            .map(|active| &mut active.seq)
            .collect::<Vec<_>>();
        model.run_mtp_propose_batched(&tokens, &positions, &stash, num_drafts, &mut seqs, 0, None)
    };

    match result {
        Ok(Some(proposals)) if proposals_match_batch(&proposals, n, 4) => {
            let width = proposals[0].len();
            for (active, drafts) in refs.iter_mut().zip(proposals) {
                active.pending_drafts = drafts;
            }
            tracing::info!(
                "DFLASH BATCH bootstrap: n={n} gamma={width} proposer_ms={:.1}",
                started.elapsed().as_secs_f64() * 1000.0,
            );
            candidates
        }
        Ok(None) => Vec::new(),
        Ok(Some(proposals)) => {
            tracing::error!(
                "DFlash batch bootstrap returned invalid proposal geometry: n={n}, widths={:?}",
                proposals.iter().map(Vec::len).collect::<Vec<_>>(),
            );
            for active in &mut refs {
                active.finished = true;
            }
            candidates
        }
        Err(error) => {
            tracing::error!("DFlash batch bootstrap proposal failed: {error:#}");
            for active in &mut refs {
                active.finished = true;
            }
            candidates
        }
    }
}

#[cfg(test)]
mod tests {
    use super::proposals_match_batch;

    #[test]
    fn proposal_geometry_must_cover_every_sequence_at_one_width() {
        assert!(proposals_match_batch(&[vec![1; 7], vec![2; 7]], 2, 4));
        assert!(!proposals_match_batch(&[vec![1; 7]], 2, 4));
        assert!(!proposals_match_batch(&[vec![1; 7], vec![2; 6]], 2, 4));
        assert!(!proposals_match_batch(&[vec![1; 3], vec![2; 3]], 2, 4));
    }
}
