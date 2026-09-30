// SPDX-License-Identifier: AGPL-3.0-only

//! KV budget arithmetic against the GB10 TP2 starts of 2026-09-30
//! (`--gpu-memory-utilization=0.93`, 121.7 GiB, 0.8 + 0.1 GiB reserves).

use super::*;

const GIB: f64 = (1u64 << 30) as f64;
/// Blocks the reference start sized from its 10.1 GiB budget.
const NORMAL_BLOCKS: usize = 102_066;

fn gib(x: f64) -> usize {
    (x * GIB) as usize
}

/// A start on the reference pair: `baseline`/`free_now`/`tracked` in GiB.
fn start(baseline: f64, free_now: f64, tracked: Option<f64>) -> Inputs {
    Inputs {
        total: gib(121.7),
        free_now: gib(free_now),
        utilization: 0.93,
        reserve: gib(0.8) + gib(0.1),
        baseline_free: Some(gib(baseline)),
        manual_external: None,
        tracked_own: tracked.map(gib),
    }
}

/// The sizing this module replaced, kept verbatim as the reference a start
/// with quiet co-tenants must still reproduce to the byte.
fn legacy(i: &Inputs) -> usize {
    let mut used = i.total.saturating_sub(i.free_now);
    if let Some(ext) = i.manual_external {
        used = used.saturating_sub(ext);
    } else if let Some(baseline) = i.baseline_free {
        let own = baseline.saturating_sub(i.free_now);
        if own > 0 && own <= used {
            used = own;
        }
    }
    ((i.total as f64 * i.utilization) as usize)
        .saturating_sub(used)
        .saturating_sub(i.reserve)
        .min(i.free_now.saturating_sub(i.reserve))
}

/// Blocks a budget buys, at the reference start's bytes per block.
fn blocks(bytes: usize) -> usize {
    let per_block = size(&start(117.1, 14.9, None)).bytes / NORMAL_BLOCKS;
    bytes / per_block
}

#[test]
fn normal_start_keeps_the_legacy_pool_to_the_byte() {
    // Rank 0: baseline-free 117.1 − free-now 14.9 = Atlas-own 102.2 → 10.1 GiB.
    let quiet = start(117.1, 14.9, None);
    let want = legacy(&quiet);
    assert!((want as f64 / GIB - 10.08).abs() < 0.01, "{want}");
    // Whatever a tracked measure at or below the delta says, nothing moves.
    for tracked in [None, Some(0.0), Some(60.0), Some(101.4), Some(102.2)] {
        let got = size(&start(117.1, 14.9, tracked));
        assert_eq!(got.bytes, want, "tracked {tracked:?}");
        assert_eq!(got.basis, Basis::Delta);
        assert_eq!(got.own, gib(117.1) - gib(14.9));
        assert!(!got.headroom_limited);
    }
    assert_eq!(blocks(want), NORMAL_BLOCKS);
}

#[test]
fn memory_released_by_another_process_is_not_credited_to_atlas() {
    // The bad start: a co-tenant freed 2.2 GiB (rank 0) and 3.4 GiB (rank 1)
    // during the load, so the delta read 100.1 and 97.9 for a process that
    // holds about 102.3. The delta alone sized 123500 and 143479 blocks.
    assert!(blocks(legacy(&start(114.3, 14.2, None))) > 120_000);
    assert!(blocks(legacy(&start(112.9, 15.0, None))) > 140_000);

    let rank0 = size(&start(114.3, 14.2, Some(102.3)));
    assert_eq!(rank0.own, gib(102.3));
    assert_eq!(rank0.basis_own, gib(114.3) - gib(14.2));
    assert_eq!(
        rank0.disagreement(),
        Some(Disagreement::TrackedAbove(gib(102.3) - rank0.basis_own))
    );
    assert!(!rank0.headroom_limited);
    // About the normal pool: 0.1 GiB under it, since own is 102.3 not 102.2.
    assert!(
        (100_000..=NORMAL_BLOCKS).contains(&blocks(rank0.bytes)),
        "{}",
        blocks(rank0.bytes)
    );
    // What it leaves free is what a normal start leaves with these co-tenants.
    let left = gib(14.2) - rank0.bytes;
    assert!(left > gib(4.0), "{left}");

    let rank1 = size(&start(112.9, 15.0, Some(101.3)));
    assert_eq!(rank1.own, gib(101.3));
    assert!(blocks(rank1.bytes) < 112_000, "{}", blocks(rank1.bytes));
    // Both ranks then agree on the minimum, which is rank 0's.
    assert!(rank0.bytes < rank1.bytes);
}

#[test]
fn without_a_tracked_measure_the_headroom_floor_bounds_the_damage() {
    // Same bad start, no independent measure at all: the pool is still capped
    // by what is free now, less a third of the 8.5 GiB outside the budget.
    let got = size(&start(114.3, 14.2, None));
    assert!(got.headroom_limited);
    assert_eq!(got.headroom, (gib(121.7) - got.total_budget) / 3);
    assert_eq!(got.bytes, gib(14.2) - (gib(0.8) + gib(0.1)) - got.headroom);
    assert!(blocks(got.bytes) < 107_000, "{}", blocks(got.bytes));
    assert!(gib(14.2) - got.bytes > gib(3.7));
}

#[test]
fn a_co_tenant_that_allocates_during_the_load_shrinks_the_pool() {
    // 3 GiB taken by someone else mid-load: the delta over-measures (105.2),
    // the larger figure is kept, and the pool is 3 GiB smaller. Safe.
    let input = start(117.1, 11.9, Some(102.0));
    let got = size(&input);
    assert_eq!(got.own, gib(117.1) - gib(11.9));
    assert_eq!(got.bytes, legacy(&input));
    assert_eq!(
        got.bytes + (gib(14.9) - gib(11.9)),
        size(&start(117.1, 14.9, None)).bytes
    );
    assert_eq!(
        got.disagreement(),
        Some(Disagreement::TrackedBelow(got.own - gib(102.0)))
    );
}

#[test]
fn co_tenants_may_not_take_the_last_third_of_the_margin() {
    // 7.4 GiB of co-tenant that stays: own is measured right (102.3), the
    // utilization budget would allow 9.98 GiB, but that would leave 2.0 GiB
    // for a process that still grows by about 2.
    let got = size(&start(114.3, 12.0, Some(102.3)));
    assert!(got.headroom_limited);
    assert_eq!(got.bytes, gib(12.0) - (gib(0.8) + gib(0.1)) - got.headroom);
    assert!(got.bytes < legacy(&start(114.3, 12.0, Some(102.3))));
}

#[test]
fn manual_external_reserve_still_wins_over_the_baseline() {
    // ATLAS_KV_EXTERNAL_RESERVE_GB=4.6 on the normal start: raw used 106.8
    // less 4.6 is the same 102.2, whatever the baseline says.
    let mut input = start(90.0, 14.9, None);
    input.manual_external = Some(gib(4.6));
    let got = size(&input);
    assert_eq!(got.basis, Basis::Manual);
    assert_eq!(got.own, gib(121.7) - gib(14.9) - gib(4.6));
    assert_eq!(got.bytes, legacy(&input));

    // An override that discounts more than the co-tenants hold cannot push
    // own below what the process provably allocated.
    input.manual_external = Some(gib(20.0));
    input.tracked_own = Some(gib(102.3));
    let got = size(&input);
    assert_eq!(got.basis_own, gib(121.7) - gib(14.9) - gib(20.0));
    assert_eq!(got.own, gib(102.3));
    assert!(matches!(
        got.disagreement(),
        Some(Disagreement::TrackedAbove(_))
    ));

    // Larger than everything in use: saturates, never wraps.
    input.manual_external = Some(gib(500.0));
    input.tracked_own = None;
    assert_eq!(size(&input).own, 0);
}

#[test]
fn zero_and_absent_measures_fall_back_to_raw_used() {
    let raw_used = gib(121.7) - gib(14.9);

    // No baseline (mock backend) and no tracked measure: co-tenants count.
    let mut input = start(117.1, 14.9, None);
    input.baseline_free = None;
    let got = size(&input);
    assert_eq!((got.basis, got.own), (Basis::Raw, raw_used));
    assert_eq!(got.bytes, legacy(&input));
    assert_eq!(got.disagreement(), None);

    // Baseline at or under free-now (delta zero) or above total: implausible.
    for baseline in [14.9, 10.0, 140.0] {
        let input = start(baseline, 14.9, Some(0.0));
        let got = size(&input);
        assert_eq!(
            (got.basis, got.own),
            (Basis::Implausible, raw_used),
            "{baseline}"
        );
        assert_eq!(got.bytes, legacy(&input));
    }

    // More tracked than is in use system-wide (106.8): the counters and free
    // memory are not measuring the same pool, so the figure is set aside.
    for tracked in [107.0, 500.0] {
        let got = size(&start(117.1, 14.9, Some(tracked)));
        assert_eq!(got.tracked, None, "{tracked}");
        assert_eq!(got.own, gib(117.1) - gib(14.9));
        assert_eq!(got.bytes, legacy(&start(117.1, 14.9, None)));
    }

    // With no baseline own is already everything in use; tracked adds nothing.
    let mut input = start(117.1, 14.9, Some(105.0));
    input.baseline_free = None;
    assert_eq!(size(&input).own, raw_used);
}

#[test]
fn nothing_left_is_zero_not_a_wraparound() {
    assert_eq!(size(&start(117.1, 0.5, Some(102.0))).bytes, 0);
    let mut input = start(117.1, 14.9, None);
    input.reserve = gib(200.0);
    assert_eq!(size(&input).bytes, 0);
    input = start(117.1, 14.9, None);
    input.utilization = 0.5;
    assert_eq!(size(&input).bytes, 0);
}

#[test]
fn measures_within_half_a_gib_do_not_disagree() {
    assert_eq!(
        size(&start(117.1, 14.9, Some(102.2 - 0.4))).disagreement(),
        None
    );
    assert_eq!(
        size(&start(117.1, 14.9, Some(102.2 + 0.4))).disagreement(),
        None
    );
    assert!(
        size(&start(117.1, 14.9, Some(102.2 - 0.6)))
            .disagreement()
            .is_some()
    );
    assert!(
        size(&start(117.1, 14.9, Some(102.2 + 0.6)))
            .disagreement()
            .is_some()
    );
    // Raw used includes co-tenants by design; a smaller tracked figure there
    // says nothing about either measure.
    let mut input = start(117.1, 14.9, Some(90.0));
    input.baseline_free = None;
    assert_eq!(size(&input).disagreement(), None);
}
