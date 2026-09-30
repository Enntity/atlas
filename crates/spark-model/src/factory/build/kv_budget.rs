// SPDX-License-Identifier: AGPL-3.0-only

//! How many bytes `build_model` may give the paged KV cache.
//!
//! The user-facing contract (vLLM / sparkrun convention) is that
//! `total × --gpu-memory-utilization` is the ceiling on everything THIS
//! process holds. The KV pool gets what is left of that ceiling after the
//! process's own pre-KV footprint and the reserves. Co-tenants (GB10 is
//! shared: ComfyUI, voice, a docker build) are excluded, so a low utilization
//! does not starve the pool for memory Atlas never took.
//!
//! **Own footprint** is the larger of two measures:
//!
//! 1. Free memory at context init minus free memory now
//!    (`gpu::baseline_free_bytes`), or, when
//!    `ATLAS_KV_EXTERNAL_RESERVE_GB` is set, raw used minus that figure.
//!    Exact while co-tenants hold still. When one RELEASES memory during the
//!    load, that memory is credited to Atlas as footprint it never had and
//!    the pool is sized into it: on 2026-09-30 two ranks read 100.1 and
//!    97.9 GiB for a process that holds 102.3, the pool came out 2.1 GiB
//!    large, and the host ran out of unified memory two requests later.
//! 2. What the process itself allocated (`GpuBackend::own_footprint`):
//!    nothing another process does can lower it.
//!
//! A co-tenant that ALLOCATES during the load inflates (1), and a counter
//! that misses something deflates (2); either way the larger one is kept, so
//! the estimate errs toward a smaller pool. The two are logged side by side
//! and a gap over [`DISAGREE_WARN_BYTES`] is a WARN.
//!
//! **Headroom floor.** Whatever was measured, the pool never takes free
//! memory below the reserves plus a third of the memory outside the
//! utilization ceiling. Free-now is read directly, so this bounds the damage
//! of any mis-measured footprint, and it stops co-tenants that fill the
//! margin from leaving a process that still grows after sizing (CUDA graphs,
//! JIT, lazy buffers) with nothing. With quiet co-tenants under two thirds of
//! the margin the floor is slack and the pool is byte-for-byte what measure
//! (1) alone gives.

use spark_runtime::gpu::GpuBackend;

/// Two own-footprint measures further apart than this are logged at WARN.
pub(super) const DISAGREE_WARN_BYTES: usize = 512 << 20;
/// The pool leaves `1 / HEADROOM_DIVISOR` of `total × (1 − utilization)` free.
const HEADROOM_DIVISOR: usize = 3;

/// Everything the arithmetic needs, in bytes, measured by the caller.
#[derive(Clone, Copy, Debug)]
pub(super) struct Inputs {
    pub total: usize,
    pub free_now: usize,
    pub utilization: f64,
    /// Inference reserve plus derived (lazy BF16) reserve.
    pub reserve: usize,
    /// Free memory at GPU-context init; `None` under the mock backend.
    pub baseline_free: Option<usize>,
    /// `ATLAS_KV_EXTERNAL_RESERVE_GB`, when set above zero.
    pub manual_external: Option<usize>,
    /// The process's own allocations since context init, when the backend
    /// keeps that account.
    pub tracked_own: Option<usize>,
}

/// Which free-memory measure produced [`KvBudget::basis_own`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Basis {
    /// Raw used minus `ATLAS_KV_EXTERNAL_RESERVE_GB`.
    Manual,
    /// Baseline-free minus free-now.
    Delta,
    /// No baseline: raw used, co-tenants included.
    Raw,
    /// A baseline that cannot be right (delta zero, or above raw used).
    Implausible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Disagreement {
    /// Tracked exceeds the free-memory measure by this much: something else
    /// released memory during the load.
    TrackedAbove(usize),
    /// The delta exceeds tracked by this much: something else allocated
    /// during the load, or the process holds memory the counters miss.
    TrackedBelow(usize),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct KvBudget {
    /// `total − free_now`: everything in use, co-tenants included.
    pub used_raw: usize,
    pub basis: Basis,
    pub basis_own: usize,
    /// The tracked measure, if present and no larger than `used_raw`.
    pub tracked: Option<usize>,
    /// Pre-KV footprint the budget was charged with.
    pub own: usize,
    pub total_budget: usize,
    pub headroom: usize,
    /// True when the headroom floor, not the utilization ceiling, set `bytes`.
    pub headroom_limited: bool,
    pub bytes: usize,
}

impl KvBudget {
    pub(super) fn disagreement(&self) -> Option<Disagreement> {
        let tracked = self.tracked?;
        if tracked > self.basis_own.saturating_add(DISAGREE_WARN_BYTES) {
            Some(Disagreement::TrackedAbove(tracked - self.basis_own))
        } else if self.basis == Basis::Delta
            && self.basis_own > tracked.saturating_add(DISAGREE_WARN_BYTES)
        {
            Some(Disagreement::TrackedBelow(self.basis_own - tracked))
        } else {
            None
        }
    }
}

pub(super) fn size(i: &Inputs) -> KvBudget {
    let used_raw = i.total.saturating_sub(i.free_now);
    let (basis, basis_own) = match (i.manual_external, i.baseline_free) {
        (Some(external), _) => (Basis::Manual, used_raw.saturating_sub(external)),
        (None, Some(baseline)) => match baseline.saturating_sub(i.free_now) {
            delta if delta > 0 && delta <= used_raw => (Basis::Delta, delta),
            _ => (Basis::Implausible, used_raw),
        },
        (None, None) => (Basis::Raw, used_raw),
    };
    // A process cannot hold more than is in use system-wide. A tracked figure
    // above that means its counters and free memory are not measuring the
    // same pool (a backend where device memory is also resident memory, or
    // where "free" is not this device's), so it proves nothing here.
    let tracked = i.tracked_own.filter(|&t| t <= used_raw);
    let own = basis_own.max(tracked.unwrap_or(0));
    let total_budget = (i.total as f64 * i.utilization) as usize;
    let headroom = i.total.saturating_sub(total_budget) / HEADROOM_DIVISOR;
    let by_budget = total_budget.saturating_sub(own).saturating_sub(i.reserve);
    let by_free = i
        .free_now
        .saturating_sub(i.reserve)
        .saturating_sub(headroom);
    KvBudget {
        used_raw,
        basis,
        basis_own,
        tracked,
        own,
        total_budget,
        headroom,
        headroom_limited: by_free < by_budget,
        bytes: by_budget.min(by_free),
    }
}

/// Take the measurements, size the budget, and say what was decided.
pub(super) fn measure(
    gpu: &dyn GpuBackend,
    total: usize,
    free_now: usize,
    utilization: f64,
    reserve: usize,
) -> KvBudget {
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let manual_gb = std::env::var("ATLAS_KV_EXTERNAL_RESERVE_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&gb| gb > 0.0);
    let baseline = spark_runtime::gpu::baseline_free_bytes();
    let footprint = gpu.own_footprint();
    let budget = size(&Inputs {
        total,
        free_now,
        utilization,
        reserve,
        baseline_free: baseline,
        manual_external: manual_gb.map(|gb| (gb * 1024.0 * 1024.0 * 1024.0) as usize),
        tracked_own: footprint.map(|f| f.total()),
    });
    match (budget.basis, manual_gb, baseline) {
        (Basis::Manual, Some(gb), _) => tracing::info!(
            "ATLAS_KV_EXTERNAL_RESERVE_GB={gb} (manual override): discounting \
             external/co-tenant memory from KV budget — used_so_far {:.1} GB → \
             Atlas-own {:.1} GB",
            gib(budget.used_raw),
            gib(budget.basis_own),
        ),
        (Basis::Delta, _, Some(baseline)) => tracing::info!(
            "KV budget self-relative (auto): baseline-free {:.1} GB − free-now \
             {:.1} GB = Atlas-own {:.1} GB; co-tenants {:.1} GB excluded \
             (set ATLAS_KV_EXTERNAL_RESERVE_GB to override)",
            gib(baseline),
            gib(free_now),
            gib(budget.basis_own),
            gib(budget.used_raw - budget.basis_own),
        ),
        (Basis::Implausible, _, Some(baseline)) => tracing::warn!(
            "KV budget auto-measure implausible (baseline {:.1} GB, free-now \
             {:.1} GB, used {:.1} GB) — using raw used_so_far",
            gib(baseline),
            gib(free_now),
            gib(budget.used_raw),
        ),
        _ => {}
    }
    if let Some(f) = footprint {
        let tracked = f.total();
        tracing::info!(
            "KV own-footprint cross-check: free-memory measure {:.2} GB, tracked \
             {:.2} GB (device {:.2} GB by {} + host {:.2} GB) → {:.2} GB pre-KV",
            gib(budget.basis_own),
            gib(tracked),
            gib(f.device),
            f.device_source.label(),
            gib(f.host),
            gib(budget.own),
        );
        match budget.disagreement() {
            Some(Disagreement::TrackedAbove(gap)) => tracing::warn!(
                "KV own-footprint: Atlas allocated {:.2} GB but free memory fell by \
                 only {:.2} GB — {:.2} GB was released by something else during the \
                 load and would have been sized into the KV pool; sizing from the \
                 tracked figure",
                gib(tracked),
                gib(budget.basis_own),
                gib(gap),
            ),
            Some(Disagreement::TrackedBelow(gap)) => tracing::warn!(
                "KV own-footprint: free memory fell by {:.2} GB but Atlas tracked \
                 only {:.2} GB — {:.2} GB was taken by something else during the \
                 load, or is Atlas memory the {} misses; keeping the larger \
                 figure (the KV pool may be smaller than it needs to be)",
                gib(budget.basis_own),
                gib(tracked),
                gib(gap),
                f.device_source.label(),
            ),
            None if budget.tracked.is_none() => tracing::warn!(
                "KV own-footprint: tracked {:.2} GB exceeds the {:.2} GB in use on \
                 the whole device — its counters and free memory do not measure \
                 the same pool; ignoring the tracked figure",
                gib(tracked),
                gib(budget.used_raw),
            ),
            None => {}
        }
    }
    if budget.headroom_limited {
        tracing::warn!(
            "KV pool limited by free memory: {:.2} GB free now; keeping {:.2} GB \
             (a third of the {:.2} GB outside --gpu-memory-utilization) plus the \
             {:.2} GB reserve unallocated → {:.2} GB for KV, not the {:.2} GB the \
             utilization budget allows. Other processes hold {:.2} GB.",
            gib(free_now),
            gib(budget.headroom),
            gib(total.saturating_sub(budget.total_budget)),
            gib(reserve),
            gib(budget.bytes),
            gib(budget
                .total_budget
                .saturating_sub(budget.own)
                .saturating_sub(reserve)),
            gib(budget.used_raw.saturating_sub(budget.own)),
        );
    }
    budget
}

#[cfg(test)]
#[path = "kv_budget_tests.rs"]
mod tests;
