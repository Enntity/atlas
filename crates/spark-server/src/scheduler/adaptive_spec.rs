// SPDX-License-Identifier: AGPL-3.0-only

//! Adaptive speculation (`ATLAS_DFLASH_ADAPTIVE=1`).
//!
//! Speculation pays only when τ (= mean accepted + 1 bonus) exceeds
//! step_time / serial_time — ≈3.5 at the measured 222ms γ16 verify step vs
//! 63ms serial decode, i.e. mean accepted ≥ ~2.5. Measured 2026-07-08 on
//! coherent output: MinHeap code runs τ≈5.7 (spec wins, +30% vs serial),
//! Volvo prose runs τ≈2.3 (spec LOSES ~20% vs serial). Content decides.
//!
//! Policy: per-sequence rolling window of `accepted` over the last
//! [`WINDOW`] K=γ verify steps. Window full and mean below the threshold →
//! SUSPEND speculation for that sequence (no proposing; the scheduler's
//! bootstrap path serial-decodes it at full NOSPEC pace). After
//! `sched.levers.dflash_adaptive_reprobe` serial tokens, UN-suspend and re-probe: the window
//! must refill before suspension can re-trigger, so a probe costs WINDOW
//! spec steps (~2.7s) once per re-probe interval — a few percent on pure
//! prose, nothing on accepting content, and mixed documents (prose→code)
//! re-engage speculation automatically.
//!
//! Net posture: never materially slower than plain decode, +30% where
//! acceptance supports it. State is transient (reset on swap/restore —
//! a resumed sequence just re-measures).
//!
//! Knobs (env, read once): `ATLAS_DFLASH_ADAPTIVE=1` master switch;
//! `ATLAS_DFLASH_ADAPTIVE_MIN` mean-accepted suspend threshold (default
//! 2.0); `ATLAS_DFLASH_ADAPTIVE_REPROBE` serial tokens between probes
//! (default 256).

use crate::scheduler::ActiveSeq;

/// Rolling accept window + suspend state, embedded in [`ActiveSeq`].
#[derive(Default)]
pub(crate) struct AdaptState {
    window: Vec<u32>,
    suspended: bool,
    serial_tokens: u32,
    depth_window: Vec<u32>,
    shallow_depth: bool,
}

const WINDOW: usize = 12;
const DEPTH_WINDOW: usize = 3;
const SHALLOW_DRAFTS: usize = 7;
const DEPTH_DEMOTE_MEAN: f32 = 6.0;
const DEPTH_PROMOTE_MEAN: f32 = 6.5;

pub(crate) fn dflash_depth_ladder_enabled() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ATLAS_GLM_DFLASH_DEPTH_LADDER")
            .ok()
            .as_deref()
            == Some("1")
    })
}

fn next_shallow_depth(current: bool, accepted: &[u32]) -> bool {
    if accepted.len() < DEPTH_WINDOW {
        return current;
    }
    let mean = accepted.iter().sum::<u32>() as f32 / accepted.len() as f32;
    if current {
        mean < DEPTH_PROMOTE_MEAN
    } else {
        mean < DEPTH_DEMOTE_MEAN
    }
}

/// Apply the content-sensitive half of the GLM DFlash depth policy.
/// Width-based clamping is handled by `mtp_step`; this selects shallow γ=7
/// for a solo sequence whose recent deep proposals are not paying back.
pub(crate) fn configured_dflash_depth_limit(a: &ActiveSeq, max_drafts: usize) -> usize {
    if dflash_depth_ladder_enabled() && a.spec_adapt.shallow_depth {
        max_drafts.min(SHALLOW_DRAFTS)
    } else {
        max_drafts
    }
}

/// Record one K=γ verify step's accept count; may trip suspension.
/// Call after `num_accepted` is known (verify_dflash_step).
pub(crate) fn record_verify(
    a: &mut ActiveSeq,
    num_accepted: usize,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    if dflash_depth_ladder_enabled() {
        let st = &mut a.spec_adapt;
        st.depth_window.push(num_accepted as u32);
        if st.depth_window.len() > DEPTH_WINDOW {
            st.depth_window.remove(0);
        }
        let next = next_shallow_depth(st.shallow_depth, &st.depth_window);
        if next != st.shallow_depth {
            let mean = st.depth_window.iter().sum::<u32>() as f32 / st.depth_window.len() as f32;
            st.shallow_depth = next;
            tracing::info!(
                "adaptive DFlash depth: {} (mean accepted {mean:.2} over {} blocks)",
                if next { "SHALLOW gamma=7" } else { "DEEP" },
                st.depth_window.len(),
            );
        }
    }
    if !sched.levers.dflash_adaptive {
        return;
    }
    let st = &mut a.spec_adapt;
    st.window.push(num_accepted as u32);
    if st.window.len() > WINDOW {
        st.window.remove(0);
    }
    if st.window.len() == WINDOW {
        let mean = st.window.iter().sum::<u32>() as f32 / WINDOW as f32;
        if mean < sched.levers.dflash_adaptive_min {
            st.suspended = true;
            st.serial_tokens = 0;
            st.window.clear();
            tracing::info!(
                "adaptive spec: SUSPENDED (mean accepted {mean:.2} < {} over {WINDOW} steps) — \
                 serial decode until re-probe",
                sched.levers.dflash_adaptive_min,
            );
        }
    }
}

/// May this sequence propose/speculate right now? Un-suspends (re-probe)
/// once enough serial tokens have passed.
pub(crate) fn spec_allowed(
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> bool {
    if !sched.levers.dflash_adaptive {
        return true;
    }
    let st = &mut a.spec_adapt;
    if !st.suspended {
        return true;
    }
    if st.serial_tokens >= sched.levers.dflash_adaptive_reprobe {
        st.suspended = false;
        st.serial_tokens = 0;
        st.window.clear();
        tracing::info!(
            "adaptive spec: RE-PROBING after {} serial tokens",
            sched.levers.dflash_adaptive_reprobe
        );
        return true;
    }
    false
}

/// Is this sequence currently in the adaptive-suspended (serial) regime?
/// Read-only peek — unlike `spec_allowed`, never mutates re-probe state.
pub(crate) fn is_suspended(a: &ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) -> bool {
    sched.levers.dflash_adaptive && a.spec_adapt.suspended
}

/// Count a serially-decoded token toward the re-probe interval.
pub(crate) fn tick_serial(a: &mut ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) {
    if sched.levers.dflash_adaptive && a.spec_adapt.suspended {
        a.spec_adapt.serial_tokens = a.spec_adapt.serial_tokens.saturating_add(1);
    }
}

// The `ATLAS_DFLASH_SERIAL_APPEND` and `ATLAS_DFLASH_UNIFIED_CTX` statics
// that lived here are now `SchedLevers::dflash_serial_append` /
// `::dflash_unified_ctx`, resolved once per run and read through `SchedCtx`.

// The `ATLAS_DFLASH_ADAPTIVE*` statics are now `SchedLevers::dflash_adaptive*`.

#[cfg(test)]
mod tests {
    use super::next_shallow_depth;

    #[test]
    fn depth_hysteresis_demotes_weak_deep_blocks() {
        assert!(!next_shallow_depth(false, &[2, 3]));
        assert!(next_shallow_depth(false, &[2, 3, 4]));
        assert!(!next_shallow_depth(false, &[7, 8, 9]));
    }

    #[test]
    fn depth_hysteresis_requires_near_perfect_shallow_acceptance_to_promote() {
        assert!(next_shallow_depth(true, &[6, 6, 7]));
        assert!(!next_shallow_depth(true, &[7, 7, 7]));
    }
}
