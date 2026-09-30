// SPDX-License-Identifier: AGPL-3.0-only

//! Expert-aware DFlash verify step cost (`ATLAS_GLM_VERIFY_COST_MODEL=1`).
//!
//! GLM decode is routed-MoE bound: a verify step mostly reads the weights of
//! the distinct experts its rows route to, so an extra row costs roughly the
//! experts it touches that the other rows do not. [`Coeffs::step_ms`] prices
//! a step as
//!
//! ```text
//! ms = c0 + c_owner*owners + c_row*owners*rows + c_exp*U(D) + c_lone2*[1 owner, 2 rows]
//! U(D) = E * (1 - (1 - K/E)^D)       expected distinct experts per MoE layer
//! D    = owners + a*novel + b*repeat  effective independent top-K draws
//! ```
//!
//! Each owner's first row is a fresh draw; a later row counts `a` when its
//! token is new to the step and `b` when an earlier row already carried it
//! (routing follows the token, so repeats mostly re-hit the same experts:
//! code indentation, list separators, near-identical owners). The shape is a
//! pure function of the draft tokens the rank-0 scheduler already holds, and
//! the chosen width reaches the worker rank through the verify broadcast
//! (`EP_CMD_GLM_LONG_VERIFY`), so TP ranks cannot disagree.
//!
//! The defaults are the fit of the 2026-09-28 prose table in
//! `dflash_width::step_ms` (`scripts/verify_cost_calib.py table`, rms 4.4 ms
//! over 43 cells); that table carries no repeat evidence, so `b = a` until a
//! prose + code sweep calibrates it. `ATLAS_GLM_VERIFY_COST_COEFFS=c0,c_owner,
//! c_row,c_exp,c_lone2,a,b` overrides them; `ATLAS_GLM_VERIFY_COST_SWEEP=1`
//! round-robins the width for that calibration sweep.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::dflash_width::MAX_DRAFTS;

/// GLM-5.3 Flash routed MoE: `n_routed_experts`, `num_experts_per_tok`.
const EXPERTS: f32 = 288.0;
const TOP_K: f32 = 8.0;

/// Step cost coefficients (ms; `a`, `b` are draw weights in 0..=1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Coeffs {
    pub(crate) c0: f32,
    pub(crate) c_owner: f32,
    pub(crate) c_row: f32,
    pub(crate) c_exp: f32,
    pub(crate) c_lone2: f32,
    pub(crate) a: f32,
    pub(crate) b: f32,
}

pub(crate) const DEFAULT_COEFFS: Coeffs = Coeffs {
    c0: 47.45,
    c_owner: 7.183,
    c_row: 0.0,
    c_exp: 1.05,
    c_lone2: 12.61,
    a: 1.0,
    b: 1.0,
};

/// Token shape of one candidate verify step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StepShape {
    pub(crate) owners: usize,
    /// Rows per owner (drafts + 1).
    pub(crate) rows: usize,
    /// Rows past an owner's first whose token is new to the step.
    pub(crate) novel: usize,
    /// Rows past an owner's first whose token an earlier row carried.
    pub(crate) repeat: usize,
}

/// Shape of a step verifying the first `width` drafts of each
/// `(last_token, drafts)` owner, in owner order.
pub(crate) fn shape<'a>(owners: impl Iterator<Item = (u32, &'a [u32])>, width: usize) -> StepShape {
    let mut seen: Vec<u32> = Vec::with_capacity(8 * (MAX_DRAFTS + 1));
    let mut s = StepShape {
        rows: width + 1,
        ..StepShape::default()
    };
    for (last, drafts) in owners {
        s.owners += 1;
        if !seen.contains(&last) {
            seen.push(last);
        }
        for &t in &drafts[..width.min(drafts.len())] {
            if seen.contains(&t) {
                s.repeat += 1;
            } else {
                s.novel += 1;
                seen.push(t);
            }
        }
    }
    s
}

impl Coeffs {
    /// Seven comma-separated finite values; cost terms non-negative, draw
    /// weights in 0..=1 (`c_lone2` is a path quirk and may take either sign).
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        let v: Vec<f32> = raw
            .split(',')
            .map(|x| x.trim().parse().ok())
            .collect::<Option<_>>()?;
        let [c0, c_owner, c_row, c_exp, c_lone2, a, b] = v[..] else {
            return None;
        };
        let ok = v.iter().all(|x| x.is_finite())
            && [c0, c_owner, c_row, c_exp].iter().all(|&x| x >= 0.0)
            && [a, b].iter().all(|x| (0.0..=1.0).contains(x));
        ok.then_some(Self {
            c0,
            c_owner,
            c_row,
            c_exp,
            c_lone2,
            a,
            b,
        })
    }

    /// Expected step milliseconds for `s`.
    pub(crate) fn step_ms(&self, s: &StepShape) -> f32 {
        let draws = s.owners as f32 + self.a * s.novel as f32 + self.b * s.repeat as f32;
        let experts = EXPERTS * (1.0 - (1.0 - TOP_K / EXPERTS).powf(draws));
        let lone2 = if s.owners == 1 && s.rows == 2 {
            self.c_lone2
        } else {
            0.0
        };
        self.c0
            + self.c_owner * s.owners as f32
            + self.c_row * (s.owners * s.rows) as f32
            + self.c_exp * experts
            + lone2
    }
}

fn flag(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_GLM_VERIFY_COST_MODEL"))
}

pub(crate) fn coeffs() -> Coeffs {
    static C: OnceLock<Coeffs> = OnceLock::new();
    *C.get_or_init(|| match std::env::var("ATLAS_GLM_VERIFY_COST_COEFFS") {
        Err(_) => DEFAULT_COEFFS,
        Ok(raw) => Coeffs::parse(&raw).unwrap_or_else(|| {
            tracing::warn!(
                "ATLAS_GLM_VERIFY_COST_COEFFS={raw:?} is malformed (want 7 values \
                 c0,c_owner,c_row,c_exp,c_lone2,a,b); keeping the defaults"
            );
            DEFAULT_COEFFS
        }),
    })
}

/// Calibration sweep (`ATLAS_GLM_VERIFY_COST_SWEEP=1`): the width of the
/// `step`-th verify cycles 1..=MAX_DRAFTS, capped at `max`.
pub(crate) fn sweep_pick(step: usize, max: usize) -> usize {
    (step % MAX_DRAFTS + 1).min(max)
}

fn sweep_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_GLM_VERIFY_COST_SWEEP"))
}

pub(crate) fn sweep_width(max: usize) -> Option<usize> {
    static STEP: AtomicUsize = AtomicUsize::new(0);
    sweep_on().then(|| sweep_pick(STEP.fetch_add(1, Ordering::Relaxed), max))
}

/// The step-shape fields appended to the width log line (the calibration
/// input), when the cost model or its sweep is on.
pub(crate) fn log_suffix<'a>(
    owners: impl Iterator<Item = (u32, &'a [u32])>,
    width: usize,
) -> String {
    if !(enabled() || sweep_on()) {
        return String::new();
    }
    let s = shape(owners, width);
    format!(" novel={} repeat={}", s.novel, s.repeat)
}

#[cfg(test)]
#[path = "verify_cost_tests.rs"]
mod tests;
