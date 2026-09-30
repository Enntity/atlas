// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::verify_cost::{Coeffs, DEFAULT_COEFFS, shape};

/// Owners with these curves and no draft tokens (the table ignores tokens).
fn owners<'a>(curves: impl Iterator<Item = &'a DraftSurvival>) -> Vec<VerifyOwner<'a>> {
    curves
        .map(|survival| VerifyOwner {
            survival,
            last_token: 0,
            drafts: &[],
        })
        .collect()
}

/// The width the measured prose table picks (flag off).
fn best_table<'a>(curves: impl Iterator<Item = &'a DraftSurvival>, max: usize) -> usize {
    let o = owners(curves);
    best(&o, max, |w| step_ms(o.len(), w + 1))
}

/// The width the expert-aware model picks under `c` (flag on).
fn best_model(o: &[VerifyOwner<'_>], max: usize, c: &Coeffs) -> usize {
    best(o, max, |w| c.step_ms(&shape(VerifyOwner::tokens(o), w)))
}

fn curve(drafted: usize, accepted: usize, steps: usize) -> DraftSurvival {
    let mut s = DraftSurvival::default();
    for _ in 0..steps {
        s.record(drafted, accepted);
    }
    s
}

/// Hazard of a position with no evidence: the prior plus the full bonus.
fn unobserved() -> f32 {
    (DEFAULT_PARAMS.prior_h + DEFAULT_PARAMS.ucb).min(1.0)
}

#[test]
fn record_observes_only_reached_positions() {
    let s = curve(7, 2, 60);
    // Positions 0 and 1 always accepted, 2 always rejected, 3.. never reached.
    assert!(s.hazard(0) > 0.95 && s.hazard(1) > 0.95);
    assert!(s.hazard(2) < 0.2);
    assert!((s.hazard(3) - unobserved()).abs() < 1e-6);
}

#[test]
fn unobserved_positions_drift_back_to_the_prior() {
    let mut s = curve(7, 0, 60);
    assert!(s.hazard(0) < 0.2);
    for _ in 0..600 {
        s.record(0, 0);
    }
    assert!((s.hazard(0) - unobserved()).abs() < 0.02);
}

#[test]
fn predictable_text_verifies_wide_and_prose_narrow() {
    let counting = curve(7, 7, 60);
    let prose = curve(7, 1, 60);
    assert_eq!(best_table([&counting].into_iter(), 7), 7);
    assert!(best_table([&prose].into_iter(), 7) <= 3);
    // The batch shares one width: four prose owners narrow it at least as
    // far as one, since every extra row costs more with more owners.
    let four = [&prose, &prose, &prose, &prose];
    assert!(best_table(four.into_iter(), 7) <= best_table([&prose].into_iter(), 7));
}

#[test]
fn width_respects_the_cap() {
    let counting = curve(7, 7, 60);
    assert_eq!(best_table([&counting].into_iter(), 3), 3);
}

#[test]
fn a_full_accept_reopens_the_width() {
    // Bursty text: rejections at the fourth draft narrowed the width to
    // three, then a verify of three drafts accepted all three.
    let mut s = curve(7, 3, 20);
    assert!(best_table([&s].into_iter(), 7) <= 3);
    s.record(3, 3);
    assert!(best_table([&s].into_iter(), 7) > 3);
}

#[test]
fn eight_owners_stay_within_the_row_budget() {
    let counting = curve(7, 7, 60);
    let eight = [&counting; 8];
    // 8 owners x 4 rows is the widest batch the 32-row budget admits.
    assert_eq!(best_table(eight.into_iter(), 7), 3);
}

// ── Expert-aware cost model (ATLAS_GLM_VERIFY_COST_MODEL) ──

/// Distinct draft tokens (prose-like: no row repeats another).
const DISTINCT: [u32; 7] = [11, 12, 13, 14, 15, 16, 17];

#[test]
fn default_coefficients_reproduce_the_measured_table() {
    let (mut sse, mut cells) = (0.0f32, 0usize);
    for owners in 1..=8 {
        for rows in 2..=8 {
            let measured = step_ms(owners, rows);
            if !measured.is_finite() {
                continue;
            }
            let s = shape(
                (0..owners as u32).map(|o| (100 + o, &DISTINCT[..])),
                rows - 1,
            );
            let err = DEFAULT_COEFFS.step_ms(&s) - measured;
            assert!(err.abs() < 12.0, "owners={owners} rows={rows} err={err}");
            sse += err * err;
            cells += 1;
        }
    }
    assert_eq!(cells, 43);
    assert!((sse / cells as f32).sqrt() < 5.0);
    // The lone 2-row path quirk survives the fit.
    let lone = |w| DEFAULT_COEFFS.step_ms(&shape(std::iter::once((1, &DISTINCT[..])), w));
    assert!(lone(1) > lone(2));
}

#[test]
fn default_model_keeps_the_table_widths_on_distinct_text() {
    for (drafted, accepted) in [(7, 7), (7, 5), (7, 3), (7, 1)] {
        let c = curve(drafted, accepted, 60);
        for n in 1..=4 {
            let curves = vec![&c; n];
            let o: Vec<VerifyOwner<'_>> = (0..n)
                .map(|i| VerifyOwner {
                    survival: &c,
                    last_token: 100 + i as u32,
                    drafts: &DISTINCT,
                })
                .collect();
            let table = best_table(curves.into_iter(), 7);
            let model = best_model(&o, 7, &DEFAULT_COEFFS);
            assert!(
                table.abs_diff(model) <= 1,
                "accepted={accepted} owners={n} table={table} model={model}"
            );
        }
    }
}

#[test]
fn repeated_tokens_widen_the_verify_when_repeats_are_cheap() {
    // Code-like: every draft survives its predecessor ~80% of the time, near
    // the single-owner break-even where row cost decides the width.
    let code = DraftSurvival {
        verified: [100.0; MAX_DRAFTS],
        accepted: [80.0; MAX_DRAFTS],
        burst: None,
    };
    let calibrated = Coeffs {
        b: 0.2,
        ..DEFAULT_COEFFS
    };
    let owner = |drafts: &'static [u32]| {
        [VerifyOwner {
            survival: &code,
            last_token: 1,
            drafts,
        }]
    };
    const REPEATS: [u32; 7] = [1, 2, 1, 2, 1, 2, 1];
    let distinct = best_model(&owner(&DISTINCT), 7, &calibrated);
    let repeats = best_model(&owner(&REPEATS), 7, &calibrated);
    assert!(repeats > distinct, "repeats={repeats} distinct={distinct}");
    // With repeats priced like new tokens (b = a) the tokens do not matter.
    assert_eq!(
        best_model(&owner(&REPEATS), 7, &DEFAULT_COEFFS),
        best_model(&owner(&DISTINCT), 7, &DEFAULT_COEFFS)
    );
}

#[test]
fn model_width_respects_the_cap_and_is_deterministic() {
    let counting = curve(7, 7, 60);
    let o = [VerifyOwner {
        survival: &counting,
        last_token: 1,
        drafts: &DISTINCT,
    }];
    assert_eq!(best_model(&o, 3, &DEFAULT_COEFFS), 3);
    // A pure function of rank-0 inputs: the same owners pick the same width.
    let first = best_model(&o, 7, &DEFAULT_COEFFS);
    assert!((0..16).all(|_| best_model(&o, 7, &DEFAULT_COEFFS) == first));
}
