// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn shape_counts_novel_and_repeated_rows() {
    let a: &[u32] = &[5, 6, 5, 7];
    let b: &[u32] = &[6, 9, 9];
    // Owner a: 5, 6 new; 5 repeat. Owner b's last token 7 is its own fresh
    // row; its drafts 6 (seen in a) repeat, 9 new, 9 repeat.
    let s = shape([(4, a), (7, b)].into_iter(), 3);
    assert_eq!(
        s,
        StepShape {
            owners: 2,
            rows: 4,
            novel: 3,
            repeat: 3,
        }
    );
    // The width bounds the rows each owner contributes.
    let s = shape([(4, a)].into_iter(), 1);
    assert_eq!((s.rows, s.novel, s.repeat), (2, 1, 0));
    // A draft equal to the owner's last token is a repeat.
    let s = shape([(5, a)].into_iter(), 1);
    assert_eq!((s.novel, s.repeat), (0, 1));
}

#[test]
fn step_cost_grows_with_rows_owners_and_novelty() {
    let c = Coeffs {
        b: 0.25,
        ..DEFAULT_COEFFS
    };
    let at = |owners, rows, novel, repeat| {
        c.step_ms(&StepShape {
            owners,
            rows,
            novel,
            repeat,
        })
    };
    assert!(at(1, 4, 3, 0) < at(1, 5, 4, 0));
    assert!(at(2, 4, 6, 0) > at(1, 4, 3, 0));
    // A repeated row costs less than a new one, but not nothing.
    assert!(at(1, 5, 2, 2) < at(1, 5, 4, 0));
    assert!(at(1, 5, 2, 2) > at(1, 3, 2, 0));
    // Expected distinct experts saturate at the pool, not at rows x top-K.
    let huge = at(64, 8, 64 * 7, 0);
    let cap = c.c0 + c.c_owner * 64.0 + c.c_exp * EXPERTS;
    assert!(huge <= cap + 1e-3);
}

#[test]
fn coefficients_parse_strictly() {
    assert_eq!(
        Coeffs::parse("47.45, 7.183, 0, 1.05, 12.61, 1, 1"),
        Some(DEFAULT_COEFFS)
    );
    // c_lone2 may be negative (the quirk may invert on another build).
    assert!(Coeffs::parse("40,5,0.5,1,-3,0.8,0.2").is_some());
    for bad in [
        "",
        "1,2,3",
        "1,2,3,4,5,6,7,8",
        "40,5,0.5,1,3,0.8,x",
        "40,5,0.5,NaN,3,0.8,0.2",
        "40,5,0.5,inf,3,0.8,0.2",
        "-1,5,0.5,1,3,0.8,0.2",
        "40,5,0.5,1,3,1.5,0.2",
        "40,5,0.5,1,3,0.8,-0.1",
    ] {
        assert_eq!(Coeffs::parse(bad), None, "{bad:?}");
    }
}

#[test]
fn sweep_cycles_every_width_within_the_cap() {
    let widths: Vec<usize> = (0..2 * MAX_DRAFTS)
        .map(|s| sweep_pick(s, MAX_DRAFTS))
        .collect();
    assert_eq!(&widths[..MAX_DRAFTS], &[1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(&widths[MAX_DRAFTS..], &widths[..MAX_DRAFTS]);
    assert!((0..MAX_DRAFTS).all(|s| sweep_pick(s, 3) <= 3));
}
