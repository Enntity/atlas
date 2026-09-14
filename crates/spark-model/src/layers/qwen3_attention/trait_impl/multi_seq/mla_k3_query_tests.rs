// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn flags_are_strict_and_compare_requires_batchm() {
    assert!(!parse_flag(None).unwrap());
    assert!(!parse_flag(Some("0")).unwrap());
    assert!(parse_flag(Some("1")).unwrap());
    for value in ["", "2", "true", " 1"] {
        assert!(parse_flag(Some(value)).is_err());
    }
    assert!(!parse_compare(None, false).unwrap());
    assert!(parse_compare(Some("1"), true).unwrap());
    assert!(parse_compare(Some("1"), false).is_err());
    for value in ["", "2", "true", " 1"] {
        assert!(parse_compare(Some(value), true).is_err());
    }
}

#[test]
fn rows_use_contiguous_latent_q_and_padded_index_tail() {
    let plan = QueryPlan::new(
        DevicePtr(0x1000),
        ROWS * HIDDEN * BF16,
        DevicePtr(0x20000),
        LATENT_BYTES,
        DevicePtr(0x40000),
        QUERY_BYTES,
        None,
        &[],
    )
    .unwrap();
    for (row, (latent, q, index)) in [
        (
            0,
            (0x20000u64, 0x40000u64, 0x40000u64 + INDEX_OFFSET as u64),
        ),
        (
            1,
            (
                0x20000u64 + LATENT_ROW as u64,
                0x40000u64 + Q_ROW as u64,
                0x40000u64 + INDEX_OFFSET as u64 + INDEX_ROW as u64,
            ),
        ),
        (
            2,
            (
                0x20000u64 + 2 * LATENT_ROW as u64,
                0x40000u64 + 2 * Q_ROW as u64,
                0x40000u64 + INDEX_OFFSET as u64 + 2 * INDEX_ROW as u64,
            ),
        ),
    ] {
        let actual = plan.row(row).unwrap();
        assert_eq!(actual.q_latent.0, latent);
        assert_eq!(actual.q_full.0, q);
        assert_eq!(actual.index_query.0, index);
    }
    assert!(plan.row(ROWS).is_err());
}

#[test]
fn plan_rejects_short_or_aliasing_ranges() {
    let good = || {
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16,
            DevicePtr(0x20000),
            LATENT_BYTES,
            DevicePtr(0x40000),
            QUERY_BYTES,
            None,
            &[],
        )
    };
    assert!(good().is_ok());
    assert!(
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16 - 1,
            DevicePtr(0x20000),
            LATENT_BYTES,
            DevicePtr(0x40000),
            QUERY_BYTES,
            None,
            &[]
        )
        .is_err()
    );
    assert!(
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16,
            DevicePtr(0x20000),
            LATENT_BYTES - 1,
            DevicePtr(0x40000),
            QUERY_BYTES,
            None,
            &[]
        )
        .is_err()
    );
    assert!(
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16,
            DevicePtr(0x20000),
            LATENT_BYTES,
            DevicePtr(0x40000),
            QUERY_BYTES - 1,
            None,
            &[]
        )
        .is_err()
    );
    assert!(
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16,
            DevicePtr(0x20000),
            LATENT_BYTES,
            DevicePtr(0x40000),
            QUERY_BYTES,
            None,
            &[(DevicePtr(0x20000 + LATENT_ROW as u64), 16)]
        )
        .is_err()
    );
    assert!(
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16,
            DevicePtr(0x20000),
            LATENT_BYTES,
            DevicePtr(0x40000),
            QUERY_BYTES,
            None,
            &[(DevicePtr(0x40000 + INDEX_OFFSET as u64), 16)]
        )
        .is_err()
    );
}

#[test]
fn diagnostic_scratch_uses_dead_qkvz_arena() {
    let plan = QueryPlan::new(
        DevicePtr(0x1000),
        ROWS * HIDDEN * BF16,
        DevicePtr(0x20000),
        LATENT_BYTES,
        DevicePtr(0x40000),
        QUERY_BYTES,
        Some((DevicePtr(0x60000), DIAGNOSTIC_BYTES)),
        &[],
    )
    .unwrap();
    assert_eq!(plan.diagnostic.unwrap().0, 0x60000);
    assert!(
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16,
            DevicePtr(0x20000),
            LATENT_BYTES,
            DevicePtr(0x40000),
            QUERY_BYTES,
            Some((DevicePtr(0x60000), DIAGNOSTIC_BYTES - 1)),
            &[],
        )
        .is_err()
    );
    // The exact qkvz base is the intentional dead-scratch borrow; any
    // distinct live range overlapping the diagnostic span is rejected.
    assert!(
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16,
            DevicePtr(0x20000),
            LATENT_BYTES,
            DevicePtr(0x40000),
            QUERY_BYTES,
            Some((DevicePtr(0x60000), DIAGNOSTIC_BYTES)),
            &[(DevicePtr(0x60000 + 16), 32)],
        )
        .is_err()
    );
}

#[test]
fn diagnostic_scratch_does_not_hide_equal_base_live_alias() {
    assert!(
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16,
            DevicePtr(0x20000),
            LATENT_BYTES,
            DevicePtr(0x40000),
            QUERY_BYTES,
            Some((DevicePtr(0x60000), DIAGNOSTIC_BYTES)),
            // Same base, different logical span: this is an accidental
            // weight/live alias and must not be mistaken for qkvz owner.
            &[(DevicePtr(0x60000), 16)],
        )
        .is_err()
    );
}

#[test]
fn diagnostic_scratch_rejects_duplicate_qkvz_owner_tuple() {
    let qkvz = (DevicePtr(0x60000), DIAGNOSTIC_BYTES);
    assert!(
        QueryPlan::new(
            DevicePtr(0x1000),
            ROWS * HIDDEN * BF16,
            DevicePtr(0x20000),
            LATENT_BYTES,
            DevicePtr(0x40000),
            QUERY_BYTES,
            Some(qkvz),
            &[qkvz, qkvz],
        )
        .is_err()
    );
}
