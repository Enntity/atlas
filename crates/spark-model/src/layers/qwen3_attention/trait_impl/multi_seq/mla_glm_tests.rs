// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn batch4_is_opt_in_and_never_changes_other_widths() {
    for rows in 1..=5 {
        assert_eq!(
            batch4_kernel(rows, false, false, KernelHandle(0))
                .unwrap()
                .0,
            0
        );
        assert_eq!(
            batch4_kernel(rows, true, true, KernelHandle(42)).unwrap().0,
            if rows == 4 { 42 } else { 0 }
        );
    }
}

#[test]
fn batch4_requires_c4_and_a_live_exact_row_kernel() {
    assert!(batch4_kernel(4, true, false, KernelHandle(42)).is_err());
    assert!(batch4_kernel(3, true, false, KernelHandle(42)).is_err());
    assert!(batch4_kernel(4, true, true, KernelHandle(0)).is_err());
}

#[test]
fn glm_mla_multiseq_is_gated() {
    if std::env::var_os("ATLAS_GLM_MLA_MULTI_SEQ").is_none() {
        assert!(!enabled());
    }
}
