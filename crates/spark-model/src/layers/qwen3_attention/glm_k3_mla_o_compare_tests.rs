// SPDX-License-Identifier: AGPL-3.0-only
use super::super::StagePlan;
use super::*;
#[test]
fn compare_is_strict_and_requires_the_staged_route() {
    assert!(!parse_compare(None, false).unwrap());
    assert!(!parse_compare(Some("0"), false).unwrap());
    assert!(parse_compare(Some("1"), true).unwrap());
    assert!(parse_compare(Some("1"), false).is_err());
    for value in ["", "2", "true", " 1"] {
        assert!(parse_compare(Some(value), true).is_err());
    }
}
#[test]
fn comparison_output_requires_capacity_and_independent_storage() {
    let plan = StagePlan::new(DevicePtr(0x1000), 49664, DevicePtr(0x20000), OUTPUT, &[]).unwrap();
    plan.with_compare(DevicePtr(0x30000), OUTPUT, &[]).unwrap();
    for (ptr, size) in [
        (0, OUTPUT),
        (0x30001, OUTPUT),
        (0x30000, OUTPUT - 1),
        (0x1200, OUTPUT),
        (0x20000, OUTPUT),
        (u64::MAX - 1, OUTPUT),
    ] {
        assert!(plan.with_compare(DevicePtr(ptr), size, &[]).is_err());
    }
    assert!(
        plan.with_compare(DevicePtr(0x30000), OUTPUT, &[(DevicePtr(0x30000), 16)])
            .is_err()
    );
}
#[test]
fn full_bf16_comparison_rejects_late_and_nonfinite_bit_differences() {
    let reference = vec![0; OUTPUT];
    let mut candidate = reference.clone();
    compare_bytes(&reference, &candidate).unwrap();
    for bits in [0x7fc0_u16, 0x7f80_u16, 0xff80_u16] {
        let mut invalid = reference.clone();
        invalid[OUTPUT - 2..].copy_from_slice(&bits.to_le_bytes());
        let error = compare_bytes(&invalid, &invalid).unwrap_err().to_string();
        assert!(
            error.contains("nonfinite") && error.contains("first_index=12287"),
            "{error}"
        );
    }

    candidate[14..16].copy_from_slice(&0x3f80_u16.to_le_bytes());
    candidate[OUTPUT - 2..].copy_from_slice(&0x4000_u16.to_le_bytes());
    let error = compare_bytes(&reference, &candidate)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("mismatches=2")
            && error.contains("first_index=7")
            && error.contains("max_abs=2"),
        "{error}"
    );
    candidate.fill(0);
    candidate[OUTPUT - 2..].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    assert!(compare_bytes(&reference, &candidate).is_err());
    assert!(compare_bytes(&reference[..OUTPUT - 2], &reference).is_err());
    // Identical signed-zero bits pass; a sign-bit change is still a mismatch.
    candidate.fill(0);
    candidate[0..2].copy_from_slice(&0x8000_u16.to_le_bytes());
    assert!(compare_bytes(&reference, &candidate).is_err());
}
