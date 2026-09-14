// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
#[test]
fn staging_preserves_index_prefix_and_all_three_rows() {
    let p = StagePlan::new(DevicePtr(0x1000), 49664, DevicePtr(0x20000), 24576, &[]).unwrap();
    assert_eq!(p.row(0).unwrap(), DevicePtr(0x1200));
    assert_eq!(p.row(1).unwrap(), DevicePtr(0x5200));
    assert_eq!(p.row(2).unwrap(), DevicePtr(0x9200));
    assert!(p.row(3).is_err());
    StagePlan::new(DevicePtr(0x1000), 49664, DevicePtr(0x20002), 24576, &[]).unwrap();
    assert_eq!(p.row(2).unwrap().0 + 16384, 0x1000 + 49664);
}
#[test]
fn staging_rejects_short_alias_unaligned_and_overflow_ranges() {
    for (ptr, bytes, out, outbytes) in [
        (0x1000, 49663, 0x20000, 24576),
        (0x1002, 49664, 0x20000, 24576),
        (0x1000, 49664, 0x20001, 24576),
        (0x1000, 49664, 0x20000, 24575),
        (0x1000, 49664, 0x1200, 24576),
        (u64::MAX - 15, 49664, 0x20000, 24576),
        (0x1000, 49664, u64::MAX - 1, 24576),
        (0, 49664, 0x20000, 24576),
    ] {
        assert!(StagePlan::new(DevicePtr(ptr), bytes, DevicePtr(out), outbytes, &[]).is_err());
    }
    assert!(
        StagePlan::new(
            DevicePtr(0x1000),
            49664,
            DevicePtr(0x20000),
            24576,
            &[(DevicePtr(0x9200), 16)]
        )
        .is_err()
    );
    // Exactly adjacent index scratch and output boundaries do not overlap.
    StagePlan::new(
        DevicePtr(0x1000),
        49664,
        DevicePtr(0xd200),
        24576,
        &[(DevicePtr(0x1000), 512)],
    )
    .unwrap();
}
#[test]
fn strict_flag_and_exact_local_weight_geometry() {
    assert!(!parse_flag(None).unwrap());
    assert!(!parse_flag(Some("0")).unwrap());
    assert!(parse_flag(Some("1")).unwrap());
    for s in ["", "true", "2", " 1"] {
        assert!(parse_flag(Some(s)).is_err());
    }
    validate_weight(DevicePtr(0x1000), 4096, 8192).unwrap();
    for (p, n, k) in [
        (0, 4096, 8192),
        (0x1002, 4096, 8192),
        (0x1000, 4096, 16384),
        (0x1000, 2048, 8192),
    ] {
        assert!(validate_weight(DevicePtr(p), n, k).is_err());
    }
}
