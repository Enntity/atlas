// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

// The MoE tests load the same recording backend under their own module.
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../moe/gate_up_btile_test_gpu.rs"]
mod recording;
use recording::{Arg, Event};

fn on(mode: &str) -> Settings {
    parse(Some(mode), None, None, None).unwrap().unwrap()
}

fn region(ptr: u64, row_bytes: u32, ld: u64, rows: u32) -> L2Region {
    L2Region {
        ptr: DevicePtr(ptr),
        ld,
        row_bytes,
        rows,
    }
}

#[test]
fn off_unless_asked() {
    assert_eq!(parse(None, None, None, None).unwrap(), None);
    assert_eq!(parse(Some("0"), Some("99"), None, Some("x")).unwrap(), None);
}

#[test]
fn modes_and_defaults() {
    for (name, mode) in [
        ("lines", 0),
        ("sectors", 1),
        ("last", 2),
        ("1", 3),
        ("touch", 3),
        ("bulk", 4),
    ] {
        let s = on(name);
        assert_eq!(s.mode, mode, "{name}");
        assert_eq!((s.budget, s.ctas, s.sites), (12 << 20, 32, [true; 4]));
    }
    for bad in ["", "2", "Touch", "on"] {
        assert!(parse(Some(bad), None, None, None).is_err(), "{bad:?}");
    }
}

#[test]
fn budget_ctas_and_sites_are_checked() {
    let s = parse(Some("1"), Some("24"), Some("96"), Some("qo"))
        .unwrap()
        .unwrap();
    assert_eq!(
        (s.budget, s.ctas, s.sites),
        (24 << 20, 96, [false, false, true, true])
    );
    for (mb, ctas) in [("0", "8"), ("25", "8"), ("x", "8"), ("8", "0"), ("8", "97")] {
        assert!(
            parse(Some("1"), Some(mb), Some(ctas), None).is_err(),
            "{mb} {ctas}"
        );
    }
    for sites in ["", "afqox", "A"] {
        assert!(
            parse(Some("1"), None, None, Some(sites)).is_err(),
            "{sites:?}"
        );
    }
}

#[test]
fn budget_takes_whole_regions_then_a_prefix_or_whole_rows() {
    let whole = [region(0x1000, 100, 100, 1), region(0x2000, 50, 50, 1)];
    assert_eq!(l2_budgeted(&whole, 150), whole.to_vec());
    assert_eq!(
        l2_budgeted(&whole, 130),
        vec![whole[0], region(0x2000, 30, 30, 1)]
    );
    let strided = [region(0x3000, 64, 128, 10)];
    assert_eq!(l2_budgeted(&strided, 200), vec![region(0x3000, 64, 128, 3)]);
    assert_eq!(l2_budgeted(&strided, 63), vec![]);
    assert_eq!(l2_budgeted(&whole, 0), vec![]);
}

#[test]
fn budget_skips_empty_regions_and_caps_the_count() {
    let mut many = vec![L2Region::NONE, region(0, 64, 64, 1)];
    many.extend((0..10).map(|i| region(0x1000 * (i + 1), 16, 16, 1)));
    let out = l2_budgeted(&many, 1 << 20);
    assert_eq!(out.len(), L2_REGIONS);
    assert_eq!(out[0].ptr, DevicePtr(0x1000));
}

#[test]
fn nvfp4_regions_are_scales_then_values() {
    let w = QuantizedWeight {
        weight: DevicePtr(0x10_0000),
        weight_scale: DevicePtr(0x20_0000),
        ..QuantizedWeight::null()
    };
    // A whole [1024, 4096] weight: two contiguous runs.
    assert_eq!(
        L2Region::nvfp4(&w, 1024, 4096, 2048, 256),
        [
            L2Region::whole(DevicePtr(0x20_0000), 1024 * 256),
            L2Region::whole(DevicePtr(0x10_0000), 1024 * 2048),
        ]
    );
    // A K-slice of 1024 columns of a 2048-wide weight: strided rows.
    assert_eq!(
        L2Region::nvfp4(&w, 4096, 1024, 1024, 128),
        [
            region(0x20_0000, 64, 128, 4096),
            region(0x10_0000, 512, 1024, 4096)
        ]
    );
    let mx = Mxfp8Weight {
        data: DevicePtr(0x30_0000),
        scales: DevicePtr(0x40_0000),
    };
    assert_eq!(
        L2Region::mxfp8(&mx, 8192, 1536),
        [
            L2Region::whole(DevicePtr(0x40_0000), 8192 * 48),
            L2Region::whole(DevicePtr(0x30_0000), 8192 * 1536),
        ]
    );
}

const LANE_FIXTURE: Lane = Lane {
    kernel: KernelHandle(91),
    stream: 0x5150,
    event: 0x77,
};

fn u32_arg(v: u32) -> Arg {
    Arg::Bytes(v.to_le_bytes().to_vec())
}

fn u64_arg(v: u64) -> Arg {
    Arg::Bytes(v.to_le_bytes().to_vec())
}

#[test]
fn fork_launches_the_budgeted_regions_on_the_side_stream() {
    let gpu = recording::Gpu::new();
    let s = Settings {
        budget: 48,
        ..on("sectors")
    };
    let regions = [region(0x1000, 32, 32, 1), region(0x2000, 8, 64, 4)];
    fork(&gpu, &LANE_FIXTURE, s, 0x99, L2Site::Ffn, &regions).unwrap();
    let mut args = Vec::new();
    for r in [region(0x1000, 32, 32, 1), region(0x2000, 8, 64, 2)]
        .into_iter()
        .chain(std::iter::repeat_n(L2Region::NONE, L2_REGIONS - 2))
    {
        args.extend([
            Arg::Ptr(r.ptr),
            u64_arg(r.ld),
            u32_arg(r.row_bytes),
            u32_arg(r.rows),
        ]);
    }
    args.push(u32_arg(1));
    assert_eq!(
        gpu.trace(),
        vec![Event::Launch(91, [32, 1, 1], [256, 1, 1], 0, 0x5150, args)]
    );
}

#[test]
fn fork_skips_unlisted_sites_captures_and_empty_budgets() {
    let gpu = recording::Gpu::new();
    let regions = [region(0x1000, 32, 32, 1)];
    let qo = parse(Some("1"), None, None, Some("qo")).unwrap().unwrap();
    fork(&gpu, &LANE_FIXTURE, qo, 0x99, L2Site::Attn, &regions).unwrap();
    fork(
        &gpu,
        &LANE_FIXTURE,
        qo,
        0x99,
        L2Site::Index,
        &[L2Region::NONE],
    )
    .unwrap();
    gpu.capture
        .store(true, std::sync::atomic::Ordering::Relaxed);
    fork(&gpu, &LANE_FIXTURE, qo, 0x99, L2Site::Output, &regions).unwrap();
    assert!(gpu.trace().is_empty());
}

#[test]
fn a_failed_fork_turns_the_lane_off_without_an_error() {
    let (gpu, failed) = (recording::Gpu::new(), AtomicBool::new(false));
    let regions = [region(0x1000, 32, 32, 1)];
    let s = on("touch");
    gpu.fail.store(1, Ordering::Relaxed);
    best_effort(&failed, || {
        fork(&gpu, &LANE_FIXTURE, s, 0x99, L2Site::Attn, &regions)
    });
    assert!(failed.load(Ordering::Relaxed));
    gpu.clear();
    best_effort(&failed, || {
        fork(&gpu, &LANE_FIXTURE, s, 0x99, L2Site::Attn, &regions)
    });
    assert!(gpu.trace().is_empty());
}
