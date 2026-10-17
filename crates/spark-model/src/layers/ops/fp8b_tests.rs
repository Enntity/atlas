// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../moe/gate_up_btile_test_gpu.rs"]
mod recording;

use recording::{Arg, Event};

fn ptr(p: u64) -> Arg {
    Arg::Ptr(DevicePtr(p))
}

fn u32a(v: u32) -> Arg {
    Arg::Bytes(v.to_le_bytes().to_vec())
}

fn tiers() -> MxTiers {
    MxTiers {
        row_major: [KernelHandle(8), KernelHandle(16), KernelHandle(32)],
        tiled: [KernelHandle(108), KernelHandle(116), KernelHandle(132)],
    }
}

fn launches(gpu: &recording::Gpu) -> Vec<(u64, [u32; 3], Vec<Arg>)> {
    gpu.trace()
        .into_iter()
        .filter_map(|e| match e {
            Event::Launch(f, grid, block, _, _, args) => {
                assert_eq!(block, [256, 1, 1]);
                Some((f, grid, args))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn prefill_switch_parses_three_modes() {
    assert_eq!(Fp8bPrefill::parse(None).unwrap(), Fp8bPrefill::W8a8);
    assert_eq!(Fp8bPrefill::parse(Some("w8a8")).unwrap(), Fp8bPrefill::W8a8);
    assert_eq!(
        Fp8bPrefill::parse(Some("w8a16")).unwrap(),
        Fp8bPrefill::W8a16
    );
    assert_eq!(Fp8bPrefill::parse(Some("bf16")).unwrap(), Fp8bPrefill::Bf16);
    for v in ["1", "W8A8", "fp8"] {
        assert!(Fp8bPrefill::parse(Some(v)).is_err(), "{v}");
    }
    assert!(Fp8bPrefill::Bf16.needs_bf16() && !Fp8bPrefill::W8a8.needs_bf16());
}

#[test]
fn gemv_rows_chunk_at_32_on_the_layout_tier() {
    for (tiled, base) in [(false, 0u64), (true, 100)] {
        let gpu = recording::Gpu::new();
        let w = Fp8BlockWeight {
            data: DevicePtr(0x200),
            scales: DevicePtr(0x300),
            tiled,
        };
        // [n = 256, k = 128], 40 rows: a 32-row launch and an 8-row one.
        fp8b_gemv_rows(
            &gpu,
            &tiers(),
            &w,
            DevicePtr(0x10_0000),
            DevicePtr(0x20_0000),
            40,
            256,
            128,
            256,
            0,
        )
        .unwrap();
        let got = launches(&gpu);
        assert_eq!(got.len(), 2);
        for (i, (rows, tier)) in [(32u32, 32u64), (8, 8)].into_iter().enumerate() {
            let first = 32 * i as u64;
            assert_eq!(got[i].0, base + tier);
            assert_eq!(got[i].1, [16, 1, 1]);
            // (A, W, S, C, M, N, K, out_stride)
            assert_eq!(
                got[i].2,
                vec![
                    ptr(0x10_0000 + first * 128 * 2),
                    ptr(0x200),
                    ptr(0x300),
                    ptr(0x20_0000 + first * 256 * 2),
                    u32a(rows),
                    u32a(256),
                    u32a(128),
                    u32a(256),
                ]
            );
        }
    }
}

#[test]
fn gemv_rows_refuse_off_grid_shapes_and_missing_tiers() {
    let gpu = recording::Gpu::new();
    let w = Fp8BlockWeight {
        data: DevicePtr(0x200),
        scales: DevicePtr(0x300),
        tiled: false,
    };
    let x = DevicePtr(0x1000);
    // k off the 128 grid, n off 16, out_stride < n.
    for (n, k, os) in [(256, 96, 256), (250, 128, 256), (256, 128, 128)] {
        assert!(fp8b_gemv_rows(&gpu, &tiers(), &w, x, x, 4, n, k, os, 0).is_err());
    }
    assert!(fp8b_gemv_rows(&gpu, &MxTiers::default(), &w, x, x, 4, 256, 128, 256, 0).is_err());
    assert!(launches(&gpu).is_empty());
}

#[test]
fn multi_launch_packs_three_projections_into_one_grid() {
    let gpu = recording::Gpu::new();
    let parts = [
        MxPart {
            data: DevicePtr(0x1),
            scales: DevicePtr(0x2),
            out: DevicePtr(0x10_0000),
            n: 1536,
        },
        MxPart {
            data: DevicePtr(0x3),
            scales: DevicePtr(0x4),
            out: DevicePtr(0x20_0000),
            n: 512,
        },
    ];
    // 9 rows: the 16-row tier; grid = 1536/16 + 512/16 tiles; the third
    // projection slot is null with N = 0.
    mx_multi_rows(
        &gpu,
        &tiers(),
        false,
        DevicePtr(0x40_0000),
        &parts,
        9,
        4096,
        0,
    )
    .unwrap();
    let got = launches(&gpu);
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].0, got[0].1), (16, [96 + 32, 1, 1]));
    assert_eq!(
        got[0].2,
        vec![
            ptr(0x40_0000),
            ptr(0x1),
            ptr(0x2),
            ptr(0x10_0000),
            u32a(1536),
            ptr(0x3),
            ptr(0x4),
            ptr(0x20_0000),
            u32a(512),
            ptr(0),
            ptr(0),
            ptr(0),
            u32a(0),
            u32a(9),
            u32a(4096),
        ]
    );
}

#[test]
fn multi_launch_offsets_every_output_per_32_row_chunk() {
    let gpu = recording::Gpu::new();
    let part = |o: u64| MxPart {
        data: DevicePtr(0x1),
        scales: DevicePtr(0x2),
        out: DevicePtr(o),
        n: 64,
    };
    let parts = [part(0x10_0000), part(0x20_0000), part(0x30_0000)];
    mx_multi_rows(
        &gpu,
        &tiers(),
        true,
        DevicePtr(0x40_0000),
        &parts,
        33,
        128,
        0,
    )
    .unwrap();
    let got = launches(&gpu);
    assert_eq!(got.len(), 2);
    assert_eq!(got[1].0, 108);
    assert_eq!(got[1].2[0], ptr(0x40_0000 + 32 * 128 * 2));
    for (i, base) in [0x10_0000u64, 0x20_0000, 0x30_0000].into_iter().enumerate() {
        assert_eq!(got[1].2[3 + 4 * i], ptr(base + 32 * 64 * 2));
    }
    assert!(mx_multi_rows(&gpu, &tiers(), false, DevicePtr(0), &[], 1, 128, 0).is_err());
}

#[test]
fn ld_rows_pass_the_slice_stride() {
    let gpu = recording::Gpu::new();
    let tiers = [KernelHandle(8), KernelHandle(16), KernelHandle(32)];
    // down's K-slice: [n = 4096, k = 1024] of a [4096, 2048] weight.
    fp8b_gemv_ld_rows(
        &gpu,
        &tiers,
        DevicePtr(0x200),
        DevicePtr(0x300),
        DevicePtr(0x1000),
        DevicePtr(0x2000),
        5,
        4096,
        1024,
        2048,
        0,
    )
    .unwrap();
    let got = launches(&gpu);
    assert_eq!((got[0].0, got[0].1), (8, [256, 1, 1]));
    assert_eq!(got[0].2[8], u32a(2048));
    let x = DevicePtr(0x1000);
    assert!(fp8b_gemv_ld_rows(&gpu, &tiers, x, x, x, x, 5, 4096, 1024, 1000, 0).is_err());
}
