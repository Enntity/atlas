// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

// The MoE tests load the same recording backend under their own module.
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../moe/gate_up_btile_test_gpu.rs"]
mod recording;

const TIERS: [KernelHandle; 2] = [KernelHandle(8), KernelHandle(16)];
/// W_uv: 32 heads of [n = 256, k = 512], token rows packed.
const W_UV: [u32; 5] = [32, 512, 256, 32 * 512, 32 * 256];

/// Grouped launch of `m` rows with `[g, k, n, a_stride, c_stride]`.
fn grouped(gpu: &dyn GpuBackend, m: u32, [g, k, n, a_stride, c_stride]: [u32; 5]) -> Result<()> {
    mxfp8_gemv_grouped(
        gpu,
        &TIERS,
        DevicePtr(0x100),
        DevicePtr(0x200),
        DevicePtr(0x300),
        DevicePtr(0x400),
        m,
        g,
        k,
        n,
        a_stride,
        c_stride,
        0,
    )
}

#[test]
fn grouped_gemv_launches_one_head_plane_per_z_on_the_row_tier() {
    for (m, tier) in [(1, 8), (8, 8), (9, 16), (16, 16)] {
        let gpu = MockGpuBackend::new();
        grouped(&gpu, m, W_UV).unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].func, tier, "m={m}");
        assert_eq!(launches[0].grid, [256 / 16, 1, 32]);
        assert_eq!(launches[0].block, [256, 1, 1]);
    }
}

#[test]
fn grouped_gemv_passes_the_kernel_abi_in_order() {
    use recording::{Arg, Event};
    let gpu = recording::Gpu::new();
    grouped(&gpu, 5, W_UV).unwrap();
    let ptr = |p| Arg::Ptr(DevicePtr(p));
    let u32 = |v: u32| Arg::Bytes(v.to_le_bytes().to_vec());
    // (A, W, S, C, M, N, K, lda, out_stride)
    let abi = vec![
        ptr(0x100),
        ptr(0x200),
        ptr(0x300),
        ptr(0x400),
        u32(5),
        u32(256),
        u32(512),
        u32(32 * 512),
        u32(32 * 256),
    ];
    assert_eq!(
        gpu.trace(),
        [Event::Launch(8, [16, 1, 32], [256, 1, 1], 0, 0, abi)]
    );
}

#[test]
fn grouped_gemv_rejects_unservable_shapes() {
    let gpu = MockGpuBackend::new();
    let [g, k, n, a, c] = W_UV;
    for (m, shape) in [
        (0, W_UV),
        (17, W_UV),
        (8, [g, 500, n, a, c]),
        (8, [0, k, n, a, c]),
        (8, [g, k, n, a - 8, c]),
        (8, [g, k, n, a, c - 1]),
        (8, [g, k, n, a + 4, c]),
    ] {
        assert!(grouped(&gpu, m, shape).is_err(), "m={m} {shape:?}");
    }
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn tiled_switch_is_explicit() {
    assert!(!parse_tiled(None).unwrap());
    assert!(!parse_tiled(Some("0")).unwrap());
    assert!(parse_tiled(Some("1")).unwrap());
    for value in ["", "true", "2", " 1"] {
        assert!(parse_tiled(Some(value)).is_err(), "{value:?}");
    }
}

#[test]
fn plain_tiers_split_at_8_and_16_rows() {
    for (m, tier) in [(1, 0), (8, 0), (9, 1), (16, 1), (17, 2), (32, 2)] {
        assert_eq!(mxfp8_tier(m), tier, "m={m}");
    }
}

/// `mxfp8_gemv_rows` at (`rows`): the launches' (kernel, M, input offset,
/// output offset) for a [n = 48, k = 64] twin.
fn row_launches(rows: usize) -> Vec<(u64, Vec<recording::Arg>)> {
    use recording::Event;
    let gpu = recording::Gpu::new();
    let w = Mxfp8Weight {
        data: DevicePtr(0x200),
        scales: DevicePtr(0x300),
        tiled: false,
    };
    let tiers = [KernelHandle(8), KernelHandle(16), KernelHandle(32)];
    mxfp8_gemv_rows(
        &gpu,
        &tiers,
        &w,
        DevicePtr(0x10_0000),
        DevicePtr(0x20_0000),
        rows,
        48,
        64,
        48,
        0,
    )
    .unwrap();
    gpu.trace()
        .into_iter()
        .map(|e| match e {
            Event::Launch(f, grid, block, _, _, args) => {
                assert_eq!((grid, block), ([3, 1, 1], [256, 1, 1]));
                (f, args)
            }
            other => panic!("unexpected {other:?}"),
        })
        .collect()
}

#[test]
fn rows_helper_chunks_at_32_rows_on_each_chunk_tier() {
    use recording::Arg;
    let u32 = |v: u32| Arg::Bytes(v.to_le_bytes().to_vec());
    for (rows, want) in [
        (1usize, vec![(8u64, 1u32)]),
        (8, vec![(8, 8)]),
        (9, vec![(16, 9)]),
        (32, vec![(32, 32)]),
        (33, vec![(32, 32), (8, 1)]),
        (80, vec![(32, 32), (32, 32), (16, 16)]),
    ] {
        let launches = row_launches(rows);
        assert_eq!(launches.len(), want.len(), "rows={rows}");
        for (i, ((func, args), (tier, m))) in launches.iter().zip(&want).enumerate() {
            let first = 32 * i;
            // (A, W, S, C, M, N, K, out_stride): chunk i reads rows 32i.. of
            // A (k = 64 BF16 a row) and writes rows 32i.. of C (n = 48).
            let abi = vec![
                Arg::Ptr(DevicePtr(0x10_0000 + (first * 64 * 2) as u64)),
                Arg::Ptr(DevicePtr(0x200)),
                Arg::Ptr(DevicePtr(0x300)),
                Arg::Ptr(DevicePtr(0x20_0000 + (first * 48 * 2) as u64)),
                u32(*m),
                u32(48),
                u32(64),
                u32(48),
            ];
            assert_eq!((*func, args), (*tier, &abi), "rows={rows} chunk {i}");
        }
    }
    assert!(row_launches(0).is_empty());
}

#[test]
fn rows_helper_refuses_a_missing_tier() {
    let gpu = MockGpuBackend::new();
    let w = Mxfp8Weight {
        data: DevicePtr(0x200),
        scales: DevicePtr(0x300),
        tiled: false,
    };
    let tiers = [KernelHandle(8), KernelHandle(0), KernelHandle(32)];
    let run = |rows| {
        mxfp8_gemv_rows(
            &gpu,
            &tiers,
            &w,
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            rows,
            48,
            64,
            48,
            0,
        )
    };
    assert!(run(8).is_ok());
    assert!(run(12).is_err());
    assert_eq!(gpu.launch_count(), 1);
}

#[test]
fn rows_helper_writes_rows_out_stride_apart() {
    use recording::{Arg, Event};
    let gpu = recording::Gpu::new();
    let w = Mxfp8Weight {
        data: DevicePtr(0x200),
        scales: DevicePtr(0x300),
        tiled: false,
    };
    let tiers = [KernelHandle(8), KernelHandle(16), KernelHandle(32)];
    let run = |stride| {
        mxfp8_gemv_rows(
            &gpu,
            &tiers,
            &w,
            DevicePtr(0x10_0000),
            DevicePtr(0x20_0000),
            40,
            48,
            64,
            stride,
            0,
        )
    };
    // n = 48 into rows 100 elements apart: chunk 2 starts at row 32's slot.
    run(100).unwrap();
    let launches: Vec<Vec<Arg>> = gpu
        .trace()
        .into_iter()
        .filter_map(|e| match e {
            Event::Launch(.., args) => Some(args),
            _ => None,
        })
        .collect();
    assert_eq!(launches.len(), 2);
    assert_eq!(
        launches[1][3],
        Arg::Ptr(DevicePtr(0x20_0000 + 32 * 100 * 2))
    );
    assert_eq!(launches[1][7], Arg::Bytes(100u32.to_le_bytes().to_vec()));
    // A stride shorter than a row would overlap the rows.
    assert!(run(47).is_err());
}
