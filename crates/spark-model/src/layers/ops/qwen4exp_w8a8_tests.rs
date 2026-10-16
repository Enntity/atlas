// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::*;

const ROWS: usize = 512;

fn arena(gpu: &MockGpuBackend) -> BufferArena {
    BufferArena::new(&ModelConfig::qwen3_next_80b_nvfp4(), ROWS, 4096, 16, 8, gpu).unwrap()
}

/// The row counts of the quant launches since `from` (each launch is one
/// thread per 8 values: `m * k / 8` threads in 256-thread blocks).
fn quant_rows(gpu: &MockGpuBackend, from: usize, k: usize) -> Vec<usize> {
    gpu.launches_snapshot()[from..]
        .iter()
        .map(|l| l.grid[0] as usize * 256 * 8 / k)
        .collect()
}

#[test]
fn only_zero_is_off_and_garbage_is_the_default() {
    assert!(!parse(Some("0")));
    for v in [
        None,
        Some("1"),
        Some(""),
        Some("true"),
        Some("off"),
        Some(" 0"),
    ] {
        assert!(parse(v), "{v:?}");
    }
}

#[test]
fn a_matching_pending_range_leaves_only_its_complement() {
    let x = DevicePtr(0x1000);
    let q = |total, row0, rows| {
        let p = Pending {
            x,
            total,
            hidden: 2560,
            row0,
            rows,
        };
        quant_ranges(Some(p), x, total, 2560)
    };
    // SP rank 0 (rows from 0), rank 1 (from the split), no split, and a
    // ragged tail, which ends at m.
    assert_eq!(q(4096, 0, 2048), [0..0, 2048..4096]);
    assert_eq!(q(4096, 2048, 2048), [0..2048, 4096..4096]);
    assert_eq!(q(4096, 0, 4096), [0..0, 4096..4096]);
    assert_eq!(q(3048, 2048, 1000), [0..2048, 3048..3048]);
}

#[test]
fn a_pending_range_for_anything_else_is_ignored() {
    let x = DevicePtr(0x1000);
    let p = Pending {
        x,
        total: 4096,
        hidden: 2560,
        row0: 0,
        rows: 2048,
    };
    let all = [0..4096, 4096..4096];
    assert_eq!(quant_ranges(None, x, 4096, 2560), all);
    assert_eq!(quant_ranges(Some(p), DevicePtr(0x2000), 4096, 2560), all);
    assert_eq!(quant_ranges(Some(p), x, 4096, 3072), all);
    assert_eq!(quant_ranges(Some(p), x, 4000, 2560)[0], 0..4000);
    // Off the 128-row tile grid: its scales would not start a tile.
    let off = Pending { rows: 2000, ..p };
    assert_eq!(quant_ranges(Some(off), x, 4096, 2560), all);
}

#[test]
fn a_produced_collapse_saves_its_rows_from_the_quant() {
    let gpu = MockGpuBackend::new();
    let b = arena(&gpu);
    let (x, h, k) = (b.norm_output(), 2048usize, KernelHandle(0xDEAD));
    drop(Producer::arm(&b, x, ROWS, 256, 256, h).unwrap());
    assert!(
        take_target(256, h as u32).is_none(),
        "dropping a producer disarms it"
    );

    // SP rank 1: this rank collapses rows [256, 512) of the pass.
    let p = Producer::arm(&b, x, ROWS, 256, 256, h).unwrap();
    let t = take_target(256, h as u32).unwrap();
    assert!(
        take_target(256, h as u32).is_none(),
        "one collapse takes it"
    );
    let pad = mx_lt_inner_pad(h);
    assert_eq!(t.act.q, b.fp8_act().offset(256 * h));
    assert_eq!(t.act.sf, b.fp8_act_scale().offset(256 * pad));
    assert_eq!(t.inner_pad() as usize, pad);
    assert_eq!(t.slab(128).unwrap().q, t.act.q.offset(128 * h));
    mark_produced();
    p.finish();

    let n0 = gpu.launch_count();
    activation(&gpu, k, &b, x, ROWS, h, 0).unwrap();
    assert_eq!(quant_rows(&gpu, n0, h), vec![256], "only the peer's rows");
    // Taken: the next call quantizes everything.
    let n1 = gpu.launch_count();
    activation(&gpu, k, &b, x, ROWS, h, 0).unwrap();
    assert_eq!(quant_rows(&gpu, n1, h), vec![ROWS]);
}

#[test]
fn an_unproduced_collapse_leaves_nothing_pending() {
    let gpu = MockGpuBackend::new();
    let b = arena(&gpu);
    let (x, h, k) = (b.norm_output(), 2048usize, KernelHandle(0xDEAD));
    // Armed, but the collapse took the default arm (never marked).
    let p = Producer::arm(&b, x, ROWS, 0, ROWS, h).unwrap();
    let _ = take_target(ROWS as u32, h as u32);
    p.finish();
    let n0 = gpu.launch_count();
    activation(&gpu, k, &b, x, ROWS, h, 0).unwrap();
    assert_eq!(quant_rows(&gpu, n0, h), vec![ROWS]);
    // Off the tile grid: not armed at all.
    assert!(Producer::arm(&b, x, ROWS, 100, 100, h).is_none());
    assert!(take_target(100, h as u32).is_none());
}

#[test]
fn the_scratch_refuses_what_it_cannot_hold() {
    let gpu = MockGpuBackend::new();
    let b = arena(&gpu);
    let x = b.norm_output();
    let k = KernelHandle(0xDEAD);
    let e = activation(&gpu, k, &b, x, ROWS + 128, 1 << 14, 0).unwrap_err();
    assert!(e.to_string().contains("FP8 scratch"), "{e}");
    assert!(Producer::arm(&b, x, ROWS * 4, 0, ROWS, 1 << 14).is_none());
}
