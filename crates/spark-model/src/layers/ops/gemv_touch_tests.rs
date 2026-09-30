// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

// The MoE tests load the same recording backend under their own module.
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../moe/gate_up_btile_test_gpu.rs"]
mod recording;
use recording::{Arg, Event};

const TOUCH: GemvTouch = GemvTouch {
    kernel: KernelHandle(77),
    bytes: 12 << 20,
    ctas: 32,
};

fn weight(base: u64) -> QuantizedWeight {
    QuantizedWeight {
        weight: DevicePtr(base),
        weight_scale: DevicePtr(base + 0x10),
        weight_scale_2: 0.25,
        ..QuantizedWeight::null()
    }
}

fn ptr(p: u64) -> Arg {
    Arg::Ptr(DevicePtr(p))
}

fn u32s(values: &[u32]) -> Vec<Arg> {
    values
        .iter()
        .map(|v| Arg::Bytes(v.to_le_bytes().to_vec()))
        .collect()
}

fn f32_arg(v: f32) -> Arg {
    Arg::Bytes(v.to_le_bytes().to_vec())
}

/// The only launch on `gpu`: `(kernel, grid, args)`.
fn launch(gpu: &recording::Gpu) -> (u64, [u32; 3], Vec<Arg>) {
    match gpu.trace().as_slice() {
        [Event::Launch(kernel, grid, [256, 1, 1], 0, 0, args)] => (*kernel, *grid, args.clone()),
        other => panic!("expected one launch, got {other:?}"),
    }
}

#[test]
fn geometry_bounds_rows_by_bytes_and_ctas_by_the_grid() {
    // KDA q (4096 x 4096 NVFP4): 2304 B a row, the whole weight inside 12 MiB.
    assert_eq!(TOUCH.geometry(4096, 2304, 256), (4096, 32));
    // Dense FFN gate (12288 rows): 12 MiB of it.
    assert_eq!(TOUCH.geometry(12288, 2304, 768), (5461, 32));
    // A grid narrower than the CTA cap (kv_a: 32 CTAs at 16; here 8).
    assert_eq!(TOUCH.geometry(128, 4224, 8), (128, 8));
    // The LM head shape: never more rows than the byte bound.
    assert_eq!(TOUCH.geometry(77440, 4224, 4840), (2978, 32));
}

#[test]
fn tc8_twin_passes_the_strided_abi_with_the_touch_geometry() {
    // Shared down: this rank's 1024-column slice of a 2048-wide weight.
    let gpu = recording::Gpu::new();
    TOUCH
        .w4a16_tc(
            &gpu,
            DevicePtr(0x100),
            &weight(0x200),
            DevicePtr(0x300),
            5,
            4096,
            1024,
            1024,
            128,
            0,
        )
        .unwrap();
    let (kernel, grid, args) = launch(&gpu);
    assert_eq!((kernel, grid), (77, [256, 1, 1]));
    // (A, B_packed, B_scale, scale2, C, M, N, K, ld_half, ld_groups, touch_rows, touch_ctas)
    let mut abi = vec![
        ptr(0x100),
        ptr(0x200),
        ptr(0x210),
        f32_arg(0.25),
        ptr(0x300),
    ];
    abi.extend(u32s(&[5, 4096, 1024, 1024, 128, 4096, 32]));
    assert_eq!(args, abi);
}

#[test]
fn tc8_twin_rejects_rows_and_strides_the_kernel_cannot_serve() {
    let gpu = recording::Gpu::new();
    let w = weight(0x200);
    let call = |m, k, ld_half, ld_groups| {
        TOUCH.w4a16_tc(
            &gpu,
            DevicePtr(0x100),
            &w,
            DevicePtr(0x300),
            m,
            4096,
            k,
            ld_half,
            ld_groups,
            0,
        )
    };
    assert!(call(33, 4096, 2048, 256).is_err());
    assert!(call(0, 4096, 2048, 256).is_err());
    assert!(call(8, 4104, 2052, 256).is_err());
    assert!(call(8, 4096, 2047, 256).is_err());
    assert!(call(8, 4096, 2048, 255).is_err());
    assert!(gpu.trace().is_empty());
}

#[test]
fn twins_follow_the_tier_row_counts() {
    assert_eq!(w4a16_tc_twin(8), Some("w4a16_gemv_tc8_touch"));
    assert_eq!(w4a16_tc_twin(16), Some("w4a16_gemv_tc16_touch"));
    assert_eq!(w4a16_tc_twin(32), Some("w4a16_gemv_tc32_touch"));
    assert_eq!(w4a16_tc_twin(5), None);
    for (m, twin) in [
        (1, "tc8"),
        (8, "tc8"),
        (9, "tc16"),
        (16, "tc16"),
        (17, "tc32"),
        (32, "tc32"),
    ] {
        assert_eq!(mxfp8_tc_twin(m), format!("mxfp8_gemv_{twin}_touch"));
    }
    // Every twin a site can name is one the resolver looks up.
    let names: Vec<_> = TWINS.iter().map(|&(_, f)| f).collect();
    for rows in [8, 16, 32] {
        assert!(names.contains(&w4a16_tc_twin(rows).unwrap()));
        assert!(names.contains(&mxfp8_tc_twin(rows)));
    }
}

#[test]
fn batch3_twin_takes_the_row_count_the_fixed_tiers_bake_in() {
    for m in [2, 3] {
        let gpu = recording::Gpu::new();
        TOUCH
            .w4a16_batch3(
                &gpu,
                DevicePtr(0x100),
                &weight(0x200),
                DevicePtr(0x300),
                m,
                4096,
                4096,
                0,
            )
            .unwrap();
        let (kernel, grid, args) = launch(&gpu);
        assert_eq!((kernel, grid), (77, [1024, 1, 1]));
        // (A, B_packed, B_scale, scale2, C, M, N, K, touch_rows, touch_ctas)
        let mut abi = vec![
            ptr(0x100),
            ptr(0x200),
            ptr(0x210),
            f32_arg(0.25),
            ptr(0x300),
        ];
        abi.extend(u32s(&[m, 4096, 4096, 4096, 32]));
        assert_eq!(args, abi);
    }
    let gpu = recording::Gpu::new();
    for m in [1, 4] {
        let w = weight(0x200);
        let out = DevicePtr(0x300);
        assert!(
            TOUCH
                .w4a16_batch3(&gpu, DevicePtr(0x100), &w, out, m, 4096, 4096, 0)
                .is_err()
        );
    }
}

#[test]
fn mxfp8_twin_passes_the_abi_with_the_touch_geometry() {
    // MLA kv_a: 512 x 4096, a 32-CTA grid.
    let gpu = recording::Gpu::new();
    TOUCH
        .mxfp8_tc(
            &gpu,
            DevicePtr(0x100),
            DevicePtr(0x200),
            DevicePtr(0x300),
            DevicePtr(0x400),
            8,
            512,
            4096,
            512,
            0,
        )
        .unwrap();
    let (kernel, grid, args) = launch(&gpu);
    assert_eq!((kernel, grid), (77, [32, 1, 1]));
    // (A, W, S, C, M, N, K, out_stride, touch_rows, touch_ctas)
    let mut abi = vec![ptr(0x100), ptr(0x200), ptr(0x300), ptr(0x400)];
    abi.extend(u32s(&[8, 512, 4096, 512, 512, 32]));
    assert_eq!(args, abi);
}

/// `w4a16_gemv_batch5_qkv` with and without the twin: the same leading
/// arguments, and the twin's two trailing ones only with it.
#[test]
fn batch5_qkv_appends_the_touch_geometry_only_for_the_twin() {
    let (q, k, v) = (weight(0x200), weight(0x400), weight(0x600));
    let run = |touch| {
        let gpu = recording::Gpu::new();
        crate::layers::ops::w4a16_gemv_batch5_qkv(
            &gpu,
            KernelHandle(5),
            touch,
            DevicePtr(0x100),
            &q,
            &k,
            &v,
            DevicePtr(0x800),
            5,
            4096,
            4096,
            0,
        )
        .unwrap();
        launch(&gpu)
    };
    let (plain_kernel, plain_grid, plain) = run(None);
    let (twin_kernel, twin_grid, twin) = run(Some(TOUCH));
    assert_eq!((plain_kernel, twin_kernel), (5, 77));
    assert_eq!(plain_grid, [1024, 1, 3]);
    assert_eq!(twin_grid, plain_grid);
    assert_eq!(plain.len(), 14);
    assert_eq!(twin[..14], plain[..]);
    assert_eq!(twin[14..], u32s(&[4096, 32])[..]);
}

/// With the flag unset (every test process) no twin is handed out and none is
/// looked up, so the launch sites stay on the original kernels.
#[test]
fn twins_are_neither_resolved_nor_handed_out_with_the_flag_off() {
    let gpu = recording::Gpu::new();
    gemv_touch_resolve(&gpu);
    for (_, func) in TWINS {
        assert!(gemv_touch(&gpu, func).is_none());
    }
    let (w, a, c) = (weight(0x200), DevicePtr(0x100), DevicePtr(0x300));
    for m in 2..=8 {
        assert!(w4a16_verify_touch(&gpu, KernelHandle(8), a, &w, c, m, 4096, 4096, 0).is_none());
    }
    assert!(gpu.trace().is_empty());
}
