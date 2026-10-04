// SPDX-License-Identifier: AGPL-3.0-only
use super::super::glm_decode_fuse::parse_groups;
use super::super::glm_decode_fuse::tests::{Capture, ptr, word};
use super::*;

const SHIPPED: &[(&str, &str)] = &[
    ("rms_norm_vanilla", "rms_norm_vanilla"),
    ("glm_rms_norm_regs", "rms_norm_vanilla_regs"),
    ("norm", "rms_norm"),
    (MODULE, "glm_hc_decode_partial_bf16"),
    (MODULE, "glm_hc_decode_post_partial_bf16"),
    (MODULE, "glm_hc_decode_partial_rows_bf16"),
    (MODULE, "glm_hc_decode_post_partial_rows_bf16"),
    (MODULE, "glm_hc_decode_partial_rows_touch_bf16"),
    (MODULE, "glm_hc_decode_post_partial_rows_touch_bf16"),
    (MODULE, "glm_hc_decode_finalize_norm_bf16"),
];

fn float(v: f32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

fn kernel(gpu: &Capture, name: &str) -> KernelHandle {
    let module = SHIPPED.iter().find(|&&(_, n)| n == name).unwrap().0;
    KernelHandle(gpu.handle(module, name))
}

const WEIGHT: DenseWeight = DenseWeight {
    weight: DevicePtr(0x9000),
};
/// Highway, partials, hc_scale, hc_base, seam output, post, comb.
const PTRS: [DevicePtr; 7] = [
    DevicePtr(0x1000),
    DevicePtr(0x2000),
    DevicePtr(0x3000),
    DevicePtr(0x4000),
    DevicePtr(0x5000),
    DevicePtr(0x6000),
    DevicePtr(0x7000),
];
const OUT: DevicePtr = DevicePtr(0x8000);

#[test]
fn flag_is_explicit_and_mask_selects_groups() {
    let parse = |fuse, mask| parse_groups(NAME, ALL, fuse, mask);
    assert_eq!(parse(None, None).unwrap(), 0);
    assert_eq!(parse(Some("0"), Some("3")).unwrap(), 0);
    assert_eq!(parse(Some("1"), None).unwrap(), HC_NORM | HC_TOUCH);
    assert_eq!(parse(Some("1"), Some("2")).unwrap(), HC_TOUCH);
    assert!(parse(Some("yes"), None).is_err());
    assert!(parse(Some("1"), Some("4")).is_err());
    assert_eq!(ALL, 3);
}

#[test]
fn touch_twin_replaces_only_the_rows_twins_when_its_group_is_on() {
    let gpu = Capture::new(SHIPPED);
    for (rows, touch) in TOUCH {
        let rows = kernel(&gpu, rows);
        let touch = kernel(&gpu, touch).0;
        assert_eq!(hc_partial_for(ALL, &gpu, rows).0, touch);
        assert_eq!(hc_partial_for(HC_TOUCH, &gpu, rows).0, touch);
        assert_eq!(hc_partial_for(HC_NORM, &gpu, rows).0, rows.0);
        assert_eq!(hc_partial_for(0, &gpu, rows).0, rows.0);
    }
    // The original partial kernels (DECODE_FUSE group 2 off) stay.
    for name in [
        "glm_hc_decode_partial_bf16",
        "glm_hc_decode_post_partial_bf16",
    ] {
        let original = kernel(&gpu, name);
        assert_eq!(hc_partial_for(ALL, &gpu, original).0, original.0);
    }
    assert_eq!(hc_partial_for(ALL, &gpu, KernelHandle(0)).0, 0);
    // A target without the touch twins keeps the rows twin.
    let old = Capture::new(&SHIPPED[..7]);
    let rows = KernelHandle(old.handle(MODULE, "glm_hc_decode_post_partial_rows_bf16"));
    assert_eq!(hc_partial_for(ALL, &old, rows).0, rows.0);
    assert!(gpu.launches().is_empty());
}

#[test]
fn finalize_norm_takes_the_seam_and_its_norm_in_one_launch() {
    let gpu = Capture::new(SHIPPED);
    for name in ["rms_norm_vanilla", "rms_norm_vanilla_regs"] {
        let norm = SeamNorm::new(kernel(&gpu, name), &WEIGHT, OUT, 1e-5);
        let fused = hc_finalize_norm_for(
            HC_NORM,
            &gpu,
            &norm,
            true,
            PTRS,
            [3, 4096, 20],
            [2e-5, 1e-6],
            7,
        );
        assert!(fused.unwrap());
        // The caller's norm then launches nothing.
        norm.run(&gpu, PTRS[4], 3, 4096, 7).unwrap();
    }
    let launches = gpu.launches();
    assert_eq!(launches.len(), 2);
    let mut want: Vec<Vec<u8>> = PTRS.iter().map(|p| ptr(p.0)).collect();
    want.extend([word(3), word(20), float(2e-5), float(1e-6)]);
    want.extend([ptr(WEIGHT.weight.0), ptr(OUT.0), float(1e-5)]);
    for (k, grid, block, args) in &launches {
        let fused = kernel(&gpu, "glm_hc_decode_finalize_norm_bf16").0;
        assert_eq!((*k, *grid, *block), (fused, [3, 1, 1], [1024, 1, 1]));
        assert_eq!(args[..], want[..]);
    }
}

#[test]
fn finalize_norm_leaves_every_other_seam_to_the_two_launches() {
    let gpu = Capture::new(SHIPPED);
    let regs = kernel(&gpu, "rms_norm_vanilla_regs");
    let off = |groups, kernel, bf16, ptrs, dims: [u32; 3]| {
        let norm = SeamNorm::new(kernel, &WEIGHT, OUT, 1e-5);
        !hc_finalize_norm_for(groups, &gpu, &norm, bf16, ptrs, dims, [1e-5, 1e-6], 7).unwrap()
            && norm.fused.get().is_none()
    };
    assert!(off(0, regs, true, PTRS, [3, 4096, 20]));
    assert!(off(HC_TOUCH, regs, true, PTRS, [3, 4096, 20]));
    assert!(off(ALL, regs, false, PTRS, [3, 4096, 20]));
    assert!(off(ALL, regs, true, PTRS, [3, 2048, 20]));
    assert!(off(ALL, regs, true, PTRS, [0, 4096, 20]));
    assert!(off(ALL, regs, true, PTRS, [33, 4096, 20]));
    // The offset-convention norm is another computation.
    assert!(off(
        ALL,
        kernel(&gpu, "rms_norm"),
        true,
        PTRS,
        [3, 4096, 20]
    ));
    // A norm in place over the seam output.
    let mut in_place = PTRS;
    in_place[4] = OUT;
    assert!(off(ALL, regs, true, in_place, [3, 4096, 20]));
    assert!(gpu.launches().is_empty());
    // A target without the fused finalizer.
    let old = Capture::new(&SHIPPED[..9]);
    let norm = SeamNorm::new(
        KernelHandle(old.handle("glm_rms_norm_regs", "rms_norm_vanilla_regs")),
        &WEIGHT,
        OUT,
        1e-5,
    );
    assert!(
        !hc_finalize_norm_for(ALL, &old, &norm, true, PTRS, [3, 4096, 20], [1e-5, 1e-6], 7)
            .unwrap()
    );
    // No norm handed in: nothing to fuse.
    assert!(!hc_finalize_norm(&gpu, None, true, PTRS, [3, 4096, 20], [1e-5, 1e-6], 7).unwrap());
}

#[test]
fn an_unfused_seam_norm_is_the_rms_norm_launch() {
    let gpu = Capture::new(SHIPPED);
    let regs = kernel(&gpu, "rms_norm_vanilla_regs");
    SeamNorm::new(regs, &WEIGHT, OUT, 1e-5)
        .run(&gpu, PTRS[4], 5, 4096, 7)
        .unwrap();
    super::super::rms_norm(&gpu, regs, PTRS[4], &WEIGHT, OUT, 5, 4096, 1e-5, 7).unwrap();
    let launches = gpu.launches();
    assert_eq!(launches.len(), 2);
    assert_eq!(launches[0], launches[1]);
    assert_eq!((launches[0].1, launches[0].2), ([5, 1, 1], [1024, 1, 1]));
}

#[test]
fn a_fused_norm_refuses_other_rows() {
    let gpu = Capture::new(SHIPPED);
    let norm = SeamNorm::new(kernel(&gpu, "rms_norm_vanilla"), &WEIGHT, OUT, 1e-5);
    assert!(
        hc_finalize_norm_for(ALL, &gpu, &norm, true, PTRS, [4, 4096, 20], [1e-5, 1e-6], 7).unwrap()
    );
    assert!(norm.run(&gpu, PTRS[4], 3, 4096, 7).is_err());
    assert!(norm.run(&gpu, PTRS[0], 4, 4096, 7).is_err());
    norm.run(&gpu, PTRS[4], 4, 4096, 7).unwrap();
    assert_eq!(gpu.launches().len(), 1);
}
