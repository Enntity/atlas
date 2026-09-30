// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

const TIERS: [KernelHandle; 2] = [KernelHandle(8), KernelHandle(16)];

/// W_uv-shaped grouped launch (32 heads, [256, 512] each) of `m` rows.
fn grouped(gpu: &MockGpuBackend, m: u32, k: u32) -> Result<()> {
    mxfp8_gemv_grouped(
        gpu,
        &TIERS,
        DevicePtr(0x100),
        DevicePtr(0x200),
        DevicePtr(0x300),
        DevicePtr(0x400),
        m,
        32,
        k,
        256,
        32 * 512,
        32 * 256,
        0,
    )
}

#[test]
fn grouped_gemv_launches_one_head_plane_per_z_on_the_row_tier() {
    for (m, tier) in [(1, 8), (8, 8), (9, 16), (16, 16)] {
        let gpu = MockGpuBackend::new();
        grouped(&gpu, m, 512).unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].func, tier, "m={m}");
        assert_eq!(launches[0].grid, [256 / 16, 1, 32]);
        assert_eq!(launches[0].block, [256, 1, 1]);
    }
}

#[test]
fn grouped_gemv_rejects_unservable_shapes() {
    let gpu = MockGpuBackend::new();
    assert!(grouped(&gpu, 0, 512).is_err());
    assert!(grouped(&gpu, 17, 512).is_err());
    assert!(grouped(&gpu, 8, 500).is_err());
    assert_eq!(gpu.launch_count(), 0);
}
