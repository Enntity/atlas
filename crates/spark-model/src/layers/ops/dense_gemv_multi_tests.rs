// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::{MockGpuBackend, MockLaunch};

/// The one launch `dense_gemv_batchm_dual` issues for eight rows of N=4096.
fn batchm_dual_launch(k: u32) -> MockLaunch {
    let gpu = MockGpuBackend::new();
    let weights = [DevicePtr(0x100), DevicePtr(0x200)].map(|weight| DenseWeight { weight });
    dense_gemv_batchm_dual(
        &gpu,
        KernelHandle(7),
        [DevicePtr(0x300), DevicePtr(0x400)],
        [&weights[0], &weights[1]],
        [DevicePtr(0x500), DevicePtr(0x600)],
        8,
        4096,
        k,
        0,
    )
    .unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    launches[0].clone()
}

#[test]
fn kda_gate_pair_at_k128_uses_the_short_k_dual_tier() {
    // MockGpuBackend::kernel resolves every symbol to 0xDEAD; the caller's
    // generic handle is 7.
    let launch = batchm_dual_launch(128);
    assert_eq!(launch.func, 0xDEAD);
    assert_eq!(launch.grid, [4096 / 16, 1, 2]);
    assert_eq!(launch.block, [256, 1, 1]);
}

#[test]
fn other_k_keeps_the_generic_dual() {
    let launch = batchm_dual_launch(4096);
    assert_eq!(launch.func, 7);
    assert_eq!(launch.grid, [4096 / 4, 1, 2]);
    assert_eq!(launch.block, [256, 1, 1]);
}
