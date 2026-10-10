// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn explicit_native_query_dispatch_uses_f32_weights_and_rejects_capacity_before_launch() {
    let gpu = MockGpuBackend::new();
    let mut exec = Glm5Exl3IndexExec::new(&gpu, 3, 32, 4096, 77).unwrap();
    exec.validate(2, 32, 128, 4096, 256, 77).unwrap();
    for (rows, heads, dim, hidden, bytes, stream) in [
        (4, 32, 128, 4096, 512, 77),
        (2, 32, 128, 4096, 255, 77),
        (2, 32, 128, 4096, 256, 78),
        (2, 32, 64, 4096, 256, 77),
        (2, 16, 128, 4096, 256, 77),
        (2, 32, 128, 2048, 256, 77),
    ] {
        assert!(
            exec.validate(rows, heads, dim, hidden, bytes, stream)
                .is_err()
        );
    }
    assert_eq!(gpu.launch_count(), 0);
    exec.query_and_weights(
        &gpu,
        DevicePtr(100),
        DevicePtr(200),
        DevicePtr(300),
        DevicePtr(400),
        2,
    )
    .unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 3);
    assert_eq!(launches[0].block, [128, 1, 1]);
    assert_eq!(launches[1].grid, [64, 1, 1]);
    assert_eq!(launches[2].block, [256, 1, 1]);
    gpu.fail_next_free();
    assert!(exec.release(&gpu).is_err());
    assert!(exec.validate(1, 32, 128, 4096, 128, 77).is_ok());
    exec.release(&gpu).unwrap();
    assert!(exec.validate(1, 32, 128, 4096, 128, 77).is_err());
    assert_eq!(gpu.sweep_unreleased(), 0);
}
