// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
#[test]
fn guarded_allocator_preserves_256byte_alignment_and_checks_all_live_operands() {
    let backend = MockGpuBackend::new();
    let gpu = GuardedGpu::new(&backend);
    let p = gpu.alloc(256).unwrap();
    gpu.copy_h2d(&[1; 256], p).unwrap();
    gpu.check_all().unwrap();
    backend.copy_h2d(&[0], DevicePtr(p.0 - 1)).unwrap();
    assert!(gpu.check_all().is_err());
    assert!(gpu.free(p).is_err());
    backend.copy_h2d(&[0xa5], DevicePtr(p.0 - 1)).unwrap();
    gpu.free(p).unwrap();
    assert_eq!(backend.alloc_count(), 0);
    assert_eq!(gpu.outstanding(), 0);
}
#[test]
fn failed_underlying_free_retains_guarded_allocation_for_retry() {
    let backend = MockGpuBackend::new();
    let gpu = GuardedGpu::new(&backend);
    let p = gpu.alloc(128).unwrap();
    backend.fail_next_free();
    assert!(gpu.free(p).is_err());
    gpu.check_all().unwrap();
    gpu.free(p).unwrap();
    assert_eq!(backend.alloc_count(), 0);
    assert_eq!(gpu.outstanding(), 0);
}
