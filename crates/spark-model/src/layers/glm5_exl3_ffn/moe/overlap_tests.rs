// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn prime_uses_classic_blas_and_drains_before_cleanup() {
    let gpu = MockGpuBackend::new();
    prime_side_stream_blas(&gpu, 41).unwrap();
    let requests = gpu.bf16_gemm_requests_snapshot();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].stream, 41);
    assert_eq!(
        requests[0].numerics,
        spark_runtime::gpu::Bf16GemmNumerics::Glm53Classic
    );
    assert_eq!(gpu.sync_count(), 1);
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn failed_gemm_still_drains_and_frees_all_operands() {
    let gpu = MockGpuBackend::new();
    gpu.fail_next_bf16_gemm();
    let error = prime_side_stream_blas(&gpu, 42).unwrap_err();
    assert!(format!("{error:#}").contains("mock BF16 GEMM failed"));
    assert_eq!(
        gpu.sync_count(),
        1,
        "a failed GEMM may have partially enqueued work"
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn partial_operand_allocation_is_rolled_back_on_every_failure_position() {
    for fail_at in 0..3 {
        let mut allocated = 0;
        let mut freed = Vec::new();
        let result = allocate_prime_operands(
            || {
                if allocated == fail_at {
                    anyhow::bail!("allocation failed");
                }
                allocated += 1;
                Ok(DevicePtr(allocated as u64))
            },
            |ptr| {
                freed.push(ptr);
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(freed.len(), fail_at);
        for ptr in 1..=fail_at {
            assert!(freed.contains(&DevicePtr(ptr as u64)));
        }
    }
}

#[test]
fn cached_side_stream_is_primed_once() {
    let gpu = MockGpuBackend::new();
    let side = overlap_side_stream(&gpu, 0x72635f7072696d65).unwrap();
    assert_eq!(overlap_side_stream(&gpu, 0x72635f7072696d65).unwrap(), side);
    assert_eq!(gpu.bf16_gemm_requests_snapshot().len(), 1);
}

#[test]
fn cleanup_attempts_every_operand_even_when_one_free_fails() {
    let gpu = MockGpuBackend::new();
    gpu.fail_next_free();
    assert!(prime_side_stream_blas(&gpu, 43).is_err());
    assert_eq!(gpu.sync_count(), 1);
    assert_eq!(
        gpu.alloc_count(),
        1,
        "only the failed free remains allocated"
    );
}
