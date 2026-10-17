// SPDX-License-Identifier: AGPL-3.0-only
use super::scratch::*;
use atlas_core::{config::ModelConfig, scope::ModelResource};
use spark_runtime::gpu::mock::MockGpuBackend;

fn geometry() -> MoeScratchGeometry {
    MoeScratchGeometry::new(
        &ModelConfig {
            hidden_size: 128,
            moe_intermediate_size: 256,
            tp_world_size: 2,
            num_experts: 2,
            num_experts_per_tok: 2,
            tp_rank: 0,
            ..ModelConfig::qwen3_next_80b_nvfp4()
        },
        3,
        77,
    )
    .unwrap()
}
#[test]
fn scratch_allocated_once_for_many_borrowers_and_released_once() {
    let gpu = MockGpuBackend::new();
    let mut owner = MoeScratch::allocate(&gpu, geometry()).unwrap();
    let borrowers: Vec<_> = (0..42).map(|_| owner.borrow()).collect();
    assert_eq!(gpu.alloc_count(), 11);
    for b in &borrowers {
        assert_eq!(b.pointers().unwrap(), borrowers[0].pointers().unwrap());
    }
    owner.release(&gpu).unwrap();
    assert!(borrowers[0].pointers().is_err());
    owner.release(&gpu).unwrap();
}
#[test]
fn scratch_geometry_and_stream_are_checked_before_use() {
    let gpu = MockGpuBackend::new();
    let mut owner = MoeScratch::allocate(&gpu, geometry()).unwrap();
    let b = owner.borrow();
    assert!(b.validate(geometry(), 3, 77).is_ok());
    assert!(b.validate(geometry(), 4, 77).is_err());
    assert!(b.validate(geometry(), 3, 78).is_err());
    let mut wrong = geometry();
    wrong.hidden = 256;
    assert!(b.validate(wrong, 3, 77).is_err());
    owner.release(&gpu).unwrap();
}
#[test]
fn failed_scratch_free_is_retryable() {
    let gpu = MockGpuBackend::new();
    let mut owner = MoeScratch::allocate(&gpu, geometry()).unwrap();
    let p = owner.borrow().pointers().unwrap()[0];
    gpu.fail_next_free();
    assert!(owner.release(&gpu).is_err());
    assert!(gpu.read_alloc(p).is_some());
    owner.release(&gpu).unwrap();
    assert!(gpu.read_alloc(p).is_none());
}
#[test]
fn actual_tp2_geometry_accounts_all_eleven_buffers() {
    let g = MoeScratchGeometry::new(
        &ModelConfig {
            hidden_size: 4096,
            moe_intermediate_size: 2048,
            tp_world_size: 2,
            num_experts: 288,
            num_experts_per_tok: 8,
            tp_rank: 0,
            ..ModelConfig::qwen3_next_80b_nvfp4()
        },
        129,
        77,
    )
    .unwrap();
    assert_eq!(g.total_bytes().unwrap(), 60_468_096);
    assert_eq!(g.sizes().unwrap()[9], 1152);
}

#[path = "scratch_test_gpu.rs"]
mod backend;
#[test]
fn unknown_stream_completion_invalidates_borrowers_without_freeing_operands() {
    use std::sync::atomic::Ordering;
    let gpu = backend::FaultGpu::new();
    let mut owner = MoeScratch::allocate(&gpu, geometry()).unwrap();
    let b = owner.borrow();
    let pointers = b.pointers().unwrap();
    gpu.fail_sync.store(true, Ordering::Relaxed);
    assert!(owner.release(&gpu).is_err());
    for p in pointers {
        assert!(gpu.read_alloc(p).is_some());
    }
    assert!(b.pointers().is_err());
    gpu.fail_sync.store(false, Ordering::Relaxed);
    owner.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}
#[test]
fn partial_allocation_failure_rolls_back_existing_scratch() {
    let gpu = backend::FaultGpu::new();
    gpu.fail_alloc
        .store(4, std::sync::atomic::Ordering::Relaxed);
    assert!(MoeScratch::allocate(&gpu, geometry()).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn scratch_abi_overflow_rejects_before_allocator() {
    let mut cfg = ModelConfig {
        hidden_size: 128,
        moe_intermediate_size: 256,
        num_experts: 2,
        num_experts_per_tok: 1,
        tp_rank: 0,
        tp_world_size: 2,
        ..ModelConfig::qwen3_next_80b_nvfp4()
    };
    cfg.hidden_size = u32::MAX as usize + 1;
    assert!(MoeScratchGeometry::new(&cfg, 3, 77).is_err());
    cfg.hidden_size = 128;
    cfg.moe_intermediate_size = u32::MAX as usize + 1;
    assert!(MoeScratchGeometry::new(&cfg, 3, 77).is_err());
    cfg.moe_intermediate_size = 256;
    assert!(MoeScratchGeometry::new(&cfg, usize::MAX, 77).is_err());
}
