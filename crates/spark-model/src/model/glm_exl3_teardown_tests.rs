// SPDX-License-Identifier: AGPL-3.0-only
//! Exercise real model teardown with a failure at the GPU completion boundary.
#[path = "trait_impl/prefill_stream_test_fixture.rs"]
mod fixture;
use anyhow::{Result, ensure};
use fixture::Fixture;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

struct CompletionGpu {
    inner: Box<dyn GpuBackend>,
    state: Arc<CompletionState>,
}
struct CompletionState {
    fail: AtomicBool,
    fail_sync: AtomicBool,
    fail_pointer: AtomicU64,
    freed: Mutex<Vec<u64>>,
    events: Mutex<Vec<&'static str>>,
}
impl GpuBackend for CompletionGpu {
    fn quiesce_bf16_resources(&self) -> Result<()> {
        self.state.events.lock().unwrap().push("quiesce");
        ensure!(
            !self.state.fail.load(Ordering::Relaxed),
            "unknown BLAS completion"
        );
        self.inner.quiesce_bf16_resources()
    }
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.state.events.lock().unwrap().push("free");
        ensure!(
            ptr.is_null()
                || self
                    .state
                    .fail_pointer
                    .compare_exchange(ptr.0, 0, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err(),
            "injected owned pointer free failure"
        );
        self.inner.free(ptr)?;
        if !ptr.is_null() {
            self.state.freed.lock().unwrap().push(ptr.0);
        }
        Ok(())
    }
    fn sweep_unreleased(&self) -> usize {
        self.state.events.lock().unwrap().push("sweep");
        self.inner.sweep_unreleased()
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(src, dst)
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(src, dst)
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }
    fn launch(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.inner
            .launch(kernel, grid, block, shared, stream, params)
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.state.events.lock().unwrap().push("sync");
        ensure!(
            !self.state.fail_sync.load(Ordering::Relaxed),
            "unknown routed completion"
        );
        self.inner.synchronize(stream)
    }
    fn default_stream(&self) -> u64 {
        self.inner.default_stream()
    }
    fn kernel(&self, module: &str, name: &str) -> Result<KernelHandle> {
        self.inner.kernel(module, name)
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.inner.sm_count()
    }
}

/// Completed `sweep_unreleased` calls seen by the wrapper.
fn sweeps(state: &CompletionState) -> usize {
    let events = state.events.lock().unwrap();
    events.iter().filter(|&&e| e == "sweep").count()
}

#[test]
fn unknown_blas_completion_preserves_model_owners_until_successful_retry() {
    let mut fixture = Fixture::new(1, 1, 0);
    let gpu = Arc::new(CompletionState {
        fail: AtomicBool::new(true),
        fail_sync: AtomicBool::new(false),
        fail_pointer: AtomicU64::new(0),
        freed: Mutex::new(vec![]),
        events: Mutex::new(vec![]),
    });
    let inner = std::mem::replace(
        &mut fixture.model.gpu,
        Box::new(spark_runtime::gpu::mock::MockGpuBackend::new()),
    );
    fixture.model.gpu = Box::new(CompletionGpu {
        inner,
        state: gpu.clone(),
    });
    for _ in 0..2 {
        assert!(fixture.model.release_pools().is_err());
    }
    assert_eq!(*gpu.events.lock().unwrap(), ["quiesce", "quiesce"]);
    assert_eq!(sweeps(&gpu), 0);
    gpu.fail.store(false, Ordering::Relaxed);
    fixture.model.release_pools().unwrap();
    let events = gpu.events.lock().unwrap();
    assert_eq!(&events[..3], ["quiesce", "quiesce", "quiesce"]);
    assert!(
        events[3..].contains(&"free"),
        "real model owners must be released after completion"
    );
    assert_eq!(events.last(), Some(&"sweep"));
    assert_eq!(events.iter().filter(|&&e| e == "sweep").count(), 1);
}

#[test]
fn unknown_routed_completion_preserves_all_model_operands_before_release() {
    use crate::layers::glm5_exl3_ffn::{MoeScratch, MoeScratchGeometry};
    let mut fixture = Fixture::new(1, 1, 0);
    let state = Arc::new(CompletionState {
        fail: AtomicBool::new(false),
        fail_sync: AtomicBool::new(false),
        fail_pointer: AtomicU64::new(0),
        freed: Mutex::new(vec![]),
        events: Mutex::new(vec![]),
    });
    let inner = std::mem::replace(
        &mut fixture.model.gpu,
        Box::new(spark_runtime::gpu::mock::MockGpuBackend::new()),
    );
    fixture.model.gpu = Box::new(CompletionGpu {
        inner,
        state: state.clone(),
    });
    let cfg = atlas_core::config::ModelConfig {
        hidden_size: 128,
        moe_intermediate_size: 256,
        num_experts: 2,
        num_experts_per_tok: 1,
        tp_rank: 0,
        tp_world_size: 2,
        ..atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4()
    };
    let scratch = MoeScratch::allocate(
        fixture.model.gpu.as_ref(),
        MoeScratchGeometry::new(&cfg, 3, 77).unwrap(),
    )
    .unwrap();
    let borrowed = scratch.borrow();
    fixture.model.attach_exl3_moe_scratch(scratch).unwrap();
    state.fail_sync.store(true, Ordering::Relaxed);
    assert!(fixture.model.release_pools().is_err());
    let events = state.events.lock().unwrap().clone();
    assert_eq!(events, ["quiesce", "sync"]);
    assert_eq!(sweeps(&state), 0);
    assert!(borrowed.pointers().is_err());
    state.fail_sync.store(false, Ordering::Relaxed);
    fixture.model.release_pools().unwrap();
    let events = state.events.lock().unwrap();
    assert_eq!(&events[..4], ["quiesce", "sync", "quiesce", "sync"]);
    assert!(events[4..].contains(&"free"));
    assert_eq!(events.last(), Some(&"sweep"));
}

#[test]
fn failed_shared_owner_free_defers_sweep_and_retries_without_duplicate_frees() {
    use crate::layers::glm5_exl3_ffn::{MoeScratch, MoeScratchGeometry};
    let mut fixture = Fixture::new(1, 1, 0);
    let state = Arc::new(CompletionState {
        fail: AtomicBool::new(false),
        fail_sync: AtomicBool::new(false),
        fail_pointer: AtomicU64::new(0),
        freed: Mutex::new(vec![]),
        events: Mutex::new(vec![]),
    });
    let inner = std::mem::replace(
        &mut fixture.model.gpu,
        Box::new(spark_runtime::gpu::mock::MockGpuBackend::new()),
    );
    fixture.model.gpu = Box::new(CompletionGpu {
        inner,
        state: state.clone(),
    });
    let cfg = atlas_core::config::ModelConfig {
        hidden_size: 128,
        moe_intermediate_size: 256,
        num_experts: 2,
        num_experts_per_tok: 1,
        tp_rank: 0,
        tp_world_size: 2,
        ..atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4()
    };
    let scratch = MoeScratch::allocate(
        fixture.model.gpu.as_ref(),
        MoeScratchGeometry::new(&cfg, 3, 77).unwrap(),
    )
    .unwrap();
    let pointer = scratch.borrow().pointers().unwrap()[0].0;
    fixture.model.attach_exl3_moe_scratch(scratch).unwrap();
    state.fail_pointer.store(pointer, Ordering::Relaxed);
    assert!(fixture.model.release_pools().is_err());
    assert!(!state.events.lock().unwrap().contains(&"sweep"));
    assert!(!state.freed.lock().unwrap().contains(&pointer));
    fixture.model.release_pools().unwrap();
    let freed = state.freed.lock().unwrap();
    assert_eq!(freed.iter().filter(|&&p| p == pointer).count(), 1);
    let unique: std::collections::HashSet<_> = freed.iter().collect();
    assert_eq!(
        unique.len(),
        freed.len(),
        "a successful free must never be retried"
    );
    assert_eq!(
        state
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|&&e| e == "sweep")
            .count(),
        1
    );
    assert_eq!(sweeps(&state), 1);
}
