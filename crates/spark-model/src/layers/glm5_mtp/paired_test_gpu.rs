// SPDX-License-Identifier: AGPL-3.0-only
//! Constructor/lifecycle fault backend. Real host bytes, no GPU numerical claim.
//! Recycled backing belongs to MockGpuBackend until drop, not to the live ledger.

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, mock::MockGpuBackend};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Alloc(DevicePtr, usize),
    Alias(DevicePtr, usize),
    Reuse(DevicePtr, usize),
    Free(DevicePtr),
    Sync(u64),
}

#[derive(Default)]
pub(super) struct TestGpu {
    pub inner: MockGpuBackend,
    /// One-shot allocator lie; 0 disables. Does not create or change an owner.
    pub next_alloc_alias: AtomicU64,
    /// One-based ordinal since clear(); 0 disables. Failed operations count.
    pub fail_sync_at: AtomicUsize,
    pub fail_free_at: AtomicUsize,
    pub fail_alloc_at: AtomicUsize,
    pub fail_required_kernel: AtomicBool,
    alloc_attempts: AtomicUsize,
    pub reuse_freed: AtomicBool,
    syncs: AtomicUsize,
    frees: AtomicUsize,
    events: Mutex<Vec<Event>>,
    live: Mutex<HashMap<u64, usize>>,
    recycled: Mutex<Vec<(DevicePtr, usize)>>,
}

impl TestGpu {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset fault ordinals and trace, never ownership or recycled backing.
    pub fn clear(&self) {
        self.events.lock().clear();
        self.syncs.store(0, Ordering::Relaxed);
        self.frees.store(0, Ordering::Relaxed);
        self.next_alloc_alias.store(0, Ordering::Relaxed);
        self.fail_sync_at.store(0, Ordering::Relaxed);
        self.fail_free_at.store(0, Ordering::Relaxed);
    }

    pub fn trace(&self) -> Vec<Event> {
        self.events.lock().clone()
    }

    pub fn live_allocations(&self) -> HashMap<u64, usize> {
        self.live.lock().clone()
    }

    pub fn alloc_count(&self) -> usize {
        self.live.lock().len()
    }

    pub fn sync_count(&self) -> usize {
        self.syncs.load(Ordering::Relaxed)
    }

    pub fn free_count(&self) -> usize {
        self.frees.load(Ordering::Relaxed)
    }

    /// Inspection can see retired backing; presence does NOT prove live ownership.
    pub fn read_alloc(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
        self.inner.read_alloc(ptr)
    }

    pub fn read_span(&self, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        let mut result = vec![0; bytes];
        self.inner.copy_d2h(ptr, &mut result)?;
        Ok(result)
    }
}

impl GpuBackend for TestGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let attempt = self.alloc_attempts.fetch_add(1, Ordering::Relaxed) + 1;
        ensure!(
            attempt != self.fail_alloc_at.load(Ordering::Relaxed),
            "injected constructor allocation failure {attempt}"
        );
        let alias = self.next_alloc_alias.swap(0, Ordering::Relaxed);
        if alias != 0 {
            let ptr = DevicePtr(alias);
            self.events.lock().push(Event::Alias(ptr, bytes));
            return Ok(ptr);
        }
        let reused = if self.reuse_freed.load(Ordering::Relaxed) {
            let mut recycled = self.recycled.lock();
            recycled
                .iter()
                .position(|(_, size)| *size == bytes)
                .map(|i| recycled.swap_remove(i).0)
        } else {
            None
        };
        let ptr = match reused {
            Some(ptr) => ptr,
            None => self.inner.alloc(bytes)?,
        };
        ensure!(
            self.live.lock().insert(ptr.0, bytes).is_none(),
            "test allocator attempted to replace live ownership"
        );
        self.events.lock().push(match reused {
            Some(_) => Event::Reuse(ptr, bytes),
            None => Event::Alloc(ptr, bytes),
        });
        Ok(ptr)
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        let ordinal = self.frees.fetch_add(1, Ordering::Relaxed) + 1;
        self.events.lock().push(Event::Free(ptr));
        // Native CudaBackend removes its allocation record before cuMemFree.
        // On injected failure, leave backing inaccessible to allocation reuse.
        let bytes = self
            .live
            .lock()
            .remove(&ptr.0)
            .ok_or_else(|| anyhow::anyhow!("test backend duplicate/foreign free: {ptr:?}"))?;
        ensure!(
            ordinal != self.fail_free_at.load(Ordering::Relaxed),
            "injected paired free failure at ordinal {ordinal} for {ptr:?}"
        );
        if self.reuse_freed.load(Ordering::Relaxed) {
            self.recycled.lock().push((ptr, bytes));
            Ok(())
        } else {
            self.inner.free(ptr)
        }
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        let ordinal = self.syncs.fetch_add(1, Ordering::Relaxed) + 1;
        self.events.lock().push(Event::Sync(stream));
        ensure!(
            ordinal != self.fail_sync_at.load(Ordering::Relaxed),
            "injected paired synchronization failure at ordinal {ordinal}"
        );
        self.inner.synchronize(stream)
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
    fn default_stream(&self) -> u64 {
        7
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn kernel(&self, module: &str, name: &str) -> Result<KernelHandle> {
        ensure!(
            !(name == "rms_norm_vanilla" && self.fail_required_kernel.load(Ordering::Relaxed)),
            "injected required kernel resolution failure"
        );
        self.inner.kernel(module, name)
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
    }
    fn launch(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.inner.launch(kernel, grid, block, shared, stream, args)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_is_one_shot_and_does_not_change_original_owner_or_bytes() {
        let gpu = TestGpu::new();
        let original = gpu.alloc(64).unwrap();
        gpu.copy_h2d(&[0x6d; 64], original).unwrap();
        let before = gpu.live_allocations();
        gpu.clear();
        gpu.next_alloc_alias.store(original.0, Ordering::Relaxed);
        assert_eq!(gpu.alloc(4096).unwrap(), original);
        assert_eq!(gpu.live_allocations(), before);
        assert_eq!(gpu.read_alloc(original).unwrap(), [0x6d; 64]);
        assert_eq!(gpu.trace(), [Event::Alias(original, 4096)]);
        assert_ne!(gpu.alloc(64).unwrap(), original);
    }

    #[test]
    fn only_successfully_freed_exact_size_backing_is_reused() {
        let gpu = TestGpu::new();
        gpu.reuse_freed.store(true, Ordering::Relaxed);
        let a = gpu.alloc(64).unwrap();
        let b = gpu.alloc(64).unwrap();
        gpu.clear();
        gpu.fail_free_at.store(1, Ordering::Relaxed);
        assert!(gpu.free(a).is_err());
        assert!(!gpu.live_allocations().contains_key(&a.0));
        assert!(
            gpu.read_alloc(a).is_some(),
            "failed free is not backend-ledger ownership"
        );
        gpu.free(b).unwrap();
        assert_ne!(gpu.alloc(32).unwrap(), b);
        assert_eq!(gpu.alloc(64).unwrap(), b);
        assert_ne!(gpu.alloc(64).unwrap(), a);
        assert!(gpu.free(a).is_err());
    }

    #[test]
    fn synchronization_fault_counts_and_clear_does_not_reset_ownership() {
        let gpu = TestGpu::new();
        let ptr = gpu.alloc(16).unwrap();
        gpu.clear();
        gpu.fail_sync_at.store(2, Ordering::Relaxed);
        gpu.synchronize(7).unwrap();
        assert!(gpu.synchronize(37).is_err());
        assert_eq!(gpu.sync_count(), 2);
        assert_eq!(gpu.inner.sync_count(), 1);
        assert_eq!(gpu.trace(), [Event::Sync(7), Event::Sync(37)]);
        gpu.clear();
        assert_eq!(gpu.live_allocations().get(&ptr.0), Some(&16));
        gpu.synchronize(7).unwrap();
        assert_eq!(gpu.sync_count(), 1);
    }
}
