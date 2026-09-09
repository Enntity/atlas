// SPDX-License-Identifier: AGPL-3.0-only
//! Native-owner Drop/sweep spy around actual owned CPU backend allocations.
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, mock::MockGpuBackend};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Default)]
pub(super) struct Probe {
    pub drops: AtomicUsize,
    pub sweeps: AtomicUsize,
    pub kernels: AtomicUsize,
    pub fail_kernel: AtomicBool,
    pub pinned_frees: AtomicUsize,
}
pub(super) struct Gpu {
    inner: MockGpuBackend,
    pub probe: Arc<Probe>,
}
impl Gpu {
    pub fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            probe: Arc::new(Probe::default()),
        }
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        self.probe.drops.fetch_add(1, Ordering::Relaxed);
        self.sweep_unreleased();
    }
}
impl GpuBackend for Gpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn alloc_host_pinned(&self, n: usize) -> Result<*mut u8> {
        self.inner.alloc_host_pinned(n)
    }
    fn free_host_pinned(&self, p: *mut u8, n: usize) -> Result<()> {
        self.probe.pinned_frees.fetch_add(1, Ordering::Relaxed);
        self.inner.free_host_pinned(p, n)
    }
    fn sweep_unreleased(&self) -> usize {
        self.probe.sweeps.fetch_add(1, Ordering::Relaxed);
        0
    }
    fn copy_h2d(&self, b: &[u8], p: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(b, p)
    }
    fn copy_d2h(&self, p: DevicePtr, b: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(p, b)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(s, d, n)
    }
    fn launch(
        &self,
        k: KernelHandle,
        g: [u32; 3],
        b: [u32; 3],
        m: u32,
        s: u64,
        p: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.inner.launch(k, g, b, m, s, p)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.inner.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        self.inner.default_stream()
    }
    fn kernel(&self, module: &str, name: &str) -> Result<KernelHandle> {
        self.probe.kernels.fetch_add(1, Ordering::Relaxed);
        ensure!(
            !self.probe.fail_kernel.load(Ordering::Relaxed),
            "injected actual constructor kernel failure"
        );
        self.inner.kernel(module, name)
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.inner.memset(p, v, n)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, s: u64) -> Result<()> {
        self.inner.memset_async(p, v, n, s)
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
