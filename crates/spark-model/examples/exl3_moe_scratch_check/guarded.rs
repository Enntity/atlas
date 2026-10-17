// SPDX-License-Identifier: AGPL-3.0-only
//! Example-only aligned sentinels; every compute/copy operation delegates unchanged.
use anyhow::{Context, Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{
    Bf16Gemm, Bf16GemmNumerics, DevicePtr, GpuBackend, KernelArg, KernelHandle,
};
use std::collections::HashMap;
const GUARD: usize = 256;
pub struct GuardedGpu<'a> {
    inner: &'a dyn GpuBackend,
    allocations: Mutex<HashMap<u64, (DevicePtr, usize)>>,
}
impl<'a> GuardedGpu<'a> {
    pub fn new(inner: &'a dyn GpuBackend) -> Self {
        Self {
            inner,
            allocations: Mutex::new(HashMap::new()),
        }
    }
    fn allocate(&self, bytes: usize, managed: bool) -> Result<DevicePtr> {
        let n = bytes
            .checked_add(2 * GUARD)
            .context("guarded allocation size overflow")?;
        let base = if managed {
            self.inner.alloc_managed(n)?
        } else {
            self.inner.alloc(n)?
        };
        let result = (|| {
            self.inner.copy_h2d(&[0xa5; GUARD], base)?;
            self.inner
                .copy_h2d(&[0xa5; GUARD], base.offset(GUARD + bytes))?;
            Ok(base.offset(GUARD))
        })();
        match result {
            Ok(p) => {
                self.allocations.lock().insert(p.0, (base, bytes));
                Ok(p)
            }
            Err(error) => match self.inner.free(base) {
                Ok(()) => Err(error),
                Err(cleanup) => {
                    Err(error.context(format!("guard initialization rollback failed: {cleanup:#}")))
                }
            },
        }
    }
    fn check(&self, base: DevicePtr, bytes: usize) -> Result<()> {
        for ptr in [base, base.offset(GUARD + bytes)] {
            let mut guard = [0; GUARD];
            self.inner.copy_d2h(ptr, &mut guard)?;
            ensure!(
                guard == [0xa5; GUARD],
                "GPU allocation sentinel overwritten"
            );
        }
        Ok(())
    }
    pub fn check_all(&self) -> Result<()> {
        for &(base, bytes) in self.allocations.lock().values() {
            self.check(base, bytes)?;
        }
        Ok(())
    }
    pub fn outstanding(&self) -> usize {
        self.allocations.lock().len()
    }
}
impl GpuBackend for GuardedGpu<'_> {
    fn require_bf16_gemm(&self) -> Result<()> {
        self.inner.require_bf16_gemm()
    }
    fn require_bf16_gemm_numerics(&self, n: Bf16GemmNumerics) -> Result<()> {
        self.inner.require_bf16_gemm_numerics(n)
    }
    fn quiesce_bf16_resources(&self) -> Result<()> {
        self.inner.quiesce_bf16_resources()
    }
    fn bf16_gemm_act_weight_t(&self, r: Bf16Gemm) -> Result<()> {
        self.inner.bf16_gemm_act_weight_t(r)
    }
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.allocate(n, false)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.allocate(n, true)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        let (base, n) = *self
            .allocations
            .lock()
            .get(&p.0)
            .context("guarded allocator rejects interior/unowned free")?;
        self.check(base, n)?;
        self.inner.free(base)?;
        self.allocations.lock().remove(&p.0);
        Ok(())
    }
    fn copy_h2d(&self, s: &[u8], d: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(s, d)
    }
    fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(s, d)
    }
    fn copy_d2h_on_stream(&self, s: DevicePtr, d: &mut [u8], stream: u64) -> Result<()> {
        self.inner.copy_d2h_on_stream(s, d, stream)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(s, d, n)
    }
    fn copy_h2d_async(&self, s: &[u8], d: DevicePtr, t: u64) -> Result<()> {
        self.inner.copy_h2d_async(s, d, t)
    }
    fn copy_h2d_async_retained(&self, s: &[u8], d: DevicePtr, t: u64) -> Result<()> {
        self.inner.copy_h2d_async_retained(s, d, t)
    }
    fn copy_d2h_async(&self, s: DevicePtr, d: &mut [u8], t: u64) -> Result<()> {
        self.inner.copy_d2h_async(s, d, t)
    }
    fn copy_d2d_async(&self, s: DevicePtr, d: DevicePtr, n: usize, t: u64) -> Result<()> {
        self.inner.copy_d2d_async(s, d, n, t)
    }
    fn copy_d2d_2d_async(
        &self,
        s: DevicePtr,
        sp: usize,
        d: DevicePtr,
        dp: usize,
        w: usize,
        h: usize,
        t: u64,
    ) -> Result<()> {
        self.inner.copy_d2d_2d_async(s, sp, d, dp, w, h, t)
    }
    fn launch(
        &self,
        f: KernelHandle,
        g: [u32; 3],
        b: [u32; 3],
        m: u32,
        t: u64,
        p: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.inner.launch(f, g, b, m, t, p)
    }
    fn launch_typed(
        &self,
        f: KernelHandle,
        g: [u32; 3],
        b: [u32; 3],
        m: u32,
        t: u64,
        p: &[KernelArg<'_>],
    ) -> Result<()> {
        self.inner.launch_typed(f, g, b, m, t, p)
    }
    fn synchronize(&self, t: u64) -> Result<()> {
        self.inner.synchronize(t)
    }
    fn default_stream(&self) -> u64 {
        self.inner.default_stream()
    }
    fn kernel(&self, m: &str, n: &str) -> Result<KernelHandle> {
        self.inner.kernel(m, n)
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn debug_sync_kernels(&self) -> bool {
        self.inner.debug_sync_kernels()
    }
    fn stream_is_capturing(&self, t: u64) -> bool {
        self.inner.stream_is_capturing(t)
    }
    #[cfg(feature = "cuda")]
    fn kernel_registry(&self) -> Option<std::sync::Arc<atlas_core::registry::AtlasRegistry>> {
        self.inner.kernel_registry()
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.inner.memset(p, v, n)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, t: u64) -> Result<()> {
        self.inner.memset_async(p, v, n, t)
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
#[path = "guarded_tests.rs"]
mod tests;
