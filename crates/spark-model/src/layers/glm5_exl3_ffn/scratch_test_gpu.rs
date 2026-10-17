// SPDX-License-Identifier: AGPL-3.0-only

//! Fault injection retains the real mock allocator and copies.
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, mock::MockGpuBackend};
use std::{collections::HashMap, sync::Mutex};

pub(super) struct FaultGpu {
    pub fail_sync: std::sync::atomic::AtomicBool,
    pub fail_alloc: std::sync::atomic::AtomicUsize,
    memory: MockGpuBackend,
    kernels: Mutex<HashMap<(String, String), u64>>,
}
impl FaultGpu {
    pub fn new() -> Self {
        Self {
            fail_sync: std::sync::atomic::AtomicBool::new(false),
            fail_alloc: std::sync::atomic::AtomicUsize::new(usize::MAX),
            memory: MockGpuBackend::new(),
            kernels: Mutex::new(HashMap::new()),
        }
    }
}
impl std::ops::Deref for FaultGpu {
    type Target = MockGpuBackend;
    fn deref(&self) -> &Self::Target {
        &self.memory
    }
}
impl GpuBackend for FaultGpu {
    fn require_bf16_gemm(&self) -> Result<()> {
        self.memory.require_bf16_gemm()
    }
    fn require_bf16_gemm_numerics(
        &self,
        numerics: spark_runtime::gpu::Bf16GemmNumerics,
    ) -> Result<()> {
        self.memory.require_bf16_gemm_numerics(numerics)
    }
    fn bf16_gemm_act_weight_t(&self, request: spark_runtime::gpu::Bf16Gemm) -> Result<()> {
        self.memory.bf16_gemm_act_weight_t(request)
    }

    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        let remaining = self.fail_alloc.load(std::sync::atomic::Ordering::Relaxed);
        if remaining != usize::MAX {
            anyhow::ensure!(remaining > 0, "injected allocation failure");
            self.fail_alloc
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.memory.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.memory.alloc_managed(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.memory.free(p)
    }
    fn copy_h2d(&self, s: &[u8], p: DevicePtr) -> Result<()> {
        self.memory.copy_h2d(s, p)
    }
    fn copy_d2h(&self, p: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.memory.copy_d2h(p, d)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.memory.copy_d2d(s, d, n)
    }
    fn launch(
        &self,
        f: KernelHandle,
        g: [u32; 3],
        b: [u32; 3],
        m: u32,
        s: u64,
        p: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.memory.launch(f, g, b, m, s, p)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        anyhow::ensure!(
            !self.fail_sync.load(std::sync::atomic::Ordering::Relaxed),
            "unknown stream completion"
        );
        self.memory.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        self.memory.default_stream()
    }
    fn kernel(&self, module: &str, name: &str) -> Result<KernelHandle> {
        let mut kernels = self.kernels.lock().unwrap();
        let next = kernels.len() as u64 + 1;
        Ok(KernelHandle(
            *kernels
                .entry((module.to_owned(), name.to_owned()))
                .or_insert(next),
        ))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.memory.op_cache()
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.memory.memset(p, v, n)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, s: u64) -> Result<()> {
        self.memory.memset_async(p, v, n, s)
    }
    fn total_memory(&self) -> Result<usize> {
        self.memory.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.memory.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.memory.sm_count()
    }
}
