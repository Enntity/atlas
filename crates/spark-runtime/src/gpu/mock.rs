// SPDX-License-Identifier: AGPL-3.0-only
//! Mock GPU backend for unit tests (no GPU required).

use super::*;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

mod counters;

#[derive(Debug)]
pub struct MockAlloc {
    pub bytes: usize,
    pub data: Vec<u8>,
}

/// Records kernel launches and memory operations for test assertions.
pub struct MockGpuBackend {
    op_cache: crate::op_cache::OpCache,
    allocs: Mutex<HashMap<u64, MockAlloc>>,
    next_ptr: Mutex<u64>,
    launches: Mutex<Vec<MockLaunch>>,
    bf16_gemms: Mutex<Vec<Bf16Gemm>>,
    pending_gemm_failures: AtomicUsize,
    kernel_requests: Mutex<Vec<(String, String)>>,
    pending_kernel_failures: AtomicUsize,
    /// Copy/sync shape counters. These exist so tests can assert the SHAPE of a
    /// bulk transfer, not just its bytes: the SSM snapshot spill regressed to
    /// 60 blocking `copy_d2h` calls (one full stream drain each, ~400 ms for
    /// 66 MB) and nothing caught it, because the bytes were correct.
    syncs: AtomicUsize,
    d2h_blocking: AtomicUsize,
    d2h_async: AtomicUsize,
    /// `copy_d2d`/`copy_d2d_async` calls — one eager launch each on the real
    /// backend. The SSM verify rollback issued 2 per SSM layer per sequence
    /// (96 on the 27B), so this counter is what proves a batched form
    /// actually batched rather than merely looking different.
    d2d: AtomicUsize,
    /// `copy_d2d_2d_async` calls — ONE `cudaMemcpy2DAsync` each on the real
    /// backend regardless of `height`. Counted apart from `d2d` so a test can
    /// assert the SHAPE of the transfer, not just the bytes.
    d2d_2d: AtomicUsize,
    /// Pitched host↔device copies — ONE enqueue each on the real backend,
    /// whatever the row count.
    host_pitched: AtomicUsize,
    /// `memset_async` calls — one eager launch each on the real backend.
    memsets: AtomicUsize,
    host_pinned_allocs: AtomicUsize,
    /// Pending `free` failure injections (see [`Self::fail_next_free`]).
    pending_free_failures: AtomicUsize,
    /// `free` calls on a pointer that is not the base of a live allocation
    /// (see [`Self::invalid_free_count`]).
    invalid_frees: AtomicUsize,
    /// `destroy_graph` call counter (see [`Self::destroy_graph_count`]).
    destroyed_graphs: AtomicUsize,
    /// Streams and events created and not yet destroyed (see
    /// [`Self::live_streams_and_events`]).
    live_streams: AtomicUsize,
    live_events: AtomicUsize,
}

#[derive(Debug, Clone)]
pub struct MockLaunch {
    pub func: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Kernel parameter count (`kernelParams` entries).
    pub args: usize,
}

impl Default for MockGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockGpuBackend {
    pub fn new() -> Self {
        Self {
            op_cache: crate::op_cache::OpCache::new(),
            allocs: Mutex::new(HashMap::new()),
            next_ptr: Mutex::new(0x1000_0000),
            launches: Mutex::new(Vec::new()),
            bf16_gemms: Mutex::new(Vec::new()),
            pending_gemm_failures: AtomicUsize::new(0),
            kernel_requests: Mutex::new(Vec::new()),
            pending_kernel_failures: AtomicUsize::new(0),
            syncs: AtomicUsize::new(0),
            d2h_blocking: AtomicUsize::new(0),
            d2h_async: AtomicUsize::new(0),
            d2d: AtomicUsize::new(0),
            d2d_2d: AtomicUsize::new(0),
            host_pitched: AtomicUsize::new(0),
            memsets: AtomicUsize::new(0),
            host_pinned_allocs: AtomicUsize::new(0),
            pending_free_failures: AtomicUsize::new(0),
            invalid_frees: AtomicUsize::new(0),
            destroyed_graphs: AtomicUsize::new(0),
            live_streams: AtomicUsize::new(0),
            live_events: AtomicUsize::new(0),
        }
    }

    pub fn read_alloc(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
        self.allocs.lock().get(&ptr.0).map(|a| a.data.clone())
    }

    /// `bytes` from `src` to `dst` inside the simulated device memory.
    ///
    /// Real byte movement, not a no-op: a D2D that silently succeeds without
    /// moving anything lets a test "pass" while asserting the destination is
    /// still zero — the exact shape of a rollback bug this backend exists to
    /// catch. Source is staged through a temporary so `src` and `dst` may sit
    /// in the same allocation (the borrow checker would otherwise reject it,
    /// and the real `cudaMemcpyAsync` accepts it for non-overlapping ranges).
    fn blit(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut allocs = self.allocs.lock();
        let staged = {
            let (offset, alloc) = find_alloc(&allocs, src)
                .ok_or_else(|| anyhow::anyhow!("copy_d2d: src {src} not allocated"))?;
            if offset + bytes > alloc.bytes {
                anyhow::bail!("copy_d2d: src {src} + {bytes} overruns its allocation");
            }
            alloc.data[offset..offset + bytes].to_vec()
        };
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_d2d: dst {dst} not allocated"))?;
        if offset + bytes > alloc.bytes {
            anyhow::bail!("copy_d2d: dst {dst} + {bytes} overruns its allocation");
        }
        alloc.data[offset..offset + bytes].copy_from_slice(&staged);
        Ok(())
    }
}

/// Find the allocation containing `ptr` (supports offset pointers).
fn find_alloc(allocs: &HashMap<u64, MockAlloc>, ptr: DevicePtr) -> Option<(usize, &MockAlloc)> {
    for (&base, alloc) in allocs.iter() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

/// Mutable version of find_alloc.
fn find_alloc_mut(
    allocs: &mut HashMap<u64, MockAlloc>,
    ptr: DevicePtr,
) -> Option<(usize, &mut MockAlloc)> {
    for (&base, alloc) in allocs.iter_mut() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

impl GpuBackend for MockGpuBackend {
    fn require_bf16_gemm(&self) -> Result<()> {
        Ok(())
    }
    fn require_bf16_gemm_numerics(&self, numerics: super::Bf16GemmNumerics) -> Result<()> {
        match numerics {
            super::Bf16GemmNumerics::CublasLt
            | super::Bf16GemmNumerics::Glm53Classic
            | super::Bf16GemmNumerics::Glm53AuxiliaryClassic => Ok(()),
        }
    }
    fn bf16_gemm_act_weight_t(&self, request: Bf16Gemm) -> Result<()> {
        request.validate()?;
        self.bf16_gemms.lock().push(request);
        if self
            .pending_gemm_failures
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            anyhow::bail!("mock BF16 GEMM failed (injected)");
        }
        Ok(())
    }

    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let mut next = self.next_ptr.lock();
        let ptr = *next;
        *next += bytes as u64;
        // Align to 256 bytes
        *next = (*next + 255) & !255;
        self.allocs.lock().insert(
            ptr,
            MockAlloc {
                bytes,
                data: vec![0u8; bytes],
            },
        );
        Ok(DevicePtr(ptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes) // Mock: same as regular alloc
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if self
            .pending_free_failures
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            anyhow::bail!("mock free failed (injected): ptr {ptr}");
        }
        if self.allocs.lock().remove(&ptr.0).is_none() && !ptr.is_null() {
            self.invalid_frees.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let mut allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing dst
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_h2d: ptr {dst} not allocated"))?;
        alloc.data[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.d2h_blocking.fetch_add(1, Ordering::Relaxed);
        let allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing src
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], _stream: u64) -> Result<()> {
        // Counted separately from `copy_d2h` and NOT delegating to it, so a
        // test can distinguish the batched shape from the blocking one (the
        // trait's default impl forwards, which would make them indistinguishable).
        self.d2h_async.fetch_add(1, Ordering::Relaxed);
        let allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h_async: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.d2d.fetch_add(1, Ordering::Relaxed);
        self.blit(src, dst, bytes)
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        _stream: u64,
    ) -> Result<()> {
        // NOT delegating to `copy_d2d`: the trait default forwards, which
        // would make the two indistinguishable to `d2d_count` consumers only
        // by accident. Counted here so both forms land in one counter on
        // purpose.
        self.d2d.fetch_add(1, Ordering::Relaxed);
        self.blit(src, dst, bytes)
    }

    fn copy_d2d_2d_async(
        &self,
        src: DevicePtr,
        src_pitch: usize,
        dst: DevicePtr,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
        _stream: u64,
    ) -> Result<()> {
        // ONE launch on the real backend (`cudaMemcpy2DAsync`), so ONE tick —
        // the row loop below is emulation, not dispatch, and must not inflate
        // `d2d_count` (which exists to prove a batched form batched).
        self.d2d_2d.fetch_add(1, Ordering::Relaxed);
        if width_bytes > src_pitch || width_bytes > dst_pitch {
            anyhow::bail!(
                "copy_d2d_2d_async: width {width_bytes} exceeds pitch \
                 (src {src_pitch}, dst {dst_pitch}) — rows would overlap"
            );
        }
        for r in 0..height {
            self.blit(
                src.offset(r * src_pitch),
                dst.offset(r * dst_pitch),
                width_bytes,
            )?;
        }
        Ok(())
    }

    fn host_pitched(&self) -> Option<&dyn HostPitched> {
        Some(self)
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.launches.lock().push(MockLaunch {
            func: func.0,
            grid,
            block,
            args: params.len(),
        });
        Ok(())
    }

    fn synchronize(&self, _stream: u64) -> Result<()> {
        self.syncs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        self.kernel_requests
            .lock()
            .push((module.into(), func_name.into()));
        if self
            .pending_kernel_failures
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            anyhow::bail!("mock required kernel unavailable: {module}::{func_name}");
        }
        Ok(KernelHandle(0xDEAD))
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let mut allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc_mut(&mut allocs, ptr)
            .ok_or_else(|| anyhow::anyhow!("memset: ptr {ptr} not allocated"))?;
        alloc.data[offset..offset + bytes].fill(value);
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        self.memsets.fetch_add(1, Ordering::Relaxed);
        self.memset(ptr, value, bytes)
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // Same heap allocation as the trait default (the mock cannot page-lock);
        // overridden solely to COUNT, so a test can prove a staging buffer is
        // allocated once and reused rather than per event.
        self.host_pinned_allocs.fetch_add(1, Ordering::Relaxed);
        let layout = std::alloc::Layout::from_size_align(bytes, HOST_PINNED_ALIGN)
            .map_err(|e| anyhow::anyhow!("invalid layout: {e}"))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            anyhow::bail!("host alloc failed: {bytes} bytes");
        }
        Ok(ptr)
    }

    fn total_memory(&self) -> Result<usize> {
        Ok(128 * 1024 * 1024 * 1024) // 128 GB
    }

    fn sm_count(&self) -> Result<u32> {
        // Rationale (PCND): tests that exercise occupancy-gated dispatch need
        // SOME machine width; 48 is the GB10 value the model targets, chosen
        // so mock runs take the same branch production does.
        Ok(48)
    }

    fn free_memory(&self) -> Result<usize> {
        Ok(120 * 1024 * 1024 * 1024) // 120 GB
    }

    fn destroy_graph(&self, _graph: GraphHandle) -> Result<()> {
        self.destroyed_graphs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn create_stream(&self) -> Result<u64> {
        self.live_streams.fetch_add(1, Ordering::Relaxed);
        Ok(0)
    }

    fn destroy_stream(&self, _stream: u64) -> Result<()> {
        self.live_streams.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        self.live_events.fetch_add(1, Ordering::Relaxed);
        Ok(0)
    }

    fn destroy_event(&self, _event: u64) -> Result<()> {
        self.live_events.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }
}

impl HostPitched for MockGpuBackend {
    fn h2d_retained(&self, src: &[u8], dst: DevicePtr, shape: Pitched, _stream: u64) -> Result<()> {
        // One tick per call, as `copy_d2d_2d_async`: the rows are emulation.
        self.host_pitched.fetch_add(1, Ordering::Relaxed);
        for r in 0..shape.height {
            let row = &src[r * shape.host_pitch..][..shape.width];
            self.copy_h2d(row, dst.offset(r * shape.dev_pitch))?;
        }
        Ok(())
    }

    fn d2h(&self, src: DevicePtr, dst: &mut [u8], shape: Pitched, _stream: u64) -> Result<()> {
        self.host_pitched.fetch_add(1, Ordering::Relaxed);
        let allocs = self.allocs.lock();
        for r in 0..shape.height {
            let from = src.offset(r * shape.dev_pitch);
            let (offset, alloc) = find_alloc(&allocs, from)
                .ok_or_else(|| anyhow::anyhow!("pitched D2H: ptr {from} not allocated"))?;
            dst[r * shape.host_pitch..][..shape.width]
                .copy_from_slice(&alloc.data[offset..offset + shape.width]);
        }
        Ok(())
    }
}
