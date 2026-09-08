// SPDX-License-Identifier: AGPL-3.0-only
//! Host sentinels, not CUDA numerics: identity norm, hidden-preserving EH; logits are not oracles.
use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use crate::layers::glm5_mtp::Glm5MtpHead;
use crate::model::types::TransformerModel;
use crate::speculative::DraftProposer;
use crate::traits::SequenceState;
use crate::weight_loader::glm5::Glm5MtpModule;
use crate::weight_map::DenseWeight;
use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle, mock::MockGpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[path = "glm_c2_verdict_test_numerics.rs"]
mod numerics;
use numerics::Body;
pub(super) const DEFAULT: u64 = 7;
pub(super) const CALLER: u64 = 37;
pub(super) const ROW_BYTES: usize = 8192;
pub(super) const SLAB_BYTES: usize = 98_304;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Alloc(DevicePtr, usize),
    Free(DevicePtr),
    Copy(DevicePtr, DevicePtr, usize, u64),
    Upload(DevicePtr, usize, u64),
    Read(DevicePtr, usize, u64),
    Sync(u64),
    RecordEvent(u64, u64),
    WaitEvent(u64, u64),
    BeginCapture(u64),
    EndCapture(u64),
    AbortCapture(u64),
    LaunchGraph(u64, u64),
    Memset(DevicePtr, usize, u64),
    Kernel(String, Vec<DevicePtr>, u64),
    Target(usize, usize, u64),
    Body(usize, u64),
    Kv(Vec<i64>, u64),
}
#[derive(Default)]
pub(super) struct Recorder {
    inner: MockGpuBackend,
    events: Mutex<Vec<Event>>,
    live: Mutex<HashMap<u64, usize>>,
    retired: Mutex<Vec<(DevicePtr, usize)>>,
    names: Mutex<Vec<String>>,
    eh_pairs: Mutex<Vec<(u8, u8)>>,
    next_handle: AtomicUsize,
    pub fail: AtomicUsize,
    pub reuse_freed: AtomicBool,
    pub capturing: AtomicBool,
    pub capture_handles: AtomicBool,
    pub sweeps: AtomicUsize,
    pub deterministic_logits: AtomicBool,
}
impl Recorder {
    fn event(&self, event: Event) -> Result<()> {
        let mut events = self.events.lock();
        events.push(event);
        let fail = self.fail.load(Ordering::Relaxed);
        ensure!(events.len() != fail, "injected fixture operation failure");
        Ok(())
    }
    pub fn clear(&self) {
        self.events.lock().clear();
        self.eh_pairs.lock().clear();
        self.fail.store(usize::MAX, Ordering::Relaxed);
    }
    pub fn trace(&self) -> Vec<Event> {
        self.events.lock().clone()
    }
    pub fn eh_pairs(&self) -> Vec<(u8, u8)> {
        self.eh_pairs.lock().clone()
    }
    pub fn read_span(&self, ptr: DevicePtr, bytes: usize) -> Vec<u8> {
        let mut result = vec![0; bytes];
        self.inner.copy_d2h(ptr, &mut result).unwrap();
        result
    }
    pub fn write_span(&self, ptr: DevicePtr, bytes: &[u8]) {
        self.inner.copy_h2d(bytes, ptr).unwrap();
    }
    pub fn gather_sentinel(&self, src: u64, dst: u64, bytes: usize) -> Result<()> {
        if self.deterministic_logits.load(Ordering::Relaxed) {
            ensure!(
                bytes == 8,
                "sentinel gather supports only local vocab4 BF16"
            );
            let local = self.read_span(DevicePtr(src), bytes);
            self.inner.copy_h2d(&local, DevicePtr(dst))?;
            self.inner.copy_h2d(&local, DevicePtr(dst).offset(bytes))?;
        }
        Ok(())
    }
    pub fn slab(&self) -> DevicePtr {
        let live = self.live.lock();
        let found: Vec<_> = live.iter().filter(|(_, n)| **n == SLAB_BYTES).collect();
        assert_eq!(found.len(), 1, "exactly one actual paired slab");
        DevicePtr(*found[0].0)
    }
}
struct Gpu(Arc<Recorder>);
impl GpuBackend for Gpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        let reused = if self.0.reuse_freed.load(Ordering::Relaxed) {
            let mut retired = self.0.retired.lock();
            let found = retired.iter().position(|(_, bytes)| *bytes == n);
            found.map(|i| retired.swap_remove(i).0)
        } else {
            None
        };
        let ptr = match reused {
            Some(p) => p,
            None => self.0.inner.alloc(n)?,
        };
        self.0.live.lock().insert(ptr.0, n);
        self.0.event(Event::Alloc(ptr, n))?;
        Ok(ptr)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.alloc(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        if p.is_null() {
            return Ok(());
        }
        let allocation = self.0.live.lock().remove(&p.0);
        let n = allocation.ok_or_else(|| anyhow::anyhow!("duplicate/foreign free"))?;
        self.0.event(Event::Free(p))?; // Native remove-before-free-error semantics.
        if self.0.reuse_freed.load(Ordering::Relaxed) {
            self.0.retired.lock().push((p, n));
            Ok(())
        } else {
            self.0.inner.free(p)
        }
    }
    fn copy_h2d(&self, b: &[u8], p: DevicePtr) -> Result<()> {
        self.copy_h2d_async(b, p, DEFAULT)
    }
    fn copy_h2d_async(&self, b: &[u8], p: DevicePtr, s: u64) -> Result<()> {
        self.0.event(Event::Upload(p, b.len(), s))?;
        self.0.inner.copy_h2d(b, p)
    }
    fn copy_d2h(&self, p: DevicePtr, b: &mut [u8]) -> Result<()> {
        self.copy_d2h_on_stream(p, b, DEFAULT)
    }
    fn copy_d2h_on_stream(&self, p: DevicePtr, b: &mut [u8], s: u64) -> Result<()> {
        self.0.event(Event::Read(p, b.len(), s))?;
        self.0.inner.copy_d2h(p, b)
    }
    fn copy_d2d(&self, a: DevicePtr, b: DevicePtr, n: usize) -> Result<()> {
        self.copy_d2d_async(a, b, n, DEFAULT)
    }
    fn copy_d2d_async(&self, a: DevicePtr, b: DevicePtr, n: usize, s: u64) -> Result<()> {
        self.0.event(Event::Copy(a, b, n, s))?;
        self.0.inner.copy_d2d(a, b, n)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.0.event(Event::Sync(s))?;
        self.0.inner.synchronize(s)
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.0.event(Event::RecordEvent(event, stream))
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.0.event(Event::WaitEvent(stream, event))
    }
    fn default_stream(&self) -> u64 {
        DEFAULT
    }
    fn create_stream(&self) -> Result<u64> {
        Ok(self.0.next_handle.fetch_add(1, Ordering::Relaxed) as u64 + 1024)
    }
    fn create_event(&self) -> Result<u64> {
        Ok(self.0.next_handle.fetch_add(1, Ordering::Relaxed) as u64 + 1024)
    }
    fn stream_is_capturing(&self, _: u64) -> bool {
        self.0.capturing.load(Ordering::Relaxed)
    }
    fn begin_capture(&self, stream: u64) -> Result<()> {
        self.0.event(Event::BeginCapture(stream))
    }
    fn end_capture(&self, stream: u64) -> Result<spark_runtime::gpu::GraphHandle> {
        self.0.event(Event::EndCapture(stream))?;
        let handle = if self.0.capture_handles.load(Ordering::Relaxed) {
            self.0.next_handle.fetch_add(1, Ordering::Relaxed) as u64 + 1024
        } else {
            0
        };
        Ok(spark_runtime::gpu::GraphHandle(handle))
    }
    fn abort_capture_if_active(&self, stream: u64) {
        let _ = self.0.event(Event::AbortCapture(stream));
    }
    fn launch_graph(&self, graph: spark_runtime::gpu::GraphHandle, stream: u64) -> Result<()> {
        self.0.event(Event::LaunchGraph(graph.0, stream))
    }
    fn sweep_unreleased(&self) -> usize {
        self.0.sweeps.fetch_add(1, Ordering::Relaxed);
        0
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.0.inner.op_cache()
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.memset_async(p, v, n, DEFAULT)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, s: u64) -> Result<()> {
        self.0.event(Event::Memset(p, n, s))?;
        self.0.inner.memset(p, v, n)
    }
    fn kernel(&self, _: &str, name: &str) -> Result<KernelHandle> {
        let mut names = self.0.names.lock();
        let index = names.iter().position(|s| s == name).unwrap_or_else(|| {
            names.push(name.into());
            names.len() - 1
        });
        Ok(KernelHandle(index as u64 + 1))
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        anyhow::bail!("typed fixture launch required")
    }
    fn launch_typed(
        &self,
        k: KernelHandle,
        grid: [u32; 3],
        _: [u32; 3],
        _: u32,
        s: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let name = self.0.names.lock()[k.0 as usize - 1].clone();
        let ptrs: Vec<_> = args
            .iter()
            .filter_map(|a| match a {
                KernelArg::Buffer(p) => Some(*p),
                _ => None,
            })
            .collect();
        self.0.event(Event::Kernel(name.clone(), ptrs.clone(), s))?;
        let scalar = |index: usize| -> Result<usize> {
            match args.get(index) {
                Some(KernelArg::Bytes(bytes)) if bytes.len() == 4 => {
                    Ok(u32::from_ne_bytes((*bytes).try_into().unwrap()) as usize)
                }
                _ => anyhow::bail!("expected actual u32 kernel ABI"),
            }
        };
        if name == "batched_embed" {
            let bytes = scalar(3)? * 2;
            let ids = self.0.read_span(ptrs[0], grid[0] as usize * 4);
            for (row, id) in ids.chunks_exact(4).enumerate() {
                let token = u32::from_ne_bytes(id.try_into().unwrap()) as usize;
                self.0.inner.copy_d2d(
                    ptrs[1].offset(token * bytes),
                    ptrs[2].offset(row * bytes),
                    bytes,
                )?;
            }
        } else if matches!(name.as_str(), "rms_norm_vanilla" | "rms_norm") {
            let bytes = grid[0] as usize * scalar(3)? * 2;
            self.0.inner.copy_d2d(ptrs[0], ptrs[2], bytes)?;
        } else if name == "bf16_concat" {
            let n = scalar(3)? * 2;
            self.0.inner.copy_d2d(ptrs[0], ptrs[2], n)?;
            self.0.inner.copy_d2d(ptrs[1], ptrs[2].offset(n), n)?;
        } else if (name == "dense_gemm_bf16" && scalar(5)? == 8192)
            || (name == "dense_gemv_bf16" && scalar(4)? == 8192)
        {
            let rows = if name == "dense_gemm_bf16" {
                scalar(3)?
            } else {
                1
            };
            for row in 0..rows {
                let src = ptrs[0].offset((row * 2 + 1) * ROW_BYTES);
                let dst = ptrs[2].offset(row * ROW_BYTES);
                if self.0.deterministic_logits.load(Ordering::Relaxed) {
                    let token = self
                        .0
                        .read_span(ptrs[0].offset(row * 2 * ROW_BYTES), ROW_BYTES);
                    let hidden = self.0.read_span(src, ROW_BYTES);
                    ensure!(
                        token.iter().all(|b| *b == token[0])
                            && hidden.iter().all(|b| *b == hidden[0]),
                        "nonuniform EH sentinel input"
                    );
                    self.0.eh_pairs.lock().push((token[0], hidden[0]));
                }
                self.0.inner.copy_d2d(src, dst, ROW_BYTES)?;
            }
        } else if self.0.deterministic_logits.load(Ordering::Relaxed)
            && matches!(name.as_str(), "dense_gemm_bf16" | "dense_gemv_bf16")
        {
            let (rows, n, k) = if name == "dense_gemm_bf16" {
                (scalar(3)?, scalar(4)?, scalar(5)?)
            } else {
                (1, scalar(3)?, scalar(4)?)
            };
            ensure!(
                matches!(n, 4 | 8) && k == 4096,
                "unsupported sentinel head ABI"
            );
            for row in 0..rows {
                let value = self.0.read_span(ptrs[0].offset(row * ROW_BYTES), 1)[0];
                self.0
                    .inner
                    .copy_h2d(&vec![value; n * 2], ptrs[2].offset(row * n * 2))?;
            }
        } else if matches!(name.as_str(), "argmax_bf16" | "argmax_bf16_value") {
            let token = if self.0.deterministic_logits.load(Ordering::Relaxed) {
                u32::from(self.0.read_span(ptrs[0], 1)[0] % 8)
            } else {
                1
            };
            self.0.inner.copy_h2d(&token.to_ne_bytes(), ptrs[1])?;
        }
        Ok(())
    }
    fn total_memory(&self) -> Result<usize> {
        Ok(128usize << 30)
    }
    fn free_memory(&self) -> Result<usize> {
        Ok(120usize << 30)
    }
    fn sm_count(&self) -> Result<u32> {
        Ok(48)
    }
}
struct Rank(usize, Arc<Recorder>);
macro_rules! rank_comm {
    ($($name:ident($($arg:ty),*));*) => {
        impl spark_comm::CommBackend for Rank {
            $(fn $name(&self, $(_: $arg),*) -> Result<()> { Ok(()) })*
            fn rank(&self) -> usize { self.0 }
            fn world_size(&self) -> usize { 2 }
            fn all_gather(&self, src: u64, dst: u64, n: usize) -> Result<()> {
                self.1.gather_sentinel(src, dst, n)
            }
        }
    };
}
rank_comm! { all_reduce(u64, usize);
reduce_scatter(u64, u64, usize); broadcast(u64, usize, usize); barrier();
send_to(u64, usize, usize, u64); recv_from(u64, usize, usize, u64) }
pub(super) struct Fixture {
    pub model: TransformerModel,
    pub seqs: [SequenceState; 2],
    pub head: Arc<Glm5MtpHead>,
    pub gpu: Arc<Recorder>,
}
#[path = "glm_c2_handoff_test_build.rs"]
mod build;
