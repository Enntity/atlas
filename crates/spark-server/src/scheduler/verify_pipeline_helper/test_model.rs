// SPDX-License-Identifier: AGPL-3.0-only

//! Real selection entrypoints, owned host bytes, and an injectable backend copy.
//! No GPU numerical or completion claim: every unrelated Model call panics.
use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, mock::MockGpuBackend};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

pub(super) struct CopyGpu {
    pub inner: MockGpuBackend,
    pub fail: AtomicUsize,
    pub reads: Mutex<Vec<(DevicePtr, usize)>>,
}

impl CopyGpu {
    pub fn clear(&self, fail: usize) {
        self.reads.lock().unwrap().clear();
        self.fail.store(fail, Ordering::Relaxed);
    }
}

macro_rules! delegate {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $out:ty;)*) => {$ (
        fn $name(&self, $($arg: $ty),*) -> $out { self.inner.$name($($arg),*) }
    )*};
}
impl GpuBackend for CopyGpu {
    delegate! {
        alloc(bytes: usize) -> Result<DevicePtr>;
        alloc_managed(bytes: usize) -> Result<DevicePtr>;
        free(ptr: DevicePtr) -> Result<()>;
        copy_h2d(src: &[u8], dst: DevicePtr) -> Result<()>;
        copy_d2d(src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()>;
        synchronize(stream: u64) -> Result<()>;
        default_stream() -> u64;
        kernel(module: &str, func: &str) -> Result<KernelHandle>;
        op_cache() -> &spark_runtime::op_cache::OpCache;
        memset(ptr: DevicePtr, value: u8, bytes: usize) -> Result<()>;
        memset_async(ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()>;
        total_memory() -> Result<usize>;
        free_memory() -> Result<usize>;
        sm_count() -> Result<u32>;
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let mut reads = self.reads.lock().unwrap();
        reads.push((src, dst.len()));
        if self.fail.load(Ordering::Relaxed) == reads.len() {
            anyhow::bail!("injected backend logits copy {}", reads.len());
        }
        self.inner.copy_d2h(src, dst)
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
        panic!("selection launched a kernel")
    }
}

pub(super) struct Target {
    pub gpu: CopyGpu,
    pub base: DevicePtr,
    pub vocab: usize,
    pub fp32: bool,
}
impl Target {
    pub fn new(rows: &[Vec<f32>], fp32: bool) -> Self {
        let gpu = CopyGpu {
            inner: MockGpuBackend::new(),
            fail: AtomicUsize::new(0),
            reads: Mutex::new(Vec::new()),
        };
        let vocab = rows[0].len();
        assert!(rows.iter().all(|row| row.len() == vocab));
        let bytes: Vec<u8> = rows
            .iter()
            .flatten()
            .flat_map(|x| {
                if fp32 {
                    x.to_le_bytes().to_vec()
                } else {
                    ((x.to_bits() >> 16) as u16).to_le_bytes().to_vec()
                }
            })
            .collect();
        let base = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(&bytes, base).unwrap();
        Self {
            gpu,
            base,
            vocab,
            fp32,
        }
    }
    pub fn reads(&self) -> Vec<(DevicePtr, usize)> {
        self.gpu.reads.lock().unwrap().clone()
    }
}
impl Drop for Target {
    fn drop(&mut self) {
        self.gpu.free(self.base).unwrap();
    }
}

macro_rules! unused {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $out:ty;)*) => {$ (
        fn $name(&self, $($arg: $ty),*) -> $out { $(let _ = $arg;)*
            panic!(concat!("selection called unrelated Model::", stringify!($name))) }
    )*};
}
impl Model for Target {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.base
    }
    fn logits_ptr_is_fp32(&self, ptr: DevicePtr) -> bool {
        assert_eq!(ptr, self.base);
        self.fp32
    }
    fn copy_logits_to_host(&self, ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.gpu.copy_d2h(ptr, dst)
    }
    unused! {
        prefill(t: &[u32], s: &mut SequenceState, st: u64) -> Result<DevicePtr>;
        decode(t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr>;
        prefill_chunk(t: &[u32], s: &mut SequenceState, a: usize, b: usize, l: bool, st: u64) -> Result<DevicePtr>;
        decode_batch(t: &[u32], s: &mut [&mut SequenceState], st: u64) -> Result<DevicePtr>;
        decode_verify(t: &[u32], s: &mut SequenceState, st: u64) -> Result<Vec<u32>>;
        generate_speculative(t: &[u32], p: &spark_runtime::sampler::SamplingParams, n: usize) -> Result<spark_model::engine::GenerateResult>;
        decode_verify_graphed(t: &[u32;2], s: &mut SequenceState, st: u64) -> Result<[u32;2]>;
        decode_verify_graphed_k3(t: &[u32;3], s: &mut SequenceState, st: u64) -> Result<[u32;3]>;
        decode_verify_graphed_k4(t: &[u32;4], s: &mut SequenceState, st: u64) -> Result<[u32;4]>;
        run_mtp_propose(t: u32, p: usize, s: &mut SequenceState, st: u64) -> Result<Option<u32>>;
        run_mtp_propose_multi(t: u32, p: usize, n: usize, s: &mut SequenceState, st: u64, g: Option<&[i32]>) -> Result<Vec<u32>>;
        trim_proposer_state(s: &mut SequenceState, a: usize, st: u64) -> Result<()>;
        save_hidden_for_mtp(r: usize, st: u64) -> Result<()>;
        commit_accepted_prefix(s: &mut SequenceState, a: usize, k: usize) -> Result<()>;
        bind_gpu_to_thread() -> Result<()>;
        alloc_sequence() -> Result<SequenceState>;
        argmax_on_device(p: DevicePtr, st: u64) -> Result<u32>;
        argmax_batch(p: DevicePtr, n: usize, st: u64) -> Result<Vec<u32>>;
        hidden_after_norm() -> DevicePtr;
        checkpoint_ssm_states(s: &mut SequenceState) -> Result<()>;
        rollback_ssm_states(s: &mut SequenceState, n: usize) -> Result<()>;
        has_proposer() -> bool;
        has_self_speculative() -> bool;
        decode_draft(t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr>;
        cache_sequence(s: &SequenceState) -> ();
        free_sequence(s: &mut SequenceState) -> Result<()>;
        compact_sequence(s: &mut SequenceState, n: usize) -> Result<()>;
        detach_slot_for_reuse(s: &mut SequenceState) -> ();
    }
}
