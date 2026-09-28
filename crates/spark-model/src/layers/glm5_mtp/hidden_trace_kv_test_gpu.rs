// SPDX-License-Identifier: AGPL-3.0-only
//! Real host-backed pool bytes; no CUDA numerical kernel simulation.
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Read(DevicePtr, usize, u64),
    Allocate(usize),
    Write,
    Sync,
}
pub(super) struct Gpu {
    pub inner: MockGpuBackend,
    pub events: Mutex<Vec<Event>>,
    pub fail: AtomicUsize,
    pub capturing: AtomicBool,
    pub capture_queries: AtomicUsize,
    pub bad_allocation: AtomicUsize,
}
impl Gpu {
    pub fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            events: Mutex::new(vec![]),
            fail: AtomicUsize::new(usize::MAX),
            capturing: AtomicBool::new(false),
            capture_queries: AtomicUsize::new(0),
            bad_allocation: AtomicUsize::new(0),
        }
    }
}
impl GpuBackend for Gpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.events.lock().push(Event::Allocate(n));
        match self.bad_allocation.load(Ordering::Relaxed) {
            1 => return Ok(DevicePtr::NULL),
            2 => return Ok(DevicePtr(0x1001)),
            3 => return Ok(DevicePtr(u64::MAX - 1)),
            4 => return Ok(DevicePtr(0x1000)), // both actual pool owners alias
            _ => {}
        }
        self.inner.alloc(n)
    }
    fn alloc_managed(&self, _: usize) -> Result<DevicePtr> {
        anyhow::bail!("unexpected managed allocation")
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn copy_h2d(&self, b: &[u8], p: DevicePtr) -> Result<()> {
        self.events.lock().push(Event::Write);
        self.inner.copy_h2d(b, p)
    }
    fn copy_d2h(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        anyhow::bail!("must use explicit stream")
    }
    fn copy_d2h_on_stream(&self, p: DevicePtr, b: &mut [u8], stream: u64) -> Result<()> {
        self.events.lock().push(Event::Read(p, b.len(), stream));
        ensure!(
            self.events.lock().len() != self.fail.load(Ordering::Relaxed),
            "injected KV copy failure"
        );
        self.inner.copy_d2h_on_stream(p, b, stream)
    }
    fn copy_d2d(&self, _: DevicePtr, _: DevicePtr, _: usize) -> Result<()> {
        anyhow::bail!("unexpected D2D")
    }
    fn synchronize(&self, _: u64) -> Result<()> {
        self.events.lock().push(Event::Sync);
        anyhow::bail!("copy API owns synchronization")
    }
    fn stream_is_capturing(&self, _: u64) -> bool {
        self.capture_queries.fetch_add(1, Ordering::Relaxed);
        self.capturing.load(Ordering::Relaxed)
    }
    fn default_stream(&self) -> u64 {
        7
    }
    fn kernel(&self, _: &str, _: &str) -> Result<KernelHandle> {
        anyhow::bail!("unexpected kernel lookup")
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn memset(&self, _: DevicePtr, _: u8, _: usize) -> Result<()> {
        anyhow::bail!("unexpected memset")
    }
    fn memset_async(&self, _: DevicePtr, _: u8, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected memset")
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
        anyhow::bail!("unexpected kernel")
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

pub(super) fn context(gpu: &Gpu, run: impl FnOnce(&ForwardContext)) {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.kv_lora_rank = 512;
    config.qk_rope_head_dim = 0;
    config.index_topk = 2048;
    let buffers =
        spark_runtime::buffers::BufferArena::new(&config, 1, 2044, 16, 1, &gpu.inner).unwrap();
    let dispatch = ops::GemmDispatch::defaults();
    let derived = ops::DerivedWeights::new();
    let levers = ops::ModelLevers::defaults();
    let stats = ops::ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        ssm_batch: None,
        host_token_ids: None,
        token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Skip,
    };
    run(&ctx);
}

pub(super) fn cache(gpu: &Gpu, rows: usize, reverse: bool) -> (PagedKvCache, Vec<u32>) {
    let config = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: 1,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let mut cache = PagedKvCache::new(config, 128, gpu).unwrap();
    let mut blocks: Vec<_> = (0..(rows + 1).div_ceil(16))
        .map(|_| cache.alloc_block().unwrap())
        .collect();
    if reverse {
        blocks.reverse();
    }
    for logical in 0..rows + 1 {
        for (side, pool) in [cache.k_pool_ptr(0), cache.v_pool_ptr(0)]
            .into_iter()
            .enumerate()
        {
            let bytes: Vec<_> = (0..1024)
                .map(|i| (logical.wrapping_mul(73) ^ i ^ (i >> 8) ^ (side * 31)) as u8)
                .collect();
            let p =
                pool.offset(blocks[logical / 16] as usize * SIDE_BLOCK + logical % 16 * SIDE_ROW);
            gpu.inner.copy_h2d(&bytes, p).unwrap();
        }
    }
    gpu.events.lock().clear();
    (cache, blocks)
}
