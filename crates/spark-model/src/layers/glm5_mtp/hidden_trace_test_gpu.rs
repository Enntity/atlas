// SPDX-License-Identifier: AGPL-3.0-only
//! Host-backed test backend. Sentinel math tests hook order, not CUDA numerics.
use super::*;
use spark_runtime::gpu::{KernelArg, mock::MockGpuBackend};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
};

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Event {
    Read(DevicePtr, usize, u64),
    Kernel(u64, Vec<DevicePtr>),
    Body,
}
pub(super) struct TraceGpu {
    pub inner: MockGpuBackend,
    pub events: Arc<Mutex<Vec<Event>>>,
    pub capture_queries: AtomicUsize,
    pub capturing: AtomicBool,
    pub fail_read: AtomicBool,
    pub fail_final_read: AtomicBool,
    pub fail_post_eh_read: AtomicBool,
    pub eh_last_byte: AtomicU8,
    pub read_hashes: Mutex<Vec<[u8; 32]>>,
    pub fail_body: Arc<AtomicBool>,
    pub kv_reads: AtomicUsize,
    pub fail_kv_read_at: AtomicUsize,
}
impl TraceGpu {
    pub fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            events: Arc::new(Mutex::new(Vec::new())),
            capture_queries: AtomicUsize::new(0),
            capturing: AtomicBool::new(false),
            fail_read: AtomicBool::new(false),
            fail_final_read: AtomicBool::new(false),
            fail_post_eh_read: AtomicBool::new(false),
            eh_last_byte: AtomicU8::new(0x5a),
            read_hashes: Mutex::new(Vec::new()),
            fail_body: Arc::new(AtomicBool::new(false)),
            kv_reads: AtomicUsize::new(0),
            fail_kv_read_at: AtomicUsize::new(usize::MAX),
        }
    }
}
impl GpuBackend for TraceGpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn copy_h2d(&self, b: &[u8], p: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(b, p)
    }
    fn copy_d2h(&self, p: DevicePtr, b: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(p, b)
    }
    fn copy_d2h_on_stream(&self, p: DevicePtr, b: &mut [u8], s: u64) -> Result<()> {
        self.events.lock().push(Event::Read(p, b.len(), s));
        if matches!(b.len(), 1024 | 2048) {
            let ordinal = self.kv_reads.fetch_add(1, Ordering::Relaxed) + 1;
            anyhow::ensure!(
                ordinal != self.fail_kv_read_at.load(Ordering::Relaxed),
                "injected KV copy failure"
            );
        }
        if self.fail_read.load(Ordering::Relaxed)
            || (self.fail_post_eh_read.load(Ordering::Relaxed)
                && !self.events.lock().contains(&Event::Body)
                && self
                    .events
                    .lock()
                    .iter()
                    .filter(|e| matches!(e, Event::Read(..)))
                    .count()
                    == 2)
            || (self.fail_final_read.load(Ordering::Relaxed)
                && self.events.lock().contains(&Event::Body)
                && b.len() == ROW_BYTES)
        {
            anyhow::bail!("injected hidden read failure");
        }
        self.inner.copy_d2h_on_stream(p, b, s)?;
        self.read_hashes.lock().push(Sha256::digest(b).into());
        Ok(())
    }
    fn copy_d2d(&self, a: DevicePtr, b: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(a, b, n)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.inner.synchronize(s)
    }
    fn stream_is_capturing(&self, _: u64) -> bool {
        self.capture_queries.fetch_add(1, Ordering::Relaxed);
        self.capturing.load(Ordering::Relaxed)
    }
    fn default_stream(&self) -> u64 {
        7
    }
    fn kernel(&self, _: &str, _: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(1))
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
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        anyhow::bail!("typed launch required");
    }
    fn launch_typed(
        &self,
        kernel: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let ptrs: Vec<_> = args
            .iter()
            .filter_map(|a| match a {
                KernelArg::Buffer(p) => Some(*p),
                _ => None,
            })
            .collect();
        self.events
            .lock()
            .push(Event::Kernel(kernel.0, ptrs.clone()));
        if kernel.0 == 101 {
            self.inner.copy_d2d(ptrs[0], ptrs[2], ROW_BYTES)?;
        }
        if kernel.0 == 102
            && matches!(args.last(), Some(KernelArg::Bytes(k))
            if *k == 8192u32.to_ne_bytes())
        {
            let mut row = [0xa5; ROW_BYTES];
            row[ROW_BYTES - 1] = self.eh_last_byte.load(Ordering::Relaxed);
            self.inner.copy_h2d(&row, ptrs[2])?;
        }
        if kernel.0 == 104 {
            self.inner.copy_h2d(&7u32.to_le_bytes(), ptrs[1])?;
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

pub(super) struct Rank(pub usize);
impl spark_comm::CommBackend for Rank {
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        Ok(())
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        Ok(())
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        Ok(())
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        Ok(())
    }
    fn barrier(&self) -> Result<()> {
        Ok(())
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
    fn rank(&self) -> usize {
        self.0
    }
    fn world_size(&self) -> usize {
        2
    }
}

struct Body(Arc<Mutex<Vec<Event>>>, Arc<AtomicBool>);
impl crate::layer::TransformerLayer for Body {
    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn crate::layer::LayerState>> {
        Ok(Box::new(crate::layer::EmptyLayerState))
    }
    fn decode(
        &self,
        _: DevicePtr,
        _: DevicePtr,
        _: &mut dyn crate::layer::LayerState,
        cache: &mut PagedKvCache,
        row: usize,
        blocks: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        ctx: &ForwardContext,
        _: u64,
    ) -> Result<()> {
        self.0.lock().push(Event::Body);
        anyhow::ensure!(!self.1.load(Ordering::Relaxed), "injected body failure");
        for (side, pool) in [cache.k_pool_ptr(0), cache.v_pool_ptr(0)]
            .into_iter()
            .enumerate()
        {
            let ptr = pool.offset(blocks[row / 16] as usize * 16384 + row % 16 * 1024);
            ctx.gpu
                .copy_h2d(&[0x60 + (side as u8) * 16 + row as u8; 1024], ptr)?;
        }
        let bytes: Vec<_> = (0..4096)
            .flat_map(|_| [0, 0x3f + (row % 8) as u8])
            .collect();
        ctx.gpu.copy_h2d(&bytes, ctx.buffers.hidden_states())
    }
}

pub(super) fn fixture(
    rank: usize,
    run: impl FnOnce(&Glm5MtpHead, &mut ForwardContext, &TraceGpu, DevicePtr),
) {
    let gpu = TraceGpu::new();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.vocab_size = 8;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 1;
    config.kv_lora_rank = 512;
    config.qk_rope_head_dim = 0;
    config.intermediate_size = 2048;
    config.moe_intermediate_size = 2048;
    config.shared_expert_intermediate_size = 2048;
    config.num_experts = 2;
    config.num_experts_per_tok = 2;
    config.linear_num_key_heads = 1;
    config.linear_num_value_heads = 1;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.index_kpool = 0;
    config.index_head_dim = 0;
    config.index_topk = 2048;
    config.tp_world_size = 2;
    config.ep_world_size = 2;
    config.adapter_max_rank = 0;
    let dense = DenseWeight {
        weight: DevicePtr(0x2000_0000_0000),
    };
    let module = Glm5MtpModule {
        body: Box::new(Body(gpu.events.clone(), gpu.fail_body.clone())),
        enorm: dense,
        hnorm: dense,
        eh_proj: dense,
        eh_proj_nvfp4: None,
        norm: dense,
    };
    let embed = DenseWeight {
        weight: gpu.alloc(8 * ROW_BYTES).unwrap(),
    };
    let mut head = Glm5MtpHead::new(module, embed, dense, None, &config, &gpu, 8, 2044).unwrap();
    head.hidden_trace_enabled = true;
    head.rms_norm_k = KernelHandle(101);
    head.dense_gemv_k = KernelHandle(102);
    head.argmax_k = KernelHandle(104);
    let buffers = spark_runtime::buffers::BufferArena::new(&config, 5, 2044, 64, 1, &gpu).unwrap();
    let mut dispatch = ops::GemmDispatch::defaults();
    dispatch.cublas_gemm = false;
    let derived = ops::DerivedWeights::new();
    let stats = ops::ModelStats::new();
    let mut levers = ops::ModelLevers::defaults();
    levers.max_decode_seqs = 1;
    levers.drafter.prefill = true;
    levers.drafter.carry = false;
    let comm = Rank(rank);
    let mut ctx = ForwardContext {
        buffers: &buffers,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        ssm_batch: None,
        attn_metadata: None,
        profile: false,
        comm: Some(&comm),
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::lora::resolve_moe_lora_route(-1, -1, false),
    };
    let saved = gpu.alloc(ROW_BYTES).unwrap();
    gpu.copy_h2d(&[0x3c; ROW_BYTES], saved).unwrap();
    gpu.events.lock().clear();
    run(&head, &mut ctx, &gpu, saved);
}
