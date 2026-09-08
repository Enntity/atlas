// SPDX-License-Identifier: AGPL-3.0-only
//! Small real model with recording boundary implementations. No numerical emulation.
use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use crate::model::types::TransformerModel;
use crate::speculative::{DraftProposer, ProposerState};
use crate::traits::SequenceState;
use crate::weight_map::DenseWeight;
use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle, mock::MockGpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

pub(super) const DEFAULT: u64 = 7;
pub(super) const CALLER: u64 = 37;
const CAPACITY: usize = 8;
const ROW_BYTES: usize = 8192;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Target(usize, u64),
    Capture(usize, u64),
    Primer(usize, u64),
}
#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<Event>>,
    capture: AtomicU64,
    target_failure: AtomicBool,
    capture_failure: AtomicBool,
}

struct Gpu {
    inner: MockGpuBackend,
    record: Arc<Recorder>,
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
    fn copy_h2d(&self, b: &[u8], p: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(b, p)
    }
    fn copy_d2h(&self, p: DevicePtr, b: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(p, b)
    }
    fn copy_d2d(&self, a: DevicePtr, b: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(a, b, n)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.inner.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        DEFAULT
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
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, stream: u64) -> Result<()> {
        self.inner.memset_async(p, v, n, stream)
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
        anyhow::bail!("typed launch required")
    }
    fn launch_typed(
        &self,
        _: KernelHandle,
        grid: [u32; 3],
        _: [u32; 3],
        _: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let base = self.record.capture.load(Ordering::Relaxed);
        if let Some(KernelArg::Buffer(dst)) = args.get(2) {
            if base != 0 && dst.0 >= base && dst.0 < base + (CAPACITY * ROW_BYTES) as u64 {
                // Real rms_norm argument ABI, actual owned capture destination.
                anyhow::ensure!(args.len() == 5, "capture must use actual norm ABI");
                self.record
                    .events
                    .lock()
                    .push(Event::Capture(grid[0] as usize, stream));
                anyhow::ensure!(
                    !self.record.capture_failure.load(Ordering::Relaxed),
                    "injected capture failure"
                );
            }
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

struct Layer(Arc<Recorder>);
impl TransformerLayer for Layer {
    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }
    fn decode(
        &self,
        _: DevicePtr,
        _: DevicePtr,
        _: &mut dyn LayerState,
        _: &mut PagedKvCache,
        _: usize,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &ForwardContext,
        _: u64,
    ) -> Result<()> {
        anyhow::bail!("stream fixture requires multi-token prefill")
    }
    fn prefill(
        &self,
        _: DevicePtr,
        _: DevicePtr,
        n: usize,
        _: &mut dyn LayerState,
        _: &mut PagedKvCache,
        _: usize,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: usize,
        _: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.0.events.lock().push(Event::Target(n, stream));
        anyhow::ensure!(
            !self.0.target_failure.load(Ordering::Relaxed),
            "injected target failure"
        );
        Ok(())
    }
}

struct State(usize);
impl ProposerState for State {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
struct Proposer(Arc<Recorder>);
impl DraftProposer for Proposer {
    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(State(0)))
    }
    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state.as_any().downcast_ref::<State>().unwrap().0
    }
    fn prefill_drafter(
        &self,
        tokens: &[u32],
        source: DevicePtr,
        state: &mut dyn ProposerState,
        _: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let state = state.as_any_mut().downcast_mut::<State>().unwrap();
        if state.0 != 0 {
            return Ok(0);
        }
        anyhow::ensure!(
            source.0 == self.0.capture.load(Ordering::Relaxed),
            "foreign capture consumed"
        );
        self.0
            .events
            .lock()
            .push(Event::Primer(tokens.len() - 1, stream));
        state.0 = tokens.len() - 1;
        Ok(state.0)
    }
    fn propose(
        &self,
        _: u32,
        _: DevicePtr,
        _: usize,
        _: usize,
        _: &mut dyn ProposerState,
        _: &ForwardContext,
        _: u64,
        _: Option<DevicePtr>,
        _: Option<&[i32]>,
        _: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("stream fixture never proposes")
    }
    fn after_verify(&self, _: usize, _: &mut dyn ProposerState, _: u64) -> Result<()> {
        Ok(())
    }
}

struct Rank(usize);
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

pub(super) struct Fixture {
    pub model: TransformerModel,
    pub seq: SequenceState,
    record: Arc<Recorder>,
}
impl Fixture {
    pub fn new(tp: usize, ep: usize, rank: usize) -> Self {
        let record = Arc::new(Recorder::default());
        let gpu = Box::new(Gpu {
            inner: MockGpuBackend::new(),
            record: record.clone(),
        });
        let mut cfg = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        cfg.model_type = "glm5_next".into();
        cfg.hidden_size = 4096;
        cfg.vocab_size = 8;
        cfg.num_hidden_layers = 1;
        cfg.layer_types = vec![atlas_core::config::LayerType::FullAttention];
        cfg.num_attention_heads = 32;
        cfg.num_key_value_heads = 1;
        cfg.head_dim = 128;
        cfg.kv_lora_rank = 512;
        cfg.qk_rope_head_dim = 0;
        cfg.index_kpool = 0;
        cfg.index_head_dim = 0;
        cfg.index_topk = 2048;
        cfg.intermediate_size = 2048;
        cfg.moe_intermediate_size = 2048;
        cfg.shared_expert_intermediate_size = 2048;
        cfg.num_experts = 2;
        cfg.num_experts_per_tok = 2;
        cfg.linear_num_key_heads = 0;
        cfg.linear_num_value_heads = 0;
        cfg.num_mtp_modules = 0;
        cfg.tp_world_size = tp;
        cfg.ep_world_size = ep;
        cfg.tp_rank = rank;
        cfg.ep_rank = if ep > 1 { rank } else { 0 };
        cfg.adapter_max_rank = 0;
        let buffers =
            spark_runtime::buffers::BufferArena::new(&cfg, CAPACITY, 32, 16, 1, gpu.as_ref())
                .unwrap();
        let kv = PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 512,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            4,
            gpu.as_ref(),
        )
        .unwrap();
        let dense = DenseWeight {
            weight: gpu.alloc(8 * ROW_BYTES).unwrap(),
        };
        let comm =
            (tp > 1 || ep > 1).then(|| Arc::new(Rank(rank)) as Arc<dyn spark_comm::CommBackend>);
        let mut model = TransformerModel::new(
            cfg,
            dense,
            dense,
            dense,
            None,
            None,
            None,
            vec![Box::new(Layer(record.clone()))],
            buffers,
            kv,
            vec![],
            gpu,
            32,
            1,
            crate::layers::MtpQuantization::Bf16,
            false,
            false,
            Box::new(spark_runtime::prefix_cache::NoPrefixCaching),
            8,
            comm,
            false,
            4,
            None,
            1,
            16,
        )
        .unwrap();
        model.levers.max_decode_seqs = 1;
        model.levers.drafter.prefill = true;
        model.levers.drafter.carry = false;
        model.mtp_prefill_hidden = model.gpu.alloc(CAPACITY * ROW_BYTES).unwrap();
        model.mtp_prefill_capacity = CAPACITY;
        record
            .capture
            .store(model.mtp_prefill_hidden.0, Ordering::Relaxed);
        model.proposer = Some(Arc::new(Proposer(record.clone())));
        let mut seq = SequenceState::host_only(0);
        seq.prompt_len = 4;
        seq.layer_states = vec![Box::new(EmptyLayerState)];
        seq.disk_last_offloaded_per_layer = vec![0];
        seq.proposer_state = Some(Box::new(State(0)));
        record.events.lock().clear();
        Self { model, seq, record }
    }
    pub fn events(&self) -> Vec<Event> {
        self.record.events.lock().clone()
    }
    pub fn fail(&self, capture: bool) {
        if capture {
            self.record.capture_failure.store(true, Ordering::Relaxed);
        } else {
            self.record.target_failure.store(true, Ordering::Relaxed);
        }
    }
    pub fn disable_capture(&mut self) {
        self.model.mtp_prefill_hidden = DevicePtr::NULL;
    }
}
