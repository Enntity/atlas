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
    pub fail: AtomicUsize,
    pub reuse_freed: AtomicBool,
    pub capturing: AtomicBool,
    pub sweeps: AtomicUsize,
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
        self.fail.store(usize::MAX, Ordering::Relaxed);
    }
    pub fn trace(&self) -> Vec<Event> {
        self.events.lock().clone()
    }
    pub fn read_span(&self, ptr: DevicePtr, bytes: usize) -> Vec<u8> {
        let mut result = vec![0; bytes];
        self.inner.copy_d2h(ptr, &mut result).unwrap();
        result
    }
    pub fn write_span(&self, ptr: DevicePtr, bytes: &[u8]) {
        self.inner.copy_h2d(bytes, ptr).unwrap();
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
    fn default_stream(&self) -> u64 {
        DEFAULT
    }
    fn stream_is_capturing(&self, _: u64) -> bool {
        self.0.capturing.load(Ordering::Relaxed)
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
                self.0.inner.copy_d2d(src, dst, ROW_BYTES)?;
            }
        } else if matches!(name.as_str(), "argmax_bf16" | "argmax_bf16_value") {
            self.0.inner.copy_h2d(&1u32.to_ne_bytes(), ptrs[1])?;
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
struct Body {
    record: Arc<Recorder>,
    target: bool,
}
impl Body {
    fn target_rows(&self, h: DevicePtr, n: usize, pos: usize, s: u64) -> Result<()> {
        self.record.event(Event::Target(n, pos, s))?;
        for row in 0..n {
            let dst = h.offset(row * ROW_BYTES);
            let source = self.record.read_span(dst, 1)[0];
            let sentinel = source.wrapping_add(0x20).wrapping_add((pos + row) as u8);
            self.record
                .inner
                .copy_h2d(&vec![sentinel; ROW_BYTES], dst)?;
        }
        Ok(())
    }
}
impl TransformerLayer for Body {
    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }
    fn supports_mla_kv_only(&self) -> bool {
        !self.target
    }
    fn prefill(
        &self,
        h: DevicePtr,
        _: DevicePtr,
        rows: usize,
        _: &mut dyn LayerState,
        _: &mut PagedKvCache,
        position: usize,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: usize,
        _: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(self.target, "private body must use real KV-only entry");
        self.target_rows(h, rows, position, stream)
    }
    fn decode(
        &self,
        h: DevicePtr,
        _: DevicePtr,
        _: &mut dyn LayerState,
        _: &mut PagedKvCache,
        position: usize,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.target {
            self.target_rows(h, 1, position, stream)
        } else {
            self.record.event(Event::Body(position, stream))
        }
    }
    fn prefill_mla_kv_only(
        &self,
        h: DevicePtr,
        rows: usize,
        cache: &mut PagedKvCache,
        slots: DevicePtr,
        _: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        ensure!(!self.target, "target cannot act as private KV body");
        let raw = self.record.read_span(slots, rows * 8);
        let slots: Vec<_> = raw
            .chunks_exact(8)
            .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        self.record.event(Event::Kv(slots.clone(), stream))?;
        for (row, slot) in slots.into_iter().enumerate() {
            ensure!(slot >= 0, "negative actual KV slot");
            let block = slot as usize / cache.block_size();
            let offset = slot as usize % cache.block_size() * 1024;
            let bytes = self.record.read_span(h.offset(row * ROW_BYTES), 1024);
            let k = cache.k_cache_ptr(0, block as u32).offset(offset);
            let v = cache.v_cache_ptr(0, block as u32).offset(offset);
            self.record.inner.copy_h2d(&bytes, k)?;
            self.record.inner.copy_h2d(&bytes, v)?;
        }
        Ok(true)
    }
}
struct Rank(usize);
macro_rules! rank_comm {
    ($($name:ident($($arg:ty),*));*) => {
        impl spark_comm::CommBackend for Rank {
            $(fn $name(&self, $(_: $arg),*) -> Result<()> { Ok(()) })*
            fn rank(&self) -> usize { self.0 }
            fn world_size(&self) -> usize { 2 }
        }
    };
}
rank_comm! { all_reduce(u64, usize); all_gather(u64, u64, usize);
reduce_scatter(u64, u64, usize); broadcast(u64, usize, usize); barrier();
send_to(u64, usize, usize, u64); recv_from(u64, usize, usize, u64) }
pub(super) struct Fixture {
    pub model: TransformerModel,
    pub seqs: [SequenceState; 2],
    pub head: Arc<Glm5MtpHead>,
    pub gpu: Arc<Recorder>,
}
impl Fixture {
    pub fn new(rank: usize) -> Self {
        assert!(rank < 2);
        let record = Arc::new(Recorder::default());
        let gpu = Box::new(Gpu(record.clone()));
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
        cfg.linear_num_key_heads = 64;
        cfg.linear_num_value_heads = 64;
        cfg.num_mtp_modules = 0;
        cfg.tp_world_size = 2;
        cfg.ep_world_size = 2;
        cfg.tp_rank = rank;
        cfg.ep_rank = rank;
        cfg.adapter_max_rank = 0;
        let dense = |bytes| DenseWeight {
            weight: gpu.alloc(bytes).unwrap(),
        };
        let embed = dense(8 * ROW_BYTES);
        let final_norm = dense(ROW_BYTES);
        let lm_head = dense(8 * ROW_BYTES);
        for token in 0..8 {
            record.write_span(
                embed.weight.offset(token * ROW_BYTES),
                &vec![token as u8; ROW_BYTES],
            );
        }
        let head = Arc::new(
            Glm5MtpHead::new_paired(
                Glm5MtpModule {
                    body: Box::new(Body {
                        record: record.clone(),
                        target: false,
                    }),
                    enorm: dense(ROW_BYTES),
                    hnorm: dense(ROW_BYTES),
                    norm: dense(ROW_BYTES),
                    eh_proj: dense(4096 * 4096 * 4),
                    eh_proj_nvfp4: None,
                },
                embed,
                lm_head,
                None,
                &cfg,
                gpu.as_ref(),
                8,
                2044,
            )
            .unwrap(),
        );
        let buffers =
            spark_runtime::buffers::BufferArena::new(&cfg, 8, 2044, 16, 1, gpu.as_ref()).unwrap();
        let cache = PagedKvCache::new(
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
            256,
            gpu.as_ref(),
        )
        .unwrap();
        let mut model = TransformerModel::new(
            cfg,
            embed,
            final_norm,
            lm_head,
            None,
            None,
            None,
            vec![Box::new(Body {
                record: record.clone(),
                target: true,
            })],
            buffers,
            cache,
            vec![],
            gpu,
            2044,
            2,
            crate::layers::MtpQuantization::Bf16,
            false,
            false,
            Box::new(spark_runtime::prefix_cache::NoPrefixCaching),
            8,
            Some(Arc::new(Rank(rank))),
            false,
            4,
            None,
            2,
            16,
        )
        .unwrap();
        model.levers.max_decode_seqs = 2;
        model.levers.drafter.prefill = true;
        model.levers.drafter.carry = false;
        model.dispatch.cublas_gemm = false;
        model.mtp_prefill_hidden = model.gpu.alloc(2044 * ROW_BYTES).unwrap();
        model.mtp_prefill_capacity = 2044;
        model.mtp_hidden_save = model.gpu.alloc(ROW_BYTES).unwrap();
        model.proposer = Some(head.clone());
        let seqs = std::array::from_fn(|slot| {
            let mut seq = SequenceState::host_only(slot);
            seq.prompt_len = 4;
            seq.layer_states = vec![Box::new(EmptyLayerState)];
            seq.disk_last_offloaded_per_layer = vec![0];
            seq.proposer_state = Some(head.alloc_state(model.gpu.as_ref()).unwrap());
            seq
        });
        record.clear();
        Self {
            model,
            seqs,
            head,
            gpu: record,
        }
    }
}
