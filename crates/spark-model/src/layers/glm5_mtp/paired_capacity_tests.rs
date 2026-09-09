// SPDX-License-Identifier: AGPL-3.0-only
//! Actual head/cache construction, not a capacity-formula-only test.
use super::*;
use crate::layer::{EmptyLayerState, TransformerLayer};
use spark_runtime::gpu::mock::MockGpuBackend;
#[path = "paired_test_gpu.rs"]
mod gpu;
use gpu::TestGpu;
use std::sync::atomic::Ordering;

impl Glm5MtpHead {
    pub(crate) fn paired_test_free_blocks(&self) -> usize {
        self.kv_cache.lock().num_free_blocks()
    }
    pub(crate) fn paired_test_kv_rows(
        &self,
        state: &dyn ProposerState,
        backend: &dyn GpuBackend,
        rows: usize,
    ) -> Result<Vec<(DevicePtr, DevicePtr)>> {
        let state = state
            .as_any()
            .downcast_ref::<Glm5MtpProposerState>()
            .context("inspection requires actual GLM state")?;
        self.validate_paired_live(state, backend)?;
        let pool = self
            .paired
            .as_ref()
            .context("inspection requires paired pool")?
            .lock();
        let slot = &pool.slots[state.paired.as_ref().expect("validated lease").slot];
        let cache = self.kv_cache.lock();
        ensure!(
            rows <= slot.blocks.len() * cache.block_size(),
            "inspection exceeds canonical reserve"
        );
        let k_stride = cache.k_block_stride_bytes_for_layer(0) / cache.block_size();
        let v_stride = cache.v_block_stride_bytes_for_layer(0) / cache.block_size();
        Ok((0..rows)
            .map(|row| {
                let block = slot.blocks[row / cache.block_size()];
                let offset = row % cache.block_size();
                (
                    cache.k_cache_ptr(0, block).offset(offset * k_stride),
                    cache.v_cache_ptr(0, block).offset(offset * v_stride),
                )
            })
            .collect())
    }
}

struct Body(bool);
impl TransformerLayer for Body {
    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        ensure!(!self.0, "injected body state allocation failure");
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
        anyhow::bail!("capacity test never executes numerical body")
    }
}

fn head(gpu: &dyn GpuBackend) -> Glm5MtpHead {
    build_head(gpu).unwrap()
}

fn build_head(gpu: &dyn GpuBackend) -> Result<Glm5MtpHead> {
    configured_head(gpu, true, false)
}

fn configured_head(gpu: &dyn GpuBackend, paired: bool, indexed: bool) -> Result<Glm5MtpHead> {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.kv_lora_rank = 512;
    config.qk_rope_head_dim = 0;
    config.index_kpool = if indexed { 4 } else { 0 };
    config.index_head_dim = if indexed { 128 } else { 0 };
    config.tp_world_size = 2;
    config.ep_world_size = 2;
    config.tp_rank = 0;
    config.ep_rank = 0;
    config.adapter_max_rank = 0;
    let dense = |n| DenseWeight {
        weight: gpu.alloc(n).unwrap(),
    };
    let module = Glm5MtpModule {
        body: Box::new(Body(false)),
        enorm: dense(8192),
        hnorm: dense(8192),
        norm: dense(8192),
        eh_proj: dense(4096 * 4096 * 4),
        eh_proj_nvfp4: None,
    };
    let constructor = if paired {
        Glm5MtpHead::new_paired
    } else {
        Glm5MtpHead::new
    };
    constructor(
        module,
        dense(8 * 8192),
        dense(8 * 8192),
        None,
        &config,
        gpu,
        8,
        2044,
    )
}

#[test]
fn body_allocation_failure_cannot_publish_or_retry_its_slot() {
    let gpu = TestGpu::new();
    let mut head = head(&gpu);
    head.module.body = Box::new(Body(true));
    assert!(head.alloc_state_inner(&gpu).is_err());
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 256);
    head.module.body = Box::new(Body(false));
    let state = head.alloc_state_inner(&gpu).unwrap();
    assert_eq!(state.paired.as_ref().unwrap().slot, 1);
    assert!(head.alloc_state_inner(&gpu).is_err());
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 128);
}

#[test]
fn failed_constructor_publishes_no_head_and_retains_exact_build_abort_ledger() {
    for (allocation, kernel, remaining) in [
        (8, false, 7),
        (9, false, 8),
        (10, false, 8),
        (0, true, 10),
        (11, false, 10),
    ] {
        let gpu = TestGpu::new();
        gpu.fail_alloc_at.store(allocation, Ordering::Relaxed);
        gpu.fail_required_kernel.store(kernel, Ordering::Relaxed);
        let error = configured_head(&gpu, true, true)
            .err()
            .expect("constructor must fail");
        assert!(format!("{error:#}").contains("injected"));
        let live = gpu.live_allocations();
        assert_eq!(live.len(), remaining);
        let original: Vec<_> = gpu
            .trace()
            .iter()
            .filter_map(|event| {
                if let gpu::Event::Alloc(ptr, bytes) = event {
                    Some((*ptr, *bytes))
                } else {
                    None
                }
            })
            .take(6)
            .collect();
        assert_eq!(original.len(), 6);
        for (ptr, bytes) in original {
            assert_eq!(live.get(&ptr.0), Some(&bytes));
        }
        assert!(!live.values().any(|&bytes| bytes == SLAB_BYTES));
        // The caller abandons this failed build, not retries a partial head.
        // The test explicitly reclaims its ledger; native backend Drop owns
        // that policy. This is not a transactional constructor rollback.
        gpu.synchronize(gpu.default_stream()).unwrap();
        for ptr in live.keys() {
            gpu.free(DevicePtr(*ptr)).unwrap();
        }
        assert!(gpu.live_allocations().is_empty());
    }
}

#[test]
fn recycled_slab_address_cannot_rebind_a_lease_from_a_closed_head() {
    use crate::speculative::glm_repair::GlmPairedHandoff;
    let gpu = TestGpu::new();
    gpu.reuse_freed.store(true, Ordering::Relaxed);
    let old = head(&gpu);
    let state = old.alloc_state_inner(&gpu).unwrap();
    let address = old.paired.as_ref().unwrap().lock().slab;
    old.close(&gpu, gpu.default_stream()).unwrap();
    let new = head(&gpu);
    assert_eq!(new.paired.as_ref().unwrap().lock().slab, address);
    let current = new.alloc_state_inner(&gpu).unwrap();
    assert_eq!(state.block_table, current.block_table);
    gpu.clear();
    assert!(new.validate_paired_live(&state, &gpu).is_err());
    assert!(old.validate_paired_live(&state, &gpu).is_err());
    new.validate_paired_live(&current, &gpu).unwrap();
    assert!(gpu.trace().is_empty());
}

#[test]
fn actual_indexed_constructor_sizes_all_256_blocks_and_rejects_slab_aliases() {
    let gpu = TestGpu::new();
    let head = configured_head(&gpu, true, true).unwrap();
    let cache = head.kv_cache.lock();
    let index = cache.sparse_index_config().unwrap();
    assert_eq!((index.tokens_per_pool, index.head_dim), (4, 128));
    assert_eq!(cache.num_blocks(), 256);
    let pointers = [
        (
            cache.sparse_index_pool_ptr(0),
            cache.sparse_index_block_stride_bytes(0),
        ),
        (
            cache.sparse_index_tail_pool_ptr(0),
            cache.sparse_index_tail_block_stride_bytes(0),
        ),
    ];
    assert_eq!(
        pointers.iter().map(|(_, bytes)| bytes).sum::<usize>(),
        index.block_bytes(16).unwrap()
    );
    assert!(cache.sparse_index_scale_pool_ptr(0).is_null());
    for (ptr, stride) in pointers {
        assert_eq!(gpu.read_alloc(ptr).unwrap().len(), 256 * stride);
        let before = gpu.live_allocations();
        gpu.clear();
        gpu.next_alloc_alias.store(ptr.0, Ordering::Relaxed);
        assert!(
            Pool::new(&gpu, 2044, &cache, 0).is_err(),
            "index owner is not slab authority"
        );
        assert_eq!(gpu.free_count(), 0);
        assert_eq!(gpu.live_allocations(), before);
    }
}

#[test]
fn legacy_free_refuses_paired_state_before_touching_either_block_ledger() {
    let gpu = TestGpu::new();
    let paired = head(&gpu);
    let legacy = configured_head(&gpu, false, false).unwrap();
    assert!(legacy.paired.is_none());
    assert_eq!(legacy.kv_cache.lock().num_blocks(), 128);
    let mut state = paired.alloc_state_inner(&gpu).unwrap();
    let held: Vec<_> = (0..128)
        .map(|_| legacy.kv_cache.lock().alloc_block().unwrap())
        .collect();
    let paired_before = state.block_table.clone();
    gpu.clear();
    assert!(
        legacy.free_state(&gpu, &mut state).is_err(),
        "legacy head must reject paired authority"
    );
    assert_eq!(legacy.kv_cache.lock().num_free_blocks(), 0);
    assert_eq!(paired.kv_cache.lock().num_free_blocks(), 128);
    assert_eq!(state.block_table, paired_before);
    assert!(gpu.trace().is_empty());
    legacy.kv_cache.lock().free_blocks(&held);
}

#[test]
fn actual_pool_alias_refusal_does_not_free_original_kv_owner() {
    for use_v in [false, true] {
        let gpu = TestGpu::new();
        let head = head(&gpu);
        let cache = head.kv_cache.lock();
        let original = if use_v {
            cache.v_cache_ptr(0, 0)
        } else {
            cache.k_cache_ptr(0, 0)
        };
        let before = gpu.live_allocations();
        gpu.clear();
        gpu.next_alloc_alias.store(original.0, Ordering::Relaxed);
        assert!(Pool::new(&gpu, 2044, &cache, 0).is_err());
        assert_eq!(
            gpu.free_count(),
            0,
            "known KV owner must never become slab cleanup ownership"
        );
        assert_eq!(gpu.live_allocations(), before);
        assert!(gpu.read_span(original, 16).is_ok());
        assert_eq!(gpu.alloc_count(), before.len());
    }
}

#[test]
fn actual_free_completion_failure_cannot_be_erased_by_later_success() {
    let gpu = TestGpu::new();
    let head = head(&gpu);
    let mut a = head.alloc_state_inner(&gpu).unwrap();
    let mut b = head.alloc_state_inner(&gpu).unwrap();
    gpu.clear();
    gpu.fail_sync_at.store(1, Ordering::Relaxed);
    assert!(head.free_state(&gpu, &mut a).is_err());
    gpu.clear();
    assert!(head.free_state(&gpu, &mut a).is_err());
    assert_eq!(gpu.sync_count(), 0);
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 0);
    head.free_state(&gpu, &mut b).unwrap();
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 128);
    assert!(head.alloc_state_inner(&gpu).is_ok());
    assert!(head.alloc_state_inner(&gpu).is_err());
}

#[test]
fn actual_paired_constructor_reserves_two_full_private_prefixes() {
    let gpu = MockGpuBackend::new();
    let head = head(&gpu);
    let cache = head.kv_cache.lock();
    assert_eq!(cache.num_blocks(), 256);
    assert_eq!(cache.num_free_blocks(), 256);
    assert_eq!(cache.block_size(), 16);
    assert_eq!(cache.config().dtype, KvCacheDtype::Bf16);
    let pool = head.paired.as_ref().unwrap().lock();
    assert_eq!(gpu.read_alloc(pool.slab).unwrap().len(), 98_304);
}

#[test]
fn actual_state_allocation_reserves_each_prefix_without_initializing_rows() {
    let gpu = MockGpuBackend::new();
    let head = head(&gpu);
    let a = head.alloc_state_inner(&gpu).unwrap();
    let b = head.alloc_state_inner(&gpu).unwrap();
    assert_eq!(
        a.block_table.len(),
        128,
        "request A must own its full reserve"
    );
    assert_eq!(b.block_table.len(), 128);
    assert_eq!((a.seq_len, b.seq_len), (0, 0));
    assert!(a.block_table.iter().all(|x| !b.block_table.contains(x)));
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 0);
    assert!(head.alloc_state_inner(&gpu).is_err());
    for state in [&a, &b] {
        let bytes = pack_mtp_attn_meta(0, 0, 0, &state.block_table, 768).unwrap();
        assert_eq!(bytes.len(), 768);
        assert_eq!(&bytes[16..20], &0i32.to_le_bytes());
        assert!(pack_mtp_attn_meta(0, 0, 0, &state.block_table, 767).is_err());
    }
}

#[test]
fn actual_free_retires_once_and_reuses_only_completed_owner() {
    let gpu = MockGpuBackend::new();
    let head = head(&gpu);
    let mut a = head.alloc_state_inner(&gpu).unwrap();
    let b = head.alloc_state_inner(&gpu).unwrap();
    let old_generation = a.paired.as_ref().unwrap().generation;
    let b_blocks = b.block_table.clone();
    head.free_state(&gpu, &mut a).unwrap();
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 128);
    head.free_state(&gpu, &mut a).unwrap();
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 128);
    let next = head
        .alloc_state_inner(&gpu)
        .expect("completed slot must be reusable");
    assert!(next.paired.as_ref().unwrap().generation > old_generation);
    assert_eq!(b.block_table, b_blocks);
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 0);
}

#[test]
fn actual_free_refuses_foreign_block_view_without_returning_peer_blocks() {
    let gpu = MockGpuBackend::new();
    let head = head(&gpu);
    let mut a = head.alloc_state_inner(&gpu).unwrap();
    let b = head.alloc_state_inner(&gpu).unwrap();
    a.block_table = b.block_table.clone();
    assert!(head.free_state(&gpu, &mut a).is_err());
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 0);
    assert!(head.alloc_state_inner(&gpu).is_err());
}

#[test]
fn partial_reserve_returns_only_blocks_claimed_by_this_attempt() {
    let gpu = MockGpuBackend::new();
    let head = head(&gpu);
    let held: Vec<_> = (0..130)
        .map(|_| head.kv_cache.lock().alloc_block().unwrap())
        .collect();
    assert!(head.alloc_state_inner(&gpu).is_err());
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 126);
    assert!(head.paired.as_ref().unwrap().lock().slots[0].failed);
    head.kv_cache.lock().free_blocks(&held);
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 256);
    let b = head.alloc_state_inner(&gpu).unwrap();
    assert_eq!(b.paired.as_ref().unwrap().slot, 1);
    assert!(head.alloc_state_inner(&gpu).is_err());
}

#[test]
fn foreign_backend_claim_does_no_allocation_or_reservation() {
    let gpu = MockGpuBackend::new();
    let head = head(&gpu);
    let foreign = MockGpuBackend::new();
    assert!(head.alloc_state_inner(&foreign).is_err());
    assert_eq!(foreign.alloc_count(), 0);
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 256);
}

#[test]
fn paired_constructor_refuses_legacy_trace_before_cache_or_slab_allocation() {
    if std::env::var("ATLAS_PAIRED_TRACE_TEST_CHILD").as_deref() != Ok("1") {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", "layers::glm5_mtp::paired::capacity_tests::paired_constructor_refuses_legacy_trace_before_cache_or_slab_allocation", "--nocapture"]);
        for name in [
            "ATLAS_PAIRED_TRACE_TEST_CHILD",
            "ATLAS_GLM_MTP_HIDDEN_TRACE",
            "ATLAS_GLM_MTP_REPAIR",
            "ATLAS_GLM_MTP_DISTRIBUTED",
            "ATLAS_GLM_MTP_BATCHED_PREFILL",
            "ATLAS_MTP_SPEC_THINK",
            "ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE",
        ] {
            cmd.env(name, "1");
        }
        for name in [
            "ATLAS_GLM_MTP_SERIAL_PREFILL",
            "ATLAS_NO_MTP_DRAFTER_CONTEXT",
            "ATLAS_MTP_SINGLE_DEPTH_ADAPT",
            "ATLAS_DFLASH_ADAPTIVE",
            "ATLAS_DFLASH_RESUME_GUARD",
            "ATLAS_MTP_CATCHUP",
            "ATLAS_MTP_REFEED_ACCEPTED",
            "ATLAS_DRAFT_CONF_TAU",
        ] {
            cmd.env_remove(name);
        }
        assert!(cmd.status().unwrap().success());
        return;
    }
    assert!(hidden_trace::configured().unwrap());
    let gpu = MockGpuBackend::new();
    assert!(
        build_head(&gpu).is_err(),
        "paired head must not accept C1 trace"
    );
    // Six input weights were constructed by this fixture; neither cache nor
    // slab may have been allocated by the rejecting actual constructor.
    assert_eq!(gpu.alloc_count(), 6);
}
