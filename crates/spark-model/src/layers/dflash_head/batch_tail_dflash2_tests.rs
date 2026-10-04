// SPDX-License-Identifier: AGPL-3.0-only
//! Selector dispatch and typed ABI regressions after upstream integration.
use std::sync::Mutex;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::DenseWeight;

use super::BlockDiffusionDraftHead;
use super::free_state_tests::zero_head;
use super::selector::Dflash2CandidateSelector;

#[derive(Debug, Clone, PartialEq)]
enum RecordedArg {
    Buffer(DevicePtr),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone)]
struct RecordedLaunch {
    func: u64,
    grid: [u32; 3],
    args: Vec<RecordedArg>,
}

/// MockGpuBackend records function/grid/block but intentionally drops typed
/// arguments. Keep its allocation/copy behavior and add only the typed-ABI
/// recording needed to prove the per-sequence anchor/ban pointers.
struct RecordingGpu {
    inner: MockGpuBackend,
    launches: Mutex<Vec<RecordedLaunch>>,
}

impl RecordingGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            launches: Mutex::new(Vec::new()),
        }
    }

    fn alloc(&self, bytes: usize) -> DevicePtr {
        self.inner.alloc(bytes).unwrap()
    }

    fn launches(&self) -> Vec<RecordedLaunch> {
        self.launches.lock().unwrap().clone()
    }
}

impl GpuBackend for RecordingGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.inner.free(ptr)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(src, dst)
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.inner
            .launch(func, grid, block, shared_mem, stream, params)
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        self.launches.lock().unwrap().push(RecordedLaunch {
            func: func.0,
            grid,
            args: args
                .iter()
                .map(|arg| match arg {
                    KernelArg::Buffer(ptr) => RecordedArg::Buffer(*ptr),
                    KernelArg::Bytes(bytes) => RecordedArg::Bytes(bytes.to_vec()),
                })
                .collect(),
        });
        // The mock's default typed implementation packs these same args and
        // delegates to its launch recorder; no real CUDA work occurs.
        self.inner
            .launch_typed(func, grid, block, shared_mem, stream, args)
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        self.inner.synchronize(stream)
    }

    fn default_stream(&self) -> u64 {
        self.inner.default_stream()
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        self.inner.kernel(module, func_name)
    }

    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
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

fn ptr(arg: &RecordedArg) -> DevicePtr {
    match arg {
        RecordedArg::Buffer(ptr) => *ptr,
        RecordedArg::Bytes(_) => panic!("expected a device-buffer argument"),
    }
}

fn selector_launches(gpu: &RecordingGpu) -> Vec<RecordedLaunch> {
    gpu.launches()
        .into_iter()
        .filter(|launch| launch.func == 10 || launch.func == 11)
        .collect()
}

fn make_head(gpu: &RecordingGpu, batch_size: usize, gamma: usize) -> BlockDiffusionDraftHead {
    let hidden = 8usize;
    let vocab = 8usize;
    let rank = 2usize;
    let mut head = zero_head();

    head.hidden_size = hidden;
    head.vocab_size = vocab;
    head.gamma = gamma;
    head.batch_capacity = batch_size;
    head.drafter_cublas = false;

    head.candidate_selector = Some(Dflash2CandidateSelector {
        hidden_projection: DenseWeight {
            weight: gpu.alloc(rank * hidden * 2),
        },
        predecessor_codebook: DenseWeight {
            weight: gpu.alloc(vocab * rank * 2),
        },
        successor_codebook: DenseWeight {
            weight: gpu.alloc(vocab * rank * 2),
        },
        predecessor_host: None,
        successor_host: None,
        rank,
        top_k: 1,
        vocab_size: vocab,
        hidden_size: hidden,
    });

    head.batch_norm = gpu.alloc(batch_size * gamma * hidden * 2);
    head.batch_logits = gpu.alloc(batch_size * gamma * vocab * 2);
    head.batch_tokens = gpu.alloc(batch_size * gamma * 4);
    // [batch_capacity anchors][batch_capacity ban-depth words].
    head.batch_markov_prev = gpu.alloc(batch_size * 8);
    head.batch_dflash2_projected = gpu.alloc(batch_size * gamma * rank * 2);
    head.batch_dflash2_selector_scratch = gpu.alloc(4096);

    // Distinct handles let the test ignore the projection launch and assert
    // the selector ABI branch directly.
    head.kernels.dflash2_candidate_selector_batched = KernelHandle(10);
    head.kernels.dflash2_candidate_selector = Some(KernelHandle(11));
    head.kernels.dense_gemm_pipelined = KernelHandle(12);
    head.kernels.dense_gemv_batchm = KernelHandle(0);
    head.kernels.dense_gemv_tc16 = KernelHandle(0);
    head.kernels.dense_gemv_tc32 = KernelHandle(0);
    head.kernels.small_m_gemv = false;

    head
}

fn with_fixture(
    gamma: usize,
    run: impl FnOnce(&mut BlockDiffusionDraftHead, &ForwardContext<'_>, &RecordingGpu),
) {
    let gpu = RecordingGpu::new();
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let buffers = BufferArena::new(&config, 2, 16, 16, 1, &gpu).unwrap();
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
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
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Skip,
    };
    let mut head = make_head(&gpu, 2, gamma);
    run(&mut head, &ctx, &gpu);
}

#[test]
fn all_zero_bans_use_one_legacy_batched_selector_launch() {
    with_fixture(4, |head, ctx, gpu| {
        head.run_batched_dflash2_tail(2, &[101, 202], &[0, 0], ctx, 7)
            .unwrap();
        let launches = selector_launches(gpu);
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].func, 10);
        assert_eq!(launches[0].grid[0], 2);
    });
}

#[test]
fn mixed_bans_use_single_selector_per_sequence_with_matching_pointers() {
    with_fixture(4, |head, ctx, gpu| {
        head.run_batched_dflash2_tail(2, &[101, 202], &[0, 1], ctx, 7)
            .unwrap();
        let launches = selector_launches(gpu);
        assert_eq!(launches.len(), 2);
        assert!(launches.iter().all(|launch| launch.func == 11));

        for (sequence, launch) in launches.iter().enumerate() {
            // dflash2_candidate_selector's real pointer order is:
            // logits, projected, predecessor, successor, draft_tokens,
            // anchor, ban_depth, then scalar/end-id arguments.
            assert_eq!(
                ptr(&launch.args[5]),
                head.batch_markov_prev.offset(sequence * 4)
            );
            assert_eq!(
                ptr(&launch.args[6]),
                head.batch_ban_depth().offset(sequence * 4)
            );
        }
    });
}

#[test]
fn projected_hidden_uses_gamma_tc_tier_per_sequence() {
    with_fixture(16, |head, ctx, gpu| {
        // gamma=16 should use the TC16 arm twice. A single aggregate B*gamma
        // projection would select TC32 (m=32), changing accumulation order.
        head.kernels.small_m_gemv = true;
        head.kernels.dense_gemv_tc16 = KernelHandle(20);
        head.kernels.dense_gemv_tc32 = KernelHandle(21);
        head.kernels.dense_gemm_pipelined = KernelHandle(22);

        head.run_batched_dflash2_tail(2, &[101, 202], &[0, 0], ctx, 7)
            .unwrap();
        let projection: Vec<_> = gpu
            .launches()
            .into_iter()
            .filter(|launch| matches!(launch.func, 20..=22))
            .collect();
        assert_eq!(projection.len(), 2);
        assert!(projection.iter().all(|launch| launch.func == 20));
    });
}
