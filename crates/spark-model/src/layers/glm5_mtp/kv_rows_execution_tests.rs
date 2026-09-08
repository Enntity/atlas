// SPDX-License-Identifier: AGPL-3.0-only

//! Exercise the production entrypoint and chain. Mock kernels only record
//! launches; the body stand-in checks the real uploaded paged destinations.

use super::*;
use crate::layer::{EmptyLayerState, TransformerLayer};
use spark_runtime::{buffers::BufferArena, gpu::mock::MockGpuBackend};
use std::sync::Arc;

#[path = "prompt_writer_tests.rs"]
mod prompt_tests;
#[path = "repair_execution_tests.rs"]
mod repair_tests;

struct SlotBody(Arc<Mutex<Vec<i64>>>, bool);
impl TransformerLayer for SlotBody {
    fn supports_mla_kv_only(&self) -> bool {
        self.1
    }
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
        anyhow::bail!("the KV-only chain must not execute full decode")
    }
    fn prefill_mla_kv_only(
        &self,
        _: DevicePtr,
        rows: usize,
        cache: &mut PagedKvCache,
        slots: DevicePtr,
        ctx: &ForwardContext,
        _: u64,
    ) -> Result<bool> {
        anyhow::ensure!(
            !self.0.lock().contains(&-1),
            "injected actual KV body failure"
        );
        assert!(ctx.comm.is_none() && !ctx.graph_capture);
        let mut raw = vec![0; rows * 8];
        ctx.gpu.copy_d2h(slots, &mut raw)?;
        for bytes in raw.chunks_exact(8) {
            let slot = i64::from_le_bytes(bytes.try_into().unwrap());
            self.0.lock().push(slot);
            let block = slot as usize / cache.block_size();
            let offset = slot as usize % cache.block_size() * 1024;
            // Sentinel proves actual block+row addressing, not CUDA numerics.
            ctx.gpu.copy_h2d(
                &[0xab; 1024],
                cache.k_cache_ptr(0, block as u32).offset(offset),
            )?;
            ctx.gpu.copy_h2d(
                &[0xab; 1024],
                cache.v_cache_ptr(0, block as u32).offset(offset),
            )?;
        }
        Ok(true)
    }
}

fn fixture(
    run: impl FnOnce(&Glm5MtpHead, &ForwardContext, &MockGpuBackend, &Arc<Mutex<Vec<i64>>>),
) {
    fixture_rows(2, run);
}

fn fixture_rows(
    max_rows: usize,
    run: impl FnOnce(&Glm5MtpHead, &ForwardContext, &MockGpuBackend, &Arc<Mutex<Vec<i64>>>),
) {
    fixture_capable(true, max_rows, run);
}

fn fixture_capable(
    capable: bool,
    max_rows: usize,
    run: impl FnOnce(&Glm5MtpHead, &ForwardContext, &MockGpuBackend, &Arc<Mutex<Vec<i64>>>),
) {
    fixture_geometry(capable, max_rows, 512, 64, run);
}
fn fixture_geometry(
    capable: bool,
    max_rows: usize,
    hidden: usize,
    context: usize,
    run: impl FnOnce(&Glm5MtpHead, &ForwardContext, &MockGpuBackend, &Arc<Mutex<Vec<i64>>>),
) {
    let gpu = MockGpuBackend::new();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = hidden;
    config.vocab_size = 8;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.kv_lora_rank = 512;
    config.qk_rope_head_dim = 0;
    config.intermediate_size = 1024;
    config.moe_intermediate_size = 1024;
    config.num_experts = 2;
    config.linear_num_key_heads = 64;
    config.linear_num_value_heads = 64;
    config.linear_key_head_dim = 4;
    config.linear_value_head_dim = 4;
    if hidden > 512 {
        config.num_attention_heads = 32;
        config.head_dim = 128;
        config.linear_key_head_dim = hidden / 64;
        config.linear_value_head_dim = hidden / 64;
    }
    config.index_kpool = 0;
    config.index_head_dim = 0;
    config.index_topk = 2048;
    let dense = |n| DenseWeight {
        weight: gpu.alloc(n).unwrap(),
    };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let module = Glm5MtpModule {
        body: Box::new(SlotBody(seen.clone(), capable)),
        enorm: dense(hidden * 2),
        hnorm: dense(hidden * 2),
        eh_proj: dense(hidden * hidden * 4),
        eh_proj_nvfp4: None,
        norm: dense(hidden * 2),
    };
    let head = Glm5MtpHead::new(
        module,
        dense(8 * hidden * 2),
        dense(8 * hidden * 2),
        None,
        &config,
        &gpu,
        8,
        context,
    )
    .unwrap();
    for token in 0..8 {
        gpu.copy_h2d(
            &vec![token as u8; hidden * 2],
            head.embed_tokens.weight.offset(token * hidden * 2),
        )
        .unwrap();
    }
    let buffers = BufferArena::new(&config, max_rows, context, 16, 1, &gpu).unwrap();
    let mut dispatch = ops::GemmDispatch::defaults();
    dispatch.cublas_gemm = false;
    let derived = ops::DerivedWeights::new();
    let levers = ops::ModelLevers::defaults();
    let stats = ops::ModelStats::new();
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
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Skip,
    };
    run(&head, &ctx, &gpu, &seen);
}

#[test]
fn primer_keeps_shift_chunking_block_zero_and_noop_contract() {
    fixture(|head, ctx, gpu, seen| {
        let source = gpu.alloc(6 * 1024).unwrap().offset(1024);
        let mut state = head.alloc_state_inner(gpu).unwrap();
        let allocations = gpu.alloc_count();
        assert_eq!(
            head.prefill_kv_batched(&[7, 1, 3, 2, 7, 0], source, &mut state, ctx, 0)
                .unwrap(),
            5
        );
        assert_eq!(state.seq_len, 5);
        assert_eq!(state.last_num_drafted, 0);
        assert_eq!(state.block_table, [0]);
        assert_eq!(*seen.lock(), [0, 1, 2, 3, 4]);
        assert_eq!(gpu.d2d_count(), 5);
        assert_eq!(gpu.launch_count(), 14); // three chunks: 2 RMS + row concat + EH
        assert_eq!(gpu.sync_count(), 3);
        assert_eq!(gpu.alloc_count(), allocations);
        // The last one-row chunk must use prompt[5], not prompt[4].
        assert_eq!(
            &gpu.read_alloc(ctx.buffers.ssm_deinterleaved()).unwrap()[..1024],
            &[0; 1024]
        );
        assert_eq!(
            head.prefill_kv_batched(&[7, 1, 3], source, &mut state, ctx, 0)
                .unwrap(),
            0
        );
        assert_eq!(gpu.launch_count(), 14);
    });
}

#[test]
fn arbitrary_rows_use_owned_shuffled_blocks_and_preserve_other_rows() {
    fixture(|head, ctx, gpu, seen| {
        let blocks: Vec<_> = (0..3)
            .map(|_| head.kv_cache.lock().alloc_block().unwrap())
            .collect();
        let source = DeviceSpan {
            ptr: gpu.alloc(5 * 1024).unwrap(),
            bytes: 5 * 1024,
        };
        head.write_kv_rows(
            &[1, 3, 2, 7, 0],
            source,
            15,
            &[blocks[2], blocks[0]],
            ctx,
            0,
        )
        .unwrap();
        assert_eq!(*seen.lock(), [47, 0, 1, 2, 3]);
        let cache = head.kv_cache.lock();
        let k = gpu.read_alloc(cache.k_cache_ptr(0, 0)).unwrap();
        assert_eq!(&k[..4096], &[0xab; 4096]);
        assert!(k[4096..47 * 1024].iter().all(|&b| b == 0));
        assert_eq!(&k[47 * 1024..48 * 1024], &[0xab; 1024]);
        assert!(k[48 * 1024..].iter().all(|&b| b == 0));
    });
}

#[test]
fn malformed_requests_do_not_launch_copy_allocate_or_mutate_state() {
    fixture(|head, ctx, gpu, seen| {
        let source = DeviceSpan {
            ptr: gpu.alloc(5 * 1024).unwrap(),
            bytes: 5 * 1024,
        };
        let mut state = head.alloc_state_inner(gpu).unwrap();
        let allocations = gpu.alloc_count();
        let free = head.kv_cache.lock().num_free_blocks();
        assert!(
            head.prefill_kv_batched(&[0, 1, 2, 8], source.ptr, &mut state, ctx, 0)
                .is_err()
        );
        assert!(state.block_table.is_empty());
        assert_eq!(state.seq_len, 0);
        assert_eq!(head.kv_cache.lock().num_free_blocks(), free);
        let block = head.kv_cache.lock().alloc_block().unwrap();
        for (tokens, span, row, table) in [
            (vec![1, 8], source, 0, vec![block]),
            (
                vec![1],
                DeviceSpan {
                    ptr: source.ptr,
                    bytes: 7,
                },
                0,
                vec![block],
            ),
            (
                vec![1],
                DeviceSpan {
                    ptr: ctx.buffers.norm_output(),
                    bytes: 1024,
                },
                0,
                vec![block],
            ),
            (vec![1], source, usize::MAX, vec![block]),
            (vec![1], source, 0, vec![block, block]),
            (vec![1], source, 0, vec![4]), // in range but unallocated
        ] {
            assert!(
                head.write_kv_rows(&tokens, span, row, &table, ctx, 0)
                    .is_err()
            );
        }
        head.kv_cache.lock().inc_ref(block);
        assert!(
            head.write_kv_rows(&[1], source, 0, &[block], ctx, 0)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        assert_eq!(gpu.d2d_count(), 0);
        assert_eq!(gpu.sync_count(), 0);
        assert_eq!(gpu.alloc_count(), allocations);
        assert!(seen.lock().is_empty());
    });
}

#[test]
fn unsupported_body_is_rejected_before_gpu_work_or_primer_allocation() {
    fixture_capable(false, 2, |head, ctx, gpu, seen| {
        let source = gpu.alloc(2048).unwrap();
        let mut state = head.alloc_state_inner(gpu).unwrap();
        let free = head.kv_cache.lock().num_free_blocks();
        let allocations = gpu.alloc_count();
        assert!(
            head.prefill_kv_batched(&[0, 1, 2], source, &mut state, ctx, 0)
                .is_err()
        );
        assert!(state.block_table.is_empty());
        assert_eq!(state.seq_len, 0);
        assert_eq!(head.kv_cache.lock().num_free_blocks(), free);
        assert_eq!(gpu.alloc_count(), allocations);
        assert_eq!(
            (gpu.launch_count(), gpu.d2d_count(), gpu.sync_count()),
            (0, 0, 0)
        );
        assert!(seen.lock().is_empty());
    });
}

#[test]
fn resident_oracle_executes_real_reference_and_writer_with_shuffled_blocks() {
    fixture(|head, ctx, gpu, seen| {
        let source = DeviceSpan {
            ptr: gpu.alloc(4096).unwrap(),
            bytes: 4096,
        };
        let mut cache = head.kv_cache.lock();
        for _ in 0..3 {
            cache.alloc_block().unwrap();
        }
        let blocks = [1, 2, 0];
        let tokens = [1, 4, 2, 3];
        let plan = KvRowsPlan::new(
            &tokens,
            source,
            512,
            8,
            31,
            16,
            &blocks,
            cache.num_blocks(),
            2,
            ctx.buffers.scratch_bytes(),
            &[],
        )
        .unwrap();
        let allocations = gpu.alloc_count();
        head.verify_kv_rows(&tokens, &plan, source, &blocks, &mut cache, ctx, 0)
            .unwrap();
        assert_eq!(*seen.lock(), [16, 17, 18, 19, 47, 0, 1, 2]);
        assert_eq!(gpu.launch_count(), 20);
        assert_eq!(gpu.alloc_count(), allocations);
        let (reference_k, reference_v) = cache.read_block(0, 1, gpu).unwrap();
        assert!(reference_k.iter().chain(&reference_v).all(|&b| b == 0));
    });
}
