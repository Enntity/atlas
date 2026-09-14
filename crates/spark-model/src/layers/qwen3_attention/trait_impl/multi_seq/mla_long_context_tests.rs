// SPDX-License-Identifier: AGPL-3.0-only
//! Actual layer dispatch; recorded kernels do not establish CUDA numerics.
use super::*;
use crate::layer::ForwardContext;
use crate::layers::{
    FfnComponent,
    qwen3_attention::{GlmIndexerWeights, MlaWeights},
};
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{GpuBackend, KernelHandle, mock::MockGpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, SparseIndexCacheConfig};
#[path = "mla_long_context_test_gpu.rs"]
mod gpu;
use gpu::TestGpu;
#[path = "mla_split_context_tests.rs"]
mod split_tests;
fn fixture(run: impl FnOnce(&TestGpu, &atlas_core::config::ModelConfig, &Qwen3AttentionLayer)) {
    let gpu = TestGpu::default();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.num_hidden_layers = 1;
    config.layer_types = vec![atlas_core::config::LayerType::FullAttention];
    config.num_attention_heads = 32;
    config.num_key_value_heads = 32;
    config.head_dim = 256;
    config.q_lora_rank = 1536;
    config.kv_lora_rank = 512;
    config.qk_nope_head_dim = 256;
    config.qk_rope_head_dim = 0;
    config.v_head_dim = 256;
    config.index_n_heads = 32;
    config.index_head_dim = 128;
    config.index_topk = 2048;
    config.index_kpool = 4;
    config.max_position_embeddings = 32768;
    config.tp_world_size = 2;
    config.ep_world_size = 2;
    let dense = |_bytes| DenseWeight {
        weight: gpu.alloc(2).unwrap(),
    };
    let absent = DenseWeight {
        weight: DevicePtr::NULL,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        dense(8192),
        AttentionWeights {
            q_proj: dense(32 * 256 * 4096 * 2),
            k_proj: dense(32 * 256 * 4096 * 2),
            v_proj: dense(32 * 256 * 4096 * 2),
            o_proj: QuantizedWeight::null(),
            q_norm: dense(512),
            k_norm: dense(512),
            q_norm_full: None,
            k_norm_full: None,
            k_scale: 1.0,
            v_scale: 1.0,
        },
        dense(8192),
        FfnComponent::None,
        0,
        None,
        None,
        None,
        &gpu,
        KvCacheDtype::Bf16,
        0,
        &config,
    )
    .unwrap();
    layer.set_mla_weights(MlaWeights {
        wq_a: dense(1536 * 4096 * 2),
        wq_a_nvfp4: None,
        wq_a_fp8: None,
        wq_b: dense(32 * 256 * 1536 * 2),
        wq_b_nvfp4: None,
        wq_b_fp8: None,
        q_a_norm: dense(1536 * 2),
        wkv_a: dense(512 * 4096 * 2),
        wkv_a_nvfp4: None,
        wkv_a_fp8: None,
        wkv_b: dense(32 * 512 * 512 * 2),
        kv_a_norm: dense(512 * 2),
        wkv_a_rope: absent,
        wkv_a_merged: dense(512 * 4096 * 2),
        // Comparison mode validates the real operand range, not a two-byte stub.
        wo: DenseWeight {
            weight: gpu.alloc(4096 * 32 * 256 * 2).unwrap(),
        },
        wo_nvfp4: None,
        wo_a: absent,
        wo_a_nvfp4: None,
        wo_a_fp8: None,
        wo_b: absent,
        wo_b_nvfp4: None,
        wo_b_fp8: None,
        w_uk_t: dense(32 * 256 * 512 * 2),
        w_uv: dense(32 * 512 * 256 * 2),
        wq_b_rope: absent,
        w_qk_absorbed: absent,
        w_uk_block_diag: absent,
        w_uv_block_diag: absent,
        yarn_inv_freq: DevicePtr::NULL,
        main_inv_freq: DevicePtr::NULL,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        o_lora_rank: 0,
        nope: 256,
        rope: 0,
        v_dim: 256,
        glm_indexer: Some(GlmIndexerWeights {
            wq_b: dense(2),
            wk: dense(2),
            weights_proj: dense(2),
            kpool_gate: dense(2),
            kpool_ape: dense(2),
            k_norm_weight: dense(2),
            k_norm_bias: dense(2),
        }),
        compressor: None,
        attn_sink: DevicePtr::NULL,
    });

    layer.dense_gemv_k = KernelHandle(806);
    layer.dense_gemv_batchm_k = KernelHandle(807);
    layer.mla_batched_gemv_k = KernelHandle(808);
    layer.glm_index_tail_write_k = KernelHandle(801);
    layer.glm_index_kpool_finalize_k = KernelHandle(802);
    layer.glm_index_topk_expand_k = KernelHandle(803);
    layer.glm_sparse_attn_decode_k = KernelHandle(804);
    run(&gpu, &config, &layer);
}
#[test]
fn actual_prompt_index_and_causal_verify_dispatch() {
    const CHILD: &str = "ATLAS_TEST_LONG_MTP_ATTENTION";
    if std::env::var_os(CHILD).is_none() {
        let name = concat!(
            module_path!(),
            "::actual_prompt_index_and_causal_verify_dispatch"
        );
        for (tc, o_batch, compare) in [
            ("0", "0", "0"),
            ("1", "0", "0"),
            ("0", "1", "0"),
            ("1", "1", "0"),
            ("0", "1", "1"),
            ("1", "1", "1"),
        ] {
            let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
            cmd.args(["--exact", name.split_once("::").unwrap().1, "--nocapture"]);
            for (k, _) in std::env::vars_os() {
                if k.to_string_lossy().starts_with("ATLAS_") {
                    cmd.env_remove(k);
                }
            }
            let out = cmd
                .env(CHILD, "1")
                .env("ATLAS_GLM_MTP_LONG_CONTEXT", "1")
                .env("ATLAS_GLM_MTP_REPAIR", "1")
                .env("ATLAS_GLM_SPARSE_DECODE_TC", tc)
                .env("ATLAS_GLM_K3_MLA_O_BATCHM", o_batch)
                .env("ATLAS_GLM_K3_MLA_O_COMPARE", compare)
                .output()
                .unwrap();
            assert!(String::from_utf8_lossy(&out.stdout).contains("running 1 test"));
            assert!(
                out.status.success(),
                "TC={tc}, batchm={o_batch}, compare={compare}: {}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        return;
    }
    fixture(|gpu, config, layer| {
        if std::env::var("ATLAS_GLM_K3_MLA_O_BATCHM").as_deref() == Ok("1") {
            use crate::layers::qwen3_attention::glm_k3_mla_o::initialize;
            let ptr = layer.mla.as_ref().unwrap().wo.weight;
            initialize(gpu, config, KvCacheDtype::Bf16, ptr, 4096, 8192).unwrap();
            let mut wrong = config.clone();
            wrong.num_attention_heads = 64;
            assert!(initialize(gpu, &wrong, KvCacheDtype::Bf16, ptr, 4096, 8192).is_err());
            assert!(initialize(gpu, config, KvCacheDtype::Fp8, ptr, 4096, 8192).is_err());
        }
        let sparse_kernel = if ops::glm_sparse_decode_tc_enabled(&config.model_type).unwrap() {
            gpu.kernel(
                "glm_sparse_prefill_kv_reuse",
                "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad",
            )
            .unwrap()
            .0
        } else {
            804
        };
        let arena = BufferArena::new(config, 16, 32768, 16, 1, gpu).unwrap();
        let mut dispatch = ops::GemmDispatch::defaults();
        dispatch.cublas_gemm = false;
        let derived = ops::DerivedWeights::new();
        let levers = ops::ModelLevers::defaults();
        let stats = ops::ModelStats::new();
        let ptr = gpu.alloc(3 * 2049 * 4 + 1024).unwrap();
        let meta = AttnMetadataDev {
            positions: ptr,
            positions_h: ptr,
            positions_w: ptr,
            slot: ptr.offset(256),
            seq_len: ptr.offset(512),
            block_table: ptr.offset(768),
            max_blocks_per_seq: 2049,
            num_seqs: 3,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let ctx = ForwardContext {
            buffers: &arena,
            gpu,
            config,
            dispatch: &dispatch,
            derived: &derived,
            levers: &levers,
            stats: &stats,
            ssm_batch: None,
            attn_metadata: Some(meta),
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let mut cache = PagedKvCache::new(
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
            2050,
            gpu,
        )
        .unwrap();
        cache
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
            .unwrap();
        let before = gpu.launch_count();
        layer
            .prefill_mla_kv_only_impl(arena.hidden_states(), 3, &mut cache, meta.slot, &ctx, 0)
            .unwrap();
        let launches = gpu.launches_snapshot();
        let index: Vec<_> = launches[before..]
            .iter()
            .filter(|x| (801..=804).contains(&x.func))
            .map(|x| (x.func, x.grid[0]))
            .collect();
        assert_eq!(
            index,
            vec![(801, 3), (802, 3)],
            "KV-only prompt/repair must populate raw and pooled index"
        );
        for positions in [[2046, 2048, 2049], [32767, 32768, 32769]] {
            let c = MultiSeqCtx::new(
                layer,
                &ctx,
                arena.hidden_states(),
                arena.residual(),
                3,
                &positions,
                16,
                0,
            );
            let before = gpu.launch_count();
            assert!(layer.ms_mla_decode(&c, &mut cache, meta).is_err());
            assert_eq!(
                gpu.launch_count(),
                before,
                "bad row extents must refuse before kernels"
            );
        }
        let graph_ctx = ForwardContext {
            graph_capture: true,
            midchunk_capture: None,
            ..ctx
        };
        let positions = [2046, 2047, 2048];
        let c = MultiSeqCtx::new(
            layer,
            &graph_ctx,
            arena.hidden_states(),
            arena.residual(),
            3,
            &positions,
            16,
            0,
        );
        let before = gpu.launch_count();
        assert!(layer.ms_mla_decode(&c, &mut cache, meta).is_err());
        assert_eq!(gpu.launch_count(), before);
        for positions in [[2046, 2047, 2048], [32764, 32765, 32766]] {
            let c = MultiSeqCtx::new(
                layer,
                &ctx,
                arena.hidden_states(),
                arena.residual(),
                3,
                &positions,
                16,
                0,
            );
            if std::env::var("ATLAS_GLM_K3_MLA_O_BATCHM").as_deref() == Ok("1") {
                let compare = std::env::var("ATLAS_GLM_K3_MLA_O_COMPARE").as_deref() == Ok("1");
                for bad in 0..if compare { 3 } else { 2 } {
                    let mut rejected = MultiSeqCtx::new(
                        layer,
                        &ctx,
                        arena.hidden_states(),
                        arena.residual(),
                        3,
                        &positions,
                        16,
                        0,
                    );
                    if bad == 0 {
                        rejected.nq = 64;
                    } else if bad == 1 {
                        rejected.normed = arena.ssm_qkvz().offset(512);
                    } else {
                        rejected.normed = arena.attn_output();
                    }
                    let count = gpu.launch_count();
                    assert!(layer.ms_mla_decode(&rejected, &mut cache, meta).is_err());
                    assert_eq!(
                        gpu.launch_count(),
                        count,
                        "bad O plan must reject before cache/index writes"
                    );
                }
            }
            let before = gpu.launch_count();
            let args_before = gpu.1.lock().unwrap().len();
            let alloc_before = gpu.alloc_count();
            let copies_before = gpu.d2h_blocking_count();
            layer.ms_mla_decode(&c, &mut cache, meta).unwrap();
            assert_eq!(gpu.alloc_count(), alloc_before, "no forward allocation");
            let args = gpu.1.lock().unwrap();
            let calls = &args[args_before..];
            let batched = std::env::var("ATLAS_GLM_K3_MLA_O_BATCHM").as_deref() == Ok("1");
            let compare = std::env::var("ATLAS_GLM_K3_MLA_O_COMPARE").as_deref() == Ok("1");
            assert_eq!(
                gpu.d2h_blocking_count() - copies_before,
                if compare { 2 } else { 0 },
                "only diagnostic mode reads both full output buffers"
            );
            let o_weight = layer
                .mla
                .as_ref()
                .unwrap()
                .wo
                .weight
                .0
                .to_ne_bytes()
                .to_vec();
            let o_calls: Vec<_> = calls
                .iter()
                .enumerate()
                .filter(|(_, (k, a))| (*k == 806 || *k == 807) && a[1] == o_weight)
                .collect();
            assert_eq!(
                o_calls.len(),
                if compare {
                    4
                } else if batched {
                    1
                } else {
                    3
                }
            );
            let extracted: Vec<_> = calls
                .iter()
                .enumerate()
                .filter(|(_, (k, a))| *k == 808 && a[3] == 256_u32.to_ne_bytes())
                .collect();
            assert_eq!(extracted.len(), 3);
            let prefix_writes: Vec<_> = calls
                .iter()
                .filter(|(k, a)| {
                    *k == 806
                        && (a[2] == arena.ssm_qkvz().0.to_ne_bytes()
                            || a[2] == arena.ssm_qkvz().offset(256).0.to_ne_bytes())
                        && a[3] == 128_u32.to_ne_bytes()
                })
                .collect();
            assert_eq!(
                prefix_writes.len(),
                6,
                "three serial key/gate writes stay in512-byte prefix"
            );
            for (i, (_, (_, a))) in extracted.iter().enumerate() {
                let offset = if batched { 512 + i * 8192 * 2 } else { 0 };
                assert_eq!(a[2], arena.ssm_qkvz().offset(offset).0.to_ne_bytes());
            }
            if compare {
                let scalar: Vec<_> = o_calls.iter().filter(|(_, (k, _))| *k == 806).collect();
                assert_eq!(scalar.len(), 3);
                for i in 0..3 {
                    assert_eq!(
                        scalar[i].0,
                        extracted[i].0 + 1,
                        "scalar baseline must immediately follow its V extract"
                    );
                    let a = &scalar[i].1.1;
                    assert_eq!(
                        a[0],
                        arena.ssm_qkvz().offset(512 + i * 8192 * 2).0.to_ne_bytes()
                    );
                    assert_eq!(
                        a[2],
                        arena.moe_output().offset(i * 4096 * 2).0.to_ne_bytes()
                    );
                }
            }
            if batched {
                let batched_o = o_calls.iter().find(|(_, (k, _))| *k == 807).unwrap();
                assert!(
                    batched_o.0 > extracted[2].0,
                    "O must follow third V extraction"
                );
                assert_eq!(calls.last().unwrap().0, 807);
                let a = &batched_o.1.1;
                assert_eq!(a[0], arena.ssm_qkvz().offset(512).0.to_ne_bytes());
                let expected_output = if compare {
                    arena.attn_output()
                } else {
                    arena.moe_output()
                };
                assert_eq!(a[2], expected_output.0.to_ne_bytes());
                assert_eq!(
                    &a[3..],
                    &[3_u32, 4096, 8192, 4096].map(|v| v.to_ne_bytes().to_vec())
                );
            }
            drop(args);
            let launches = gpu.launches_snapshot();
            let index: Vec<_> = launches[before..]
                .iter()
                .filter(|x| (801..=804).contains(&x.func) || x.func == sparse_kernel)
                .map(|x| x.func)
                .collect();
            let mut expected = vec![];
            for pos in positions {
                expected.extend([801, 802]);
                if pos + 1 > 2048 {
                    expected.extend([803, sparse_kernel]);
                }
            }
            assert_eq!(
                index, expected,
                "causal rows must maintain index before selecting sparse attention"
            );
        }
    });
}
