// SPDX-License-Identifier: AGPL-3.0-only
//! Actual MLA constructor/trait/arena; recorded dispatch is not CUDA arithmetic.
use super::*;
use crate::layer::glm_owner_verify::{GlmOwnerBatchShape, GlmOwnerBatchWorkspace};
use crate::layer::glm_pair_verify::{GlmPairFfn, GlmPairLayerInput, GlmPairWorkspace};
use crate::layers::qwen3_attention::{HcHeadWeights, MlaWeights, Qwen3AttentionLayer};
use crate::weight_map::AttentionWeights;

#[path = "paged_glm_projection_tests.rs"]
mod paged_projection;

struct ForeignState;
impl LayerState for ForeignState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn with_mla(
    rank: usize,
    run: impl FnOnce(&Gpu, &atlas_core::config::ModelConfig, &Qwen3AttentionLayer),
) {
    let gpu = Gpu::new();
    let (_store, mut config, mut ffn) = resident_tests::setup(&gpu, rank);
    ffn.transpose_for_prefill_unified_keep_shared(&gpu, &config)
        .unwrap();
    ffn.unified_layout = true;
    ffn.nvfp4_fused_silu_quant = true;
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
    config.hc_mult = 4;
    let dense = |bytes| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
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
        FfnComponent::Moe(ffn),
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
        wo: dense(4096 * 32 * 256 * 2),
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
        glm_indexer: None,
        compressor: None,
        attn_sink: DevicePtr::NULL,
    });
    let site = || HcSiteWeights {
        hc_fn: gpu.alloc(24 * 4 * 4096 * 4).unwrap(),
        hc_base: gpu.alloc(24 * 4).unwrap(),
        hc_scale: gpu.alloc(12).unwrap(),
        lowrank: None,
    };
    layer.set_hc_weights(HcWeights {
        attn: site(),
        ffn: site(),
        head: Some(HcHeadWeights {
            hc_fn: gpu.alloc(4 * 4 * 4096 * 4).unwrap(),
            hc_base: gpu.alloc(16).unwrap(),
            hc_scale: gpu.alloc(4).unwrap(),
            lowrank: None,
        }),
        hc_mult: 4,
        sinkhorn_iters: 1,
        hc_eps: 1e-6,
        is_first_model_layer: true,
        is_last_model_layer: true,
    });
    run(&gpu, &config, &layer);
}

fn inputs<'a>(
    states: &'a mut [Box<dyn LayerState>],
    blocks: &'a [[u32; 1]],
    positions: &'a [usize; 5],
    arena: &BufferArena,
) -> Vec<GlmPairLayerInput<'a>> {
    states
        .iter_mut()
        .enumerate()
        .map(|(i, state)| GlmPairLayerInput {
            hidden: arena.hidden_states().offset(i * 40960),
            state: state.as_mut(),
            positions,
            block_table: &blocks[i],
        })
        .collect()
}

#[test]
fn actual_mla_owner_batch_layer_entry() {
    const CHILD: &str = "ATLAS_TEST_MLA_OWNER_BATCH";
    if std::env::var_os(CHILD).is_none() {
        let name = concat!(module_path!(), "::actual_mla_owner_batch_layer_entry");
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", name.split_once("::").unwrap().1, "--nocapture"]);
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("ATLAS_") {
                command.env_remove(key);
            }
        }
        command
            .env(CHILD, "1")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .env("ATLAS_GLM_MLA_MULTI_SEQ", "1")
            .env("ATLAS_GLM_K5_GROUPED_MOE", "1")
            .env("ATLAS_GLM_K5_BATCHED_SHARED", "0")
            .env("ATLAS_MOE_PREFILL_EXACT_TILES", "1");
        for key in [
            "ATLAS_GLM_K5_HC_CUBLAS",
            "ATLAS_HOST_TRANSPOSE",
            "ATLAS_GLM_MOE_GATE_UP_M16",
            "ATLAS_GLM_K5_FUSED_SHARED_GATE_UP",
            "ATLAS_GLM_K5_COMPACT_MOE",
            "ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP",
            "ATLAS_GLM_K5_FUSED_MOE_HC",
        ] {
            command.env(key, "0");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        return;
    }
    for rank in 0..2 {
        with_mla(rank, |gpu, config, layer| {
            // Existing pair is the constructor/context control before wider RED.
            for count in 2usize..=8 {
                let arena =
                    BufferArena::new(config, (count * 10).max(40), 2048, 16, count, gpu).unwrap();
                let resources = ContextResources::new();
                let mut levers = ops::ModelLevers::defaults();
                levers.max_decode_seqs = count as u32;
                let comm = Comm {
                    gpu,
                    rank,
                    reductions: Mutex::new(vec![]),
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
                    8,
                    gpu,
                )
                .unwrap();
                let mut blocks: Vec<_> =
                    (0..count).map(|_| [cache.alloc_block().unwrap()]).collect();
                let positions = [3usize, 4, 5, 6, 7];
                let metas: Vec<_> = (0..count)
                    .map(|owner| {
                        let ptr = gpu.alloc(256).unwrap();
                        let mut data = [0u32; 64];
                        for row in 0..5 {
                            data[row] = positions[row] as u32;
                            data[8 + row] = blocks[owner][0] * 16 + positions[row] as u32;
                            data[16 + row] = positions[row] as u32 + 1;
                            data[24 + row] = blocks[owner][0];
                            data[32 + row] = u32::MAX;
                        }
                        gpu.copy_h2d(
                            &data
                                .iter()
                                .flat_map(|n| n.to_le_bytes())
                                .collect::<Vec<_>>(),
                            ptr,
                        )
                        .unwrap();
                        AttnMetadataDev {
                            positions: ptr,
                            positions_h: ptr,
                            positions_w: ptr,
                            slot: ptr.offset(32),
                            seq_len: ptr.offset(64),
                            block_table: ptr.offset(96),
                            max_blocks_per_seq: 1,
                            num_seqs: 5,
                            seq_slot: ptr.offset(128),
                            moe_row_adapter: DevicePtr::NULL,
                        }
                    })
                    .collect();
                let contexts: Vec<_> = metas
                    .iter()
                    .map(|&meta| {
                        let mut ctx = resources.view(&arena, config, gpu);
                        ctx.levers = &levers;
                        ctx.comm = Some(&comm);
                        ctx.attn_metadata = Some(meta);
                        ctx
                    })
                    .collect();
                let refs: Vec<_> = contexts.iter().collect();
                let mut states: Vec<_> = (0..count)
                    .map(|_| layer.alloc_state(gpu).unwrap())
                    .collect();
                if count == 3 {
                    let actual = std::mem::replace(&mut states[2], Box::new(ForeignState));
                    let mut bad = inputs(&mut states, &blocks, &positions, &arena);
                    let mut workspace = GlmOwnerBatchWorkspace::new(
                        refs[0],
                        GlmOwnerBatchShape::new(count).unwrap(),
                    )
                    .unwrap();
                    gpu.clear();
                    let error = layer
                        .decode_glm_owner_verify(
                            &mut bad,
                            &mut cache,
                            &mut workspace,
                            &refs,
                            gpu.default_stream(),
                        )
                        .unwrap_err();
                    assert!(error.to_string().contains("actual state/block map invalid"));
                    assert!(
                        gpu.trace().is_empty(),
                        "foreign final owner must precede writers"
                    );
                    drop(bad);
                    states[2] = actual;
                }
                let mut owners = inputs(&mut states, &blocks, &positions, &arena);
                gpu.clear();
                if count == 2 {
                    layer
                        .validate_glm_pair_verify(refs[0], GlmPairFfn::TwoK5, gpu.default_stream())
                        .unwrap();
                } else {
                    layer
                        .validate_glm_owner_verify(
                            refs[0],
                            GlmOwnerBatchShape::new(count).unwrap(),
                            gpu.default_stream(),
                        )
                        .expect(
                            "actual MLA wider preflight must support three through eight owners",
                        );
                }
                assert!(
                    gpu.trace().is_empty(),
                    "actual MLA preflight must not write"
                );
                if count == 2 {
                    let mut workspace = GlmPairWorkspace::new(refs[0], GlmPairFfn::TwoK5).unwrap();
                    layer
                        .decode_glm_pair_verify(
                            [owners.remove(0), owners.remove(0)],
                            &mut cache,
                            &mut workspace,
                            [refs[0], refs[1]],
                            gpu.default_stream(),
                        )
                        .unwrap();
                } else {
                    let mut workspace = GlmOwnerBatchWorkspace::new(
                        refs[0],
                        GlmOwnerBatchShape::new(count).unwrap(),
                    )
                    .unwrap();
                    layer
                        .decode_glm_owner_verify(
                            &mut owners,
                            &mut cache,
                            &mut workspace,
                            &refs,
                            gpu.default_stream(),
                        )
                        .unwrap();
                }
                let trace = gpu.trace();
                assert!(
                    !trace
                        .iter()
                        .any(|e| matches!(e, Event::Alloc(..) | Event::Free(..)))
                );
                for owner in 0..count {
                    assert_eq!(
                        trace
                            .iter()
                            .filter(|e| **e
                                == Event::Copy(
                                    arena.norm_output(),
                                    arena.norm_output().offset((count * 5 + owner * 5) * 8192),
                                    40960,
                                    gpu.default_stream()
                                ))
                            .count(),
                        1
                    );
                    assert_eq!(
                        trace
                            .iter()
                            .filter(|e| **e
                                == Event::Copy(
                                    arena.hc_streams(),
                                    arena.hc_streams().offset((owner + 1) * 327680),
                                    327680,
                                    gpu.default_stream()
                                ))
                            .count(),
                        2
                    );
                }
                drop(owners);
                // Final owner aliases the first owner's actual writable K5 cache slots.
                // All-owner validation must refuse before even owner0's attention write.
                blocks[count - 1] = blocks[0];
                let mut owners = inputs(&mut states, &blocks, &positions, &arena);
                gpu.clear();
                let error = if count == 2 {
                    let mut workspace = GlmPairWorkspace::new(refs[0], GlmPairFfn::TwoK5).unwrap();
                    layer
                        .decode_glm_pair_verify(
                            [owners.remove(0), owners.remove(0)],
                            &mut cache,
                            &mut workspace,
                            [refs[0], refs[1]],
                            gpu.default_stream(),
                        )
                        .unwrap_err()
                } else {
                    let mut workspace = GlmOwnerBatchWorkspace::new(
                        refs[0],
                        GlmOwnerBatchShape::new(count).unwrap(),
                    )
                    .unwrap();
                    layer
                        .decode_glm_owner_verify(
                            &mut owners,
                            &mut cache,
                            &mut workspace,
                            &refs,
                            gpu.default_stream(),
                        )
                        .unwrap_err()
                };
                assert!(
                    error.to_string().contains("writable cache slots alias"),
                    "{error:#}"
                );
                assert!(
                    gpu.trace().is_empty(),
                    "later-owner alias must precede every writer"
                );
            }
        });
    }
}
