// SPDX-License-Identifier: AGPL-3.0-only

//! Actual KDA trait entry with real allocations; launch/copy evidence, not CUDA
//! arithmetic or selected request/transaction authority.

use super::*;
use crate::layer::glm_pair_verify::{GlmPairFfn, GlmPairLayerInput, GlmPairWorkspace};

#[test]
fn actual_kda_pair_layer_entry() {
    const ENV: &str = "ATLAS_TEST_KDA_PAIR_LAYER";
    if std::env::var_os(ENV).is_none() {
        let name = concat!(module_path!(), "::actual_kda_pair_layer_entry");
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name.split_once("::").unwrap().1, "--nocapture"])
            .env(ENV, "1")
            .env("ATLAS_GLM_INDEPENDENT_DECODE", "0")
            .env("ATLAS_GLM_K5_HC_CUBLAS", "0")
            .env("ATLAS_KDA_REGRESIDENT_PREFILL", "0")
            .env("ATLAS_GLM_K5_FUSED_QKV", "0")
            .env("ATLAS_GLM_K5_FUSED_DENSE_PAIRS", "0")
            .env("ATLAS_GLM_K5_FUSED_DENSE_TRIPLE", "0")
            .env("ATLAS_GLM_K5_BATCHED_CONV_SNAPSHOT", "0")
            .env("ATLAS_GLM_K5_BATCHED_RECURRENT_SNAPSHOT", "0")
            .env("ATLAS_GLM_K5_FUSED_TP_HC", "0")
            .env("ATLAS_GLM_KDA_BATCHED_FFN", "1")
            .env("ATLAS_GLM_K5_GROUPED_MOE", "1")
            .env("ATLAS_GLM_K5_BATCHED_SHARED", "0")
            .env("ATLAS_MOE_PREFILL_EXACT_TILES", "1")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .env("ATLAS_HOST_TRANSPOSE", "0")
            .env("ATLAS_GLM_MOE_GATE_UP_M16", "0")
            .env("ATLAS_GLM_MOE_GATE_UP_M16_VERIFY", "0")
            .env("ATLAS_GLM_K5_COMPACT_MOE", "0")
            .env("ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP", "0")
            .env("ATLAS_GLM_K5_FUSED_SHARED_GATE_UP", "0")
            .env("ATLAS_GLM_K5_FUSED_MOE_HC", "0")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    for rank in 0..2 {
        for mode in [GlmPairFfn::TwoK5, GlmPairFfn::Joint] {
            with_kda_ffn(rank, true, |gpu, config, layer| {
                let arena = BufferArena::new(config, 20, 2048, 16, 20, gpu).unwrap();
                let resources = ContextResources::new();
                let mut levers = ops::ModelLevers::defaults();
                levers.max_decode_seqs = 2;
                let comm = Comm {
                    gpu,
                    rank,
                    reductions: Mutex::new(vec![]),
                };
                let mut ctx = resources.view(&arena, config, gpu);
                ctx.levers = &levers;
                ctx.comm = Some(&comm);
                let mut workspace = GlmPairWorkspace::new(&ctx, mode).unwrap();
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
                let blocks = [
                    [cache.alloc_block().unwrap()],
                    [cache.alloc_block().unwrap()],
                ];
                let mut a = layer.alloc_state(gpu).unwrap();
                let mut b = layer.alloc_state(gpu).unwrap();
                for state in [&mut a, &mut b] {
                    let state = state.as_any_mut().downcast_mut::<SsmLayerState>().unwrap();
                    state.h_state_intermediates =
                        (0..4).map(|_| gpu.alloc(2097152).unwrap()).collect();
                    state.conv_state_intermediates =
                        (0..5).map(|_| gpu.alloc(196608).unwrap()).collect();
                }
                let snapshot_spans: Vec<_> = [&a, &b]
                    .into_iter()
                    .flat_map(|state| {
                        let s = state.as_any().downcast_ref::<SsmLayerState>().unwrap();
                        (0..4)
                            .flat_map(|row| {
                                [
                                    (s.h_state, s.h_state_intermediates[row], 2097152),
                                    (s.conv_state, s.conv_state_intermediates[row], 196608),
                                ]
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let positions = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]];
                let inputs = [
                    GlmPairLayerInput {
                        hidden: arena.hidden_states(),
                        state: a.as_mut(),
                        positions: &positions[0],
                        block_table: &blocks[0],
                    },
                    GlmPairLayerInput {
                        hidden: arena.hidden_states().offset(5 * 8192),
                        state: b.as_mut(),
                        positions: &positions[1],
                        block_table: &blocks[1],
                    },
                ];
                gpu.clear();
                layer
                    .validate_glm_pair_verify(&ctx, mode, gpu.default_stream())
                    .unwrap();
                assert!(gpu.trace().is_empty(), "pair preflight must not write");
                layer
                    .decode_glm_pair_verify(
                        inputs,
                        &mut cache,
                        &mut workspace,
                        [&ctx, &ctx],
                        gpu.default_stream(),
                    )
                    .expect("actual KDA pair compute must reach both temporal owners");
                assert!(layer.supports_glm_pair_verify());
                let trace = gpu.trace();
                assert!(
                    !trace
                        .iter()
                        .any(|event| matches!(event, Event::Alloc(..) | Event::Free(..)))
                );
                for (src, dst, bytes) in snapshot_spans {
                    assert_eq!(
                        trace
                            .iter()
                            .filter(|event| **event
                                == Event::Copy(src, dst, bytes, gpu.default_stream()))
                            .count(),
                        1,
                        "each owner's exact K5 snapshot copied once"
                    );
                }
                for owner in 0..2 {
                    assert_eq!(
                        trace
                            .iter()
                            .filter(|event| **event
                                == Event::Copy(
                                    arena.norm_output(),
                                    arena.norm_output().offset((10 + owner * 5) * 8192),
                                    40960,
                                    gpu.default_stream()
                                ))
                            .count(),
                        1
                    );
                    assert_eq!(
                        trace
                            .iter()
                            .filter(|event| **event
                                == Event::Copy(
                                    arena.hc_streams(),
                                    arena.hc_streams().offset((owner + 1) * 327680),
                                    327680,
                                    gpu.default_stream()
                                ))
                            .count(),
                        2,
                        "attention and completed FFN highways preserved"
                    );
                }
            });
        }
    }
}
