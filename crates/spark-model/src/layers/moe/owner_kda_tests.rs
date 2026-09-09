// SPDX-License-Identifier: AGPL-3.0-only
//! Real KDA layer/arena/state with recorded kernels, not native arithmetic.
use super::*;
use crate::layer::glm_owner_verify::{GlmOwnerBatchShape, GlmOwnerBatchWorkspace};
use crate::layer::glm_pair_verify::GlmPairLayerInput;

#[test]
fn actual_kda_owner_batch_layer_entry() {
    const CHILD: &str = "ATLAS_TEST_KDA_OWNER_BATCH";
    if std::env::var_os(CHILD).is_none() {
        let name = concat!(module_path!(), "::actual_kda_owner_batch_layer_entry");
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
            .env("ATLAS_GLM_K5_GROUPED_MOE", "1")
            .env("ATLAS_GLM_K5_BATCHED_SHARED", "0")
            .env("ATLAS_MOE_PREFILL_EXACT_TILES", "1");
        for key in [
            "ATLAS_GLM_K5_HC_CUBLAS",
            "ATLAS_HOST_TRANSPOSE",
            "ATLAS_KDA_REGRESIDENT_PREFILL",
            "ATLAS_GLM_MOE_GATE_UP_M16",
            "ATLAS_GLM_K5_BATCHED_CONV_SNAPSHOT",
            "ATLAS_GLM_K5_BATCHED_RECURRENT_SNAPSHOT",
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
        for count in 3..=8 {
            with_kda_ffn(rank, true, |gpu, config, layer| {
                let shape = GlmOwnerBatchShape::new(count).unwrap();
                let arena = BufferArena::new(config, count * 10, 2048, 16, count, gpu).unwrap();
                let resources = ContextResources::new();
                let mut levers = ops::ModelLevers::defaults();
                levers.max_decode_seqs = count as u32;
                let comm = Comm {
                    gpu,
                    rank,
                    reductions: Mutex::new(vec![]),
                };
                let mut ctx = resources.view(&arena, config, gpu);
                ctx.levers = &levers;
                ctx.comm = Some(&comm);
                // Exercise actual trait preflight independently of workspace construction.
                gpu.clear();
                layer
                    .validate_glm_owner_verify(&ctx, shape, gpu.default_stream())
                    .expect("actual KDA wider preflight must support three through eight owners");
                assert!(gpu.trace().is_empty());
                let mut workspace = GlmOwnerBatchWorkspace::new(&ctx, shape).unwrap();
                let mut states: Vec<_> = (0..count)
                    .map(|_| layer.alloc_state(gpu).unwrap())
                    .collect();
                for state in &mut states {
                    let state = state.as_any_mut().downcast_mut::<SsmLayerState>().unwrap();
                    state.h_state_intermediates =
                        (0..4).map(|_| gpu.alloc(2097152).unwrap()).collect();
                    state.conv_state_intermediates =
                        (0..5).map(|_| gpu.alloc(196608).unwrap()).collect();
                }
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
                let blocks: Vec<_> = (0..count).map(|_| [cache.alloc_block().unwrap()]).collect();
                let positions = [3, 4, 5, 6, 7];
                let contexts = [&ctx; 8];
                let mut inputs: Vec<_> = states
                    .iter_mut()
                    .enumerate()
                    .map(|(i, state)| GlmPairLayerInput {
                        hidden: arena.hidden_states().offset(i * 40960),
                        state: state.as_mut(),
                        positions: &positions,
                        block_table: &blocks[i],
                    })
                    .collect();
                gpu.clear();
                layer
                    .decode_glm_owner_verify(
                        &mut inputs,
                        &mut cache,
                        &mut workspace,
                        &contexts[..count],
                        gpu.default_stream(),
                    )
                    .unwrap();
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
                                    arena
                                        .norm_output()
                                        .offset((shape.rows() + owner * 5) * 8192),
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
                drop(inputs);
                let alias = states[0]
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .unwrap()
                    .h_state;
                let last = states[count - 1]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .unwrap();
                let original = last.h_state;
                last.h_state = alias;
                let mut workspace = GlmOwnerBatchWorkspace::new(&ctx, shape).unwrap();
                let mut inputs: Vec<_> = states
                    .iter_mut()
                    .enumerate()
                    .map(|(i, state)| GlmPairLayerInput {
                        hidden: arena.hidden_states().offset(i * 40960),
                        state: state.as_mut(),
                        positions: &positions,
                        block_table: &blocks[i],
                    })
                    .collect();
                gpu.clear();
                assert!(
                    layer
                        .decode_glm_owner_verify(
                            &mut inputs,
                            &mut cache,
                            &mut workspace,
                            &contexts[..count],
                            gpu.default_stream()
                        )
                        .is_err()
                );
                assert!(
                    gpu.trace().is_empty(),
                    "last owner alias must refuse before any writer"
                );
                drop(inputs);
                states[count - 1]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .unwrap()
                    .h_state = original;
            });
        }
    }
}
