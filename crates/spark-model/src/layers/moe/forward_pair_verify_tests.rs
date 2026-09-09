// SPDX-License-Identifier: AGPL-3.0-only
//! Actual constructor, arena and dispatch recorder; not GPU numerical proof.
use super::*;

#[test]
fn actual_pair_verify_ffn_dispatch() {
    const SENTINEL: &str = "ATLAS_TEST_PAIR_VERIFY_FFN";
    if std::env::var_os(SENTINEL).is_none() {
        let name = concat!(module_path!(), "::actual_pair_verify_ffn_dispatch");
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", name.split_once("::").unwrap().1, "--nocapture"])
            .env(SENTINEL, "1")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .env("ATLAS_GLM_K5_GROUPED_MOE", "1")
            .env("ATLAS_GLM_K5_BATCHED_SHARED", "0")
            .env("ATLAS_GLM_K5_FUSED_SHARED_GATE_UP", "1")
            .env("ATLAS_MOE_PREFILL_EXACT_TILES", "1");
        for key in [
            "ATLAS_GLM_INDEPENDENT_DECODE",
            "ATLAS_NVFP4_PREQUANT_MOE",
            "ATLAS_NVFP4_FUSED_SILU_QUANT",
            "ATLAS_GLM_MOE_GATE_UP_M16",
            "ATLAS_GLM_MOE_GATE_UP_M16_VERIFY",
            "ATLAS_GLM_M5_ROUTER_BN4",
            "ATLAS_GLM_M5_ROUTER_BN4_VERIFY",
            "ATLAS_GLM_M5_SHARED_M16",
            "ATLAS_GLM_M5_SHARED_M16_VERIFY",
            "ATLAS_GLM_C2_COMPACT_MOE",
            "ATLAS_GLM_C3_GROUPED_MOE",
            "ATLAS_GLM_C4_GROUPED_MOE",
            "ATLAS_GLM_K5_COMPACT_MOE",
            "ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP",
            "ATLAS_GLM_K5_HC_CUBLAS",
            "ATLAS_MOE_GROUPED_CUTLASS",
            "ATLAS_HOLO_MOE_GROUPED_CUTLASS",
            "ATLAS_HOLO_MOE_GROUPED_DOWN",
            "ATLAS_NVFP4_MMQ_MOE",
            "ATLAS_GLM_TARGET_SHARED_FP8",
            "ATLAS_GLM_TARGET_SHARED_FP8_VERIFY",
            "ATLAS_DUMP_EXPERT_IDS",
            "ATLAS_HOST_TRANSPOSE",
        ] {
            cmd.env(key, "0");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    for rank in 0..2 {
        for vector in [false, true] {
            let gpu = Gpu::new();
            let (_store, config, mut layer) = resident_tests::setup(&gpu, rank);
            layer
                .transpose_for_prefill_unified_keep_shared(&gpu, &config)
                .unwrap();
            layer.unified_layout = true;
            layer.nvfp4_vecscale = vector;
            layer.nvfp4_fused_silu_quant = true;
            let arena = BufferArena::new(&config, 10, 2048, 16, 2, &gpu).unwrap();
            let resources = ContextResources::new();
            let comm = Comm {
                gpu: &gpu,
                rank,
                reductions: Mutex::new(vec![]),
            };
            let mut ctx = resources.view(&arena, &config, &gpu);
            ctx.comm = Some(&comm);
            // Real base-model routing is Fold with no resident adapter, not Skip.
            ctx.moe_lora_route = crate::lora::resolve_moe_lora_route(-1, -1, false);
            assert!(ctx.ssm_batch.is_none()); // Not independent indexed decode.
            assert!(arena.sizes().norm_output >= 81920);
            let shared = layer.w4a16_gemm_t.0;
            let exact_shared = [layer.w4a16_batchm.kernel(5).0, layer.w4a16_batch5_dual_k.0];
            let shared_weights = [
                layer.shared_gate_t.unwrap(),
                layer.shared_up_t.unwrap(),
                layer.shared_down_t.unwrap(),
            ];
            let builder = layer.moe_build_tile_worklist_k.0;
            let fused = if vector {
                layer.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0
            } else {
                layer.moe_w4a4_prequant_t_k64_compact_gate_up.0
            };
            let down = if vector {
                layer.moe_w4a4_prequant_t_k64_vecscale.0
            } else {
                layer.moe_w4a4_prequant_t_k64.0
            };
            let topk = layer.moe_topk_sigmoid_batched_k.0;
            let sort = layer.moe_sort_by_expert.0;
            let unpermute = layer.moe_unpermute_reduce_ep.0;
            // Actual entry negatives must refuse before shared/router writers.
            gpu.clear();
            assert!(
                layer
                    .forward_pair_verify(arena.norm_output().offset(16), &ctx, 91)
                    .is_err()
            );
            assert!(gpu.trace().is_empty());
            let good = layer.moe_build_tile_worklist_k;
            layer.moe_build_tile_worklist_k.0 = 0;
            assert!(
                layer
                    .forward_pair_verify(arena.norm_output(), &ctx, 91)
                    .is_err()
            );
            assert!(gpu.trace().is_empty());
            layer.moe_build_tile_worklist_k = good;
            let good = layer.w4a16_gemm_t;
            layer.w4a16_gemm_t.0 = 0;
            assert!(layer.validate_pair_k5_control().is_err());
            assert!(
                layer
                    .forward_pair_verify(arena.norm_output(), &ctx, 91)
                    .is_err()
            );
            assert!(gpu.trace().is_empty());
            layer.w4a16_gemm_t = good;
            let good = layer.shared_down_t.take();
            assert!(
                layer
                    .forward_pair_verify(arena.norm_output(), &ctx, 91)
                    .is_err()
            );
            assert!(gpu.trace().is_empty());
            layer.shared_down_t = good;
            let route = ctx.moe_lora_route;
            ctx.moe_lora_route = crate::layer::MoeLoraRoute::Refuse;
            assert!(
                layer
                    .forward_pair_verify(arena.norm_output(), &ctx, 91)
                    .is_err()
            );
            assert!(gpu.trace().is_empty());
            ctx.moe_lora_route = route;
            ctx.graph_capture = true;
            assert!(
                layer
                    .forward_pair_verify(arena.norm_output(), &ctx, 91)
                    .is_err()
            );
            assert!(gpu.trace().is_empty());
            ctx.graph_capture = false;
            let short = BufferArena::new(&config, 9, 2048, 16, 2, &gpu).unwrap();
            let mut short_ctx = resources.view(&short, &config, &gpu);
            short_ctx.comm = Some(&comm);
            gpu.clear();
            assert!(
                layer
                    .forward_pair_verify(short.norm_output(), &short_ctx, 91)
                    .is_err()
            );
            assert!(gpu.trace().is_empty());
            let ffn = FfnComponent::Moe(layer);
            gpu.clear();
            ffn.validate_pair_verify(arena.norm_output(), &ctx, 91)
                .unwrap();
            ffn.validate_pair_k5(arena.norm_output(), &ctx, 91).unwrap();
            assert!(gpu.trace().is_empty());
            // Initial RED must reach the actual new-entry refusal, not a
            // constructor error or an invented historical numerical defect.
            assert_eq!(
                ffn.forward_pair_verify(arena.norm_output(), &ctx, 91)
                    .unwrap(),
                arena.moe_output()
            );
            let trace = gpu.trace();
            let calls: Vec<_> = trace
                .iter()
                .filter_map(|event| match event {
                    Event::Launch(handle, _, _, _, stream, args) => {
                        assert_eq!(*stream, 91);
                        Some((*handle, args))
                    }
                    _ => None,
                })
                .collect();
            for handle in [topk, sort, builder, fused, down, unpermute] {
                assert_ne!(handle, 0);
                assert_eq!(calls.iter().filter(|call| call.0 == handle).count(), 1);
            }
            let shared_calls: Vec<_> = calls.iter().filter(|call| call.0 == shared).collect();
            assert!(!calls.iter().any(|call| exact_shared.contains(&call.0)));
            assert_eq!(shared_calls.len(), 6); // Two literal generic-T K5 sets.
            for (index, call) in shared_calls.iter().enumerate() {
                let projection = index % 3;
                let owner = index / 3;
                let n = if projection == 2 { 4096u32 } else { 2048 };
                let k = if projection == 2 { 2048u32 } else { 4096 };
                assert_eq!(call.1.len(), 9);
                assert_eq!(call.1[5], Arg::Bytes(5u32.to_ne_bytes().to_vec()));
                assert_eq!(call.1[6], Arg::Bytes(n.to_ne_bytes().to_vec()));
                assert_eq!(call.1[7], Arg::Bytes(k.to_ne_bytes().to_vec()));
                assert_eq!(call.1[8], Arg::Bytes(n.to_ne_bytes().to_vec())); // actual ldb
                assert_eq!(call.1[1], Arg::Ptr(shared_weights[projection].weight));
                assert_eq!(call.1[2], Arg::Ptr(shared_weights[projection].weight_scale));
                let (input, output) = match projection {
                    0 => (
                        arena.norm_output().offset(owner * 40960),
                        arena.ssm_deinterleaved().offset(owner * 20480),
                    ),
                    1 => (
                        arena.norm_output().offset(owner * 40960),
                        arena.ssm_qkvz().offset(owner * 20480),
                    ),
                    _ => (
                        arena.ssm_deinterleaved().offset(owner * 20480),
                        arena.attn_output().offset(owner * 40960),
                    ),
                };
                assert_eq!(call.1[0], Arg::Ptr(input));
                assert_eq!(call.1[4], Arg::Ptr(output));
            }
            assert_eq!(
                comm.reductions
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|r| (r.0, r.1))
                    .collect::<Vec<_>>(),
                vec![(81920, Some(91))]
            );
            assert!(!trace.iter().any(|event| matches!(
                event,
                Event::Read(..) | Event::Alloc(..) | Event::Free(..)
            )));
            actual_dense_pair(&ffn, &config, &gpu, &comm);
        }
    }
}

fn actual_dense_pair(
    ffn: &FfnComponent,
    config: &atlas_core::config::ModelConfig,
    gpu: &Gpu,
    comm: &Comm<'_>,
) {
    let FfnComponent::Moe(moe) = ffn else {
        unreachable!()
    };
    // Borrow real resident NVFP4 shared projection owners for the same4096x2048
    // dense geometry, through the actual DenseFfnLayer constructor.
    let mut config = config.clone();
    config.intermediate_size = 2048;
    let layer = crate::layers::DenseFfnLayer::new(
        crate::layers::DenseFfnWeights {
            gate_proj: moe.weights.shared_expert.gate_proj,
            up_proj: moe.weights.shared_expert.up_proj,
            down_proj: moe.weights.shared_expert.down_proj,
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        gpu,
    )
    .unwrap();
    let ffn = FfnComponent::Dense(layer);
    let arena = BufferArena::new(&config, 10, 2048, 16, 2, gpu).unwrap();
    let resources = ContextResources::new();
    let mut ctx = resources.view(&arena, &config, gpu);
    ctx.comm = Some(comm);
    let before = comm.reductions.lock().unwrap().len();
    gpu.clear();
    ffn.validate_pair_verify(arena.norm_output(), &ctx, 91)
        .unwrap();
    assert!(gpu.trace().is_empty());
    assert_eq!(
        ffn.forward_pair_verify(arena.norm_output(), &ctx, 91)
            .unwrap(),
        arena.moe_output()
    );
    let trace = gpu.trace();
    let copies: Vec<_> = trace
        .iter()
        .enumerate()
        .filter(|(_, event)| matches!(event, Event::Copy(..)))
        .collect();
    assert_eq!(copies.len(), 1);
    assert_eq!(
        *copies[0].1,
        Event::Copy(
            arena.moe_output(),
            arena.moe_output().offset(40960),
            40960,
            91
        )
    );
    let at = copies[0].0;
    let launches = |events: &[Event]| {
        events
            .iter()
            .filter(|event| matches!(event, Event::Launch(..)))
            .count()
    };
    assert_eq!(launches(&trace[..at]), 4);
    assert_eq!(launches(&trace[at + 1..]), 4);
    for (owner, events) in [(1, &trace[..at]), (0, &trace[at + 1..])] {
        let Event::Launch(_, _, _, _, stream, args) = &events[0] else {
            panic!("dense K5 starts with projection")
        };
        assert_eq!(*stream, 91);
        assert_eq!(args[0], Arg::Ptr(arena.norm_output().offset(owner * 40960)));
        assert_eq!(args[5], Arg::Bytes(5u32.to_ne_bytes().to_vec()));
    }
    assert_eq!(comm.reductions.lock().unwrap().len(), before);
    assert!(
        !trace
            .iter()
            .any(|event| matches!(event, Event::Read(..) | Event::Alloc(..) | Event::Free(..)))
    );
}
