// SPDX-License-Identifier: AGPL-3.0-only
//! Actual resident FFN/typed launch ABI; recorded kernels are not numerical proof.
use super::*;
use crate::layer::glm_owner_verify::GlmOwnerBatchShape;

#[test]
fn actual_owner_verify_ffn_dispatch() {
    const SENTINEL: &str = "ATLAS_TEST_OWNER_VERIFY_FFN";
    if std::env::var_os(SENTINEL).is_none() {
        let name = concat!(module_path!(), "::actual_owner_verify_ffn_dispatch");
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", name.split_once("::").unwrap().1, "--nocapture"])
            .env(SENTINEL, "1")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .env("ATLAS_GLM_K5_GROUPED_MOE", "1")
            .env("ATLAS_GLM_K5_BATCHED_SHARED", "0")
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
    for owners in [3, 4] {
        for rank in 0..2 {
            for vector in [false, true] {
                let shape = GlmOwnerBatchShape::new(owners).unwrap();
                let rows = shape.rows();
                let gpu = Gpu::new();
                let (_store, config, mut layer) = resident_tests::setup(&gpu, rank);
                layer
                    .transpose_for_prefill_unified_keep_shared(&gpu, &config)
                    .unwrap();
                layer.unified_layout = true;
                layer.nvfp4_vecscale = vector;
                layer.nvfp4_fused_silu_quant = true;
                let handles = [
                    layer.dense_gemm_router.0,
                    layer.moe_topk_sigmoid_batched_k.0,
                    layer.moe_sort_by_expert.0,
                    layer.moe_build_tile_worklist_k.0,
                    if vector {
                        layer.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0
                    } else {
                        layer.moe_w4a4_prequant_t_k64_compact_gate_up.0
                    },
                    if vector {
                        layer.moe_w4a4_prequant_t_k64_vecscale.0
                    } else {
                        layer.moe_w4a4_prequant_t_k64.0
                    },
                    layer.moe_unpermute_reduce_ep.0,
                    layer.moe_batched_blend.0,
                ];
                let shared = layer.w4a16_gemm_t.0;
                let activation = layer.moe_act_mul.0;
                let ffn = FfnComponent::Moe(layer);
                let arena = BufferArena::new(&config, rows * 2, 2048, 16, 4, &gpu).unwrap();
                let resources = ContextResources::new();
                let comm = Comm {
                    gpu: &gpu,
                    rank,
                    reductions: Mutex::new(vec![]),
                };
                let mut ctx = resources.view(&arena, &config, &gpu);
                ctx.comm = Some(&comm);
                ctx.moe_lora_route = crate::lora::resolve_moe_lora_route(-1, -1, false);
                gpu.clear();
                // Genuine new-entry RED: actual resident layer and owned arena
                // reach the explicit scaffold refusal, not a fabricated producer.
                assert_eq!(
                    ffn.forward_owner_verify(arena.norm_output(), shape, &ctx, 91)
                        .unwrap(),
                    arena.moe_output()
                );
                let trace = gpu.trace();
                let calls: Vec<_> = trace
                    .iter()
                    .filter_map(|event| match event {
                        Event::Launch(handle, grid, _, _, stream, args) => {
                            assert_eq!(*stream, 91);
                            Some((*handle, *grid, args))
                        }
                        _ => None,
                    })
                    .collect();
                for handle in handles {
                    assert_ne!(handle, 0);
                    assert_eq!(calls.iter().filter(|call| call.0 == handle).count(), 1);
                }
                let call = |handle| calls.iter().find(|call| call.0 == handle).unwrap();
                let u32_arg = |value: usize| Arg::Bytes((value as u32).to_ne_bytes().to_vec());
                let router = call(handles[0]);
                assert_eq!(router.1, [5, (rows as u32).div_ceil(16), 1]);
                assert_eq!(router.2[0], Arg::Ptr(arena.norm_output()));
                assert_eq!(router.2[3], u32_arg(rows));
                assert_eq!(call(handles[4]).1, [(rows * 8 * 16) as u32, 2, 1]);
                assert_eq!(call(handles[4]).2[5], Arg::Ptr(arena.expert_gate_out()));
                assert_eq!(call(handles[4]).2[9], Arg::Ptr(arena.expert_up_out()));
                assert_eq!(call(handles[4]).2[17], u32_arg(rows * 8 * 16));
                assert_eq!(call(handles[5]).1, [32, 1, 288]);
                assert_eq!(call(handles[6]).2[6], u32_arg(rows));
                assert_eq!(call(handles[7]).2[5], u32_arg(rows));
                let shared_calls: Vec<_> = calls.iter().filter(|call| call.0 == shared).collect();
                assert_eq!(shared_calls.len(), 3);
                for (projection, call) in shared_calls.iter().enumerate() {
                    assert_eq!(call.2.len(), 9);
                    assert_eq!(call.2[5], u32_arg(rows));
                    let n = if projection == 2 { 4096 } else { 2048 };
                    assert_eq!(call.2[6], u32_arg(n));
                    assert_eq!(call.2[8], u32_arg(n));
                    assert_eq!(
                        call.2[4],
                        Arg::Ptr(match projection {
                            0 => arena.ssm_deinterleaved(),
                            1 => arena.ssm_qkvz(),
                            _ => arena.attn_output(),
                        })
                    );
                    assert_eq!(
                        call.2[0],
                        Arg::Ptr(if projection == 2 {
                            arena.ssm_deinterleaved()
                        } else {
                            arena.norm_output()
                        })
                    );
                }
                let silu = call(activation);
                assert_eq!(silu.2[3], u32_arg(rows * 2048));
                assert_eq!(
                    comm.reductions
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|r| (r.0, r.1))
                        .collect::<Vec<_>>(),
                    vec![(rows * 8192, Some(91))]
                );
                assert!(!trace.iter().any(|event| matches!(
                    event,
                    Event::Read(..) | Event::Alloc(..) | Event::Free(..)
                )));
                // Existing fused SiLU writes packed A/scales into down scratch
                // before copying to dead up scratch; writing up in-place races.
                // Permit exactly that copy, never a normalized-tail write.
                let copies: Vec<_> = trace
                    .iter()
                    .enumerate()
                    .filter(|(_, event)| matches!(event, Event::Copy(..)))
                    .collect();
                assert_eq!(copies.len(), 1, "actual owner FFN copies: {copies:?}");
                let packed = rows * 8 * 2048 / 2;
                let scales = rows * 8 * 2048 / 16;
                assert_eq!(
                    *copies[0].1,
                    Event::Copy(
                        arena.expert_down_out(),
                        arena.expert_up_out(),
                        packed + scales,
                        91,
                    )
                );
                let down_at = trace
                    .iter()
                    .position(
                        |event| matches!(event, Event::Launch(handle, ..) if *handle == handles[5]),
                    )
                    .unwrap();
                assert!(copies[0].0 < down_at);
                // This backend records, but does not execute kernels. Exact
                // row counts and destinations constrain the saved norm tail;
                // no claim that an unchanged byte sentinel proves CUDA stores.
                for call in &calls {
                    assert!(!call.2.iter().any(|arg| matches!(arg, Arg::Ptr(p)
                        if p.0 >= arena.norm_output().offset(rows * 8192).0
                        && p.0 < arena.norm_output().offset(rows * 2 * 8192).0)));
                }
                gpu.clear();
                ffn.validate_owner_verify(arena.norm_output(), shape, &ctx, 91)
                    .unwrap();
                assert!(gpu.trace().is_empty());
                assert!(
                    ffn.forward_owner_verify(arena.norm_output().offset(16), shape, &ctx, 91)
                        .is_err()
                );
                assert!(gpu.trace().is_empty());
                ctx.graph_capture = true;
                assert!(
                    ffn.forward_owner_verify(arena.norm_output(), shape, &ctx, 91)
                        .is_err()
                );
                assert!(gpu.trace().is_empty());
                ctx.graph_capture = false;
                let short = BufferArena::new(&config, rows - 1, 2048, 16, 4, &gpu).unwrap();
                let mut short_ctx = resources.view(&short, &config, &gpu);
                short_ctx.comm = Some(&comm);
                gpu.clear();
                assert!(
                    ffn.forward_owner_verify(short.norm_output(), shape, &short_ctx, 91)
                        .is_err()
                );
                assert!(gpu.trace().is_empty());
                if rank == 0 && !vector {
                    dense_owners(&ffn, &config, &gpu, &comm, shape);
                }
            }
        }
    }
}

fn dense_owners(
    ffn: &FfnComponent,
    config: &atlas_core::config::ModelConfig,
    gpu: &Gpu,
    comm: &Comm<'_>,
    shape: GlmOwnerBatchShape,
) {
    let FfnComponent::Moe(moe) = ffn else {
        unreachable!()
    };
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
    let arena = BufferArena::new(&config, shape.rows() * 2, 2048, 16, 4, gpu).unwrap();
    let resources = ContextResources::new();
    let mut ctx = resources.view(&arena, &config, gpu);
    ctx.comm = Some(comm);
    let reductions = comm.reductions.lock().unwrap().len();
    gpu.clear();
    assert_eq!(
        ffn.forward_owner_verify(arena.norm_output(), shape, &ctx, 91)
            .unwrap(),
        arena.moe_output()
    );
    let trace = gpu.trace();
    let mut at = 0;
    for owner in (0..shape.owners()).rev() {
        let Event::Launch(_, _, _, _, stream, args) = &trace[at] else {
            panic!("actual dense K5 starts with projection")
        };
        assert_eq!(*stream, 91);
        assert_eq!(args[0], Arg::Ptr(arena.norm_output().offset(owner * 40960)));
        assert_eq!(args[5], Arg::Bytes(5u32.to_ne_bytes().to_vec()));
        assert!(
            trace[at..at + 4]
                .iter()
                .all(|e| matches!(e, Event::Launch(..)))
        );
        at += 4;
        if owner != 0 {
            assert_eq!(
                trace[at],
                Event::Copy(
                    arena.moe_output(),
                    arena.moe_output().offset(owner * 40960),
                    40960,
                    91
                )
            );
            at += 1;
        }
    }
    assert_eq!(at, trace.len());
    assert_eq!(comm.reductions.lock().unwrap().len(), reductions);
}
