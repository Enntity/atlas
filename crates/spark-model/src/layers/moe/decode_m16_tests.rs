// SPDX-License-Identifier: AGPL-3.0-only
//! ATLAS_GLM_MOE_DECODE_M16 dispatch through the actual verify FFN entry;
//! the numerics are gated on hardware by scripts/moe-decode-bench.
use super::*;
use crate::layers::moe::decode_m16::{MAX_ROWS, m16_shape};

#[test]
fn m16_tiles_cover_one_row_slab_of_k128_aligned_experts() {
    for rows in 1..=MAX_ROWS {
        assert!(m16_shape(rows, 4096, 1024));
        assert!(m16_shape(rows, 4096, 2048));
    }
    // More rows than one m16 slab, or none: the M64 kernels keep the batch.
    assert!(!m16_shape(0, 4096, 1024));
    assert!(!m16_shape(MAX_ROWS + 1, 4096, 1024));
    // Gate/up tiles are 128 columns wide, down tiles 256.
    assert!(!m16_shape(8, 4096, 1000));
    assert!(!m16_shape(8, 4096 + 128, 1024));
}

/// One verify FFN of `rows` rows through `forward_independent`; the launches
/// as `(handle, grid, block, args)` and the arena.
fn verify_ffn(gpu: &Gpu, rank: usize, vector: bool, rows: usize) -> Vec<Event> {
    let (_store, config, mut layer) = resident_tests::setup(gpu, rank);
    layer
        .transpose_for_prefill_unified_keep_shared(gpu, &config)
        .unwrap();
    layer.unified_layout = true;
    layer.nvfp4_vecscale = vector;
    layer.nvfp4_fused_silu_quant = true;
    let arena = BufferArena::new(&config, 8, 2048, 16, 8, gpu).unwrap();
    let metadata = gpu.alloc(512).unwrap();
    let meta = AttnMetadataDev {
        positions: metadata,
        positions_h: metadata,
        positions_w: metadata,
        slot: metadata.offset(32),
        seq_len: metadata.offset(64),
        block_table: metadata.offset(96),
        max_blocks_per_seq: 1,
        num_seqs: rows as u32,
        seq_slot: metadata,
        moe_row_adapter: DevicePtr::NULL,
    };
    let resources = ContextResources::new();
    let mut levers = ops::ModelLevers::defaults();
    levers.max_decode_seqs = 8;
    let comm = Comm {
        gpu,
        rank,
        reductions: Mutex::new(vec![]),
    };
    let mut ctx = resources.view(&arena, &config, gpu);
    ctx.levers = &levers;
    ctx.comm = Some(&comm);
    ctx.attn_metadata = Some(meta);
    let h = [gpu.alloc(8 * 2097152).unwrap()];
    let conv = [gpu.alloc(8 * 196608).unwrap()];
    let pool =
        crate::layer::ssm_batch::SsmPoolView::new(&h, &conv, 2097152, 2097152, 196608, 8).unwrap();
    let slots = [7, 0, 6, 1, 5, 2, 4, 3];
    ctx.ssm_batch = Some(
        crate::layer::ssm_batch::SsmBatchView::new(pool, metadata.offset(256), &slots[..rows])
            .unwrap(),
    );
    let (up_out, scratch) = (arena.expert_up_out(), arena.moe_router_in_f32());
    gpu.clear();
    layer
        .forward_independent(arena.norm_output(), rows, &ctx, 91)
        .unwrap();
    let trace = gpu.trace();
    // The launches the flag swaps, by kernel.
    let launches = |name: &str| -> Vec<([u32; 3], [u32; 3], Vec<Arg>)> {
        let handle = gpu.kernel("moe_w4a16", name).unwrap().0;
        trace
            .iter()
            .filter_map(|e| match e {
                Event::Launch(h, grid, block, _, 91, args) if *h == handle => {
                    Some((*grid, *block, args.clone()))
                }
                _ => None,
            })
            .collect()
    };
    let position = |name: &str| {
        let handle = gpu.kernel("moe_w4a16", name).unwrap().0;
        trace
            .iter()
            .position(|e| matches!(e, Event::Launch(h, ..) if *h == handle))
    };
    let compact = if vector {
        "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up"
    } else {
        "moe_w4a4_grouped_gemm_prequant_t_k64_compact_gate_up"
    };
    // The worklist builder's launches, by their N tiles per row tile.
    let builder: Vec<Arg> = trace
        .iter()
        .filter_map(|e| match e {
            Event::Launch(h, _, _, _, 91, args) if *h == layer.moe_build_tile_worklist_k.0 => {
                Some(args[5].clone())
            }
            _ => None,
        })
        .collect();
    let n_tiles = |n: u32| vec![Arg::Bytes(n.to_ne_bytes().to_vec())];
    let silu = trace
        .iter()
        .filter(|e| matches!(e, Event::Launch(h, ..) if *h == layer.silu_mul_quant_nvfp4_k.0))
        .count();
    // ATLAS_GLM_MOE_DOWN_ZSKIP swaps only the down twin.
    let zskip = std::env::var("ATLAS_GLM_MOE_DOWN_ZSKIP").as_deref() == Ok("1");
    let (down_name, other_down) = if zskip {
        ("glm_moe_decode_m16_k128w_zskip", "glm_moe_decode_m16_k128w")
    } else {
        ("glm_moe_decode_m16_k128w", "glm_moe_decode_m16_k128w_zskip")
    };
    let [gate_up, down] = ["glm_moe_decode_m16_gate_up_silu_k128w", down_name].map(launches);
    assert!(launches(other_down).is_empty());
    if std::env::var("ATLAS_GLM_MOE_DECODE_M16").as_deref() == Ok("1") {
        // One tile per routed expert, at most one expert per routed row.
        let (routed, bound) = (rows as u32 * 8, rows as u32 * 8);
        assert_eq!((gate_up.len(), down.len()), (1, 1));
        // One worklist item per routed local expert, its count at the base.
        assert_eq!(builder, n_tiles(1));
        assert_eq!((gate_up[0].0, gate_up[0].1), ([16, bound, 1], [256, 1, 1]));
        assert_eq!((down[0].0, down[0].1), ([16, bound, 1], [256, 1, 1]));
        // The K128W argument lists: 17 fused gate/up, 12 down, the worklist
        // scratch where they take the prefix.
        assert_eq!((gate_up[0].2.len(), down[0].2.len()), (17, 12));
        assert_eq!(gate_up[0].2[11], Arg::Ptr(scratch));
        assert_eq!(down[0].2[11], Arg::Ptr(scratch));
        // Gate/up leaves the NVFP4 SiLU product where the down reads it.
        let scales = up_out.offset(routed as usize * 2048 / 2);
        assert_eq!(gate_up[0].2[15..], [Arg::Ptr(up_out), Arg::Ptr(scales)]);
        assert_eq!(down[0].2[..2], [Arg::Ptr(up_out), Arg::Ptr(scales)]);
        assert!(position("glm_moe_decode_m16_gate_up_silu_k128w") < position(down_name));
        // Nothing of the M64 compact path runs.
        assert_eq!((silu, launches(compact).len()), (0, 0));
        for dense in [
            "moe_w4a4_grouped_gemm_prequant_t_k128",
            "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale",
            "moe_w4a4_grouped_gemm_prequant_t_k64",
        ] {
            assert!(launches(dense).is_empty(), "{dense}");
        }
    } else {
        // Flag off: the base launches, none of the twins.
        assert_eq!((gate_up.len(), down.len()), (0, 0));
        assert_eq!(builder, n_tiles(16));
        assert_eq!((silu, launches(compact).len()), (1, 1));
    }
    assert_eq!(
        comm.reductions
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.0)
            .collect::<Vec<_>>(),
        vec![rows * 8192]
    );
    trace
}

#[test]
fn verify_ffn_launches_the_m16_twins_only_with_the_flag() {
    const SENTINEL: &str = "ATLAS_TEST_DECODE_M16";
    if std::env::var_os(SENTINEL).is_none() {
        let name = concat!(
            module_path!(),
            "::verify_ffn_launches_the_m16_twins_only_with_the_flag"
        );
        let name = name.split_once("::").unwrap().1;
        // Off, on, on with the zero-row skip, and the skip alone (ignored).
        for (m16, zskip) in [("0", "0"), ("1", "0"), ("1", "1"), ("0", "1")] {
            let mode = format!("{m16}{zskip}");
            let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
            cmd.args(["--exact", name, "--nocapture"])
                .env(SENTINEL, &mode)
                .env("ATLAS_GLM_MOE_DECODE_M16", m16)
                .env("ATLAS_GLM_MOE_DOWN_ZSKIP", zskip)
                .env("ATLAS_GLM_INDEPENDENT_DECODE", "1")
                .env("ATLAS_EP_PROTOCOL", "v2");
            for flag in [
                "ATLAS_NVFP4_PREQUANT_MOE",
                "ATLAS_NVFP4_FUSED_SILU_QUANT",
                "ATLAS_GLM_MOE_GATE_UP_M16",
                "ATLAS_GLM_MOE_GATE_UP_M16_VERIFY",
                "ATLAS_GLM_C2_COMPACT_MOE",
                "ATLAS_GLM_C4_GROUPED_MOE",
                "ATLAS_GLM_C3_GROUPED_MOE",
                "ATLAS_GLM_K5_COMPACT_MOE",
                "ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP",
                "ATLAS_GLM_K5_HC_CUBLAS",
            ] {
                cmd.env(flag, "0");
            }
            let output = cmd.output().unwrap();
            assert!(
                output.status.success(),
                "mode{mode}\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        }
        return;
    }
    for rank in 0..2 {
        for vector in [false, true] {
            for rows in 2..=8 {
                let gpu = Gpu::new();
                let trace = verify_ffn(&gpu, rank, vector, rows);
                // Graph-safe either way: no allocation, free or host read.
                assert!(
                    !trace
                        .iter()
                        .any(|e| matches!(e, Event::Alloc(..) | Event::Free(..) | Event::Read(..)))
                );
            }
        }
    }
}
