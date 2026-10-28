// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn shared_expert_layer_3_fp8_twins_full_and_split() -> Result<()> {
    let gpu = native_backend()?;
    let stream = gpu.default_stream();
    let nk = native_kernels(gpu);
    let p = l(3, "mlp.shared_experts");
    let stems: Vec<String> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|n| format!("{p}.{n}"))
        .collect();
    let (store, host) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;
    let config = tp2_rank(0);
    let native = crate::weight_loader::glm5::shared_native::shared_native_twins(
        NATIVE, &store, &p, &config, gpu,
    )?
    .context("FP4-Local ships the shared experts FP8 block")?;
    for (stem, twin) in stems.iter().zip([native.gate, native.up, native.down]) {
        let (w, s, n, k) = host_fp8b(&host, stem)?;
        ensure!(
            bytes(gpu, twin.data, n * k)? == w && f32s(gpu, twin.scales, s.len())? == s,
            "{stem}: twin != checkpoint"
        );
    }
    let (h, inter) = (4096usize, 2048usize);
    let (_, x) = activations(gpu, 32, h, 3)?;
    let (_, act) = activations(gpu, 32, inter, 4)?;
    let o = [0, 1, 2, 3].map(|_| gpu.alloc(32 * h * 2).unwrap());
    // gate + up fused == their own launches.
    for m in [1usize, 8, 32] {
        let parts = [
            MxPart::fp8b(&native.gate, o[0], inter),
            MxPart::fp8b(&native.up, o[1], inter),
        ];
        ops::mx_multi_rows(gpu, &nk.fp8b_multi, false, x, &parts, m, h, stream)?;
        gpu.synchronize(stream)?;
        for (i, w) in [native.gate, native.up].iter().enumerate() {
            let fused = bf16s(gpu, o[i], m * inter)?;
            ops::fp8b_gemv_rows(gpu, &nk.fp8b, w, x, o[2], m, inter, h, inter, stream)?;
            gpu.synchronize(stream)?;
            ensure!(
                fused == bf16s(gpu, o[2], m * inter)?,
                "shared fused {i} at {m} rows"
            );
        }
    }
    // The serve-time split (`run_shared_native_split`), rank by rank:
    // gate / up on the rank's row slices of the twins equal those rows of the
    // full projections bit for bit (same CTA bodies over the same rows), and
    // down's K-slice tier (`fp8b_gemv_tc*_ld`) is within 1e-2 of an f64
    // reference of the rank's partial, built from the checkpoint's bytes.
    let (dw, ds, _, _) = host_fp8b(&host, &stems[2])?;
    let act_host = bf16s(gpu, act, 32 * inter)?;
    let full_gate = {
        ops::fp8b_gemv_rows(
            gpu,
            &nk.fp8b,
            &native.gate,
            x,
            o[0],
            32,
            inter,
            h,
            inter,
            stream,
        )?;
        ops::fp8b_gemv_rows(
            gpu, &nk.fp8b, &native.up, x, o[1], 32, inter, h, inter, stream,
        )?;
        gpu.synchronize(stream)?;
        [bf16s(gpu, o[0], 32 * inter)?, bf16s(gpu, o[1], 32 * inter)?]
    };
    let full = rows_invariant(gpu, o[0], h, &[1, 4, 32], &|m| {
        ops::fp8b_gemv_rows(
            gpu,
            &nk.fp8b,
            &native.down,
            act,
            o[0],
            m,
            h,
            inter,
            h,
            stream,
        )
    })?;
    let down_ref = |row: usize, c0: usize, k: usize, ncol: usize| -> f64 {
        (0..k)
            .map(|j| {
                f(act_host[row * inter + c0 + j]) as f64
                    * fp8_e4m3_to_f32(dw[ncol * inter + c0 + j]) as f64
                    * ds[(ncol / 128) * (inter / 128) + (c0 + j) / 128] as f64
            })
            .sum()
    };
    // The pair's halves and the chain's quarters.
    for ranks in [2usize, 4] {
        let part = inter / ranks;
        let mut partial = vec![vec![0f32; 32 * h]; ranks];
        for (rank, slot) in partial.iter_mut().enumerate() {
            let split = native.split(rank, ranks, h, inter);
            let c0 = rank * part;
            // gate / up row slices == rows [c0, c0 + part) of the full outputs.
            let parts = [
                MxPart::fp8b(&split.gate, o[2], part),
                MxPart::fp8b(&split.up, o[3], part),
            ];
            ops::mx_multi_rows(gpu, &nk.fp8b_multi, false, x, &parts, 32, h, stream)?;
            gpu.synchronize(stream)?;
            for (i, out) in [o[2], o[3]].into_iter().enumerate() {
                let got = bf16s(gpu, out, 32 * part)?;
                let want: Vec<u16> = (0..32)
                    .flat_map(|r| full_gate[i][r * inter + c0..r * inter + c0 + part].to_vec())
                    .collect();
                ensure!(
                    got == want,
                    "rank {rank}/{ranks}: split {} rows != the full projection's",
                    ["gate", "up"][i]
                );
            }
            // down's K-slice on this rank's activation columns.
            let act_part = {
                let part_bytes: Vec<u8> = (0..32)
                    .flat_map(|r| act_host[r * inter + c0..r * inter + c0 + part].to_vec())
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let d = gpu.alloc(part_bytes.len())?;
                gpu.copy_h2d(&part_bytes, d)?;
                d
            };
            let got = rows_invariant(gpu, o[1], h, &[1, 4, 32], &|m| {
                ops::fp8b_gemv_ld_rows(
                    gpu,
                    &nk.fp8b_ld,
                    split.down_data,
                    split.down_scales,
                    act_part,
                    o[1],
                    m,
                    h,
                    part,
                    inter,
                    stream,
                )
            })?;
            let mut worst = 0f64;
            for row in 0..4 {
                for ncol in 0..h {
                    let y = down_ref(row, c0, part, ncol);
                    let mag: f64 = (0..part)
                        .map(|j| {
                            (f(act_host[row * inter + c0 + j]) as f64
                                * fp8_e4m3_to_f32(dw[ncol * inter + c0 + j]) as f64
                                * ds[(ncol / 128) * (inter / 128) + (c0 + j) / 128] as f64)
                                .abs()
                        })
                        .sum();
                    worst = worst.max((f(got[row * h + ncol]) as f64 - y).abs() / mag.max(1e-30));
                }
            }
            ensure!(
                worst < 1e-2,
                "rank {rank}/{ranks}: split down vs f64 reference {worst:.3e}"
            );
            eprintln!(
                "rank {rank}/{ranks}: split gate/up == full rows, split down within {worst:.2e} of f64"
            );
            *slot = got.iter().map(|&b| f(b)).collect();
            gpu.free(act_part)?;
        }
        // The partials sum to the full down within their own rounding: each is
        // a BF16 word, so the bound scales with sum |p|, not with the (possibly
        // cancelled) sum.
        let mut worst = 0f32;
        for i in 0..32 * h {
            let (sum, mag) = partial
                .iter()
                .fold((0f32, 0f32), |(s, m), p| (s + p[i], m + p[i].abs()));
            let bound = mag + f(full[i]).abs() + 1e-6;
            worst = worst.max((sum - f(full[i])).abs() / bound);
        }
        ensure!(worst < 2e-2, "{ranks} split partials vs full down: {worst}");
        eprintln!("shared: {ranks} ranks' split sum within {worst:.2e} of the full down");
    }
    eprintln!("shared: twins == checkpoint, fused == singles");
    for p in o.into_iter().chain([x, act]) {
        gpu.free(p)?;
    }
    native.free(gpu)
}
/// Every native W8A8 prefill path gives a row the same bytes at every pass
/// height and position: the pinned cuBLASLt MXFP8 kernel (KDA layer 0 q,
/// dense layer 0 gate and down) and the block-FP8 GEMM (MLA layer 3 o, shared
/// layer 3 gate). The first `h` rows at each height and a 30-row window
/// ending at row `h` computed alone equal the same rows of a 16384-row pass.
#[test]
#[ignore = "needs a GB10, cuBLASLt with MXFP8, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn native_w8a8_prefill_rows_are_height_invariant() -> Result<()> {
    use crate::layers::ops::{MxAct, mx_lt_scale_bytes, mx_quant_rows};
    use crate::weight_loader::glm5::kda_mx::{KdaMxPolicy, kda_twin};
    const HEIGHTS: [usize; 6] = [1, 30, 64, 2048, 8192, 16384];
    const TALL: usize = 16384;
    let gpu = native_backend()?;
    let stream = gpu.default_stream();
    let nk = native_kernels(gpu);
    let kda = l(0, "self_attn.q_proj");
    let dense = l(0, "mlp");
    let mla_o = l(3, "self_attn.o_proj");
    let shared = l(3, "mlp.shared_experts");
    let mut stems = vec![kda.clone(), mla_o.clone(), format!("{shared}.gate_proj")];
    stems.extend(["gate_proj", "up_proj", "down_proj"].map(|p| format!("{dense}.{p}")));
    let (store, _) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;

    // (label, n, k, run(x, out, rows)) for every path under test.
    let kda_w8a8 = kda_twin(
        KdaMxPolicy {
            decode: false,
            w8a8_prefill: true,
        },
        &store,
        &kda,
        [8192, 4096],
        TpShardKind::ColumnParallel,
        0,
        2,
        gpu,
    )?
    .and_then(|m| m.w8a8)
    .context("KDA W8A8 operand")?;
    let dense_native = crate::weight_loader::glm5::dense::native_dense(
        NATIVE,
        &store,
        &dense,
        [4096, 12288],
        gpu,
    )?
    .context("dense twins")?;
    let dense_ops = dense_native
        .w8a8
        .context("dense W8A8 operands (ATLAS_GLM_FP4L_FP8_PREFILL unset)")?;
    // MLA o: this rank's column slice of the checkpoint's FP8 block bytes.
    let raw = crate::weight_map::raw_fp8_block(&store, &mla_o)?.context("MLA o FP8 block")?;
    let (o_twin, on, ok) =
        crate::weight_map::fp8_block_twin(gpu, raw, TpShardKind::RowParallel, 0, 2, stream)?;
    let raw = crate::weight_map::raw_fp8_block(&store, &format!("{shared}.gate_proj"))?
        .context("shared gate")?;
    let (g_twin, gn, gk) =
        crate::weight_map::fp8_block_twin(gpu, raw, TpShardKind::Replicated, 0, 1, stream)?;

    let kmax = 12288;
    let quant = nk.mx_quant;
    let act = MxAct {
        q: gpu.alloc(TALL * kmax)?,
        sf: gpu.alloc(mx_lt_scale_bytes(TALL, kmax))?,
    };
    let scratch = ops::Fp8ActScratch {
        act: act.q,
        act_bytes: TALL * kmax,
        scale: act.sf,
        scale_bytes: TALL * (kmax / 128) * 4,
    };
    let mx = |w: crate::layers::ops::MxW8a8| {
        move |x: DevicePtr, out: DevicePtr, rows: usize| -> Result<()> {
            mx_quant_rows(gpu, quant, x, act, 0..rows, w.k, stream)?;
            w.gemm(act, out, rows, stream)
        }
    };
    let fb = |w: crate::weight_map::Fp8BlockWeight, n: usize, k: usize| {
        move |x: DevicePtr, out: DevicePtr, rows: usize| -> Result<()> {
            ops::fp8b_prefill_gemm(
                gpu,
                &nk,
                scratch,
                Fp8bPrefill::W8a8,
                &w,
                x,
                out,
                rows,
                n,
                k,
                stream,
            )
        }
    };
    type Run<'a> = Box<dyn Fn(DevicePtr, DevicePtr, usize) -> Result<()> + 'a>;
    let paths: Vec<(&str, usize, usize, Run)> = vec![
        (
            "KDA q MXFP8 W8A8 (pinned)",
            kda_w8a8.n,
            kda_w8a8.k,
            Box::new(mx(kda_w8a8)),
        ),
        (
            "dense gate MXFP8 W8A8 (pinned)",
            dense_ops[0].n,
            dense_ops[0].k,
            Box::new(mx(dense_ops[0])),
        ),
        (
            "dense down MXFP8 W8A8 (pinned)",
            dense_ops[2].n,
            dense_ops[2].k,
            Box::new(mx(dense_ops[2])),
        ),
        ("MLA o FP8 block W8A8", on, ok, Box::new(fb(o_twin, on, ok))),
        (
            "shared gate FP8 block W8A8",
            gn,
            gk,
            Box::new(fb(g_twin, gn, gk)),
        ),
    ];
    for (label, n, k, run) in &paths {
        let (_, x) = activations(gpu, TALL, *k, *n as u64)?;
        let (full, part) = (gpu.alloc(TALL * n * 2)?, gpu.alloc(TALL * n * 2)?);
        run(x, full, TALL)?;
        gpu.synchronize(stream)?;
        let want = bf16s(gpu, full, TALL * n)?;
        for h in HEIGHTS {
            run(x, part, h)?;
            gpu.synchronize(stream)?;
            ensure!(
                bf16s(gpu, part, h * n)? == want[..h * n],
                "{label}: the first {h} rows differ from the {TALL}-row pass"
            );
            let m = h.min(30);
            let r0 = h - m;
            run(x.offset(r0 * k * 2), part, m)?;
            gpu.synchronize(stream)?;
            ensure!(
                bf16s(gpu, part, m * n)? == want[r0 * n..h * n],
                "{label}: rows {r0}..{h} alone differ from the same rows of the {TALL}-row pass"
            );
        }
        eprintln!("{label}: rows invariant at heights {HEIGHTS:?} and alone");
        for p in [x, full, part] {
            gpu.free(p)?;
        }
    }
    drop(paths);
    gpu.free(act.q)?;
    gpu.free(act.sf)?;
    o_twin.free(gpu)?;
    g_twin.free(gpu)?;
    kda_w8a8.free(gpu)?;
    gpu.free(kda_w8a8.data)?;
    dense_native.free(gpu)
}
