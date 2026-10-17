// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`: a GB10, a glm-5.3-flash kernel build and the real
//! nvidia/GLM-5.3-Flash-FP4-Local checkpoint) of the native-format paths
//! (`ATLAS_GLM_FP4L_NATIVE`), each against host references computed from the
//! checkpoint's own bytes:
//!
//! - MLA layer 3 (both TP2 ranks): each FP8 block twin is the checkpoint's
//!   slice (values and f32 grid); `fp8b_gemv` rows at 1..64 equal the same
//!   rows at 64 (decode == verify row) and row 0 is within 1e-2 of an f64
//!   reference; the W8A16 and W8A8 prefill GEMMs are row-invariant (1..512
//!   rows) and within 1e-2 / 5e-2 of the reference; q_a + kv_a in one multi
//!   launch equal their own launches bit for bit.
//! - shared expert of layer 3: the twins are the checkpoint bytes; gate + up
//!   in one multi launch equal their own launches; the two ranks' split down
//!   partials sum to the full down within 1e-3 and the K-slice tier is
//!   row-invariant.
//! - dense MLP of layer 0: the MXFP8 twins are the checkpoint bytes; the
//!   exact expansion equals the host's bit for bit; gate + up rows at 1..64
//!   equal the same rows at 64, row 0 within 1e-2 of the reference.
//! - KDA layer 0 (both ranks): the fused q/k/v launch equals the three
//!   launches bit for bit at 1..32 rows.
//!
//! ```text
//! ATLAS_GLM_FP4L_CKPT=<checkpoint dir> cargo test --release -p spark-model --lib \
//!     glm5::fp4l::tests::native_gpu -- --ignored --nocapture --test-threads=1
//! ```

use super::gpu::{Host, backend, bf16s, f, l, tp2_rank, upload, with_parts};
use crate::layers::glm_native::NativePolicy;
use crate::layers::ops::{self, Fp8bPrefill, MxPart, native_kernels};
use crate::tp_shard::TpShardKind;
use crate::weight_map::{Mxfp8Weight, mxfp8_to_bf16_host};
use anyhow::{Context, Result, ensure};
use atlas_core::numeric::{f32_to_bf16, fp8_e4m3_to_f32};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// The CUDA backend, refusing a kernel build without every native kernel
/// (names them: a stale atlas-kernels build looks like "no kernel for N rows").
fn native_backend() -> Result<&'static spark_runtime::cuda_backend::AtlasCudaBackend> {
    let gpu = backend()?;
    let missing = ops::missing_native_kernels(gpu);
    ensure!(
        missing.is_empty(),
        "this kernel build lacks {missing:?}: rebuild atlas-kernels (touch \
         kernels/gb10/glm-5.3-flash/nvfp4/mxfp8_gemv.cu or cargo clean -p atlas-kernels)"
    );
    Ok(gpu)
}

/// `rows x k` pseudo-random BF16 activations (host bits and device copy).
fn activations(
    gpu: &dyn GpuBackend,
    rows: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<u16>, DevicePtr)> {
    let mut s = 0x9e37_79b9_7f4a_7c15_u64 ^ seed;
    let x: Vec<u16> = (0..rows * k)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f32_to_bf16(((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        })
        .collect();
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok((x, p))
}

/// The checkpoint's FP8 block tensor `stem` on the host: values, f32 grid, n, k.
fn host_fp8b(host: &Host, stem: &str) -> Result<(Vec<u8>, Vec<f32>, usize, usize)> {
    let (_, ws, w) = &host[&format!("{stem}.weight")];
    let (_, _, s) = &host[&format!("{stem}.weight_scale")];
    let grid = s
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((w.clone(), grid, ws[0], ws[1]))
}

/// Rows `[r0, r0 + ln)` x columns `[c0, c0 + lk)` of an FP8 block tensor.
fn slice_fp8b(
    (w, grid, _n, k): &(Vec<u8>, Vec<f32>, usize, usize),
    (r0, ln): (usize, usize),
    (c0, lk): (usize, usize),
) -> (Vec<u8>, Vec<f32>) {
    let values = (0..ln)
        .flat_map(|r| w[(r0 + r) * k + c0..(r0 + r) * k + c0 + lk].to_vec())
        .collect();
    let gk = k / 128;
    let grid = (0..ln / 128)
        .flat_map(|r| {
            grid[(r0 / 128 + r) * gk + c0 / 128..(r0 / 128 + r) * gk + (c0 + lk) / 128].to_vec()
        })
        .collect();
    (values, grid)
}

fn f32s(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn bytes(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// Worst `|got - ref| / sum |terms|` over output columns of row `row` against
/// `y = x · W^T` in f64, `weight(c, j)` the exact weight value.
fn worst_rel(
    got: &[u16],
    x: &[u16],
    row: usize,
    n: usize,
    k: usize,
    weight: impl Fn(usize, usize) -> f64,
) -> f64 {
    let mut worst = 0f64;
    for c in 0..n {
        let (mut y, mut mag) = (0f64, 0f64);
        for j in 0..k {
            let t = f(x[row * k + j]) as f64 * weight(c, j);
            y += t;
            mag += t.abs();
        }
        worst = worst.max((f(got[row * n + c]) as f64 - y).abs() / mag.max(1e-30));
    }
    worst
}

/// Rows of `run(rows)` (rows x n BF16 at `out`) at every width in `widths`
/// equal the same rows at the widest.
fn rows_invariant(
    gpu: &dyn GpuBackend,
    out: DevicePtr,
    n: usize,
    widths: &[usize],
    run: &dyn Fn(usize) -> Result<()>,
) -> Result<Vec<u16>> {
    let widest = *widths.iter().max().context("widths")?;
    run(widest)?;
    gpu.synchronize(gpu.default_stream())?;
    let full = bf16s(gpu, out, widest * n)?;
    for &m in widths {
        run(m)?;
        gpu.synchronize(gpu.default_stream())?;
        ensure!(
            bf16s(gpu, out, m * n)? == full[..m * n],
            "{m} rows differ from the same rows at {widest}"
        );
    }
    Ok(full)
}

const NATIVE: NativePolicy = NativePolicy {
    mla_fp8: true,
    shared_fp8: true,
    dense_mxfp8: true,
    kda_decode: true,
    kda_w8a8: false,
};

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn mla_layer_3_fp8_twins_are_checkpoint_slices_and_exact() -> Result<()> {
    let gpu = native_backend()?;
    let stream = gpu.default_stream();
    let nk = native_kernels(gpu);
    let p = l(3, "self_attn");
    let stems: Vec<String> = crate::weight_loader::glm5::mla_native::PROJECTIONS
        .iter()
        .map(|n| format!("{p}.{n}"))
        .collect();
    let (store, host) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;
    for rank in 0..2 {
        let c = tp2_rank(rank);
        let tp = crate::weight_loader::glm5::tp::MlaTpPlan::from_config(&c);
        let twins = crate::weight_loader::glm5::mla_native::mla_native_twins(
            NATIVE, &store, &p, &tp, &c, false, gpu,
        )?
        .context("FP4-Local ships the MLA projections FP8 block")?;
        let kinds = [
            TpShardKind::Replicated,
            TpShardKind::Replicated,
            TpShardKind::ColumnParallel,
            TpShardKind::RowParallel,
        ];
        for ((stem, (twin, [ln, lk])), kind) in stems.iter().zip(&twins).zip(kinds) {
            let full = host_fp8b(&host, stem)?;
            let (r0, c0) = match kind {
                TpShardKind::ColumnParallel => (rank * ln, 0),
                TpShardKind::RowParallel => (0, rank * lk),
                TpShardKind::Replicated => (0, 0),
            };
            let (want_w, want_s) = slice_fp8b(&full, (r0, *ln), (c0, *lk));
            ensure!(
                bytes(gpu, twin.data, ln * lk)? == want_w
                    && f32s(gpu, twin.scales, want_s.len())? == want_s,
                "rank {rank} {stem}: twin != checkpoint slice"
            );
            let (x, input) = activations(gpu, 512, *lk, rank as u64)?;
            let out = gpu.alloc(512 * ln * 2)?;
            let weight = |cc: usize, j: usize| {
                fp8_e4m3_to_f32(want_w[cc * lk + j]) as f64
                    * want_s[(cc / 128) * (lk / 128) + j / 128] as f64
            };
            // Decode / verify: the GEMV tiers at every width up to 64.
            let gemv = rows_invariant(
                gpu,
                out,
                *ln,
                &[1, 2, 5, 8, 9, 17, 31, 32, 33, 47, 64],
                &|m| ops::fp8b_gemv_rows(gpu, &nk.fp8b, twin, input, out, m, *ln, *lk, *ln, stream),
            )?;
            let e = worst_rel(&gemv, &x, 0, *ln, *lk, weight);
            ensure!(
                e < 1e-2,
                "rank {rank} {stem}: GEMV row 0 relative error {e}"
            );
            // Prefill: W8A16 (exact weights) and W8A8 (vLLM's recipe), both
            // row-invariant across chunk heights.
            let scratch = ops::Fp8ActScratch {
                act: gpu.alloc(512 * lk)?,
                act_bytes: 512 * lk,
                scale: gpu.alloc(512 * (lk / 128) * 4)?,
                scale_bytes: 512 * (lk / 128) * 4,
            };
            for (mode, tol) in [(Fp8bPrefill::W8a16, 1e-2), (Fp8bPrefill::W8a8, 5e-2)] {
                let full = rows_invariant(gpu, out, *ln, &[65, 128, 200, 512], &|m| {
                    ops::fp8b_prefill_gemm(
                        gpu, &nk, scratch, mode, twin, input, out, m, *ln, *lk, stream,
                    )
                })?;
                let e = worst_rel(&full, &x, 0, *ln, *lk, weight);
                ensure!(
                    e < tol,
                    "rank {rank} {stem}: {mode:?} row 0 relative error {e}"
                );
            }
            gpu.free(scratch.act)?;
            gpu.free(scratch.scale)?;
            gpu.free(input)?;
            gpu.free(out)?;
            eprintln!(
                "rank {rank} {stem}: twin == checkpoint, GEMV / prefill row-invariant, err {e:.2e}"
            );
        }
        // q_a + kv_a in one multi launch == their own launches.
        let (_, input) = activations(gpu, 32, 4096, 7)?;
        let outs = [
            gpu.alloc(32 * 1536 * 2)?,
            gpu.alloc(32 * 512 * 2)?,
            gpu.alloc(32 * 1536 * 2)?,
        ];
        for m in [1usize, 9, 32] {
            let parts = [
                MxPart::fp8b(&twins[0].0, outs[0], 1536),
                MxPart::fp8b(&twins[1].0, outs[1], 512),
            ];
            ops::mx_multi_rows(gpu, &nk.fp8b_multi, false, input, &parts, m, 4096, stream)?;
            gpu.synchronize(stream)?;
            for (i, n) in [(0usize, 1536usize), (1, 512)] {
                let fused = bf16s(gpu, outs[i], m * n)?;
                ops::fp8b_gemv_rows(
                    gpu,
                    &nk.fp8b,
                    &twins[i].0,
                    input,
                    outs[2],
                    m,
                    n,
                    4096,
                    n,
                    stream,
                )?;
                gpu.synchronize(stream)?;
                ensure!(
                    fused == bf16s(gpu, outs[2], m * n)?,
                    "q_a/kv_a fused {i} at {m} rows"
                );
            }
        }
        for o in outs.into_iter().chain([input]) {
            gpu.free(o)?;
        }
        for (t, _) in twins {
            t.free(gpu)?;
        }
    }
    Ok(())
}

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
    // Split down: rank partials (K-slices of the same twin) sum to the full.
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
    let mut partial = vec![vec![0f32; 32 * h]; 2];
    for (rank, part) in partial.iter_mut().enumerate() {
        let split = native.split(rank, h, inter);
        // This rank's half of the activation columns.
        let (_, act_half) = {
            let all = bf16s(gpu, act, 32 * inter)?;
            let half: Vec<u8> = (0..32)
                .flat_map(|r| all[r * inter + rank * 1024..r * inter + rank * 1024 + 1024].to_vec())
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let d = gpu.alloc(half.len())?;
            gpu.copy_h2d(&half, d)?;
            ((), d)
        };
        let got = rows_invariant(gpu, o[1], h, &[1, 4, 32], &|m| {
            ops::fp8b_gemv_ld_rows(
                gpu,
                &nk.fp8b_ld,
                split.down_data,
                split.down_scales,
                act_half,
                o[1],
                m,
                h,
                1024,
                inter,
                stream,
            )
        })?;
        *part = got.iter().map(|&b| f(b)).collect();
        gpu.free(act_half)?;
    }
    let mut worst = 0f32;
    for i in 0..32 * h {
        let sum = partial[0][i] + partial[1][i];
        worst = worst.max((sum - f(full[i])).abs() / f(full[i]).abs().max(1e-2));
    }
    ensure!(worst < 2e-2, "split partials vs full down: {worst}");
    eprintln!("shared: twins == checkpoint, fused == singles, split sum within {worst:.2e}");
    for p in o.into_iter().chain([x, act]) {
        gpu.free(p)?;
    }
    native.free(gpu)
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn dense_layer_0_mxfp8_twins_expand_and_project_exactly() -> Result<()> {
    let gpu = native_backend()?;
    let stream = gpu.default_stream();
    let nk = native_kernels(gpu);
    let p = l(0, "mlp");
    let stems: Vec<String> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|n| format!("{p}.{n}"))
        .collect();
    let (store, host) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;
    // Decode twins only (the W8A8 operands need a tuned cuBLASLt; m_max 0).
    let native =
        crate::weight_loader::glm5::dense::native_dense(NATIVE, &store, &p, [4096, 12288], gpu)?
            .context("FP4-Local ships the dense MLP MXFP8")?;
    for (stem, twin) in stems.iter().zip([native.gate, native.up, native.down]) {
        let (_, ws, w) = &host[&format!("{stem}.weight")];
        let (_, _, s) = &host[&format!("{stem}.weight_scale")];
        let (n, k) = (ws[0], ws[1]);
        ensure!(
            bytes(gpu, twin.data, n * k)? == *w && bytes(gpu, twin.scales, n * k / 32)? == *s,
            "{stem}: twin != checkpoint"
        );
        // The exact transient expansion == the host's BF16, bit for bit.
        let out = gpu.alloc(n * k * 2)?;
        ops::mxfp8_expand_bf16(gpu, nk.mx_expand, &twin, out, n, k, stream)?;
        gpu.synchronize(stream)?;
        let want = mxfp8_to_bf16_host(w, s, n, k)?;
        ensure!(bf16s(gpu, out, n * k)? == want, "{stem}: expansion != host");
        gpu.free(out)?;
        // GEMV rows invariant up to 64; row 0 against the f64 reference.
        let (x, input) = activations(gpu, 64, k, 11)?;
        let y = gpu.alloc(64 * n * 2)?;
        let got = rows_invariant(gpu, y, n, &[1, 3, 8, 16, 32, 33, 64], &|m| {
            ops::mxfp8_gemv_rows(gpu, &nk.mx.row_major, &twin, input, y, m, n, k, n, stream)
        })?;
        let e = worst_rel(&got, &x, 0, n.min(512), k, |c, j| f(want[c * k + j]) as f64);
        ensure!(e < 1e-2, "{stem}: row 0 relative error {e}");
        gpu.free(input)?;
        gpu.free(y)?;
        eprintln!("{stem}: twin == checkpoint, expansion exact, rows invariant, err {e:.2e}");
    }
    native.free(gpu)
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn kda_layer_0_fused_qkv_equals_three_launches() -> Result<()> {
    use crate::weight_loader::glm5::kda_mx::{KdaMxPolicy, kda_twin};
    let gpu = native_backend()?;
    let stream = gpu.default_stream();
    let nk = native_kernels(gpu);
    let stems = ["q_proj", "k_proj", "v_proj"].map(|n| l(0, &format!("self_attn.{n}")));
    let (store, _) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;
    let policy = KdaMxPolicy {
        decode: true,
        w8a8_prefill: false,
    };
    for rank in 0..2 {
        let twins: Vec<Mxfp8Weight> = stems
            .iter()
            .map(|stem| {
                kda_twin(
                    policy,
                    &store,
                    stem,
                    [8192, 4096],
                    TpShardKind::ColumnParallel,
                    rank,
                    2,
                    gpu,
                )?
                .and_then(|m| m.decode)
                .context("twin")
            })
            .collect::<Result<_>>()?;
        let (_, x) = activations(gpu, 32, 4096, rank as u64)?;
        let outs = [0, 1, 2, 3].map(|_| gpu.alloc(32 * 4096 * 2).unwrap());
        for m in [1usize, 2, 5, 8, 9, 16, 17, 32] {
            let parts: Vec<MxPart> = (0..3)
                .map(|i| MxPart::mx(&twins[i], outs[i], 4096))
                .collect();
            ops::mx_multi_rows(
                gpu,
                &nk.mx_multi,
                twins[0].tiled,
                x,
                &parts,
                m,
                4096,
                stream,
            )?;
            gpu.synchronize(stream)?;
            for i in 0..3 {
                let fused = bf16s(gpu, outs[i], m * 4096)?;
                ops::mxfp8_gemv_rows(
                    gpu,
                    &nk.mx.row_major,
                    &twins[i],
                    x,
                    outs[3],
                    m,
                    4096,
                    4096,
                    4096,
                    stream,
                )?;
                gpu.synchronize(stream)?;
                ensure!(
                    fused == bf16s(gpu, outs[3], m * 4096)?,
                    "rank {rank} {} at {m} rows",
                    stems[i]
                );
            }
        }
        eprintln!("rank {rank}: fused q/k/v == three launches at 1..32 rows");
        for p in outs.into_iter().chain([x]) {
            gpu.free(p)?;
        }
        for t in twins {
            gpu.free(t.data)?;
            gpu.free(t.scales)?;
        }
    }
    Ok(())
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
