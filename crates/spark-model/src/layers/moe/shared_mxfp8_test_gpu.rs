// SPDX-License-Identifier: AGPL-3.0-only

//! GPU (`#[ignore]`; a GB10 and a qwen3.8-flash-next kernel build): FP4-Local's
//! MXFP8 shared expert ([`Mxfp8SharedExpert`]) against f64 host references
//! of its own math, on the checkpoint's layers 0, 3 and 47 with
//! `ATLAS_FP4L_CKPT` (read straight from the shards), else on random codes at
//! the checkpoint's scale range:
//!
//! 1. decode: the stacked gate/up on the plain tiers within FP32 summation
//!    error of the f64 sums rounded once; the `_swiglu` tiers equal
//!    `mx_swiglu` of those BF16 values (the same sums); down, from the GPU's
//!    own activation, within FP32 summation error of the f64 sum;
//! 2. decode rows are row-invariant: rows of 1-, 2-, 8-, 33- and 64-row
//!    passes equal the row alone, byte for byte (decode == verify == batch);
//! 3. prefill (W8A8): its gate/up and down GEMMs within FP32 summation error
//!    of f64 GEMMs of the operands it quantized; `mxfp8_swiglu_quant_act` of
//!    the plain tiers' gate/up equals `mxfp8_quant_act_bf16` of the
//!    `_swiglu` tiers' activation byte for byte; a 3000-row pass equals its
//!    rows run as two passes cut at 1344 (the SP halves), byte for byte; its
//!    distance from decode's BF16-activation output is printed.
//!
//! ```text
//! ATLAS_FP4L_CKPT=<model dir> cargo test --release -p spark-model --lib shared_mxfp8 \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::layers::ops::rows_gpu_tests::real::tensor;
use crate::layers::ops::rows_gpu_tests::{Rng, backend, upload};
use crate::layers::ops::{MxAct, mx_lt_inner_pad, mx_lt_scale_bytes, mx_lt_sf_offset};
use crate::weight_map::RawMxfp8;

const H: usize = 2560;
const I: usize = 640;
/// Decode rows, and the pass widths they are checked at.
const ROWS: usize = 64;
const WIDTHS: [usize; 5] = [1, 2, 8, 33, 64];
/// Prefill rows and the 64-aligned (not 128-aligned) cut.
const T: usize = 3000;
const CUT: usize = 1344;
const LAYERS: [usize; 3] = [0, 3, 47];

fn f(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf(x: f32) -> u16 {
    atlas_core::numeric::f32_to_bf16(x)
}

fn d2h(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The device's `mx_swiglu` in f32 on BF16 inputs.
fn swiglu(g: u16, u: u16) -> u16 {
    let g = f(g);
    bf(f(bf(g / (1.0 + (-g).exp()))) * f(u))
}

fn e4m3(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((b >> 3) & 0xf), f64::from(b & 7));
    sign * if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

/// `(E4M3 [n, k], UE8M0 [n, k/32])` as f64 `[n, k]`; `tiled` scales are in
/// the cuBLASLt layout, else row-major.
fn dequant(q: &[u8], sf: &[u8], n: usize, k: usize, tiled: bool) -> Vec<f64> {
    let pad = mx_lt_inner_pad(k);
    (0..n * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            let s = if tiled {
                sf[mx_lt_sf_offset(r, c / 32, pad)]
            } else {
                sf[r * (k / 32) + c / 32]
            };
            e4m3(q[i]) * 2f64.powi(i32::from(s) - 127)
        })
        .collect()
}

/// The checkpoint's (or random) `[gate, up, down]` of layer `layer`, as
/// row-major `(E4M3, UE8M0)` host bytes.
fn weights(layer: usize, rng: &mut Rng) -> Result<[(Vec<u8>, Vec<u8>); 3]> {
    let shapes = [(I, H), (I, H), (H, I)];
    let Ok(dir) = std::env::var("ATLAS_FP4L_CKPT") else {
        return Ok(shapes.map(|(n, k)| {
            let w = (0..n * k)
                .map(|_| match rng.next() as u8 {
                    b @ (0x7f | 0xff) => b - 1,
                    b => b,
                })
                .collect();
            let s = (0..n * k / 32)
                .map(|_| 113 + (rng.next() % 6) as u8)
                .collect();
            (w, s)
        }));
    };
    let mut out = Vec::new();
    for (name, (n, k)) in ["gate_proj", "up_proj", "down_proj"].iter().zip(shapes) {
        let p = format!("model.language_model.layers.{layer}.mlp.shared_expert.{name}");
        let (w, s, tn, tk) = tensor(&dir, &p)?;
        anyhow::ensure!((tn, tk) == (n, k), "{p}: [{tn}, {tk}]");
        out.push((w, s));
    }
    Ok(out.try_into().expect("three projections"))
}

/// Activation-like BF16 rows: N(0, 1) times a per-channel lognormal.
fn activations(rng: &mut Rng, m: usize) -> Vec<u16> {
    let mut unit = || ((rng.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let mut gauss = || (-2.0 * unit().ln()).sqrt() * (std::f64::consts::TAU * unit()).cos();
    let ch: Vec<f64> = (0..H).map(|_| (0.5 * gauss()).exp()).collect();
    (0..m * H)
        .map(|i| bf((gauss() * ch[i % H]) as f32))
        .collect()
}

/// Whether `got` is `exact` (f64) rounded once to BF16, give or take FP32
/// summation error over `abs` (the sum of |terms|): `|got - exact| <=
/// ulp/2 + 4e-6 * abs` (2^-18: a few dozen FP32 roundings of the terms).
fn within(got: u16, exact: f64, abs: f64) -> bool {
    let g = f64::from(f(got));
    let ulp = 2f64.powi(exact.abs().max(g.abs()).max(1e-30).log2().floor() as i32 - 7);
    (g - exact).abs() <= ulp / 2.0 + 4e-6 * abs
}

/// `C[m, n] = A[m, k] W[n, k]^T` and `sum |a w|`, in f64; `a` / `w` f64.
fn gemm(a: &[f64], w: &[f64], m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut c, mut abs) = (vec![0f64; m * n], vec![0f64; m * n]);
    for r in 0..m {
        for j in 0..n {
            let (x, y) = (&a[r * k..(r + 1) * k], &w[j * k..(j + 1) * k]);
            c[r * n + j] = x.iter().zip(y).map(|(p, q)| p * q).sum();
            abs[r * n + j] = x.iter().zip(y).map(|(p, q)| (p * q).abs()).sum();
        }
    }
    (c, abs)
}

/// How many of `got` are not [`within`] their reference.
fn misses(got: &[u16], (exact, abs): &(Vec<f64>, Vec<f64>)) -> usize {
    (0..got.len())
        .filter(|&i| !within(got[i], exact[i], abs[i]))
        .count()
}

struct Dev {
    raw: [RawMxfp8; 3],
    gate_up: crate::weight_map::Mxfp8Weight,
    host_w: [Vec<f64>; 3],
}

/// The raw tensors on the device, rank `rank` of `size`'s stacked gate/up
/// twin (the plain tiers' operand) and its host dequant `[gate, up, down]`
/// slices (gate/up rows, down columns).
fn upload_weights(
    gpu: &dyn GpuBackend,
    w: &[(Vec<u8>, Vec<u8>); 3],
    (rank, size): (usize, usize),
) -> Result<Dev> {
    let shapes = [(I, H), (I, H), (H, I)];
    let mut raw = Vec::new();
    for ((q, s), (n, k)) in w.iter().zip(shapes) {
        raw.push(RawMxfp8 {
            data: upload(gpu, q)?,
            scales: upload(gpu, s)?,
            n,
            k,
        });
    }
    let raw: [RawMxfp8; 3] = raw.try_into().expect("three");
    // The plain tiers' operand: [gate; up], as the expert stacks it.
    let (gate_up, ..) = crate::weight_map::mxfp8_rows_twin(
        gpu,
        &[(raw[0], &[I]), (raw[1], &[I])],
        rank,
        size,
        gpu.default_stream(),
    )?;
    let il = I / size;
    let host_w = [0, 1, 2].map(|i| {
        let (n, k) = shapes[i];
        let full = dequant(&w[i].0, &w[i].1, n, k, false);
        if i < 2 {
            full[rank * il * k..(rank + 1) * il * k].to_vec()
        } else {
            (0..n)
                .flat_map(|r| full[r * k + rank * il..r * k + (rank + 1) * il].to_vec())
                .collect()
        }
    });
    Ok(Dev {
        raw,
        gate_up,
        host_w,
    })
}

/// Layers' weights rotated so each pass streams from DRAM, as decode does.
const COPIES: usize = 48;
const ITERS: usize = 20;

#[test]
#[ignore = "needs a GB10 GPU and a qwen3.8-flash-next kernel build"]
fn shared_mxfp8_decode_bench() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let mut rng = Rng(0xbe9c_4a11);
    let x = upload(gpu, &bytes(&activations(&mut rng, 32)))?;
    let (act, out) = (gpu.alloc(32 * I * 2)?, gpu.alloc(32 * H * 2)?);
    // The whole expert (TP1) and a TP2 rank's slice.
    for size in [1, 2] {
        let mut experts = Vec::new();
        for _ in 0..COPIES {
            let w = weights(0, &mut rng)?;
            let raw = upload_weights(gpu, &w, (0, size))?.raw;
            experts.push(Mxfp8SharedExpert::from_raw(
                gpu,
                raw,
                [H, I],
                (0, size),
                None,
            )?);
        }
        for rows in [1, 2, 4, 8, 16, 32] {
            let run = || -> Result<f64> {
                gpu.synchronize(stream)?;
                let t = std::time::Instant::now();
                for _ in 0..ITERS {
                    for e in &experts {
                        e.decode(gpu, x, rows, act, out, stream)?;
                    }
                }
                gpu.synchronize(stream)?;
                Ok(t.elapsed().as_secs_f64() * 1e6 / (ITERS * COPIES) as f64)
            };
            run()?;
            let us = (0..3).map(|_| run()).collect::<Result<Vec<_>>>()?;
            let best = us.iter().cloned().fold(f64::INFINITY, f64::min);
            let bytes = 3 * H * I / size * 33 / 32;
            eprintln!(
                "MXFP8 shared expert decode, 1/{size} slice, {rows} rows: {best:.1} us a layer \
                 (eager launches, {:.0} GB/s of weights), {:.2} ms over 48 layers",
                bytes as f64 / best / 1e3,
                best * 48.0 / 1e3
            );
        }
    }
    Ok(())
}

/// Prefill, per layer and pass of `m` rows: the W8A8 arm against the
/// reference's other mode, A16 (BF16 activations: cuBLASLt BF16 GEMMs on the
/// exact BF16 dequant, which would cost a resident BF16 copy), event-free
/// host timing of back-to-back passes.
#[test]
#[ignore = "needs a GB10 GPU and a qwen3.8-flash-next kernel build"]
fn shared_mxfp8_prefill_bench() -> Result<()> {
    use spark_runtime::cublaslt;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let mut rng = Rng(0x9f11_5ea7);
    let w = weights(0, &mut rng)?;
    let dev = upload_weights(gpu, &w, (0, 1))?;
    let m_max = 16384;
    let mx = Mxfp8SharedExpert::from_raw(gpu, dev.raw, [H, I], (0, 1), Some(m_max))?;
    let bf = |v: &[f64]| bytes(&v.iter().map(|&x| bf(x as f32)).collect::<Vec<_>>());
    let (wg, wu, wd) = (
        upload(gpu, &bf(&dev.host_w[0]))?,
        upload(gpu, &bf(&dev.host_w[1]))?,
        upload(gpu, &bf(&dev.host_w[2]))?,
    );
    let silu = gpu.kernel("moe_silu_mul", "moe_silu_mul")?;
    let x = upload(gpu, &bytes(&activations(&mut rng, m_max)))?;
    let (g, u, gu, out) = (
        gpu.alloc(m_max * I * 2)?,
        gpu.alloc(m_max * I * 2)?,
        gpu.alloc(m_max * 2 * I * 2)?,
        gpu.alloc(m_max * H * 2)?,
    );
    let mxs = MxAct {
        q: gpu.alloc(m_max * H)?,
        sf: gpu.alloc(mx_lt_scale_bytes(m_max, H))?,
    };
    let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        f()?;
        gpu.synchronize(stream)?;
        let t = std::time::Instant::now();
        for _ in 0..10 {
            f()?;
        }
        gpu.synchronize(stream)?;
        Ok(t.elapsed().as_secs_f64() * 1e2)
    };
    for m in [512, 2048, 8192, 16384] {
        let w8 = time(&mut || mx.prefill(gpu, x, m, mxs, [g, gu], out, stream))?;
        let (mm, n, k) = (m as u32, I as u32, H as u32);
        let a16 = time(&mut || {
            cublaslt::bf16_gemm_act_weight_t(x.0, wg.0, g.0, mm, n, k, stream)?;
            cublaslt::bf16_gemm_act_weight_t(x.0, wu.0, u.0, mm, n, k, stream)?;
            ops::silu_mul(gpu, silu, g, u, g, mm * n, stream)?;
            cublaslt::bf16_gemm_act_weight_t(g.0, wd.0, out.0, mm, k, n, stream)
        })?;
        eprintln!(
            "MXFP8 shared expert prefill, {m} rows: W8A8 {w8:.3} ms, A16 (BF16 GEMMs) {a16:.3} ms a layer"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "shared_mxfp8_test_gpu_extra_tests.rs"]
mod extra_tests;
