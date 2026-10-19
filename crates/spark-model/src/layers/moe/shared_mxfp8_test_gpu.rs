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

fn upload_weights(gpu: &dyn GpuBackend, w: &[(Vec<u8>, Vec<u8>); 3]) -> Result<Dev> {
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
        0,
        1,
        gpu.default_stream(),
    )?;
    let host_w = [0, 1, 2].map(|i| {
        let (n, k) = shapes[i];
        dequant(&w[i].0, &w[i].1, n, k, false)
    });
    Ok(Dev {
        raw,
        gate_up,
        host_w,
    })
}

#[test]
#[ignore = "needs a GB10 GPU and a qwen3.8-flash-next kernel build"]
fn shared_mxfp8_matches_f64_and_is_row_invariant() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let mut rng = Rng(0x5a1e_d0c7_3e1f_0b22);
    let plain = [
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc8")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc16")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc32")?,
    ];
    let quant = gpu.kernel(ops::MXFP8_ACT_MODULE, ops::MX_QUANT_KERNEL)?;
    let swiglu_quant = gpu.kernel("mxfp8_gemv", "mxfp8_swiglu_quant_act")?;
    for layer in LAYERS {
        let w = weights(layer, &mut rng)?;
        let dev = upload_weights(gpu, &w)?;
        let mx = Mxfp8SharedExpert::from_raw(gpu, dev.raw, [H, I], Some(T))?;
        let x = activations(&mut rng, T);
        let xd = upload(gpu, &bytes(&x))?;
        let (gu, act, out) = (
            gpu.alloc(T * 2 * I * 2)?,
            gpu.alloc(T * I * 2)?,
            gpu.alloc(T * H * 2)?,
        );
        let xf: Vec<f64> = x.iter().map(|&v| f64::from(f(v))).collect();

        // 1. Decode, stage by stage, on the first ROWS rows.
        ops::mxfp8_gemv_rows(
            gpu,
            &plain,
            &dev.gate_up,
            xd,
            gu,
            ROWS,
            2 * I,
            H,
            2 * I,
            stream,
        )?;
        mx.decode(gpu, xd, ROWS, act, out, stream)?;
        gpu.synchronize(stream)?;
        let gu_h = u16s(&d2h(gpu, gu, ROWS * 2 * I * 2)?);
        let act_h = u16s(&d2h(gpu, act, ROWS * I * 2)?);
        let dec = u16s(&d2h(gpu, out, ROWS * H * 2)?);
        let stacked: Vec<f64> = [&dev.host_w[0][..], &dev.host_w[1][..]].concat();
        let gu_miss = misses(&gu_h, &gemm(&xf[..ROWS * H], &stacked, ROWS, 2 * I, H));
        let (mut act_exact, mut act_off) = (0usize, 0usize);
        for r in 0..ROWS {
            for j in 0..I {
                let want = swiglu(gu_h[r * 2 * I + j], gu_h[r * 2 * I + I + j]);
                let got = act_h[r * I + j];
                act_exact += usize::from(got == want);
                act_off += usize::from((f(got) - f(want)).abs() > (f(want).abs() * 2f32.powi(-7)));
            }
        }
        let act_f: Vec<f64> = act_h.iter().map(|&v| f64::from(f(v))).collect();
        let dn_miss = misses(&dec, &gemm(&act_f, &dev.host_w[2], ROWS, H, I));
        eprintln!(
            "layer {layer} decode: gate/up {gu_miss}/{} outside f64 +- FP32 sum error; SiLU*up \
             {act_exact}/{} = mx_swiglu of the plain sums ({act_off} more than 1 ulp off); \
             down {dn_miss}/{} outside",
            ROWS * 2 * I,
            ROWS * I,
            ROWS * H
        );
        assert_eq!(gu_miss, 0, "layer {layer}: gate/up");
        assert!(
            act_off == 0 && act_exact * 1000 >= ROWS * I * 999,
            "layer {layer}: SiLU*up"
        );
        assert_eq!(dn_miss, 0, "layer {layer}: down");

        // 2. Row invariance: every pass width against the row alone.
        let pass = |w: usize| -> Result<Vec<u16>> {
            gpu.memset_async(out, 0xa5, ROWS * H * 2, stream)?;
            for r0 in (0..ROWS).step_by(w) {
                let n = w.min(ROWS - r0);
                mx.decode(
                    gpu,
                    xd.offset(r0 * H * 2),
                    n,
                    act,
                    out.offset(r0 * H * 2),
                    stream,
                )?;
            }
            gpu.synchronize(stream)?;
            Ok(u16s(&d2h(gpu, out, ROWS * H * 2)?))
        };
        let alone = pass(1)?;
        assert_eq!(alone, dec, "layer {layer}: the 64-row pass vs rows alone");
        for w in WIDTHS {
            assert!(
                pass(w)? == alone,
                "layer {layer}: {w}-row passes vs rows alone"
            );
        }

        // 3. Prefill (W8A8).
        let scratch = |m: usize| -> Result<MxAct> {
            Ok(MxAct {
                q: gpu.alloc(m * H)?,
                sf: gpu.alloc(mx_lt_scale_bytes(m, H))?,
            })
        };
        let (s1, s2) = (scratch(T)?, scratch(T)?);
        mx.prefill(gpu, xd, T, s1, [act, gu], out, stream)?;
        gpu.synchronize(stream)?;
        let pre = u16s(&d2h(gpu, out, T * H * 2)?);
        let gu_p = u16s(&d2h(gpu, gu, ROWS * 2 * I * 2)?);
        // Row r's tiled scales sit where they would in a ROWS-row operand.
        let aq = d2h(gpu, s1.q, ROWS * I)?;
        let asf = d2h(gpu, s1.sf, mx_lt_scale_bytes(ROWS, I))?;
        // The input's image, as prefill quantized it (the same kernel).
        ops::mx_quant_rows(gpu, quant, xd, s2, 0..ROWS, H, stream)?;
        gpu.synchronize(stream)?;
        let xq = dequant(
            &d2h(gpu, s2.q, ROWS * H)?,
            &d2h(gpu, s2.sf, mx_lt_scale_bytes(ROWS, H))?,
            ROWS,
            H,
            true,
        );
        let pgu_miss = misses(&gu_p, &gemm(&xq, &stacked, ROWS, 2 * I, H));
        let a = dequant(&aq, &asf, ROWS, I, true);
        let pdn_miss = misses(&pre[..ROWS * H], &gemm(&a, &dev.host_w[2], ROWS, H, I));
        // SiLU quant of the plain tiers' gate/up == quant of the `_swiglu`
        // tiers' activation (the same sums, `mx_swiglu` on both sides).
        let image = |s: MxAct| -> Result<(Vec<u8>, Vec<u8>)> {
            gpu.synchronize(stream)?;
            Ok((
                d2h(gpu, s.q, ROWS * I)?,
                d2h(gpu, s.sf, mx_lt_scale_bytes(ROWS, I))?,
            ))
        };
        let (m, i) = (ROWS as u32, I as u32);
        ops::mxfp8_gemv_rows(
            gpu,
            &plain,
            &dev.gate_up,
            xd,
            gu,
            ROWS,
            2 * I,
            H,
            2 * I,
            stream,
        )?;
        ops::mxfp8_swiglu_quant_act(gpu, swiglu_quant, gu, s2.q, s2.sf, m, i, stream)?;
        let from_plain = image(s2)?;
        mx.decode(gpu, xd, ROWS, act, out, stream)?;
        ops::mxfp8_quant_act(gpu, quant, act, i, s2.q, s2.sf, m, i, stream)?;
        let from_swiglu = image(s2)?;
        assert!(
            from_plain == from_swiglu,
            "layer {layer}: SiLU quant twins differ"
        );
        // The SP halves.
        gpu.memset_async(out, 0xa5, T * H * 2, stream)?;
        for (r0, n) in [(0, CUT), (CUT, T - CUT)] {
            let o = r0 * H * 2;
            mx.prefill(gpu, xd.offset(o), n, s1, [act, gu], out.offset(o), stream)?;
        }
        gpu.synchronize(stream)?;
        let halves = u16s(&d2h(gpu, out, T * H * 2)?);
        let (mut d2, mut n2) = (0f64, 0f64);
        for (p, q) in pre[..ROWS * H].iter().zip(&dec) {
            d2 += (f64::from(f(*p)) - f64::from(f(*q))).powi(2);
            n2 += f64::from(f(*q)).powi(2);
        }
        eprintln!(
            "layer {layer} prefill: gate/up {pgu_miss}/{} and down {pdn_miss}/{} outside f64 +- \
             FP32 sum error of their operands; SiLU quant twins equal; {T} rows == halves at \
             {CUT}: {}; rel-L2 W8A8 vs decode {:.2e}",
            ROWS * 2 * I,
            ROWS * H,
            halves == pre,
            (d2 / n2).sqrt()
        );
        assert_eq!(pgu_miss, 0, "layer {layer}: prefill gate/up");
        assert_eq!(pdn_miss, 0, "layer {layer}: prefill down");
        assert!(
            halves == pre,
            "layer {layer}: the SP halves differ from the whole chunk"
        );
        for p in [
            xd,
            gu,
            act,
            out,
            s1.q,
            s1.sf,
            s2.q,
            s2.sf,
            dev.gate_up.data,
            dev.gate_up.scales,
        ] {
            gpu.free(p)?;
        }
        for r in dev.raw {
            gpu.free(r.data)?;
            gpu.free(r.scales)?;
        }
    }
    Ok(())
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
    let mut experts = Vec::new();
    for _ in 0..COPIES {
        let w = weights(0, &mut rng)?;
        experts.push(Mxfp8SharedExpert::from_raw(
            gpu,
            upload_weights(gpu, &w)?.raw,
            [H, I],
            None,
        )?);
    }
    let x = upload(gpu, &bytes(&activations(&mut rng, 32)))?;
    let (act, out) = (gpu.alloc(32 * I * 2)?, gpu.alloc(32 * H * 2)?);
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
        let bytes = 3 * H * I * 33 / 32;
        eprintln!(
            "MXFP8 shared expert decode, {rows} rows: {best:.1} us a layer (eager launches, \
             {:.0} GB/s of weights), {:.2} ms over 48 layers",
            bytes as f64 / best / 1e3,
            best * 48.0 / 1e3
        );
    }
    Ok(())
}
