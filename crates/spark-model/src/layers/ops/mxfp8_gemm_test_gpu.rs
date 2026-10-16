// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`; a GB10, cuBLASLt with MXFP8 and a kernel target
//! with `mxfp8_act_quant`) of `cublaslt::fp8_gemm_act_weight_t_mx` at the
//! FP4-Local prefill shapes of `cublaslt::MX_SWEEP` (TP2 rank 0's cut and
//! TP1; real weights with `ATLAS_FP4L_CKPT`, else random codes at the
//! checkpoint's scale range):
//!
//! - against an f64 host reference on sampled outputs;
//! - every cuBLASLt candidate (the unmasked heuristic list and the
//!   in-order-only one tuning asks for) at M = 1 .. 16384 and odd pass
//!   offsets, against one reference: every in-order kernel must give the
//!   same bytes (what lets `mx_algo` choose a kernel per row bucket for a
//!   shape this passes on: `MX_SWEEP`'s `verified_on`, see `mx_sweep.rs`);
//!   split-K differences are reported;
//! - a microbench (`bench`) of the tuned GEMM against BF16 cuBLASLt plus the
//!   quant's cost.
//!
//! ```text
//! ATLAS_FP4L_CKPT=<checkpoint dir> cargo test --release -p spark-model --lib \
//!     mxfp8_gemm -- --ignored --nocapture --test-threads=1
//! ```

use anyhow::{Context, Result};
use spark_runtime::cublaslt;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::mxfp8::rows_gpu_tests::real::tensor;
use super::super::mxfp8::rows_gpu_tests::{Rng, backend, upload};
use super::gpu_tests::{down, kernels, u16_bytes, unit};
use super::tests::{bf16, e4m3_to_f32, quant_rows};
use super::*;

/// A swept MXFP8 projection: `(label, n, k)` from `cublaslt::MX_SWEEP` and
/// how to cut it from the checkpoint: row ranges of stacked tensors, or a K
/// column range.
pub(super) struct Shape {
    label: &'static str,
    n: usize,
    k: usize,
    /// `(tensor suffix of layer `layer`, row ranges)`; empty for a K cut.
    rows: &'static [(&'static str, &'static [(usize, usize)])],
    /// `(tensor suffix, first column)` for a K cut.
    cols: Option<(&'static str, usize)>,
    layer: usize,
}

type Cut = (
    &'static [(&'static str, &'static [(usize, usize)])],
    Option<(&'static str, usize)>,
    usize,
);

/// Where each `MX_SWEEP` label's weight comes from: TP2 is rank 0's cut,
/// TP1 the whole projection. GDN layer 0, QSA layer 3.
fn cut(label: &str) -> Option<Cut> {
    const QKV: &str = "linear_attn.in_proj_qkv";
    const Z: &str = "linear_attn.in_proj_z";
    Some(match label {
        "tp2 gdn qkvz" => (
            &[
                (QKV, &[(0, 1024), (2048, 3072), (4096, 7168)]),
                (Z, &[(0, 3072)]),
            ],
            None,
            0,
        ),
        "tp1 gdn qkvz" => (&[(QKV, &[(0, 10240)]), (Z, &[(0, 6144)])], None, 0),
        "tp2 gdn out" | "tp1 gdn out" => (&[], Some(("linear_attn.out_proj", 0)), 0),
        "tp2 qsa q" => (&[("self_attn.q_proj", &[(0, 6144)])], None, 3),
        "tp1 qsa q" => (&[("self_attn.q_proj", &[(0, 12288)])], None, 3),
        "tp2 qsa k, v" => (&[("self_attn.k_proj", &[(0, 256)])], None, 3),
        "tp1 qsa k, v" => (&[("self_attn.k_proj", &[(0, 512)])], None, 3),
        "tp2 qsa o" | "tp1 qsa o" => (&[], Some(("self_attn.o_proj", 0)), 3),
        _ => return None,
    })
}

/// Every shape of `cublaslt::MX_SWEEP`, with its cut.
pub(super) fn shapes() -> Result<Vec<Shape>> {
    cublaslt::MX_SWEEP
        .iter()
        .map(|s| {
            let (rows, cols, layer) = cut(s.label)
                .with_context(|| format!("MX_SWEEP's {:?} has no checkpoint cut", s.label))?;
            Ok(Shape {
                label: s.label,
                n: s.n as usize,
                k: s.k as usize,
                rows,
                cols,
                layer,
            })
        })
        .collect()
}

#[test]
fn every_swept_shape_has_a_checkpoint_cut() {
    let shapes = shapes().unwrap();
    assert_eq!(shapes.len(), cublaslt::MX_SWEEP.len());
    for s in &shapes {
        // A row cut stacks exactly n rows; a K cut takes k of the columns.
        let stacked: usize = s
            .rows
            .iter()
            .flat_map(|(_, r)| r.iter().map(|(lo, hi)| hi - lo))
            .sum();
        assert!(
            (s.cols.is_some() && stacked == 0) || (s.cols.is_none() && stacked == s.n),
            "{}: {stacked} rows for N = {}",
            s.label,
            s.n
        );
    }
}

/// Row-major `(E4M3 [n, k], UE8M0 [n, k/32])` for `s`: the checkpoint's cut
/// with `ATLAS_FP4L_CKPT`, else random codes at the checkpoint's scale range.
pub(super) fn weights(s: &Shape, rng: &mut Rng) -> Result<(Vec<u8>, Vec<u8>)> {
    let Ok(dir) = std::env::var("ATLAS_FP4L_CKPT") else {
        let w = (0..s.n * s.k)
            .map(|_| match rng.next() as u8 {
                b @ (0x7f | 0xff) => b - 1,
                b => b,
            })
            .collect();
        let sc = (0..s.n * s.k / 32)
            .map(|_| 111 + (rng.next() % 8) as u8)
            .collect();
        return Ok((w, sc));
    };
    let name = |suffix: &str| format!("model.language_model.layers.{}.{suffix}", s.layer);
    let (mut w, mut sc) = (Vec::new(), Vec::new());
    if let Some((suffix, c0)) = s.cols {
        let (tw, ts, n, k) = tensor(&dir, &name(suffix))?;
        anyhow::ensure!(n == s.n, "{}: {n} rows", s.label);
        for r in 0..n {
            w.extend_from_slice(&tw[r * k + c0..r * k + c0 + s.k]);
            sc.extend_from_slice(&ts[r * k / 32 + c0 / 32..r * k / 32 + (c0 + s.k) / 32]);
        }
    }
    for (suffix, ranges) in s.rows {
        let (tw, ts, _, k) = tensor(&dir, &name(suffix))?;
        anyhow::ensure!(k == s.k, "{}: K {k}", s.label);
        for &(lo, hi) in *ranges {
            w.extend_from_slice(&tw[lo * k..hi * k]);
            sc.extend_from_slice(&ts[lo * k / 32..hi * k / 32]);
        }
    }
    anyhow::ensure!(w.len() == s.n * s.k, "{}: cut {} bytes", s.label, w.len());
    Ok((w, sc))
}

/// Activation-like BF16 rows: N(0,1) x a lognormal per channel x a
/// lognormal per token (the feasibility bench's distribution 0).
pub(super) fn activations(rng: &mut Rng, m: usize, k: usize) -> Vec<u16> {
    let gauss = |rng: &mut Rng| {
        let (u1, u2) = (unit(rng).max(1e-12), unit(rng));
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let ch: Vec<f64> = (0..k).map(|_| (0.5 * gauss(rng)).exp()).collect();
    let mut x = Vec::with_capacity(m * k);
    for _ in 0..m {
        let tk = (0.3 * gauss(rng)).exp();
        for c in &ch {
            let v = (gauss(rng) * c * tk) as f32;
            x.push((v.to_bits() >> 16) as u16);
        }
    }
    x
}

/// Device operands of one shape: the weight, its tiled scales and `max_m`
/// activation rows (BF16).
pub(super) struct Operands {
    w: DevicePtr,
    w_sf: DevicePtr,
    act: DevicePtr,
    host_w: (Vec<u8>, Vec<u8>),
    host_act: Vec<u16>,
}

pub(super) fn operands(
    gpu: &dyn GpuBackend,
    s: &Shape,
    max_m: usize,
    rng: &mut Rng,
) -> Result<Operands> {
    let (_, tile) = kernels(gpu)?;
    let host_w = weights(s, rng)?;
    let w = upload(gpu, &host_w.0)?;
    let rm = upload(gpu, &host_w.1)?;
    let w_sf = gpu.alloc(mx_lt_scale_bytes(s.n, s.k))?;
    mxfp8_lt_tile_scales(
        gpu,
        tile,
        rm,
        (s.k / 32) as u32,
        w_sf,
        s.n as u32,
        s.k as u32,
        gpu.default_stream(),
    )?;
    gpu.synchronize(gpu.default_stream())?;
    gpu.free(rm)?;
    let host_act = activations(rng, max_m, s.k);
    let act = upload(gpu, &u16_bytes(&host_act))?;
    Ok(Operands {
        w,
        w_sf,
        act,
        host_w,
        host_act,
    })
}

/// Quantize activation rows `[r0, r0 + m)` into fresh buffers `(q, sf)`.
pub(super) fn quantized(
    gpu: &dyn GpuBackend,
    o: &Operands,
    k: usize,
    r0: usize,
    m: usize,
) -> Result<(DevicePtr, DevicePtr)> {
    let (quant, _) = kernels(gpu)?;
    let q = gpu.alloc(m * k)?;
    let sf = gpu.alloc(mx_lt_scale_bytes(m, k))?;
    gpu.memset(sf, 0, mx_lt_scale_bytes(m, k))?;
    mxfp8_quant_act(
        gpu,
        quant,
        o.act.offset(r0 * k * 2),
        k as u32,
        q,
        sf,
        m as u32,
        k as u32,
        gpu.default_stream(),
    )?;
    Ok((q, sf))
}

/// Row counts (and pass offsets) the invariance check sweeps; every bucket
/// from 64 to 16384 and the odd heights in between.
const SWEEP: [(usize, usize); 11] = [
    (1, 0),
    (13, 37),
    (64, 64),
    (100, 5),
    (777, 1),
    (2048, 0),
    (2048, 64),
    (4096, 3),
    (5000, 0),
    (8192, 64),
    (16384, 0),
];
const SWEEP_ROWS: usize = 16384 + 64;

/// Whether a `mx_algo_description` names an in-order kernel.
fn in_order(d: &str) -> bool {
    d.contains("splitK=1 reduction=0") || d.contains("splitK=0 reduction=0")
}

#[test]
#[ignore = "needs a GB10 GPU and cuBLASLt with MXFP8"]
fn mxfp8_gemm_prepare_tunes_one_kernel_or_every_bucket_up_to_m_max() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let desc = cublaslt::mx_algo_description;
    // Never swept: ONE kernel for every row count, above m_max too. A shape
    // no other test uses, so the cache starts empty for it.
    let (n, k) = (384u32, 1536u32);
    assert!(!cublaslt::mx_per_bucket(n, k));
    assert!(desc(n, k, 64).is_none());
    cublaslt::fp8_gemm_mx_prepare(n, k, 3000, stream)?;
    let one = desc(n, k, 1).context("untuned")?;
    assert!(in_order(&one), "{one}");
    for m in [64, 65, 1000, 3000, 4096, 4097, 16384] {
        assert_eq!(desc(n, k, m).as_ref(), Some(&one), "{m} rows");
    }
    // Swept, on a cuBLASLt it passed on: a kernel per bucket up to m_max.
    let s = cublaslt::MX_SWEEP[0];
    if cublaslt::mx_per_bucket(s.n, s.k) {
        let above_untuned = desc(s.n, s.k, 8192).is_none();
        cublaslt::fp8_gemm_mx_prepare(s.n, s.k, 3000, stream)?;
        for m in [1, 64, 65, 128, 1000, 2048, 3000, 4096] {
            let d = desc(s.n, s.k, m).with_context(|| format!("bucket of {m} rows untuned"))?;
            assert!(in_order(&d), "{m} rows: {d}");
        }
        if above_untuned {
            assert!(desc(s.n, s.k, 8192).is_none(), "tuned past m_max");
        }
    } else {
        println!("cuBLASLt {}: not a swept version", cublaslt::version());
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU, cuBLASLt with MXFP8 and the mxfp8_act_quant kernels"]
fn mxfp8_gemm_is_correct_and_every_in_order_kernel_gives_the_same_rows() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    println!("cuBLASLt {}", cublaslt::version());
    let mut rng = Rng(0x0dd_ba11);
    for s in &shapes()? {
        let (n, k) = (s.n, s.k);
        let o = operands(gpu, s, SWEEP_ROWS, &mut rng)?;
        // Reference: the first in-order candidate over all SWEEP_ROWS rows.
        let out_ref = gpu.alloc(SWEEP_ROWS * n * 2)?;
        let (q, sf) = quantized(gpu, &o, k, 0, SWEEP_ROWS)?;
        let mut ref_desc = None;
        for i in 0..cublaslt::MX_CANDIDATES {
            let r = cublaslt::fp8_gemm_act_weight_t_mx_candidate(
                q.0,
                sf.0,
                o.w.0,
                o.w_sf.0,
                out_ref.0,
                [SWEEP_ROWS as u32, n as u32, k as u32],
                i,
                true,
                stream,
            )?;
            match r {
                Some((d, true)) => {
                    ref_desc = Some(d);
                    break;
                }
                Some(_) => {}
                None => break,
            }
        }
        let ref_desc = ref_desc.context("no in-order reference kernel")?;
        gpu.synchronize(stream)?;
        let full = down(gpu, out_ref, SWEEP_ROWS * n * 2)?;
        gpu.free(q)?;
        gpu.free(sf)?;
        // Correctness: sampled outputs against f64 over the dequantized
        // operands (the quant's own bytes, from the host reference).
        let (hq, hsf) = quant_rows(&o.host_act[..256 * k], 256, k);
        let pad = mx_lt_inner_pad(k);
        let mut worst = 0f64;
        for _ in 0..256 {
            let (r, c) = (
                (rng.next() % 256) as usize,
                (rng.next() % n as u64) as usize,
            );
            let (mut acc, mut mag) = (0f64, 0f64);
            for kk in 0..k {
                let a = f64::from(e4m3_to_f32(hq[r * k + kk]))
                    * 2f64.powi(i32::from(hsf[mx_lt_sf_offset(r, kk / 32, pad)]) - 127);
                let w = f64::from(e4m3_to_f32(o.host_w.0[c * k + kk]))
                    * 2f64.powi(i32::from(o.host_w.1[c * k / 32 + kk / 32]) - 127);
                acc += a * w;
                mag += (a * w).abs();
            }
            let got = f64::from(bf16(u16::from_le_bytes([
                full[(r * n + c) * 2],
                full[(r * n + c) * 2 + 1],
            ])));
            let tol = acc.abs() / 256.0 + mag * 1e-5 + 1e-30;
            worst = worst.max((got - acc).abs() / tol);
        }
        println!(
            "{} {n}x{k}: reference kernel {ref_desc}; worst sampled error {worst:.3} of tolerance",
            s.label
        );
        assert!(
            worst <= 1.0,
            "{}: MXFP8 GEMM off the f64 reference",
            s.label
        );

        // Invariance: every candidate at every (m, offset) against the
        // reference rows; the tuned entry point too.
        for &(m, r0) in &SWEEP {
            let (q, sf) = quantized(gpu, &o, k, r0, m)?;
            let out = gpu.alloc(m * n * 2)?;
            let want = &full[r0 * n * 2..(r0 + m) * n * 2];
            let mut line = Vec::new();
            let mut in_order_bad = 0;
            // The unmasked heuristic list, then the in-order-only one tuning
            // asks for (prefix `m`).
            for (masked, i) in [false, true]
                .into_iter()
                .flat_map(|b| (0..cublaslt::MX_CANDIDATES).map(move |i| (b, i)))
            {
                gpu.memset(out, 0xff, m * n * 2)?;
                let Some((d, ordered)) = cublaslt::fp8_gemm_act_weight_t_mx_candidate(
                    q.0,
                    sf.0,
                    o.w.0,
                    o.w_sf.0,
                    out.0,
                    [m as u32, n as u32, k as u32],
                    i,
                    masked,
                    stream,
                )?
                else {
                    continue;
                };
                gpu.synchronize(stream)?;
                let got = down(gpu, out, m * n * 2)?;
                let bad = got.iter().zip(want).filter(|(a, b)| a != b).count();
                if ordered && bad > 0 {
                    in_order_bad += 1;
                }
                line.push(format!(
                    "{}#{i} [{d}]{} {}",
                    if masked { "m" } else { "" },
                    if ordered { "" } else { " split" },
                    if bad == 0 {
                        "same".to_string()
                    } else {
                        format!("DIFF {bad}")
                    }
                ));
            }
            gpu.memset(out, 0xff, m * n * 2)?;
            cublaslt::fp8_gemm_act_weight_t_mx(
                q.0,
                sf.0,
                o.w.0,
                o.w_sf.0,
                out.0,
                [m as u32, n as u32, k as u32],
                stream,
            )?;
            gpu.synchronize(stream)?;
            let tuned_bad = down(gpu, out, m * n * 2)?
                .iter()
                .zip(want)
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "  m={m} at {r0}: tuned [{}] {} | {}",
                cublaslt::mx_algo_description(n as u32, k as u32, m as u32).unwrap_or_default(),
                if tuned_bad == 0 { "same" } else { "DIFF" },
                line.join("; ")
            );
            assert_eq!(
                in_order_bad, 0,
                "{} m={m}: an in-order kernel differs",
                s.label
            );
            assert_eq!(tuned_bad, 0, "{} m={m}: the tuned kernel differs", s.label);
            for p in [q, sf, out] {
                gpu.free(p)?;
            }
        }
        for p in [o.w, o.w_sf, o.act, out_ref] {
            gpu.free(p)?;
        }
    }
    Ok(())
}

#[path = "mxfp8_gemm_bench_gpu.rs"]
pub(super) mod bench;
