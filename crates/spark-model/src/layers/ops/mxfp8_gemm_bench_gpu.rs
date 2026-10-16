// SPDX-License-Identifier: AGPL-3.0-only

//! Microbench (`#[ignore]`) of the tuned MXFP8 W8A8 GEMM against BF16
//! cuBLASLt at the TP2 per-rank FP4-Local prefill shapes, M = 2048 ..
//! 16384, plus the activation quant's cost and every candidate's
//! event-timed cost; then the projections' total for a 16K-token prompt.

use anyhow::Result;
use spark_runtime::cublaslt;
use spark_runtime::gpu::GpuBackend;

use super::super::super::mxfp8::rows_gpu_tests::{Rng, backend, upload};
use super::super::gpu_tests::{kernels, u16_bytes};
use super::super::tests::e4m3_to_f32;
use super::super::*;
use super::{SHAPES, operands, quantized};

/// Median ms per call of `f` (5 runs of `iters` back-to-back calls).
fn time_ms(gpu: &dyn GpuBackend, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    let stream = gpu.default_stream();
    for _ in 0..3 {
        f()?;
    }
    gpu.synchronize(stream)?;
    let mut t = Vec::new();
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f()?;
        }
        gpu.synchronize(stream)?;
        t.push(t0.elapsed().as_secs_f64() * 1e3 / iters as f64);
    }
    t.sort_by(f64::total_cmp);
    Ok(t[2])
}

#[test]
#[ignore = "needs a GB10 GPU, cuBLASLt with MXFP8 and the mxfp8_act_quant kernels"]
fn mxfp8_gemm_bench_against_bf16_cublaslt() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let (quant, _) = kernels(gpu)?;
    println!(
        "cuBLASLt {}; weights {}",
        cublaslt::version(),
        if std::env::var("ATLAS_FP4L_CKPT").is_ok() {
            "FP4-Local layer 0 / 3, rank 0's TP2 cut"
        } else {
            "random"
        }
    );
    println!(
        "shape | N K M | bf16 ms TF | mxfp8 tuned ms TF | quant ms | gemm x | incl quant x | \
         events: heuristic #0, best in-order, best split ms | tuned kernel"
    );
    const MS: [usize; 4] = [2048, 4096, 8192, 16384];
    let mut rng = Rng(0xbe9c_4a11);
    // Per M: [bf16, mx tuned, quant] ms summed over a 16K-token prompt.
    let mut total = [[0f64; 3]; 4];
    for s in &SHAPES {
        let (n, k) = (s.n, s.k);
        let o = operands(gpu, s, 16384, &mut rng)?;
        // BF16 twin of the weight (exact) for the baseline.
        let wb: Vec<u16> = (0..n * k)
            .map(|i| {
                let v = e4m3_to_f32(o.host_w.0[i]) * 2f32.powi(i32::from(o.host_w.1[i / 32]) - 127);
                (v.to_bits() >> 16) as u16
            })
            .collect();
        let wbf = upload(gpu, &u16_bytes(&wb))?;
        // Layers of this kind per TP2 rank, and whether it quantizes its own
        // input (kv and the indexer share the q projection's).
        let (layers, own_quant) = match s.label {
            "gdn qkvz" | "gdn out" => (36.0, true),
            "qsa q" | "qsa o" => (12.0, true),
            "qsa kv" | "qsa indexer" => (12.0, false),
            _ => (48.0, true),
        };
        for (mi, &m) in MS.iter().enumerate() {
            let (q, sf) = quantized(gpu, &o, k, 0, m)?;
            let out = gpu.alloc(m * n * 2)?;
            let dims = [m as u32, n as u32, k as u32];
            let bf = time_ms(gpu, 10, || {
                cublaslt::bf16_gemm_act_weight_t(
                    o.act.0, wbf.0, out.0, m as u32, n as u32, k as u32, stream,
                )
            })?;
            let tuned = time_ms(gpu, 10, || {
                cublaslt::fp8_gemm_act_weight_t_mx(q.0, sf.0, o.w.0, o.w_sf.0, out.0, dims, stream)
            })?;
            let qms = time_ms(gpu, 10, || {
                mxfp8_quant_act(
                    gpu, quant, o.act, k as u32, q, sf, m as u32, k as u32, stream,
                )
            })?;
            let mut cands = cublaslt::fp8_gemm_mx_candidate_times(
                q.0, sf.0, o.w.0, o.w_sf.0, out.0, dims, false, stream,
            )?;
            let first = cands.first().map_or(f32::NAN, |c| c.2);
            cands.extend(cublaslt::fp8_gemm_mx_candidate_times(
                q.0, sf.0, o.w.0, o.w_sf.0, out.0, dims, true, stream,
            )?);
            let best = |ordered: bool| {
                cands
                    .iter()
                    .filter(|c| c.1 == ordered)
                    .map(|c| c.2)
                    .fold(f32::INFINITY, f32::min)
            };
            let tf = |ms: f64| 2.0 * (m * n * k) as f64 / (ms * 1e-3) / 1e12;
            println!(
                "{} | {n} {k} {m} | {bf:.4} {:.0} | {tuned:.4} {:.0} | {qms:.4} | {:.2} | {:.2} | \
                 {first:.4} {:.4} {:.4} | {}",
                s.label,
                tf(bf),
                tf(tuned),
                bf / tuned,
                bf / (tuned + qms),
                best(true),
                best(false),
                cublaslt::mx_algo_description(n as u32, k as u32, m as u32).unwrap_or_default()
            );
            let chunks = (16384 / m) as f64;
            total[mi][0] += layers * chunks * bf;
            total[mi][1] += layers * chunks * tuned;
            if own_quant {
                total[mi][2] += layers * chunks * qms;
            }
            for p in [q, sf, out] {
                gpu.free(p)?;
            }
        }
        for p in [o.w, o.w_sf, o.act, wbf] {
            gpu.free(p)?;
        }
    }
    for (mi, &m) in MS.iter().enumerate() {
        let [bf, mx, q] = total[mi];
        println!(
            "16K-token prompt in {m}-row chunks, these projections on one TP2 rank: BF16 {bf:.0} ms, \
             MXFP8 {mx:.0} + quant {q:.0} = {:.0} ms (saves {:.0} ms; {:.0} with the quant fused)",
            mx + q,
            bf - mx - q,
            bf - mx
        );
    }
    Ok(())
}
