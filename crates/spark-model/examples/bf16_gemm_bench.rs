// SPDX-License-Identifier: AGPL-3.0-only
//! BF16 prefill GEMM throughput at GLM-5.3 Flash TP2 shapes (M = 4096 rows):
//! cuBLASLt's heuristic vs the CUTLASS configs of
//! `spark_runtime::cutlass::bf16_gemm_tuned`, plus the head-batched MLA
//! absorb / V-up GEMMs (cuBLASLt vs CUTLASS batched). Every CUTLASS result is
//! checked against cuBLASLt (worst |diff| over 2 BF16 ulps of the magnitude).
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash-nvfp4 \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example bf16_gemm_bench

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Rows (`BF16_BENCH_M` overrides, e.g. 8196 for an 8K prefill chunk).
fn rows() -> u32 {
    std::env::var("BF16_BENCH_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096)
}
const CONFIGS: [u32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const SHAPES: [(&str, u32, u32); 7] = [
    ("q_a      [1536 x 4096]", 1536, 4096),
    ("q_b      [8192 x 1536]", 8192, 1536),
    ("o        [4096 x 8192]", 4096, 8192),
    ("kv_a     [ 576 x 4096]", 576, 4096),
    ("shared g [2048 x 4096]", 2048, 4096),
    ("shared d [4096 x 2048]", 4096, 2048),
    ("square   [4096 x 4096]", 4096, 4096),
];

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut rng = 0x1234_5678u64;
    let mut fill = |p: DevicePtr, count: usize| -> Result<()> {
        let bytes: Vec<u8> = (0..count)
            .flat_map(|_| {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                bf16::from_f32(v).to_bits().to_le_bytes()
            })
            .collect();
        g.copy_h2d(&bytes, p)
    };
    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        f()?;
        g.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            f()?;
        }
        g.synchronize(0)?;
        Ok(t0.elapsed().as_secs_f64() / 20.0)
    };
    let read = |p: DevicePtr, count: usize| -> Result<Vec<f32>> {
        let mut b = vec![0u8; count * 2];
        g.copy_d2h(p, &mut b)?;
        Ok(b.chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
            .collect())
    };
    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / (x.abs().max(y.abs()) * (2.0 / 128.0) + 1e-2))
            .fold(0f32, f32::max)
    };

    #[allow(non_snake_case)]
    let M = rows();
    for (name, n, k) in SHAPES {
        let (mu, nu, ku) = (M as usize, n as usize, k as usize);
        let (a, w) = (g.alloc(mu * ku * 2)?, g.alloc(nu * ku * 2)?);
        let (c, c2) = (g.alloc(mu * nu * 2)?, g.alloc(mu * nu * 2)?);
        fill(a, mu * ku)?;
        fill(w, nu * ku)?;
        let flop = 2.0 * M as f64 * n as f64 * k as f64;
        let lt =
            time(&|| spark_runtime::cublaslt::bf16_gemm_act_weight_t(a.0, w.0, c.0, M, n, k, 0))?;
        let reference = read(c, mu * nu)?;
        let mut line = format!("{name}: cuBLASLt {:5.1}TF | CUTLASS", flop / lt / 1e12);
        let mut err = 0f32;
        for config in CONFIGS {
            let run = || {
                spark_runtime::cutlass::bf16_gemm_tuned(a.0, w.0, c2.0, M, n, k, k, n, config, 0)
            };
            match run() {
                Ok(()) => {
                    line += &format!(" {config}:{:5.1}", flop / time(&run)? / 1e12);
                    err = err.max(worst(&reference, &read(c2, mu * nu)?));
                }
                Err(_) => line += &format!(" {config}:  err"),
            }
        }
        println!("{line}  worst/tol {err:.2}");
        for p in [a, w, c, c2] {
            g.free(p)?;
        }
    }

    // MLA absorb (q_nope -> latent) and V-up (latent -> v): 32 TP-local heads,
    // token rows interleaved across heads.
    let heads = 32u32;
    for (name, gk, gn) in [
        ("absorb 32x[256->512]", 256u32, 512u32),
        ("v-up   32x[512->256]", 512, 256),
    ] {
        let (a_stride, c_stride) = (heads * gk, heads * gn);
        let rows = M as usize * c_stride as usize;
        let a = g.alloc(M as usize * a_stride as usize * 2)?;
        let w = g.alloc((heads * gk * gn) as usize * 2)?;
        let (c, c2) = (g.alloc(rows * 2)?, g.alloc(rows * 2)?);
        fill(a, M as usize * a_stride as usize)?;
        fill(w, (heads * gk * gn) as usize)?;
        let bytes = 2.0 * M as f64 * (a_stride + c_stride) as f64;
        let lt = time(&|| {
            spark_runtime::cublaslt::bf16_grouped_gemm_act_weight_t(
                a.0, w.0, c.0, M, heads, gn, gk, a_stride, c_stride, 0,
            )
        })?;
        let cu = time(&|| {
            spark_runtime::cutlass::bf16_grouped_gemm_act_weight_t(
                a.0, w.0, c2.0, M, heads, gn, gk, a_stride, c_stride, 0,
            )
        })?;
        println!(
            "{name}: cuBLASLt {:7.1}us {:4.0}GB/s | CUTLASS {:7.1}us {:4.0}GB/s  worst/tol {:.2}",
            lt * 1e6,
            bytes / lt / 1e9,
            cu * 1e6,
            bytes / cu / 1e9,
            worst(&read(c, rows)?, &read(c2, rows)?)
        );
    }
    Ok(())
}
