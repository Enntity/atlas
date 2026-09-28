// SPDX-License-Identifier: AGPL-3.0-only
//! Prefill-shaped benchmark + bitwise gate for the native-FP4 routed MoE
//! grouped GEMM (`moe_w4a4_grouped_gemm_prequant_t_k64_vecscale`, transposed
//! `[K/2, N]` expert weights, prequantized NVFP4 activations).
//!
//! Synthetic EP2 rank at a 4K chunk, launched as in serving: a grid over all
//! 288 experts, of which the second 144 are remote (NULL weights) but still
//! own sorted rows; 16384 routed rows on the 144 local experts with a
//! hot-expert skew, gate (N=2048, K=4096) and down (N=4096, K=2048).
//! Every candidate kernel in `CANDIDATES` must reproduce the reference
//! output bit for bit (same per-element K accumulation order).
//!
//! Exit: 0 pass, 1 mismatch.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example moe_fp4_prefill_bench

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const MODULE: &str = "moe_w4a16";
const REFERENCE: &str = "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale";
/// (kernel, CTA N tile, threads per CTA, reads K-major `[N, K/2]` weights,
/// compact row-tile grid over `moe_mtile_prefix`).
const CANDIDATES: &[(&str, u32, u32, bool, bool)] = &[
    (
        "moe_w4a4_grouped_gemm_prequant_t_k128",
        128,
        256,
        false,
        false,
    ),
    (
        "moe_w4a4_grouped_gemm_prequant_t_k128w_compact",
        256,
        256,
        false,
        true,
    ),
    (
        "moe_w4a4_grouped_gemm_prequant_nk_k128",
        128,
        256,
        true,
        false,
    ),
];
/// Local experts; the grid covers twice as many (the remote rank's half).
const EXPERTS: usize = 144;
const GRID_EXPERTS: usize = 2 * EXPERTS;
/// Routed rows on one EP2 rank: 16384 at a 4K chunk (top-8 over two ranks);
/// `MOE_BENCH_ROWS` overrides (32768 = an 8K chunk).
fn rows_total() -> usize {
    std::env::var("MOE_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16384)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

/// Expert row counts: a Zipf-like skew (hottest ~3K rows), summing to rows_total().
fn expert_rows() -> Vec<usize> {
    let weights: Vec<f64> = (0..EXPERTS)
        .map(|e| 1.0 / (1.0 + e as f64).powf(0.9))
        .collect();
    let total: f64 = weights.iter().sum();
    let mut rows: Vec<usize> = weights
        .iter()
        .map(|w| (w / total * rows_total() as f64) as usize)
        .collect();
    let short = rows_total() - rows.iter().sum::<usize>();
    rows[EXPERTS - 1] += short;
    rows
}

/// Per-expert weight tables for `EXPERTS` local experts (the remote half
/// NULL): Atlas transposed `[K/2, N]` + `[K/16, N]`, and the same bytes
/// K-major `[N, K/2]` + `[N, K/16]` (each byte keeps its k-pair). Returns
/// (packed, scales, packed K-major, scales K-major) pointer tables and the
/// allocations to free.
fn expert_tables(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    n: usize,
    k: usize,
) -> Result<([DevicePtr; 4], Vec<DevicePtr>)> {
    let mut tables = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut owned = Vec::new();
    for _ in 0..EXPERTS {
        let w: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
        let s: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x30 + (rng.next() % 16) as u8)
            .collect();
        let mut w_nk = vec![0u8; w.len()];
        for kp in 0..k / 2 {
            for c in 0..n {
                w_nk[c * (k / 2) + kp] = w[kp * n + c];
            }
        }
        let mut s_nk = vec![0u8; s.len()];
        for gi in 0..k / 16 {
            for c in 0..n {
                s_nk[c * (k / 16) + gi] = s[gi * n + c];
            }
        }
        for (table, bytes) in tables.iter_mut().zip([&w, &s, &w_nk, &s_nk]) {
            let d = up(g, bytes)?;
            table.extend_from_slice(&d.0.to_le_bytes());
            owned.push(d);
        }
    }
    let mut ptrs = [DevicePtr(0); 4];
    for (p, mut table) in ptrs.iter_mut().zip(tables) {
        table.resize(GRID_EXPERTS * 8, 0);
        *p = up(g, &table)?;
        owned.push(*p);
    }
    Ok((ptrs, owned))
}

fn time(g: &dyn GpuBackend, f: &dyn Fn() -> Result<()>) -> Result<f64> {
    f()?;
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() / 10.0)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    n_tile: u32,
    threads: u32,
    args: &[DevicePtr; 8],
    n: u32,
    k: u32,
    max_m_tiles: u32,
    prefix: Option<DevicePtr>,
) -> Result<()> {
    let z = if prefix.is_some() {
        1
    } else {
        GRID_EXPERTS as u32
    };
    let mut l = KernelLaunch::new(g, kernel)
        .grid([div_ceil(n, n_tile), max_m_tiles, z])
        .block([threads, 1, 1]);
    for p in args {
        l = l.arg_ptr(*p);
    }
    l = l.arg_u32(GRID_EXPERTS as u32).arg_u32(n).arg_u32(k);
    if let Some(p) = prefix {
        l = l.arg_ptr(p);
    }
    l.launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let reference = g.kernel(MODULE, REFERENCE)?;
    let rows = expert_rows();
    let max_m_tiles = rows.iter().map(|r| r.div_ceil(64)).max().unwrap_or(1) as u32;
    // Remote experts mirror the local histogram; their rows follow the local ones.
    let mut offsets = vec![0i32];
    for r in rows.iter().chain(rows.iter()) {
        offsets.push(offsets.last().unwrap() + *r as i32);
    }
    let all_rows = 2 * rows_total();
    // Compact grid bound as served without an offsets download: every local
    // row in M64 tiles plus one partial tile per local expert.
    let tile_bound = (rows_total().div_ceil(64) + EXPERTS) as u32;
    let prefix_k = g.kernel(MODULE, "moe_mtile_prefix")?;
    let d_prefix = g.alloc((GRID_EXPERTS + 1) * 4)?;
    let off_bytes: Vec<u8> = offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_off = up(g, &off_bytes)?;
    let mut rng = Lcg(0xF4F4);
    let mut fail = false;
    for (name, n, k) in [
        ("gate [2048 x 4096]", 2048u32, 4096u32),
        ("down [4096 x 2048]", 4096, 2048),
    ] {
        let (nu, ku) = (n as usize, k as usize);
        // A: packed E2M1 rows + UE4M3 group scales in a sane exponent range.
        let a: Vec<u8> = (0..all_rows * ku / 2).map(|_| rng.next() as u8).collect();
        let a_s: Vec<u8> = (0..all_rows * ku / 16)
            .map(|_| 0x30 + (rng.next() % 16) as u8)
            .collect();
        let (d_a, d_as) = (up(g, &a)?, up(g, &a_s)?);
        let ([d_pp, d_sp, d_ppk, d_spk], owned) = expert_tables(g, &mut rng, nu, ku)?;
        let scale2: Vec<u8> = (0..GRID_EXPERTS)
            .flat_map(|_| 1.0f32.to_le_bytes())
            .collect();
        let d_s2 = up(g, &scale2)?;
        let c_bytes = all_rows * nu * 2;
        let (c_ref, c_new) = (g.alloc(c_bytes)?, g.alloc(c_bytes)?);
        g.memset(c_ref, 0, c_bytes)?;
        let args = |c| [d_a, d_as, d_pp, d_sp, d_s2, c, d_off, DevicePtr(0)];
        let args_nk = |c| [d_a, d_as, d_ppk, d_spk, d_s2, c, d_off, DevicePtr(0)];
        let flop = 2.0 * rows_total() as f64 * n as f64 * k as f64;
        let run_ref = || {
            launch(
                g,
                reference,
                128,
                128,
                &args(c_ref),
                n,
                k,
                max_m_tiles,
                None,
            )
        };
        let t_ref = time(g, &run_ref)?;
        println!(
            "{name}: reference {:7.1}us {:5.1} TFLOPS (max_m_tiles {max_m_tiles})",
            t_ref * 1e6,
            flop / t_ref / 1e12
        );
        let mut want = vec![0u8; c_bytes];
        g.copy_d2h(c_ref, &mut want)?;
        for &(kname, n_tile, threads, k_major, compact) in CANDIDATES {
            let Ok(kernel) = g.kernel(MODULE, kname) else {
                println!("  {kname}: absent");
                continue;
            };
            g.memset(c_new, 0, c_bytes)?;
            let a = if k_major { args_nk(c_new) } else { args(c_new) };
            let run = || {
                if !compact {
                    return launch(g, kernel, n_tile, threads, &a, n, k, max_m_tiles, None);
                }
                KernelLaunch::new(g, prefix_k)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(d_off)
                    .arg_ptr(a[2])
                    .arg_ptr(d_prefix)
                    .arg_u32(GRID_EXPERTS as u32)
                    .launch(0)?;
                launch(
                    g,
                    kernel,
                    n_tile,
                    threads,
                    &a,
                    n,
                    k,
                    tile_bound,
                    Some(d_prefix),
                )
            };
            let t = time(g, &run)?;
            let mut got = vec![0u8; c_bytes];
            g.copy_d2h(c_new, &mut got)?;
            let diff = want
                .chunks_exact(2)
                .zip(got.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            fail |= diff != 0;
            println!(
                "  {kname}: {:7.1}us {:5.1} TFLOPS  {}",
                t * 1e6,
                flop / t / 1e12,
                if diff == 0 {
                    "bitwise".to_string()
                } else {
                    format!("MISMATCH {diff} values")
                }
            );
        }
        for p in owned.into_iter().chain([d_a, d_as, d_s2, c_ref, c_new]) {
            g.free(p)?;
        }
    }
    fail |= gate_up_silu(g, &mut rng, d_off, d_prefix, tile_bound, all_rows)?;
    fail |= tile_worklist(g, &offsets)?;
    std::process::exit(if fail { 1 } else { 0 });
}

/// The fused gate/up + SiLU·mul + NVFP4 kernel against its three-kernel
/// reference (K128W gate, K128W up, `silu_mul_quant_nvfp4`): the packed E2M1
/// and E4M3 scale bytes must match. Returns true on mismatch.
fn gate_up_silu(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    d_off: DevicePtr,
    d_prefix: DevicePtr,
    tile_bound: u32,
    all_rows: usize,
) -> Result<bool> {
    let (n, k) = (2048u32, 4096u32);
    let (nu, ku) = (n as usize, k as usize);
    let (Ok(fused), Ok(wide), Ok(silu), Ok(prefix_k)) = (
        g.kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w"),
        g.kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_t_k128w_compact"),
        g.kernel("moe_silu_mul", "silu_mul_quant_nvfp4"),
        g.kernel(MODULE, "moe_mtile_prefix"),
    ) else {
        println!("gate/up + SiLU quant: absent");
        return Ok(false);
    };
    let a: Vec<u8> = (0..all_rows * ku / 2).map(|_| rng.next() as u8).collect();
    let a_s: Vec<u8> = (0..all_rows * ku / 16)
        .map(|_| 0x30 + (rng.next() % 16) as u8)
        .collect();
    let (d_a, d_as) = (up(g, &a)?, up(g, &a_s)?);
    let ([g_pp, g_sp, ..], mut owned) = expert_tables(g, rng, nu, ku)?;
    let ([u_pp, u_sp, ..], u_owned) = expert_tables(g, rng, nu, ku)?;
    owned.extend(u_owned);
    // Output scales that put the SiLU inputs around the clamp and below it.
    let s2 = |base: f32| -> Vec<u8> {
        (0..GRID_EXPERTS)
            .flat_map(|e| (base * (1.0 + (e % 5) as f32 * 0.25)).to_le_bytes())
            .collect()
    };
    let (g_s2, u_s2) = (up(g, &s2(1.0 / 256.0))?, up(g, &s2(1.0 / 128.0))?);
    let (c_g, c_u) = (g.alloc(all_rows * nu * 2)?, g.alloc(all_rows * nu * 2)?);
    let (q_bytes, s_bytes) = (all_rows * nu / 2, all_rows * nu / 16);
    let (ref_q, ref_s, new_q, new_s) = (
        g.alloc(q_bytes)?,
        g.alloc(s_bytes)?,
        g.alloc(q_bytes)?,
        g.alloc(s_bytes)?,
    );
    for (p, b) in [
        (c_g, all_rows * nu * 2),
        (c_u, all_rows * nu * 2),
        (ref_q, q_bytes),
        (ref_s, s_bytes),
        (new_q, q_bytes),
        (new_s, s_bytes),
    ] {
        g.memset(p, 0, b)?;
    }
    let prefix = |pp| {
        KernelLaunch::new(g, prefix_k)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(d_off)
            .arg_ptr(pp)
            .arg_ptr(d_prefix)
            .arg_u32(GRID_EXPERTS as u32)
            .launch(0)
    };
    let run_ref = || -> Result<()> {
        prefix(g_pp)?;
        for (pp, sp, s2p, c) in [(g_pp, g_sp, g_s2, c_g), (u_pp, u_sp, u_s2, c_u)] {
            launch(
                g,
                wide,
                256,
                256,
                &[d_a, d_as, pp, sp, s2p, c, d_off, DevicePtr(0)],
                n,
                k,
                tile_bound,
                Some(d_prefix),
            )?;
        }
        KernelLaunch::new(g, silu)
            .grid([all_rows as u32, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(c_g)
            .arg_ptr(c_u)
            .arg_ptr(ref_q)
            .arg_ptr(ref_s)
            .arg_ptr(DevicePtr(0))
            .arg_u32(all_rows as u32)
            .arg_u32(n)
            .launch(0)
    };
    let run_fused = || -> Result<()> {
        prefix(g_pp)?;
        KernelLaunch::new(g, fused)
            .grid([n / 128, tile_bound, 1])
            .block([256, 1, 1])
            .arg_ptr(d_a)
            .arg_ptr(d_as)
            .arg_ptr(g_pp)
            .arg_ptr(g_sp)
            .arg_ptr(g_s2)
            .arg_ptr(DevicePtr(0))
            .arg_ptr(d_off)
            .arg_ptr(DevicePtr(0))
            .arg_u32(GRID_EXPERTS as u32)
            .arg_u32(n)
            .arg_u32(k)
            .arg_ptr(d_prefix)
            .arg_ptr(u_pp)
            .arg_ptr(u_sp)
            .arg_ptr(u_s2)
            .arg_ptr(new_q)
            .arg_ptr(new_s)
            .launch(0)
    };
    let (t_ref, t_new) = (time(g, &run_ref)?, time(g, &run_fused)?);
    let fetch = |p, b| -> Result<Vec<u8>> {
        let mut v = vec![0u8; b];
        g.copy_d2h(p, &mut v)?;
        Ok(v)
    };
    let (rq, rs, nq, ns) = (
        fetch(ref_q, q_bytes)?,
        fetch(ref_s, s_bytes)?,
        fetch(new_q, q_bytes)?,
        fetch(new_s, s_bytes)?,
    );
    let diff = rq.iter().zip(&nq).filter(|(a, b)| a != b).count()
        + rs.iter().zip(&ns).filter(|(a, b)| a != b).count();
    let zero_scales = rs.iter().filter(|&&b| b == 0).count();
    println!(
        "gate/up + SiLU quant: gate+up+silu {:7.1}us, fused {:7.1}us  {} ({zero_scales} of {s_bytes} reference scales zero)",
        t_ref * 1e6,
        t_new * 1e6,
        if diff == 0 {
            "bitwise".to_string()
        } else {
            format!("MISMATCH {diff} bytes")
        }
    );
    for p in owned
        .into_iter()
        .chain([d_a, d_as, g_s2, u_s2, c_g, c_u, ref_q, ref_s, new_q, new_s])
    {
        g.free(p)?;
    }
    Ok(diff != 0)
}

/// `moe_build_tile_worklist` (block scan) against a host serial walk of the
/// same offsets, local experts only, at the decode compact shape (M64 tiles,
/// 16 N tiles). Returns true on mismatch.
fn tile_worklist(g: &dyn GpuBackend, offsets: &[i32]) -> Result<bool> {
    let Ok(builder) = g.kernel("moe", "moe_build_tile_worklist") else {
        println!("tile worklist: absent");
        return Ok(false);
    };
    let (n_tiles, m_tile) = (16u32, 64u32);
    let ptrs: Vec<u8> = (0..GRID_EXPERTS)
        .flat_map(|e| (if e < EXPERTS { 0x1000u64 } else { 0 }).to_le_bytes())
        .collect();
    let mut want = Vec::new();
    for e in 0..EXPERTS {
        let rows = (offsets[e + 1] - offsets[e]) as u32;
        for mt in 0..rows.div_ceil(m_tile) {
            for nt in 0..n_tiles {
                want.extend([e as u32, (mt << 6) | nt]);
            }
        }
    }
    let off_bytes: Vec<u8> = offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (d_off, d_ptrs) = (up(g, &off_bytes)?, up(g, &ptrs)?);
    let (d_list, d_total) = (g.alloc(want.len() * 4 + 64)?, g.alloc(4)?);
    let run = || {
        KernelLaunch::new(g, builder)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_off)
            .arg_ptr(d_ptrs)
            .arg_ptr(d_list)
            .arg_ptr(d_total)
            .arg_u32(GRID_EXPERTS as u32)
            .arg_u32(n_tiles)
            .arg_u32(m_tile)
            .launch(0)
    };
    let t = time(g, &run)?;
    let mut total = [0u8; 4];
    g.copy_d2h(d_total, &mut total)?;
    let mut got = vec![0u8; want.len() * 4];
    g.copy_d2h(d_list, &mut got)?;
    let got: Vec<u32> = got
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ok = i32::from_le_bytes(total) as usize * 2 == want.len() && got == want;
    println!(
        "tile worklist: {} items {:5.1}us  {}",
        want.len() / 2,
        t * 1e6,
        if ok { "exact" } else { "MISMATCH" }
    );
    for p in [d_off, d_ptrs, d_list, d_total] {
        g.free(p)?;
    }
    Ok(!ok)
}
