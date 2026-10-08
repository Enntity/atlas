// SPDX-License-Identifier: AGPL-3.0-only
//! Kernel-level checks behind `ATLAS_QWEN4EXP_PREFILL_ROWINV=1`, at the
//! qwen4_exp TP2 rank shapes, on a GB10 (run inside atlas-release-builder for
//! the runtime image's cuBLASLt):
//!
//! 1. `proj`: the row-invariant BF16 projection (`qwen4exp_rowinv::
//!    try_bf16_gemm`: cuBLASLt's algo-21 k-chain when offered, else the tile
//!    kernel `dense_gemm_bf16_pipelined`). A solo pass over `m` rows against
//!    the same rows inside a 368..2048-row pass, at two offsets, and the
//!    k-chain against the tile kernel: differing bytes.
//! 2. `hc`: the prefill mHC collapse (`hc_pre_stage_vec` + `hc_mma_down_rows`
//!    + `hc_mma_finish_rows` over a 1920-row slab) against the DECODE
//!    collapse (`hc_mma_down` + `hc_mma_finish`, 1..32 rows) of the same
//!    rows: differing `y` / `inj` bytes. Zero means a prefill row's collapse
//!    is its decode collapse, byte for byte.
//!
//!   cargo build -p spark-model --release --features cuda,gpu-examples \
//!     --example qwen4exp_rowinv_probe

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

fn bf16s(seed: &mut u64, n: usize, scale: f32) -> Vec<u8> {
    f32s(seed, n, scale)
        .chunks(4)
        .flat_map(|b| {
            let f = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            (((f.to_bits()) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn f32s(seed: &mut u64, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            (f * scale).to_le_bytes()
        })
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// `out = a x w^T` on the tile kernel (`tile`) or cuBLASLt's k-chain.
#[allow(clippy::too_many_arguments)]
fn proj(
    g: &dyn GpuBackend,
    tile: bool,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    [m, n, k]: [u32; 3],
    s: u64,
) -> Result<&'static str> {
    if !tile
        && spark_runtime::cublaslt::bf16_gemm_act_weight_t_kchain_any(
            a.0,
            w.0,
            out.0,
            [m, n, k],
            s,
        )?
    {
        return Ok("kchain");
    }
    KernelLaunch::new(g, g.kernel("gemm", "dense_gemm_bf16_pipelined")?)
        .grid([n.div_ceil(128), m.div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(s)?;
    Ok("tile")
}

fn check_proj(g: &dyn GpuBackend, seed: &mut u64) -> Result<()> {
    let s = g.default_stream();
    // GDN in_proj / out_proj (= attention o_proj), QSA indexer q+k, the
    // BF16 attention q+gate and k/v, PLE key/value.
    let shapes = [
        ("gdn_in_proj", 8192, 2560),
        ("gdn_out/o_proj", 2560, 3072),
        ("qsa_qk", 640, 2560),
        ("attn_q_gate", 6144, 2560),
        ("attn_k_or_v", 256, 2560),
        ("ple_key", 10240, 2560),
        ("ple_value", 2560, 2560),
    ];
    let solos = [1usize, 7, 22, 46, 64, 100, 200, 513];
    for (name, n, k) in shapes {
        let w = upload(g, &bf16s(seed, n * k, 0.02))?;
        let mut bad = Vec::new();
        let mut kinds = std::collections::BTreeSet::new();
        for big in [368usize, 2048] {
            let a = upload(g, &bf16s(seed, big * k, 1.0))?;
            let ob = g.alloc(big * n * 2)?;
            kinds.insert(proj(
                g,
                false,
                a,
                w,
                ob,
                [big as u32, n as u32, k as u32],
                s,
            )?);
            let ot = g.alloc(big * n * 2)?;
            proj(g, true, a, w, ot, [big as u32, n as u32, k as u32], s)?;
            g.synchronize(s)?;
            let vb = read(g, ob, big * n * 2)?;
            let d = diff(&vb, &read(g, ot, big * n * 2)?);
            if d > 0 {
                bad.push(format!("kchain!=tile at {big}: {d} B"));
            }
            for &m in solos.iter().filter(|&&m| m < big) {
                for off in [0usize, 46.min(big - m)] {
                    let os = g.alloc(m * n * 2)?;
                    let a_off = a.offset(off * k * 2);
                    kinds.insert(proj(
                        g,
                        false,
                        a_off,
                        w,
                        os,
                        [m as u32, n as u32, k as u32],
                        s,
                    )?);
                    g.synchronize(s)?;
                    let d = diff(
                        &vb[off * n * 2..(off + m) * n * 2],
                        &read(g, os, m * n * 2)?,
                    );
                    if d > 0 {
                        bad.push(format!("solo {m}@{off} in {big}: {d} B"));
                    }
                    g.free(os)?;
                }
            }
            g.free(a)?;
            g.free(ob)?;
            g.free(ot)?;
        }
        g.free(w)?;
        println!(
            "proj {name:<15} N={n:<5} K={k:<5} kernels {kinds:?}: {}",
            if bad.is_empty() {
                "row-invariant, k-chain == tile".to_string()
            } else {
                bad.join("; ")
            }
        );
    }
    Ok(())
}

const H: usize = 2560;
const HC: usize = 4;
const RANK: usize = 320;
const D: usize = HC * H;

struct HcW {
    norm: DevicePtr,
    down: DevicePtr,
    inject: DevicePtr,
    up: DevicePtr,
}

/// stage rows of `streams` into `normed` (FP32).
fn stage(g: &dyn GpuBackend, w: &HcW, streams: DevicePtr, normed: DevicePtr, t: u32) -> Result<()> {
    KernelLaunch::new(g, g.kernel("hyper_connection", "hc_pre_stage_vec")?)
        .grid([t, if t >= 48 { 1 } else { 8 }, 1])
        .block([1024, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.norm)
        .arg_ptr(normed)
        .arg_u32(H as u32)
        .arg_u32(HC as u32)
        .arg_f32(1e-6)
        .launch(g.default_stream())
}

/// down + finish; `rows` picks the prefill twins (groups on blockIdx.z).
fn collapse(
    g: &dyn GpuBackend,
    w: &HcW,
    rows: bool,
    [normed, low, y, inj]: [DevicePtr; 4],
    t: u32,
) -> Result<()> {
    let s = g.default_stream();
    let (dn, fin, wm, gd, gf) = if rows {
        ("hc_mma_down_rows", "hc_mma_finish_rows", 7u32, 96u32, 64u32)
    } else {
        ("hc_mma_down", "hc_mma_finish", 3, 32, 32)
    };
    let out_rows = (RANK + HC) as u32;
    KernelLaunch::new(g, g.kernel("qwen4exp_hc_mma", dn)?)
        .grid([out_rows.div_ceil(16).div_ceil(wm), 8, t.div_ceil(gd)])
        .block([32 * wm, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(w.down)
        .arg_ptr(w.inject)
        .arg_ptr(low)
        .arg_ptr(inj)
        .arg_u32(H as u32)
        .arg_u32(HC as u32)
        .arg_u32(RANK as u32)
        .arg_u32(t)
        .launch(s)?;
    let nt = t.min(gf).div_ceil(8);
    let smem = if rows {
        4 * nt * 8 * 32 * 4
    } else {
        // hc_mma_finish_smem: staged fragments or the mean tile.
        (RANK as u32 / 16 * nt * 32 * 16).max(4 * nt * 8 * 32 * 4)
    };
    KernelLaunch::new(g, g.kernel("qwen4exp_hc_mma", fin)?)
        .grid([H as u32 / 32, 1, t.div_ceil(gf)])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(normed)
        .arg_ptr(low)
        .arg_ptr(w.up)
        .arg_ptr(y)
        .arg_u32(H as u32)
        .arg_u32(RANK as u32)
        .arg_u32(t)
        .launch(s)
}

fn check_hc(g: &dyn GpuBackend, seed: &mut u64) -> Result<()> {
    let w = HcW {
        norm: upload(g, &bf16s(seed, D, 0.5))?,
        down: upload(g, &bf16s(seed, RANK * D, 0.02))?,
        inject: upload(g, &bf16s(seed, HC * D, 0.02))?,
        up: upload(g, &bf16s(seed, RANK * D, 0.02))?,
    };
    let big = 1920usize;
    let streams = upload(g, &f32s(seed, big * D, 1.0))?;
    let buf = |n: usize| g.alloc(n);
    let (nb, lb, yb, ib) = (
        buf(big * D * 4)?,
        buf(big * RANK * 4)?,
        buf(big * H * 2)?,
        buf(big * HC * 4)?,
    );
    stage(g, &w, streams, nb, big as u32)?;
    collapse(g, &w, true, [nb, lb, yb, ib], big as u32)?;
    g.synchronize(g.default_stream())?;
    let (yv, iv) = (read(g, yb, big * H * 2)?, read(g, ib, big * HC * 4)?);
    let (ns, ls, ys, is) = (
        buf(32 * D * 4)?,
        buf(32 * RANK * 4)?,
        buf(32 * H * 2)?,
        buf(32 * HC * 4)?,
    );
    let mut worst = (0usize, 0usize);
    let mut cases = 0;
    for t in [1usize, 2, 5, 8, 9, 17, 31, 32] {
        for off in [0usize, 95, 96, 1000, big - t] {
            stage(g, &w, streams.offset(off * D * 4), ns, t as u32)?;
            collapse(g, &w, false, [ns, ls, ys, is], t as u32)?;
            g.synchronize(g.default_stream())?;
            let dy = diff(
                &yv[off * H * 2..(off + t) * H * 2],
                &read(g, ys, t * H * 2)?,
            );
            let di = diff(
                &iv[off * HC * 4..(off + t) * HC * 4],
                &read(g, is, t * HC * 4)?,
            );
            worst = (worst.0.max(dy), worst.1.max(di));
            cases += 1;
        }
    }
    println!(
        "hc   prefill rows twins (1920-row slab) vs decode collapse (1..32 rows), {cases} cases: \
         worst y {} B, inj {} B differ",
        worst.0, worst.1
    );
    Ok(())
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let mut seed = 0x5eed_u64;
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if which == "all" || which == "proj" {
        check_proj(g, &mut seed)?;
    }
    if which == "all" || which == "hc" {
        check_hc(g, &mut seed)?;
    }
    Ok(())
}
