// SPDX-License-Identifier: AGPL-3.0-only
//! FP4-Local native-format GEMVs against the production paths they replace,
//! at GLM-5.3-Flash's per-rank (TP2 / EP2) shapes and the decode / verify /
//! multi-sequence row counts M = 1, 2, 4, 8, 16, 32 (plus a correctness gate).
//!
//! Groups (production arm -> native arm, `ATLAS_GLM_FP4L_NATIVE=1`):
//! - KDA q/k/v [3 x 4096 x 4096]: NVFP4 `w4a16_gemv_tc8` x3 (canonical) ->
//!   MXFP8 `mxfp8_gemv_tc*` x3 and the fused `mxfp8_gemv_tc*_multi` (one launch);
//!   KDA o [4096 x 4096]: NVFP4 tc8 -> MXFP8.
//! - MLA q_a + kv_a [1536 + 512 x 4096], q_b [8192 x 1536], o [4096 x 8192]:
//!   production MXFP8 twins -> FP8 block `fp8b_gemv_tc*` (q_a + kv_a also as one
//!   `fp8b_gemv_tc*_multi` launch).
//! - shared expert, TP split (rank half): NVFP4 tc8 gate + up [1024 x 4096] and
//!   strided down [4096 x 1024 of 2048] -> `fp8b_gemv_tc*_multi` + `fp8b_gemv_tc*_ld`.
//! - dense MLP 0..2 (replicated): NVFP4 tc8 gate + up [12288 x 4096], down
//!   [4096 x 12288] -> `mxfp8_gemv_tc*_multi` + `mxfp8_gemv_tc*`.
//!
//! Weights cycle over enough copies (>= 256 MiB) that every launch streams them
//! from DRAM, like a decode step. Each line prints the group's us per call
//! (best of `REPS` batches), the bytes it reads and the GB/s; the floor column
//! is a device-to-device copy of the same bytes (read + write, so its GB/s is
//! 2 x bytes / time), a rough stream roofline.
//!
//! Gate (exit 1 on failure):
//! 1. every FP8 block tier row at M equals the same row at 32 bit for bit
//!    (decode == verify row), for each MLA / shared shape;
//! 2. the multi-projection launches equal their single-projection launches
//!    bit for bit (KDA q/k/v, MLA q_a + kv_a, shared halves);
//! 3. `fp8b_gemv` at M = 4 is within 1e-2 (relative to the row's |y| max) of
//!    an f64 host reference built from the same E4M3 bytes and f32 grid.
//!
//! Run (GB10, GLM kernel build):
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash ATLAS_TARGET_QUANT=nvfp4 \
//!     cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example glm_fp4l_native_bench [reps=7]

use anyhow::{Result, ensure};
use spark_model::layers::ops::{self, MxPart, native_kernels};
use spark_model::weight_map::{Fp8BlockWeight, Mxfp8Weight};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const ROWS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const COLD_BYTES: usize = 256 << 20;
const ITERS: usize = 20;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    /// An E4M3 byte without the NaN codes.
    fn e4m3(&mut self) -> u8 {
        let b = (self.next() >> 32) as u8;
        if b & 0x7F == 0x7F { b ^ 1 } else { b }
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn fetch(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// One weight in every format a group uses, `copies` times (cold cycling).
struct Weight {
    n: usize,
    k: usize,
    mx: Vec<Mxfp8Weight>,
    fp8b: Vec<Fp8BlockWeight>,
    nv: Vec<(DevicePtr, DevicePtr)>,
    /// Host bytes of copy 0 (E4M3 values, f32 grid) for the reference.
    host: (Vec<u8>, Vec<f32>),
}

const NV_SCALE2: f32 = 1.0 / 64.0;

impl Weight {
    fn new(g: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize) -> Result<Self> {
        let values: Vec<u8> = (0..n * k).map(|_| rng.e4m3()).collect();
        let grid: Vec<f32> = (0..(n / 128) * (k / 128))
            .map(|_| 0.01 + 0.01 * rng.unit().abs())
            .collect();
        let grid_bytes: Vec<u8> = grid.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mx_scales: Vec<u8> = (0..n * k / 32)
            .map(|_| 118 + (rng.next() % 4) as u8)
            .collect();
        let nv_packed: Vec<u8> = (0..n * k / 2).map(|_| (rng.next() >> 24) as u8).collect();
        let nv_scales: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x30 + (rng.next() % 16) as u8)
            .collect();
        let copies = (COLD_BYTES / (n * k)).clamp(2, 64);
        let mut w = Self {
            n,
            k,
            mx: Vec::new(),
            fp8b: Vec::new(),
            nv: Vec::new(),
            host: (values.clone(), grid),
        };
        for _ in 0..copies {
            let data = up(g, &values)?;
            w.mx.push(Mxfp8Weight {
                data,
                scales: up(g, &mx_scales)?,
                tiled: false,
            });
            w.fp8b.push(Fp8BlockWeight {
                data,
                scales: up(g, &grid_bytes)?,
                tiled: false,
            });
            w.nv.push((up(g, &nv_packed)?, up(g, &nv_scales)?));
        }
        Ok(w)
    }

    fn mx_bytes(&self) -> usize {
        self.n * self.k * 33 / 32
    }
    fn fp8b_bytes(&self) -> usize {
        self.n * self.k + (self.n / 128) * (self.k / 128) * 4
    }
    fn nv_bytes(&self) -> usize {
        self.n * self.k * 9 / 16
    }
}

/// The canonical NVFP4 tensor-core tier (`w4a16_gemv_tc8`, 1..32 rows).
fn nv_tc(
    g: &dyn GpuBackend,
    k8: KernelHandle,
    (w, s): (DevicePtr, DevicePtr),
    x: DevicePtr,
    y: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, k8)
        .grid([div_ceil(n as u32, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(s)
        .arg_f32(NV_SCALE2)
        .arg_ptr(y)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(0)
}

/// `w4a16_gemv_tc8_ld` on columns `[col0, col0 + k)` of a `[n, ld]` NVFP4 weight.
#[allow(clippy::too_many_arguments)]
fn nv_tc_ld(
    g: &dyn GpuBackend,
    k8ld: KernelHandle,
    (w, s): (DevicePtr, DevicePtr),
    x: DevicePtr,
    y: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ld: usize,
    col0: usize,
) -> Result<()> {
    KernelLaunch::new(g, k8ld)
        .grid([div_ceil(n as u32, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w.offset(col0 / 2))
        .arg_ptr(s.offset(col0 / 16))
        .arg_f32(NV_SCALE2)
        .arg_ptr(y)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32((ld / 2) as u32)
        .arg_u32((ld / 16) as u32)
        .launch(0)
}

/// Best-of-`reps` microseconds per call of `f(copy)` over `copies` copies.
fn time(
    g: &dyn GpuBackend,
    reps: usize,
    copies: usize,
    f: &dyn Fn(usize) -> Result<()>,
) -> Result<f64> {
    for c in 0..copies.min(4) {
        f(c)?;
    }
    g.synchronize(0)?;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t0 = Instant::now();
        for i in 0..ITERS {
            f(i % copies)?;
        }
        g.synchronize(0)?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64);
    }
    Ok(best)
}

fn line(label: &str, m: usize, us: f64, bytes: usize) {
    println!(
        "  {label:<34} M={m:<2} {us:>8.1} us  {:>7.1} MB  {:>6.1} GB/s",
        bytes as f64 / 1e6,
        bytes as f64 / us / 1e3
    );
}

fn main() -> Result<()> {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let nk = native_kernels(g);
    ensure!(
        nk.mx.complete(false)
            && nk.fp8b.complete(false)
            && nk.fp8b_multi.complete(false)
            && nk.mx_multi.complete(false)
            && nk.fp8b_ld.iter().all(|k| k.0 != 0),
        "the native mxfp8_gemv tiers are missing: build with ATLAS_TARGET_MODEL=glm-5.3-flash"
    );
    let tc8 = g.kernel("w4a16_gemv", "w4a16_gemv_tc8")?;
    let tc8_ld = g.kernel("w4a16_gemv", "w4a16_gemv_tc8_ld")?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    // Activations: 32 rows of the widest K (12288), and outputs.
    let x_host: Vec<f32> = (0..32 * 12288).map(|_| rng.unit()).collect();
    let x = up(g, &bf16_bytes(&x_host))?;
    let outs: Vec<DevicePtr> = (0..4)
        .map(|_| g.alloc(32 * 12288 * 2))
        .collect::<Result<_>>()?;
    let floor_dst = g.alloc(COLD_BYTES)?;
    let mut fail = false;

    let kda = Weight::new(g, &mut rng, 4096, 4096)?;
    let q_a = Weight::new(g, &mut rng, 1536, 4096)?;
    let kv_a = Weight::new(g, &mut rng, 512, 4096)?;
    let q_b = Weight::new(g, &mut rng, 8192, 1536)?;
    let mla_o = Weight::new(g, &mut rng, 4096, 8192)?;
    let sh_gu = Weight::new(g, &mut rng, 2048, 4096)?;
    let sh_down = Weight::new(g, &mut rng, 4096, 2048)?;
    let dense_gu = Weight::new(g, &mut rng, 12288, 4096)?;
    let dense_down = Weight::new(g, &mut rng, 4096, 12288)?;

    let floor = |w: &Weight, bytes: usize| -> Result<f64> {
        let copies = w.mx.len();
        time(g, reps, copies, &|c| {
            g.copy_d2d_async(w.mx[c].data, floor_dst, bytes.min(w.n * w.k), 0)
        })
        .map(|us| us / 2.0 * (bytes as f64 / bytes.min(w.n * w.k) as f64))
    };

    println!("== KDA q/k/v [3 x 4096 x 4096] and o [4096 x 4096] (per rank, 34 layers)");
    for m in ROWS {
        let n = 4096;
        let copies = kda.mx.len();
        let nv = time(g, reps, copies, &|c| {
            for i in 0..3 {
                nv_tc(g, tc8, kda.nv[(c + i) % copies], x, outs[i], m, n, 4096)?;
            }
            Ok(())
        })?;
        line("qkv NVFP4 tc8 x3 (production)", m, nv, 3 * kda.nv_bytes());
        let mx3 = time(g, reps, copies, &|c| {
            for i in 0..3 {
                ops::mxfp8_gemv_rows(
                    g,
                    &nk.mx.row_major,
                    &kda.mx[(c + i) % copies],
                    x,
                    outs[i],
                    m,
                    n,
                    4096,
                    n,
                    0,
                )?;
            }
            Ok(())
        })?;
        line("qkv MXFP8 x3", m, mx3, 3 * kda.mx_bytes());
        let fused = time(g, reps, copies, &|c| {
            let parts: Vec<MxPart> = (0..3)
                .map(|i| MxPart::mx(&kda.mx[(c + i) % copies], outs[i], n))
                .collect();
            ops::mx_multi_rows(g, &nk.mx_multi, false, x, &parts, m, 4096, 0)
        })?;
        line("qkv MXFP8 fused (native)", m, fused, 3 * kda.mx_bytes());
        let o_nv = time(g, reps, copies, &|c| {
            nv_tc(g, tc8, kda.nv[c], x, outs[0], m, n, 4096)
        })?;
        line("o NVFP4 tc8 (production)", m, o_nv, kda.nv_bytes());
        let o_mx = time(g, reps, copies, &|c| {
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &kda.mx[c],
                x,
                outs[0],
                m,
                n,
                4096,
                n,
                0,
            )
        })?;
        line("o MXFP8 (native)", m, o_mx, kda.mx_bytes());
    }
    let f = floor(&kda, 3 * kda.mx_bytes())?;
    line(
        "floor: d2d of the qkv MXFP8 bytes",
        0,
        f,
        3 * kda.mx_bytes(),
    );

    println!("== MLA q_a + kv_a, q_b, o (per rank, 11 layers)");
    for m in ROWS {
        let cq = q_a.mx.len().min(kv_a.mx.len());
        let mx = time(g, reps, cq, &|c| {
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &q_a.mx[c],
                x,
                outs[0],
                m,
                1536,
                4096,
                1536,
                0,
            )?;
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &kv_a.mx[c],
                x,
                outs[1],
                m,
                512,
                4096,
                512,
                0,
            )
        })?;
        line(
            "q_a + kv_a MXFP8 twins (production)",
            m,
            mx,
            q_a.mx_bytes() + kv_a.mx_bytes(),
        );
        let fb = time(g, reps, cq, &|c| {
            let parts = [
                MxPart::fp8b(&q_a.fp8b[c], outs[0], 1536),
                MxPart::fp8b(&kv_a.fp8b[c], outs[1], 512),
            ];
            ops::mx_multi_rows(g, &nk.fp8b_multi, false, x, &parts, m, 4096, 0)
        })?;
        line(
            "q_a + kv_a FP8 block fused (native)",
            m,
            fb,
            q_a.fp8b_bytes() + kv_a.fp8b_bytes(),
        );
        for (label, w) in [("q_b", &q_b), ("o", &mla_o)] {
            let copies = w.mx.len();
            let mx = time(g, reps, copies, &|c| {
                ops::mxfp8_gemv_rows(
                    g,
                    &nk.mx.row_major,
                    &w.mx[c],
                    x,
                    outs[0],
                    m,
                    w.n,
                    w.k,
                    w.n,
                    0,
                )
            })?;
            line(
                &format!("{label} MXFP8 twin (production)"),
                m,
                mx,
                w.mx_bytes(),
            );
            let fb = time(g, reps, copies, &|c| {
                ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[c], x, outs[0], m, w.n, w.k, w.n, 0)
            })?;
            line(
                &format!("{label} FP8 block (native)"),
                m,
                fb,
                w.fp8b_bytes(),
            );
        }
    }

    println!(
        "== shared expert, TP split rank half (42 layers): gate + up [1024 x 4096], down [4096 x 1024/2048]"
    );
    for m in ROWS {
        let copies = sh_gu.nv.len().min(sh_down.nv.len());
        let nv = time(g, reps, copies, &|c| {
            let (gw, gs) = sh_gu.nv[c];
            let half = (gw.offset(1024 * 4096 / 2), gs.offset(1024 * 4096 / 16));
            nv_tc(g, tc8, half, x, outs[0], m, 1024, 4096)?;
            nv_tc(
                g,
                tc8,
                sh_gu.nv[(c + 1) % copies],
                x,
                outs[1],
                m,
                1024,
                4096,
            )?;
            nv_tc_ld(
                g,
                tc8_ld,
                sh_down.nv[c],
                outs[0],
                outs[2],
                m,
                4096,
                1024,
                2048,
                1024,
            )
        })?;
        let nv_bytes = 2 * sh_gu.nv_bytes() / 2 + sh_down.nv_bytes() / 2;
        line("NVFP4 tc8 split (production)", m, nv, nv_bytes);
        let fb = time(g, reps, copies, &|c| {
            let rows = |w: &Fp8BlockWeight| Fp8BlockWeight {
                data: w.data.offset(1024 * 4096),
                scales: w.scales.offset(8 * 32 * 4),
                tiled: false,
            };
            let parts = [
                MxPart::fp8b(&rows(&sh_gu.fp8b[c]), outs[0], 1024),
                MxPart::fp8b(&rows(&sh_gu.fp8b[(c + 1) % copies]), outs[1], 1024),
            ];
            ops::mx_multi_rows(g, &nk.fp8b_multi, false, x, &parts, m, 4096, 0)?;
            let d = &sh_down.fp8b[c];
            ops::fp8b_gemv_ld_rows(
                g,
                &nk.fp8b_ld,
                d.data.offset(1024),
                d.scales.offset(8 * 4),
                outs[0],
                outs[2],
                m,
                4096,
                1024,
                2048,
                0,
            )
        })?;
        let fb_bytes = 2 * sh_gu.fp8b_bytes() / 2 + sh_down.fp8b_bytes() / 2;
        line("FP8 block split (native)", m, fb, fb_bytes);
    }

    println!(
        "== dense MLP 0..2, replicated (3 layers): gate + up [12288 x 4096], down [4096 x 12288]"
    );
    for m in ROWS {
        let copies = dense_gu.nv.len().min(dense_down.nv.len());
        let nv = time(g, reps, copies, &|c| {
            nv_tc(g, tc8, dense_gu.nv[c], x, outs[0], m, 12288, 4096)?;
            nv_tc(
                g,
                tc8,
                dense_gu.nv[(c + 1) % copies],
                x,
                outs[1],
                m,
                12288,
                4096,
            )?;
            nv_tc(g, tc8, dense_down.nv[c], outs[0], outs[2], m, 4096, 12288)
        })?;
        line(
            "NVFP4 tc8 (production)",
            m,
            nv,
            2 * dense_gu.nv_bytes() + dense_down.nv_bytes(),
        );
        let mx = time(g, reps, copies, &|c| {
            let parts = [
                MxPart::mx(&dense_gu.mx[c], outs[0], 12288),
                MxPart::mx(&dense_gu.mx[(c + 1) % copies], outs[1], 12288),
            ];
            ops::mx_multi_rows(g, &nk.mx_multi, false, x, &parts, m, 4096, 0)?;
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &dense_down.mx[c],
                outs[0],
                outs[2],
                m,
                4096,
                12288,
                4096,
                0,
            )
        })?;
        line(
            "MXFP8 fused gate+up, down (native)",
            m,
            mx,
            2 * dense_gu.mx_bytes() + dense_down.mx_bytes(),
        );
    }

    println!("== gate");
    // 1. FP8 block row invariance: rows at M == the same rows at 32.
    for (label, w) in [
        ("q_a", &q_a),
        ("q_b", &q_b),
        ("o", &mla_o),
        ("shared down", &sh_down),
    ] {
        let (n, k) = (w.n, w.k);
        ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[0], x, outs[3], 32, n, k, n, 0)?;
        g.synchronize(0)?;
        let full = fetch(g, outs[3], 32 * n * 2)?;
        for m in ROWS {
            ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[0], x, outs[2], m, n, k, n, 0)?;
            g.synchronize(0)?;
            let got = fetch(g, outs[2], m * n * 2)?;
            if got != full[..m * n * 2] {
                println!("  FAIL {label}: rows at M={m} differ from the same rows at 32");
                fail = true;
            }
        }
    }
    // 2. Multi == singles (KDA q/k/v MXFP8, MLA q_a + kv_a FP8 block).
    for m in [1usize, 9, 32] {
        let parts: Vec<MxPart> = (0..3)
            .map(|i| MxPart::mx(&kda.mx[i], outs[i], 4096))
            .collect();
        ops::mx_multi_rows(g, &nk.mx_multi, false, x, &parts, m, 4096, 0)?;
        g.synchronize(0)?;
        for i in 0..3 {
            let fused = fetch(g, outs[i], m * 4096 * 2)?;
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &kda.mx[i],
                x,
                outs[3],
                m,
                4096,
                4096,
                4096,
                0,
            )?;
            g.synchronize(0)?;
            if fused != fetch(g, outs[3], m * 4096 * 2)? {
                println!("  FAIL KDA fused projection {i} at M={m} differs from its own launch");
                fail = true;
            }
        }
        let parts = [
            MxPart::fp8b(&q_a.fp8b[0], outs[0], 1536),
            MxPart::fp8b(&kv_a.fp8b[0], outs[1], 512),
        ];
        ops::mx_multi_rows(g, &nk.fp8b_multi, false, x, &parts, m, 4096, 0)?;
        g.synchronize(0)?;
        for (i, w) in [&q_a, &kv_a].into_iter().enumerate() {
            let fused = fetch(g, outs[i], m * w.n * 2)?;
            ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[0], x, outs[3], m, w.n, 4096, w.n, 0)?;
            g.synchronize(0)?;
            if fused != fetch(g, outs[3], m * w.n * 2)? {
                println!("  FAIL MLA fused projection {i} at M={m} differs from its own launch");
                fail = true;
            }
        }
    }
    // 3. FP8 block vs the f64 host reference (q_b: K = 1536, first 256 rows).
    {
        let (m, w) = (4usize, &q_b);
        ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[0], x, outs[0], m, w.n, w.k, w.n, 0)?;
        g.synchronize(0)?;
        let got = bf16_to_f32(&fetch(g, outs[0], m * w.n * 2)?);
        let xb = bf16_to_f32(&bf16_bytes(&x_host[..m * w.k]));
        let (values, grid) = &w.host;
        let mut worst = 0f64;
        for i in 0..m {
            let mut ymax = 1e-30f64;
            let mut ys = Vec::with_capacity(256);
            for r in 0..256 {
                let mut acc = 0f64;
                for c in 0..w.k {
                    let s = grid[(r / 128) * (w.k / 128) + c / 128] as f64;
                    acc += xb[i * w.k + c] as f64 * e4m3(values[r * w.k + c]) * s;
                }
                ymax = ymax.max(acc.abs());
                ys.push(acc);
            }
            for (r, y) in ys.iter().enumerate() {
                worst = worst.max((got[i * w.n + r] as f64 - y).abs() / ymax);
            }
        }
        println!("  fp8b vs f64 reference (q_b, M=4, 256 rows): max |err| / |y|max = {worst:.2e}");
        if worst > 1e-2 {
            println!("  FAIL fp8b reference");
            fail = true;
        }
    }
    println!("{}", if fail { "gate: FAIL" } else { "gate: pass" });
    std::process::exit(i32::from(fail));
}

/// E4M3 byte to f64 (no NaN codes are generated).
fn e4m3(b: u8) -> f64 {
    let (s, e, m) = ((b >> 7) & 1, ((b >> 3) & 0xF) as i32, (b & 7) as f64);
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if s == 1 { -v } else { v }
}
