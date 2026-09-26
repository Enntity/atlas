// SPDX-License-Identifier: AGPL-3.0-only
//! Accuracy + bandwidth gate for the tensor-core `w4a16_gemv_tc8` tier
//! against the scalar `w4a16_gemv_batch8` at GLM-5.3 Flash TP2 projection
//! shapes and every M the tier serves (1..=8).
//!
//! tc8 sums in a different order, so the gate is numeric, not bytewise: the
//! worst |tc8 - batch8| must stay within 2 BF16 ulps of the output magnitude
//! (both round an FP32 accumulator once). Timings are cold-weight (a scratch
//! flush between reps) and report effective weight bandwidth.
//!
//! Exit: 0 pass, 1 any leg out of tolerance, 2 kernels absent.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash-nvfp4 \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example w4a16_tc8_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const GROUP_SIZE: usize = 16;
const SCALE2: f32 = 0.0123_f32;

/// GLM-5.3 Flash per-rank (TP2) projection shapes, plus an odd-N tail.
const SHAPES: [(&str, usize, usize); 6] = [
    ("kda qkv/o     [4096 x 4096]", 4096, 4096),
    ("shared gate/up[2048 x 4096]", 2048, 4096),
    ("shared down   [4096 x 2048]", 4096, 2048),
    ("dense ffn     [12288 x 4096]", 12288, 4096),
    ("mla q_b       [8192 x 1536]", 8192, 1536),
    ("tail odd-N    [4099 x 4096]", 4099, 4096),
];

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    tc: bool,
    a: DevicePtr,
    w: DevicePtr,
    ws: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let (grid, block) = if tc { (div_ceil(n, 16), 256) } else { (div_ceil(n, 4), 256) };
    KernelLaunch::new(g, kh)
        .grid([grid, 1, 1])
        .block([block, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(ws)
        .arg_f32(SCALE2)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let (Ok(b8), Ok(tc8)) = (
        g.kernel("w4a16_gemv", "w4a16_gemv_batch8"),
        g.kernel("w4a16_gemv", "w4a16_gemv_tc8"),
    ) else {
        eprintln!("w4a16_gemv_batch8 / w4a16_gemv_tc8 absent from this target");
        std::process::exit(2);
    };
    let mut fail = false;
    for (name, n, k) in SHAPES {
        let mut rng = Lcg(0x5EED ^ (n * 31 + k) as u64);
        let a: Vec<u8> = (0..8 * k)
            .flat_map(|_| bf16::from_f32(rng.f() * 3.0 - 1.5).to_bits().to_le_bytes())
            .collect();
        let w: Vec<u8> = (0..n * k / 2).map(|_| (rng.f() * 256.0) as u8).collect();
        let ws: Vec<u8> = (0..n * k / GROUP_SIZE)
            .map(|_| 0x30u8 + (rng.f() * 24.0) as u8)
            .collect();
        let (ad, wd, wsd) = (up(g, &a)?, up(g, &w)?, up(g, &ws)?);
        // Rotating weight copies (> L2) so back-to-back timed launches stream
        // from DRAM while their launch overhead overlaps the previous kernel.
        let copies = (96usize << 20).div_ceil(w.len() + ws.len()).max(2);
        let rot: Vec<(DevicePtr, DevicePtr)> = (0..copies)
            .map(|_| Ok((up(g, &w)?, up(g, &ws)?)))
            .collect::<Result<_>>()?;
        let (c_ref, c_tc) = (g.alloc(8 * n * 2)?, g.alloc(8 * n * 2)?);
        let weight_bytes = (n * k / 2 + n * k / GROUP_SIZE) as f64;
        for m in 1..=8u32 {
            launch(g, b8, false, ad, wd, wsd, c_ref, m, n as u32, k as u32)?;
            launch(g, tc8, true, ad, wd, wsd, c_tc, m, n as u32, k as u32)?;
            g.synchronize(0)?;
            let (r, t) = (down(g, c_ref, m as usize * n)?, down(g, c_tc, m as usize * n)?);
            let mut worst = 0f32;
            for (x, y) in r.iter().zip(&t) {
                // 2 BF16 ulps of the larger magnitude (+ a floor for ~0 outputs).
                let tol = x.abs().max(y.abs()) * (2.0 / 128.0) + 1e-3;
                worst = worst.max((x - y).abs() / tol);
            }
            let ok = worst <= 1.0;
            fail |= !ok;
            let time = |kh, tc| -> Result<f64> {
                let reps = 4 * copies;
                g.synchronize(0)?;
                let t0 = std::time::Instant::now();
                for i in 0..reps {
                    let (w, s) = rot[i % copies];
                    launch(g, kh, tc, ad, w, s, c_ref, m, n as u32, k as u32)?;
                }
                g.synchronize(0)?;
                Ok(t0.elapsed().as_secs_f64() / reps as f64)
            };
            let (t8, ttc) = (time(b8, false)?, time(tc8, true)?);
            println!(
                "{name} M={m}: worst/tol {worst:.3} {}  batch8 {:6.1}us {:5.0}GB/s  tc8 {:6.1}us {:5.0}GB/s",
                if ok { "ok" } else { "FAIL" },
                t8 * 1e6,
                weight_bytes / t8 / 1e9,
                ttc * 1e6,
                weight_bytes / ttc / 1e9,
            );
        }
    }
    std::process::exit(if fail { 1 } else { 0 });
}
