// SPDX-License-Identifier: AGPL-3.0-only
//! Gate for `glm_hc_post_mix_ss_bf16` (GLM mHC post fused with the next
//! site's pre-mix) against `hc_post_bf16` + `glm_hc_mix_ss_bf16`: the updated
//! BF16 highway must match bitwise; raw mix and sum of squares only reorder
//! FP32 sums (relative 1e-4). Also reports both paths' time at 4096 tokens.
//!
//! Exit: 0 pass, 1 mismatch.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash-nvfp4 \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example hc_post_mix_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const H: usize = 4096;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32 - 0.5
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}
fn bf(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes()).collect()
}
fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let post_k = g.kernel("hyper_connection", "hc_post_bf16")?;
    let mix_k = g.kernel("glm_hc_prefill_vec", "glm_hc_mix_ss_bf16")?;
    let fused_k = g.kernel("glm_hc_prefill_vec", "glm_hc_post_mix_ss_bf16")?;
    let mut fail = false;
    for tokens in [4096usize, 2049, 1000] {
        let mut r = Lcg(tokens as u64);
        let highway: Vec<f32> = (0..tokens * 4 * H).map(|_| r.f() * 4.0).collect();
        let block: Vec<f32> = (0..tokens * H).map(|_| r.f() * 2.0).collect();
        let post: Vec<f32> = (0..tokens * 4).map(|_| r.f() + 1.0).collect();
        let comb: Vec<f32> = (0..tokens * 16).map(|_| r.f() * 0.5 + 0.25).collect();
        let fn_: Vec<f32> = (0..24 * 4 * H).map(|_| r.f() * 0.02).collect();
        let (hw_a, hw_b) = (up(g, &bf(&highway))?, up(g, &bf(&highway))?);
        let (blk, pst, cmb, hfn) = (
            up(g, &bf(&block))?,
            up(g, &f32b(&post))?,
            up(g, &f32b(&comb))?,
            up(g, &f32b(&fn_))?,
        );
        let (mix_a, mix_b) = (g.alloc(tokens * 25 * 4)?, g.alloc(tokens * 25 * 4)?);
        let t = tokens as u32;
        let reference = |hw: DevicePtr| -> Result<()> {
            KernelLaunch::new(g, post_k)
                .grid([t, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(blk)
                .arg_ptr(hw)
                .arg_ptr(pst)
                .arg_ptr(cmb)
                .arg_ptr(hw)
                .arg_u32(H as u32)
                .arg_u32(4)
                .launch(0)?;
            KernelLaunch::new(g, mix_k)
                .grid([t.div_ceil(32), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(hw)
                .arg_ptr(hfn)
                .arg_ptr(mix_a)
                .arg_ptr(mix_a.offset(tokens * 24 * 4))
                .arg_u32(t)
                .launch(0)
        };
        let fused = |hw: DevicePtr| -> Result<()> {
            KernelLaunch::new(g, fused_k)
                .grid([t.div_ceil(32), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(blk)
                .arg_ptr(hw)
                .arg_ptr(pst)
                .arg_ptr(cmb)
                .arg_ptr(hfn)
                .arg_ptr(mix_b)
                .arg_ptr(mix_b.offset(tokens * 24 * 4))
                .arg_u32(t)
                .launch(0)
        };
        reference(hw_a)?;
        fused(hw_b)?;
        g.synchronize(0)?;
        let bytes = tokens * 4 * H * 2;
        let same_highway = down(g, hw_a, bytes)? == down(g, hw_b, bytes)?;
        let (ma, mb) = (
            as_f32(&down(g, mix_a, tokens * 25 * 4)?),
            as_f32(&down(g, mix_b, tokens * 25 * 4)?),
        );
        // Relative to the largest magnitude: the dots cancel, so per-element
        // relative error on a near-zero sum only measures the cancellation.
        let scale = ma.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = ma
            .iter()
            .zip(&mb)
            .map(|(a, b)| (a - b).abs() / scale)
            .fold(0f32, f32::max);
        let ok = same_highway && worst < 1e-4;
        fail |= !ok;
        let time = |f: &dyn Fn(DevicePtr) -> Result<()>, hw: DevicePtr| -> Result<f64> {
            f(hw)?;
            g.synchronize(0)?;
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                f(hw)?;
            }
            g.synchronize(0)?;
            Ok(t0.elapsed().as_secs_f64() / 20.0)
        };
        let (ta, tb) = (time(&reference, hw_a)?, time(&fused, hw_b)?);
        println!(
            "T={tokens}: highway {} mix worst rel {worst:.2e} {}  post+mix {:7.1}us  fused {:7.1}us",
            if same_highway { "bitwise" } else { "DIFFERS" },
            if ok { "ok" } else { "FAIL" },
            ta * 1e6,
            tb * 1e6
        );
    }
    std::process::exit(if fail { 1 } else { 0 });
}
