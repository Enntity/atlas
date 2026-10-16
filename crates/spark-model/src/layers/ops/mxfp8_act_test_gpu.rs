// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`; a GB10 and a kernel target with `mxfp8_act_quant`)
//! for the MXFP8 W8A8 prefill operands and GEMM:
//!
//! - `mxfp8_quant_act_bf16` against the ModelOpt host reference, byte for
//!   byte, over random BF16 blocks across the whole exponent range plus zero,
//!   inf, NaN, -0, subnormal and exact-power blocks;
//! - `mxfp8_lt_scale_tile` against the host relayout (random scales, a TP K
//!   slice through `src_ld`, and with `ATLAS_FP4L_CKPT` real scales).
//!
//! The GEMM's checks and bench are in `mxfp8_gemm_test_gpu.rs`.
//!
//! ```text
//! ATLAS_FP4L_CKPT=<checkpoint dir> cargo test --release -p spark-model --lib \
//!     mxfp8_act -- --ignored --nocapture --test-threads=1
//! ```

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::mxfp8::rows_gpu_tests::real::tensor;
use super::super::mxfp8::rows_gpu_tests::{Rng, backend, upload};
use super::tests::{quant_rows, tile_scales};
use super::*;

pub(super) fn down(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut v)?;
    Ok(v)
}

pub(super) fn kernels(gpu: &dyn GpuBackend) -> Result<(KernelHandle, KernelHandle)> {
    Ok((
        gpu.kernel(MXFP8_ACT_MODULE, "mxfp8_quant_act_bf16")?,
        gpu.kernel(MXFP8_ACT_MODULE, "mxfp8_lt_scale_tile")?,
    ))
}

pub(super) fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A uniform in [0, 1).
pub(super) fn unit(rng: &mut Rng) -> f64 {
    (rng.next() >> 11) as f64 / (1u64 << 53) as f64
}

/// Random BF16 rows of `k` whose 32-value blocks each pick a regime: zeros,
/// one +-inf, one NaN, -0s, subnormals, an exact 448 * 2^j amax, or random
/// values within a few binades of a random exponent (the whole range).
fn hostile_rows(rng: &mut Rng, m: usize, k: usize) -> Vec<u16> {
    let mut x = vec![0u16; m * k];
    for blk in x.chunks_mut(32) {
        let mode = rng.next() % 12;
        let base = 1 + (rng.next() % 254) as u16; // exponent field 1..=254
        for v in blk.iter_mut() {
            let sign = ((rng.next() & 1) as u16) << 15;
            let mant = (rng.next() & 0x7f) as u16;
            let exp = base.saturating_sub((rng.next() % 6) as u16).max(1);
            *v = sign | (exp << 7) | mant;
        }
        let at = (rng.next() % 32) as usize;
        match mode {
            0 => blk.fill(0),
            1 => blk[at] = 0x7f80 | (((rng.next() & 1) as u16) << 15),
            2 => blk[at] = 0x7fc1,
            3 => blk.iter_mut().for_each(|v| *v = 0x8000),
            4 => blk.iter_mut().for_each(|v| *v = (*v & 0x807f) | 1),
            5 => {
                // amax = 448 * 2^j exactly (bf16 0x43e0 = 448), the rest smaller.
                let j = (rng.next() % 200) as i32 - 100;
                let e = (0x43e0u16 >> 7) as i32 + j;
                blk.iter_mut()
                    .for_each(|v| *v = (*v & 0x807f) | (((e - 1).max(1) as u16) << 7));
                blk[at] = (0x43e0 & 0x807f) | ((e as u16) << 7);
            }
            _ => {}
        }
    }
    x
}

#[test]
#[ignore = "needs a GB10 GPU and the mxfp8_act_quant kernels"]
fn mxfp8_quant_act_matches_the_modelopt_host_reference() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let (quant, _) = kernels(gpu)?;
    let mut rng = Rng(0x5eed_a11c);
    // (m, k, lda): a ragged tile, a padded scale row (K = 320 -> 10 of 12
    // columns), a strided input, one row, a large pass.
    for (m, k, lda) in [
        (333usize, 2560usize, 2560usize),
        (77, 320, 384),
        (1, 3072, 3072),
        (4096, 6144, 6144),
    ] {
        let x = hostile_rows(&mut rng, m, lda);
        let dense: Vec<u16> = (0..m)
            .flat_map(|r| x[r * lda..r * lda + k].to_vec())
            .collect();
        let (q_ref, sf_ref) = quant_rows(&dense, m, k);
        let sf_bytes = mx_lt_scale_bytes(m, k);
        let dx = upload(gpu, &u16_bytes(&x))?;
        let (dq, dsf) = (gpu.alloc(m * k)?, gpu.alloc(sf_bytes)?);
        gpu.memset(dsf, 0xaa, sf_bytes)?;
        mxfp8_quant_act(
            gpu, quant, dx, lda as u32, dq, dsf, m as u32, k as u32, stream,
        )?;
        gpu.synchronize(stream)?;
        let (q, sf) = (down(gpu, dq, m * k)?, down(gpu, dsf, sf_bytes)?);
        let bad_q = q.iter().zip(&q_ref).filter(|(a, b)| a != b).count();
        let pad = mx_lt_inner_pad(k);
        let mut bad_sf = 0;
        let mut untouched = true;
        for r in 0..m.div_ceil(128) * 128 {
            for c in 0..pad {
                let o = mx_lt_sf_offset(r, c, pad);
                if r < m {
                    bad_sf += usize::from(sf[o] != sf_ref[o]);
                } else {
                    untouched &= sf[o] == 0xaa;
                }
            }
        }
        println!(
            "quant m={m} k={k} lda={lda}: {bad_q}/{} E4M3 and {bad_sf}/{} scale bytes differ",
            m * k,
            m * pad
        );
        assert_eq!(bad_q, 0, "E4M3 bytes differ from the host reference");
        assert_eq!(bad_sf, 0, "scale bytes differ from the host reference");
        assert!(untouched, "scales of rows past m were written");
        for p in [dx, dq, dsf] {
            gpu.free(p)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU and the mxfp8_act_quant kernels"]
fn mxfp8_lt_scale_tile_matches_the_host_relayout() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let (_, tile) = kernels(gpu)?;
    let mut rng = Rng(0x7117_5ca1);
    // (label, row-major [rows, full_inner] scales, first column, columns).
    let mut cases: Vec<(String, Vec<u8>, usize, usize, usize)> = Vec::new();
    let random: Vec<u8> = (0..300 * 96).map(|_| rng.next() as u8).collect();
    cases.push(("random 300x96".into(), random.clone(), 300, 0, 96));
    cases.push(("random 300x96 cols 48..96".into(), random, 300, 48, 48));
    cases.push((
        "random 1x10".into(),
        (0..10).map(|_| rng.next() as u8).collect(),
        1,
        0,
        10,
    ));
    if let Ok(dir) = std::env::var("ATLAS_FP4L_CKPT") {
        let l = |i: usize, rest: &str| format!("model.language_model.layers.{i}.{rest}");
        let (_, s, n, k) = tensor(&dir, &l(0, "linear_attn.in_proj_qkv"))?;
        cases.push(("real l0 in_proj_qkv".into(), s, n, 0, k / 32));
        let (_, s, n, k) = tensor(&dir, &l(0, "linear_attn.out_proj"))?;
        cases.push(("real l0 out_proj TP2 rank 1".into(), s, n, k / 64, k / 64));
        let (_, s, n, k) = tensor(&dir, &l(0, "mlp.shared_expert.down_proj"))?;
        cases.push((
            "real l0 shared down TP2 rank 1".into(),
            s,
            n,
            k / 64,
            k / 64,
        ));
    } else {
        println!("ATLAS_FP4L_CKPT unset: random scales only");
    }
    for (label, src, rows, c0, inner) in &cases {
        let full = src.len() / rows;
        let host = tile_scales(&src[*c0..], *rows, *inner, full);
        let dsrc = upload(gpu, src)?;
        let bytes = mx_lt_scale_bytes(*rows, inner * 32);
        let ddst = gpu.alloc(bytes)?;
        gpu.memset(ddst, 0x5a, bytes)?;
        mxfp8_lt_tile_scales(
            gpu,
            tile,
            dsrc.offset(*c0),
            full as u32,
            ddst,
            *rows as u32,
            (inner * 32) as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let dev = down(gpu, ddst, bytes)?;
        let bad = dev.iter().zip(&host).filter(|(a, b)| a != b).count();
        println!("scale tile {label}: {bad}/{bytes} bytes differ");
        assert_eq!(bad, 0, "{label}: relayout differs from the host's");
        gpu.free(dsrc)?;
        gpu.free(ddst)?;
    }
    Ok(())
}
