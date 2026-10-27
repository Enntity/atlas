// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`; a GB10, a glm-5.3-flash kernel build and the
//! FlashKDA bridge): the FlashKDA prefill at TP4's 16 local heads and TP2's
//! 32 against the reference recurrence (`kda_recurrent_bf16`) over two
//! consecutive 4096-row pieces (the state carries across calls). FlashKDA
//! runs the chunked algorithm, so it is compared by error, which must be the
//! same small error at both head counts; then both are timed.
//!
//!   ATLAS_KDA_FLASH_LIBRARY=/opt/atlas/native/libatlas_mango_flash.so \
//!     cargo test --release -p spark-model --lib flash_kda_heads -- --ignored --nocapture

use super::*;
use crate::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

fn uniform(seed: &mut u64, lo: f32, hi: f32) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    lo + (hi - lo) * ((*seed >> 40) as f32 / (1u64 << 24) as f32)
}

fn bf16(v: f32) -> u16 {
    ((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16
}

fn bf16_bytes(n: usize, seed: &mut u64, lo: f32, hi: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| bf16(uniform(seed, lo, hi)).to_le_bytes())
        .collect()
}

fn f32_bytes(n: usize, seed: &mut u64, lo: f32, hi: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| uniform(seed, lo, hi).to_le_bytes())
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(g.default_stream())?;
    let mut v = vec![0u8; bytes];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn as_f32(bytes: &[u8], bf: bool) -> Vec<f32> {
    if bf {
        bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// (max |a - b| / max |b|, mean |a - b| / mean |b|).
fn error(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (mut dmax, mut bmax, mut dsum, mut bsum) = (0f32, 0f32, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        assert!(x.is_finite(), "non-finite FlashKDA value");
        let d = (x - y).abs();
        dmax = dmax.max(d);
        bmax = bmax.max(y.abs());
        dsum += d as f64;
        bsum += y.abs() as f64;
    }
    (dmax / bmax, (dsum / bsum) as f32)
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_KDA_FLASH_LIBRARY"]
fn flash_kda_heads_match_reference_recurrence() -> Result<()> {
    let path = std::env::var("ATLAS_KDA_FLASH_LIBRARY")?;
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let reference = g.kernel("kda", "kda_recurrent_bf16")?;
    let stream = g.default_stream();
    let (piece, d) = (4096usize, 128usize);
    let mut errors = Vec::new();
    for heads in [16usize, 32] {
        let flash = FlashPrefill::open(&path, heads, d, -5.0)?;
        let tokens = 2 * piece;
        let p = heads * d;
        let mut seed = 0x5eed_0000 + heads as u64;
        let qkv = upload(g, &bf16_bytes(tokens * 3 * p, &mut seed, -2.0, 2.0))?;
        let gate = upload(g, &bf16_bytes(tokens * p, &mut seed, -3.0, 1.0))?;
        let beta = upload(g, &bf16_bytes(tokens * heads, &mut seed, -2.0, 2.0))?;
        let a = upload(g, &f32_bytes(heads, &mut seed, 0.0, 2.7))?;
        let bias = upload(g, &f32_bytes(p, &mut seed, -0.5, 0.5))?;
        let state0 = f32_bytes(heads * d * d, &mut seed, -0.1, 0.1);
        let state_bytes = state0.len();
        let (state_want, state_got) = (upload(g, &state0)?, upload(g, &state0)?);
        let out_bytes = tokens * p * 2;
        let (want, got) = (g.alloc(out_bytes)?, g.alloc(out_bytes)?);
        let ws = unsafe { (flash.workspace_size)(piece as i32, heads as i32, 1) } as usize;
        let aux_cap = (state_bytes + piece * heads * 2).next_multiple_of(128) + 20;
        let scratch = [
            (g.alloc(3 * piece * p * 2)?, 3 * piece * p * 2),
            (g.alloc(ws)?, ws),
            (g.alloc(aux_cap)?, aux_cap),
        ];
        let reference_run = |state: DevicePtr| {
            ops::kda_recurrent(
                g,
                reference,
                qkv,
                gate,
                beta,
                a,
                bias,
                state,
                want,
                tokens as u32,
                heads as u32,
                d as u32,
                -5.0,
                stream,
            )
        };
        let flash_run = |state: DevicePtr| -> Result<()> {
            for row in [0, piece] {
                flash.run_piece(
                    [
                        qkv.offset(row * 3 * p * 2),
                        gate.offset(row * p * 2),
                        beta.offset(row * heads * 2),
                        a,
                        bias,
                        state,
                        got.offset(row * p * 2),
                    ],
                    piece,
                    scratch,
                    stream,
                )?;
            }
            Ok(())
        };
        reference_run(state_want)?;
        flash_run(state_got)?;
        let out = error(
            &as_f32(&read(g, got, out_bytes)?, true),
            &as_f32(&read(g, want, out_bytes)?, true),
        );
        let state = error(
            &as_f32(&read(g, state_got, state_bytes)?, false),
            &as_f32(&read(g, state_want, state_bytes)?, false),
        );
        let time = |run: &dyn Fn(DevicePtr) -> Result<()>, s: DevicePtr| -> Result<f64> {
            g.synchronize(stream)?;
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                run(s)?;
            }
            g.synchronize(stream)?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / 5.0)
        };
        let ms_ref = time(&reference_run, state_want)?;
        let ms_flash = time(&flash_run, state_got)?;
        println!(
            "H{heads} T{tokens}: output err max {:.2e} mean {:.2e}; state err max {:.2e} mean {:.2e}; \
             reference {ms_ref:.2} ms, FlashKDA {ms_flash:.2} ms ({:.1}x)",
            out.0,
            out.1,
            state.0,
            state.1,
            ms_ref / ms_flash
        );
        errors.push((out, state));
        for ptr in [qkv, gate, beta, a, bias, state_want, state_got, want, got] {
            g.free(ptr)?;
        }
        for (ptr, _) in scratch {
            g.free(ptr)?;
        }
    }
    for &((out_max, out_mean), (state_max, state_mean)) in &errors {
        assert!(
            out_mean < 2e-2 && state_mean < 2e-2 && out_max < 0.1 && state_max < 0.1,
            "FlashKDA strays from the recurrence: {errors:?}"
        );
    }
    // The same algorithm at both head counts: a layout fault at 16 heads would
    // show as an error far above 32 heads'.
    let (h16, h32) = (errors[0], errors[1]);
    assert!(
        h16.0.1 < 4.0 * h32.0.1 + 1e-4 && h16.1.1 < 4.0 * h32.1.1 + 1e-4,
        "H16 error {h16:?} far above H32's {h32:?}"
    );
    Ok(())
}

/// The planar entry (bridge ABI 2) over q / k / v planes leaves the packed
/// entry's output and state bytes, piece after piece.
#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_KDA_FLASH_LIBRARY (ABI 2)"]
fn flash_kda_planes_match_packed() -> Result<()> {
    // SAFETY: set before the bridge reads it, in this test only.
    unsafe { std::env::set_var("ATLAS_GLM_KDA_FLASH_PLANES", "1") };
    let path = std::env::var("ATLAS_KDA_FLASH_LIBRARY")?;
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let pack = g.kernel("kda", "kda_pack_qkv")?;
    let (piece, d) = (4096usize, 128usize);
    for heads in [16usize, 32] {
        let flash = FlashPrefill::open(&path, heads, d, -5.0)?;
        ensure!(flash.takes_planes(), "bridge without the planar entry");
        let tokens = 2 * piece;
        let p = heads * d;
        let mut seed = 0xf1a5_0000 + heads as u64;
        let planes = upload(g, &bf16_bytes(tokens * 3 * p, &mut seed, -2.0, 2.0))?;
        let qkv = g.alloc(tokens * 3 * p * 2)?;
        ops::kda_pack_qkv(g, pack, planes, qkv, tokens as u32, p as u32, stream)?;
        let gate = upload(g, &bf16_bytes(tokens * p, &mut seed, -3.0, 1.0))?;
        let beta = upload(g, &bf16_bytes(tokens * heads, &mut seed, -2.0, 2.0))?;
        let a = upload(g, &f32_bytes(heads, &mut seed, 0.0, 2.7))?;
        let bias = upload(g, &f32_bytes(p, &mut seed, -0.5, 0.5))?;
        let state0 = f32_bytes(heads * d * d, &mut seed, -0.1, 0.1);
        let state_bytes = state0.len();
        let (state_want, state_got) = (upload(g, &state0)?, upload(g, &state0)?);
        let out_bytes = tokens * p * 2;
        let (want, got) = (g.alloc(out_bytes)?, g.alloc(out_bytes)?);
        let ws = unsafe { (flash.workspace_size)(piece as i32, heads as i32, 1) } as usize;
        let aux_cap = (state_bytes + piece * heads * 2).next_multiple_of(128) + 20;
        let (packed, workspace, aux) =
            (g.alloc(3 * piece * p * 2)?, g.alloc(ws)?, g.alloc(aux_cap)?);
        let plane = tokens * p * 2;
        for row in [0, piece] {
            let (gate, beta) = (gate.offset(row * p * 2), beta.offset(row * heads * 2));
            flash.run_piece(
                [
                    qkv.offset(row * 3 * p * 2),
                    gate,
                    beta,
                    a,
                    bias,
                    state_want,
                    want.offset(row * p * 2),
                ],
                piece,
                [(packed, 3 * piece * p * 2), (workspace, ws), (aux, aux_cap)],
                stream,
            )?;
            let [q, k, v] = [0, 1, 2].map(|i| planes.offset(i * plane + row * p * 2));
            flash.run_piece_planes(
                [
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    a,
                    bias,
                    state_got,
                    got.offset(row * p * 2),
                ],
                piece,
                [(workspace, ws), (aux, aux_cap)],
                stream,
            )?;
        }
        ensure!(
            read(g, want, out_bytes)? == read(g, got, out_bytes)?,
            "H{heads}: planar output differs"
        );
        ensure!(
            read(g, state_want, state_bytes)? == read(g, state_got, state_bytes)?,
            "H{heads}: planar state differs"
        );
        println!("H{heads}: planar FlashKDA == packed (2 x {piece} rows, output and state)");
    }
    Ok(())
}
