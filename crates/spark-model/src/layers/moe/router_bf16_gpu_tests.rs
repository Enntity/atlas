// SPDX-License-Identifier: AGPL-3.0-only

//! GPU: qwen4_exp's BF16 router (`mlp.gate`, the checkpoint's bytes) on the
//! three kernel families that read it — decode (`dense_gemv_bf16`), verify
//! and batch rows (`dense_gemv_bf16_batchm` up to 8 rows, the
//! `Qwen4ExpWideRows` 16/32-row tiers past it) and prefill (the q38 tile
//! GEMM) — against an FP64 host reference of the same weights, on real
//! router weights of the FP4-Local checkpoint. `#[ignore]`; on a GB10 with a
//! qwen3.8-flash-next kernel build:
//!
//! ```text
//! ATLAS_FP4L_CKPT=<model dir> cargo test --release -p spark-model --lib \
//!   router_bf16_gpu -- --ignored --nocapture
//! ```
//!
//! Checked: every verify/batch row is byte-identical to the decode row (the
//! spec == serial contract), and every kernel's top-10 expert set matches the
//! FP64 reference's on every row whose 10th/11th logits are not within a
//! BF16 rounding of each other (the NVFP4 copy this replaces moved the set
//! on most rows: 8.5-9.3% weight error).

use std::path::PathBuf;

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

const H: usize = 2560;
const E: usize = 512;
const TOP_K: usize = 10;
const ROWS: usize = 64;

fn bf16_to_f32(b: [u8; 2]) -> f32 {
    f32::from_bits((u16::from_le_bytes(b) as u32) << 16)
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32([c[0], c[1]]))
        .collect()
}

/// `ROWS` unit-RMS BF16 rows, a fixed LCG draw (the router reads the
/// post-norm hidden state).
fn inputs() -> Vec<u8> {
    let mut s = 0x9e37_79b9u32;
    let mut out = Vec::with_capacity(ROWS * H * 2);
    for _ in 0..ROWS {
        let row: Vec<f32> = (0..H)
            .map(|_| {
                // Sum of four uniforms: a cheap bell curve.
                (0..4)
                    .map(|_| {
                        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (s >> 8) as f32 / (1 << 24) as f32 - 0.5
                    })
                    .sum::<f32>()
            })
            .collect();
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / H as f32).sqrt();
        for v in row {
            out.extend_from_slice(&atlas_core::numeric::f32_to_bf16(v / rms).to_le_bytes());
        }
    }
    out
}

/// Top-10 expert ids of `logits` (descending, ties to the lower id), and
/// whether the 10th and 11th are farther apart than `tie`.
fn top10(logits: &[f64], tie: f64) -> (Vec<usize>, bool) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let clear = logits[order[TOP_K - 1]] - logits[order[TOP_K]] > tie;
    let mut set = order[..TOP_K].to_vec();
    set.sort_unstable();
    (set, clear)
}

#[test]
#[ignore]
fn router_bf16_gpu_kernels_match_the_reference() {
    let dir = PathBuf::from(std::env::var("ATLAS_FP4L_CKPT").expect("ATLAS_FP4L_CKPT"));
    // The wide 16/32-row tiers resolve only under the batched lane.
    unsafe { std::env::set_var("ATLAS_QWEN4EXP_BATCH_FAST", "1") };
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let gemv = g.kernel("gemv", "dense_gemv_bf16").unwrap();
    let batchm = g
        .kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
        .unwrap();
    let tile = g.kernel("gemm", "dense_gemm_bf16_pipelined").unwrap();
    let wide = ops::Qwen4ExpWideRows::resolve(g, "qwen4_exp");
    let up = |b: &[u8]| {
        let p = g.alloc(b.len()).unwrap();
        g.copy_h2d(b, p).unwrap();
        p
    };
    let down = |p: DevicePtr, n: usize| {
        g.synchronize(stream).unwrap();
        let mut v = vec![0u8; n];
        g.copy_d2h(p, &mut v).unwrap();
        v
    };

    let x_host = inputs();
    let x = up(&x_host);
    let xf = f32s(&x_host);
    let logits = g.alloc(ROWS * E * 2).unwrap();
    let mut worst = 0f64;
    for layer in [0usize, 11, 23, 35, 47] {
        let name = format!("model.language_model.layers.{layer}.mlp.gate.weight");
        let (dtype, shape, w_host) =
            crate::weight_map::test_checkpoint::read_checkpoint_tensor(&dir, &name).unwrap();
        assert_eq!(
            (dtype.as_str(), shape.as_slice()),
            ("BF16", &[E, H][..]),
            "{name}"
        );
        let w = DenseWeight {
            weight: up(&w_host),
        };
        let wf = f32s(&w_host);

        // Decode: one row a launch.
        for r in 0..ROWS {
            ops::dense_gemv(
                g,
                gemv,
                x.offset(r * H * 2),
                &w,
                logits.offset(r * E * 2),
                E as u32,
                H as u32,
                stream,
            )
            .unwrap();
        }
        let decode = down(logits, ROWS * E * 2);

        // Verify / batch rows: every width the decode loop launches.
        for m in [2usize, 3, 8, 9, 16, 17, 32] {
            let narrow = m <= ops::DENSE_GEMV_BATCHM_MAX_M as usize;
            for first in (0..ROWS - m + 1).step_by(m) {
                let (a, c) = (x.offset(first * H * 2), logits.offset(first * E * 2));
                let dims = (m as u32, E as u32, H as u32, E as u32);
                if narrow {
                    ops::dense_gemv_batchm(
                        g, batchm, a, &w, c, m as u32, E as u32, H as u32, E as u32, stream,
                    )
                    .unwrap();
                } else {
                    wide.dense_rows(g, batchm, a, &w, c, dims, stream).unwrap();
                }
            }
            let rows = (ROWS / m) * m;
            let got = down(logits, rows * E * 2);
            let diff = got.iter().zip(&decode).filter(|(a, b)| a != b).count();
            assert_eq!(
                diff, 0,
                "layer {layer}: {m}-row launches differ from decode in {diff} bytes"
            );
        }

        // Prefill: the q38 tile GEMM over the whole block.
        super::forward_prefill_route_sp::q38_router_gemm(
            g,
            tile,
            [x, w.weight, logits],
            [ROWS, E, H].map(|v| v as u32),
            stream,
        )
        .unwrap();
        let prefill = down(logits, ROWS * E * 2);

        // Both against FP64 over the same BF16 operands.
        let mut ambiguous = 0;
        for r in 0..ROWS {
            let xr = &xf[r * H..(r + 1) * H];
            let want: Vec<f64> = (0..E)
                .map(|e| {
                    wf[e * H..(e + 1) * H]
                        .iter()
                        .zip(xr)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum()
                })
                .collect();
            let norm = want.iter().map(|v| v * v).sum::<f64>().sqrt();
            // A BF16 store moves a logit by at most half an ulp: 2^-9 of it.
            let tie = want.iter().fold(0f64, |m, v| m.max(v.abs())) / 256.0;
            let (want_set, clear) = top10(&want, tie);
            ambiguous += usize::from(!clear);
            for (label, got) in [("decode", &decode), ("prefill", &prefill)] {
                let got: Vec<f64> = f32s(&got[r * E * 2..(r + 1) * E * 2])
                    .into_iter()
                    .map(f64::from)
                    .collect();
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
                    / norm;
                worst = worst.max(err);
                assert!(
                    err < 4e-3,
                    "layer {layer} row {r} {label}: logit rel err {err:.2e}"
                );
                if clear {
                    assert_eq!(
                        top10(&got, 0.0).0,
                        want_set,
                        "layer {layer} row {r} {label}"
                    );
                }
            }
        }
        eprintln!(
            "layer {layer}: decode == batch rows; top-10 == FP64 ({ambiguous} near-tied rows skipped)"
        );
        g.free(w.weight).unwrap();
    }
    eprintln!("worst logit relative error {worst:.2e}");
    g.free(logits).unwrap();
    g.free(x).unwrap();
}
