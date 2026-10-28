// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
#[ignore = "needs a GB10 GPU, cuBLASLt with MXFP8 and the mxfp8_act_quant kernels"]
fn mxfp8_gemm_is_correct_and_every_in_order_kernel_gives_the_same_rows() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    println!("cuBLASLt {}", cublaslt::version());
    let mut rng = Rng(0x0dd_ba11);
    for s in &shapes()? {
        let (n, k) = (s.n, s.k);
        let o = operands(gpu, s, SWEEP_ROWS, &mut rng)?;
        // Reference: the first in-order candidate over all SWEEP_ROWS rows.
        let out_ref = gpu.alloc(SWEEP_ROWS * n * 2)?;
        let (q, sf) = quantized(gpu, &o, k, 0, SWEEP_ROWS)?;
        let mut ref_desc = None;
        for i in 0..cublaslt::MX_CANDIDATES {
            let r = cublaslt::fp8_gemm_act_weight_t_mx_candidate(
                q.0,
                sf.0,
                o.w.0,
                o.w_sf.0,
                out_ref.0,
                [SWEEP_ROWS as u32, n as u32, k as u32],
                i,
                true,
                stream,
            )?;
            match r {
                Some((d, true)) => {
                    ref_desc = Some(d);
                    break;
                }
                Some(_) => {}
                None => break,
            }
        }
        let ref_desc = ref_desc.context("no in-order reference kernel")?;
        gpu.synchronize(stream)?;
        let full = down(gpu, out_ref, SWEEP_ROWS * n * 2)?;
        gpu.free(q)?;
        gpu.free(sf)?;
        // Correctness: sampled outputs against f64 over the dequantized
        // operands (the quant's own bytes, from the host reference).
        let (hq, hsf) = quant_rows(&o.host_act[..256 * k], 256, k);
        let pad = mx_lt_inner_pad(k);
        let mut worst = 0f64;
        for _ in 0..256 {
            let (r, c) = (
                (rng.next() % 256) as usize,
                (rng.next() % n as u64) as usize,
            );
            let (mut acc, mut mag) = (0f64, 0f64);
            for kk in 0..k {
                let a = f64::from(e4m3_to_f32(hq[r * k + kk]))
                    * 2f64.powi(i32::from(hsf[mx_lt_sf_offset(r, kk / 32, pad)]) - 127);
                let w = f64::from(e4m3_to_f32(o.host_w.0[c * k + kk]))
                    * 2f64.powi(i32::from(o.host_w.1[c * k / 32 + kk / 32]) - 127);
                acc += a * w;
                mag += (a * w).abs();
            }
            let got = f64::from(bf16(u16::from_le_bytes([
                full[(r * n + c) * 2],
                full[(r * n + c) * 2 + 1],
            ])));
            let tol = acc.abs() / 256.0 + mag * 1e-5 + 1e-30;
            worst = worst.max((got - acc).abs() / tol);
        }
        println!(
            "{} {n}x{k}: reference kernel {ref_desc}; worst sampled error {worst:.3} of tolerance",
            s.label
        );
        assert!(
            worst <= 1.0,
            "{}: MXFP8 GEMM off the f64 reference",
            s.label
        );

        // Invariance: every candidate at every (m, offset) against the
        // reference rows; the tuned entry point too.
        for &(m, r0) in &SWEEP {
            let (q, sf) = quantized(gpu, &o, k, r0, m)?;
            let out = gpu.alloc(m * n * 2)?;
            let want = &full[r0 * n * 2..(r0 + m) * n * 2];
            let mut line = Vec::new();
            let mut in_order_bad = 0;
            // The unmasked heuristic list, then the in-order-only one tuning
            // asks for (prefix `m`).
            for (masked, i) in [false, true]
                .into_iter()
                .flat_map(|b| (0..cublaslt::MX_CANDIDATES).map(move |i| (b, i)))
            {
                gpu.memset(out, 0xff, m * n * 2)?;
                let Some((d, ordered)) = cublaslt::fp8_gemm_act_weight_t_mx_candidate(
                    q.0,
                    sf.0,
                    o.w.0,
                    o.w_sf.0,
                    out.0,
                    [m as u32, n as u32, k as u32],
                    i,
                    masked,
                    stream,
                )?
                else {
                    continue;
                };
                gpu.synchronize(stream)?;
                let got = down(gpu, out, m * n * 2)?;
                let bad = got.iter().zip(want).filter(|(a, b)| a != b).count();
                if ordered && bad > 0 {
                    in_order_bad += 1;
                }
                line.push(format!(
                    "{}#{i} [{d}]{} {}",
                    if masked { "m" } else { "" },
                    if ordered { "" } else { " split" },
                    if bad == 0 {
                        "same".to_string()
                    } else {
                        format!("DIFF {bad}")
                    }
                ));
            }
            gpu.memset(out, 0xff, m * n * 2)?;
            cublaslt::fp8_gemm_act_weight_t_mx(
                q.0,
                sf.0,
                o.w.0,
                o.w_sf.0,
                out.0,
                [m as u32, n as u32, k as u32],
                stream,
            )?;
            gpu.synchronize(stream)?;
            let tuned_bad = down(gpu, out, m * n * 2)?
                .iter()
                .zip(want)
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "  m={m} at {r0}: tuned [{}] {} | {}",
                cublaslt::mx_algo_description(n as u32, k as u32, m as u32).unwrap_or_default(),
                if tuned_bad == 0 { "same" } else { "DIFF" },
                line.join("; ")
            );
            assert_eq!(
                in_order_bad, 0,
                "{} m={m}: an in-order kernel differs",
                s.label
            );
            assert_eq!(tuned_bad, 0, "{} m={m}: the tuned kernel differs", s.label);
            for p in [q, sf, out] {
                gpu.free(p)?;
            }
        }
        for p in [o.w, o.w_sf, o.act, out_ref] {
            gpu.free(p)?;
        }
    }
    Ok(())
}
