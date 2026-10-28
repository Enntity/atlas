// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
#[ignore = "needs a GB10 GPU and a qwen3.8-flash-next kernel build"]
fn shared_mxfp8_matches_f64_and_is_row_invariant() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let mut rng = Rng(0x5a1e_d0c7_3e1f_0b22);
    let plain = [
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc8")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc16")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc32")?,
    ];
    let quant = gpu.kernel(ops::MXFP8_ACT_MODULE, ops::MX_QUANT_KERNEL)?;
    let swiglu_quant = gpu.kernel("mxfp8_gemv", "mxfp8_swiglu_quant_act")?;
    for layer in LAYERS {
        let w = weights(layer, &mut rng)?;
        // The whole expert (TP1), then a TP2 rank's slices (`shared_mxfp8.rs`).
        let mut decoded = Vec::new();
        let x = activations(&mut rng, T);
        for (rank, size) in [(0, 1), (0, 2), (1, 2)] {
            let dev = upload_weights(gpu, &w, (rank, size))?;
            let il = I / size;
            let mx = Mxfp8SharedExpert::from_raw(gpu, dev.raw, [H, I], (rank, size), Some(T))?;
            let xd = upload(gpu, &bytes(&x))?;
            let (gu, act, out) = (
                gpu.alloc(T * 2 * il * 2)?,
                gpu.alloc(T * il * 2)?,
                gpu.alloc(T * H * 2)?,
            );
            let xf: Vec<f64> = x.iter().map(|&v| f64::from(f(v))).collect();

            // 1. Decode, stage by stage, on the first ROWS rows.
            ops::mxfp8_gemv_rows(
                gpu,
                &plain,
                &dev.gate_up,
                xd,
                gu,
                ROWS,
                2 * il,
                H,
                2 * il,
                stream,
            )?;
            mx.decode(gpu, xd, ROWS, act, out, stream)?;
            gpu.synchronize(stream)?;
            let gu_h = u16s(&d2h(gpu, gu, ROWS * 2 * il * 2)?);
            let act_h = u16s(&d2h(gpu, act, ROWS * il * 2)?);
            let dec = u16s(&d2h(gpu, out, ROWS * H * 2)?);
            decoded.push(dec.clone());
            let stacked: Vec<f64> = [&dev.host_w[0][..], &dev.host_w[1][..]].concat();
            let gu_miss = misses(&gu_h, &gemm(&xf[..ROWS * H], &stacked, ROWS, 2 * il, H));
            let (mut act_exact, mut act_off) = (0usize, 0usize);
            for r in 0..ROWS {
                for j in 0..il {
                    let want = swiglu(gu_h[r * 2 * il + j], gu_h[r * 2 * il + il + j]);
                    let got = act_h[r * il + j];
                    act_exact += usize::from(got == want);
                    act_off +=
                        usize::from((f(got) - f(want)).abs() > (f(want).abs() * 2f32.powi(-7)));
                }
            }
            let act_f: Vec<f64> = act_h.iter().map(|&v| f64::from(f(v))).collect();
            let dn_miss = misses(&dec, &gemm(&act_f, &dev.host_w[2], ROWS, H, il));
            eprintln!(
                "layer {layer} slice {rank}/{size} decode: gate/up {gu_miss}/{} outside f64 +- FP32 sum error; SiLU*up \
                 {act_exact}/{} = mx_swiglu of the plain sums ({act_off} more than 1 ulp off); \
                 down {dn_miss}/{} outside",
                ROWS * 2 * il,
                ROWS * il,
                ROWS * H
            );
            assert_eq!(gu_miss, 0, "layer {layer}: gate/up");
            assert!(
                act_off == 0 && act_exact * 1000 >= ROWS * il * 999,
                "layer {layer}: SiLU*up"
            );
            assert_eq!(dn_miss, 0, "layer {layer}: down");

            // 2. Row invariance: every pass width against the row alone.
            let pass = |w: usize| -> Result<Vec<u16>> {
                gpu.memset_async(out, 0xa5, ROWS * H * 2, stream)?;
                for r0 in (0..ROWS).step_by(w) {
                    let n = w.min(ROWS - r0);
                    mx.decode(
                        gpu,
                        xd.offset(r0 * H * 2),
                        n,
                        act,
                        out.offset(r0 * H * 2),
                        stream,
                    )?;
                }
                gpu.synchronize(stream)?;
                Ok(u16s(&d2h(gpu, out, ROWS * H * 2)?))
            };
            let alone = pass(1)?;
            assert_eq!(alone, dec, "layer {layer}: the 64-row pass vs rows alone");
            for w in WIDTHS {
                assert!(
                    pass(w)? == alone,
                    "layer {layer}: {w}-row passes vs rows alone"
                );
            }

            // 3. Prefill (W8A8).
            let scratch = |m: usize| -> Result<MxAct> {
                Ok(MxAct {
                    q: gpu.alloc(m * H)?,
                    sf: gpu.alloc(mx_lt_scale_bytes(m, H))?,
                })
            };
            let (s1, s2) = (scratch(T)?, scratch(T)?);
            mx.prefill(gpu, xd, T, s1, [act, gu], out, stream)?;
            gpu.synchronize(stream)?;
            let pre = u16s(&d2h(gpu, out, T * H * 2)?);
            let gu_p = u16s(&d2h(gpu, gu, ROWS * 2 * il * 2)?);
            // Row r's tiled scales sit where they would in a ROWS-row operand.
            let aq = d2h(gpu, s1.q, ROWS * il)?;
            let asf = d2h(gpu, s1.sf, mx_lt_scale_bytes(ROWS, il))?;
            // The input's image, as prefill quantized it (the same kernel).
            ops::mx_quant_rows(gpu, quant, xd, s2, 0..ROWS, H, stream)?;
            gpu.synchronize(stream)?;
            let xq = dequant(
                &d2h(gpu, s2.q, ROWS * H)?,
                &d2h(gpu, s2.sf, mx_lt_scale_bytes(ROWS, H))?,
                ROWS,
                H,
                true,
            );
            let pgu_miss = misses(&gu_p, &gemm(&xq, &stacked, ROWS, 2 * il, H));
            let a = dequant(&aq, &asf, ROWS, il, true);
            let pdn_miss = misses(&pre[..ROWS * H], &gemm(&a, &dev.host_w[2], ROWS, H, il));
            // SiLU quant of the plain tiers' gate/up == quant of the `_swiglu`
            // tiers' activation (the same sums, `mx_swiglu` on both sides).
            let image = |s: MxAct| -> Result<(Vec<u8>, Vec<u8>)> {
                gpu.synchronize(stream)?;
                Ok((
                    d2h(gpu, s.q, ROWS * il)?,
                    d2h(gpu, s.sf, mx_lt_scale_bytes(ROWS, il))?,
                ))
            };
            let (m, iu) = (ROWS as u32, il as u32);
            ops::mxfp8_gemv_rows(
                gpu,
                &plain,
                &dev.gate_up,
                xd,
                gu,
                ROWS,
                2 * il,
                H,
                2 * il,
                stream,
            )?;
            ops::mxfp8_swiglu_quant_act(gpu, swiglu_quant, gu, s2.q, s2.sf, m, iu, stream)?;
            let from_plain = image(s2)?;
            mx.decode(gpu, xd, ROWS, act, out, stream)?;
            ops::mxfp8_quant_act(gpu, quant, act, iu, s2.q, s2.sf, m, iu, stream)?;
            let from_swiglu = image(s2)?;
            assert!(
                from_plain == from_swiglu,
                "layer {layer}: SiLU quant twins differ"
            );
            // The SP halves.
            gpu.memset_async(out, 0xa5, T * H * 2, stream)?;
            for (r0, n) in [(0, CUT), (CUT, T - CUT)] {
                let o = r0 * H * 2;
                mx.prefill(gpu, xd.offset(o), n, s1, [act, gu], out.offset(o), stream)?;
            }
            gpu.synchronize(stream)?;
            let halves = u16s(&d2h(gpu, out, T * H * 2)?);
            let (mut d2, mut n2) = (0f64, 0f64);
            for (p, q) in pre[..ROWS * H].iter().zip(&dec) {
                d2 += (f64::from(f(*p)) - f64::from(f(*q))).powi(2);
                n2 += f64::from(f(*q)).powi(2);
            }
            eprintln!(
                "layer {layer} slice {rank}/{size} prefill: gate/up {pgu_miss}/{} and down {pdn_miss}/{} outside f64 +- \
                 FP32 sum error of their operands; SiLU quant twins equal; {T} rows == halves at \
                 {CUT}: {}; rel-L2 W8A8 vs decode {:.2e}",
                ROWS * 2 * il,
                ROWS * H,
                halves == pre,
                (d2 / n2).sqrt()
            );
            assert_eq!(pgu_miss, 0, "layer {layer}: prefill gate/up");
            assert_eq!(pdn_miss, 0, "layer {layer}: prefill down");
            assert!(
                halves == pre,
                "layer {layer}: the SP halves differ from the whole chunk"
            );
            for p in [
                xd,
                gu,
                act,
                out,
                s1.q,
                s1.sf,
                s2.q,
                s2.sf,
                dev.gate_up.data,
                dev.gate_up.scales,
            ] {
                gpu.free(p)?;
            }
            for r in dev.raw {
                gpu.free(r.data)?;
                gpu.free(r.scales)?;
            }
        }
        // The slices' partials sum to the whole (each rounded to BF16 once).
        let (mut d2, mut n2) = (0f64, 0f64);
        for j in 0..ROWS * H {
            let sum = f64::from(f(bf(f(decoded[1][j]) + f(decoded[2][j]))));
            d2 += (sum - f64::from(f(decoded[0][j]))).powi(2);
            n2 += f64::from(f(decoded[0][j])).powi(2);
        }
        eprintln!(
            "layer {layer}: TP2 partials' sum vs the whole, rel-L2 {:.2e}",
            (d2 / n2).sqrt()
        );
        assert!(
            (d2 / n2).sqrt() < 4e-3,
            "layer {layer}: the TP2 slices do not sum to the whole"
        );
    }
    Ok(())
}
