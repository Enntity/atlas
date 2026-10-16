// SPDX-License-Identifier: AGPL-3.0-only

//! MXFP8 probe + exact BF16 expansion.
//!
//! The golden is two real FP4-Local projections dequantized by PyTorch
//! (`scripts/mxfp8_bf16_golden.py`, which also asserts the BF16 result equals
//! the f32 product). The GPU kernel path is pinned to the host reference on
//! hardware; on CPU the host reference is pinned to PyTorch here.

use super::*;
use sha2::{Digest, Sha256};
use spark_runtime::weights::WeightTensor;
use std::collections::HashMap;

fn fixtures() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/mxfp8")
}

#[test]
fn host_dequant_matches_pytorch_on_real_projections() {
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("golden.json")).unwrap())
            .unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for c in cases {
        let tag = c["tag"].as_str().unwrap();
        let (n, k) = (
            c["shape"][0].as_u64().unwrap() as usize,
            c["shape"][1].as_u64().unwrap() as usize,
        );
        let data = std::fs::read(fixtures().join(format!("{tag}.weight.bin"))).unwrap();
        let scales = std::fs::read(fixtures().join(format!("{tag}.scale.bin"))).unwrap();
        let bf16 = mxfp8_to_bf16_host(&data, &scales, n, k).unwrap();
        let head: Vec<u16> = c["head"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u16)
            .collect();
        assert_eq!(&bf16[..8], &head[..], "{tag}");
        let bytes: Vec<u8> = bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sha: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(sha, c["bf16_sha256"].as_str().unwrap(), "{tag}");
    }
}

/// The exactness claim at the edges of the accepted range: every finite E4M3
/// code times 2^(e-127) has an f32 product whose low 16 bits are zero, i.e.
/// BF16 holds it without rounding.
#[test]
fn every_e4m3_code_is_exact_across_the_accepted_exponents() {
    for e in [*EXACT_E8M0.start(), 100, 127, 150, *EXACT_E8M0.end()] {
        let m = mxfp8_scales_f32(&[e]).unwrap()[0];
        for code in 0u8..=255 {
            if code & 0x7F == 0x7F {
                continue; // NaN codes
            }
            let v = atlas_core::numeric::fp8_e4m3_to_f32(code) * m;
            assert!(v.is_finite(), "e={e} code={code:#04x}");
            assert_eq!(v.to_bits() & 0xFFFF, 0, "e={e} code={code:#04x} rounds");
            // f32 multiply exact too: dividing back recovers the code value.
            assert_eq!(v / m, atlas_core::numeric::fp8_e4m3_to_f32(code));
        }
    }
}

#[test]
fn scales_outside_the_exact_range_are_refused() {
    for bad in [0u8, 2, 247, 255] {
        assert!(mxfp8_scales_f32(&[120, bad]).is_err(), "e={bad}");
    }
    assert_eq!(mxfp8_scales_f32(&[127]).unwrap(), vec![1.0]);
    assert_eq!(mxfp8_scales_f32(&[118]).unwrap(), vec![2f32.powi(-9)]);
}

fn store(tensors: &[(&str, WeightDtype, &[usize])]) -> WeightStore {
    let map: HashMap<String, WeightTensor> = tensors
        .iter()
        .map(|(n, d, s)| {
            (
                n.to_string(),
                WeightTensor {
                    ptr: DevicePtr::NULL,
                    shape: s.to_vec(),
                    dtype: *d,
                },
            )
        })
        .collect();
    WeightStore::from_map(map)
}

#[test]
fn probe_recognises_only_the_mxfp8_signature() {
    use WeightDtype::*;
    let p = "m.layers.0.linear_attn.out_proj";
    let w = "m.layers.0.linear_attn.out_proj.weight";
    let s = "m.layers.0.linear_attn.out_proj.weight_scale";

    let got = mxfp8_source(
        &store(&[(w, FP8E4M3, &[2560, 6144]), (s, UInt8, &[2560, 192])]),
        p,
    )
    .unwrap()
    .expect("MXFP8");
    assert_eq!((got.n, got.k, got.weight.tiled), (2560, 6144, false));
    // F8_E8M0-typed scales are the same format.
    assert!(
        mxfp8_source(
            &store(&[(w, FP8E4M3, &[64, 64]), (s, FP8E8M0, &[64, 2])]),
            p
        )
        .unwrap()
        .is_some()
    );

    // Not MXFP8: BF16 weight; BF16 per-row scale (unsloth FP8); no scale;
    // NVFP4 (U8 weight, E4M3 scale).
    for t in [
        store(&[(w, BF16, &[64, 64])]),
        store(&[(w, FP8E4M3, &[64, 64]), (s, BF16, &[64, 1])]),
        store(&[(w, FP8E4M3, &[64, 64])]),
        store(&[(w, UInt8, &[64, 32]), (s, FP8E4M3, &[64, 4])]),
    ] {
        assert!(mxfp8_source(&t, p).unwrap().is_none());
    }

    // MXFP8 signature, wrong geometry: refused, not routed elsewhere.
    assert!(mxfp8_source(&store(&[(w, FP8E4M3, &[64, 64]), (s, UInt8, &[64, 4])]), p).is_err());
    assert!(mxfp8_source(&store(&[(w, FP8E4M3, &[64, 48]), (s, UInt8, &[64, 1])]), p).is_err());
}
