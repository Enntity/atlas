// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::numeric::{f32_to_bf16, nvfp4_dequant};

/// Deterministic xorshift so the tests need no RNG dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte_in(&mut self, lo: u8, hi: u8) -> u8 {
        lo + (self.next() % u64::from(hi - lo)) as u8
    }
}

/// MXFP8 -> BF16 must be EXACT: compare against an f64 reconstruction that
/// shares no code with the decoder.
#[test]
fn mxfp8_expansion_is_exact() {
    let (out, inp) = (8, 96);
    let mut rng = Rng(42);
    let weight: Vec<u8> = (0..out * inp)
        .map(|_| {
            let b = (rng.next() & 0xFF) as u8;
            if b & 0x7F == 0x7F { 0x38 } else { b } // skip NaN codes
        })
        .collect();
    let scale: Vec<u8> = (0..out * inp / 32).map(|_| rng.byte_in(110, 135)).collect();
    let got = mxfp8_to_bf16(&weight, &scale, out, inp).unwrap();
    for (i, &w) in weight.iter().enumerate() {
        let (sign, exp, man) = (w >> 7, (w >> 3) & 0xF, w & 7);
        let mag = if exp == 0 {
            f64::from(man) / 8.0 * 2f64.powi(-6)
        } else {
            (1.0 + f64::from(man) / 8.0) * 2f64.powi(i32::from(exp) - 7)
        };
        let signed = if sign == 1 { -mag } else { mag };
        let v = signed * 2f64.powi(i32::from(scale[i / 32]) - 127);
        let bits = u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
        let back = f64::from(f32::from_bits(u32::from(bits) << 16));
        assert_eq!(
            back,
            v,
            "element {i}: byte {w:#04x} scale {}",
            scale[i / 32]
        );
    }
}

#[test]
fn mxfp8_rejects_misaligned_widths() {
    assert!(mxfp8_to_bf16(&[0; 48], &[127; 1], 1, 48).is_err());
    assert!(mxfp8_to_bf16(&[0; 64], &[127; 1], 1, 64).is_err());
}

/// NVFP4 rows decode to the same values as the existing whole-matrix
/// dequant, cast through the shared BF16 rounding.
#[test]
fn nvfp4_rows_match_the_reference_dequant() {
    let (rows, dim, group) = (5, 160, 16);
    let mut rng = Rng(7);
    let packed: Vec<u8> = (0..rows * dim / 2)
        .map(|_| (rng.next() & 0xFF) as u8)
        .collect();
    let scale: Vec<u8> = (0..rows * dim / group)
        .map(|_| rng.byte_in(0x20, 0x60))
        .collect();
    let global = 3.1e-4f32;
    let mut got = vec![0u8; rows * dim * 2];
    nvfp4_rows_to_bf16(&packed, &scale, global, dim, group, &mut got).unwrap();
    let want = nvfp4_dequant(&packed, &scale, global, rows, dim, group).unwrap();
    for (i, w) in want.iter().enumerate() {
        let bits = u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
        assert_eq!(bits, f32_to_bf16(*w), "element {i}");
    }
}
