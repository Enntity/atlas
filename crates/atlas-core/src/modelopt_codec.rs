// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side reference decoders for generic MXFP8 and NVFP4 row formats.
//!
//! * **MXFP8** (OCP microscaling): an E4M3 weight `[out, in]` with one E8M0
//!   exponent per 32 inputs, `[out, in / 32]`. Expanded to BF16 EXACTLY: an
//!   E4M3 value has 4 significant bits and the scale is a power of two, so the
//!   product always fits BF16's 8.
//!
//! * **NVFP4 rows → BF16**, for a table gathered by row (the PLE n-gram
//!   embedding), where the block scales live in a separate tensor.
//!
//! Pure arithmetic, no CUDA, no environment lookups in the hot loops.

use crate::numeric::{FP4_E2M1, FP8_E4M3_LUT};
use anyhow::{Result, ensure};

/// Inputs per MXFP8 scale.
pub const MXFP8_BLOCK: usize = 32;

/// The top 16 bits of an f32 whose low 16 bits are already zero (or whose
/// rounding the caller has accepted). Round-to-nearest-even; NaN stays NaN.
#[inline(always)]
fn bf16_rne(val: f32) -> u16 {
    let bits = val.to_bits();
    if val.is_nan() {
        return ((bits >> 16) as u16 & 0x8000) | 0x7FC0;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

/// `2^(e - 127)` for an E8M0 exponent byte. `0xFF` is NaN in OCP MX; it maps
/// to zero for the same reason the E4M3 table maps NaN to zero.
#[inline(always)]
fn e8m0_to_f32(e: u8) -> f32 {
    match e {
        0xFF => 0.0,
        // 2^-127 is an f32 subnormal: build it from its bits.
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits((e as u32) << 23),
    }
}

/// Expand an MXFP8 weight `[out, inp]` (E4M3 bytes) with its E8M0 scales
/// `[out, inp / 32]` into little-endian BF16 bytes `[out, inp]`. Exact.
pub fn mxfp8_to_bf16(weight: &[u8], scale: &[u8], out: usize, inp: usize) -> Result<Vec<u8>> {
    ensure!(
        inp.is_multiple_of(MXFP8_BLOCK),
        "MXFP8 input width {inp} is not a multiple of {MXFP8_BLOCK}"
    );
    let blocks = inp / MXFP8_BLOCK;
    ensure!(
        weight.len() == out * inp && scale.len() == out * blocks,
        "MXFP8 weight {} / scale {} bytes do not match [{out}, {inp}]",
        weight.len(),
        scale.len()
    );
    let mut dst = vec![0u8; out * inp * 2];
    for ((w, s), d) in weight
        .chunks_exact(MXFP8_BLOCK)
        .zip(scale)
        .zip(dst.chunks_exact_mut(MXFP8_BLOCK * 2))
    {
        let mult = e8m0_to_f32(*s);
        for (&b, o) in w.iter().zip(d.chunks_exact_mut(2)) {
            o.copy_from_slice(&bf16_rne(FP8_E4M3_LUT[b as usize] * mult).to_le_bytes());
        }
    }
    Ok(dst)
}

/// Dequantize NVFP4 rows to little-endian BF16. `packed` is `[rows, dim / 2]`
/// (low nibble first), `scale` is `[rows, dim / group]` E4M3, `global` is the
/// tensor's `weight_scale_2`. Writes `rows * dim` BF16 values into `dst`.
pub fn nvfp4_rows_to_bf16(
    packed: &[u8],
    scale: &[u8],
    global: f32,
    dim: usize,
    group: usize,
    dst: &mut [u8],
) -> Result<()> {
    ensure!(
        dim.is_multiple_of(2) && group > 0 && dim.is_multiple_of(group),
        "NVFP4 row width {dim} must be even and a multiple of group {group}"
    );
    let rows = packed.len() / (dim / 2);
    ensure!(
        packed.len() == rows * dim / 2
            && scale.len() == rows * dim / group
            && dst.len() == rows * dim * 2,
        "NVFP4 rows: packed {} / scale {} / dst {} bytes disagree on {rows} rows of {dim}",
        packed.len(),
        scale.len(),
        dst.len()
    );
    for (i, (&byte, o)) in packed.iter().zip(dst.chunks_exact_mut(4)).enumerate() {
        let elem = i * 2;
        // `(fp4 * block) * global`, the order `numeric::nvfp4_dequant` uses:
        // the first product is exact, so there is exactly one f32 rounding.
        let block = FP8_E4M3_LUT[scale[elem / group] as usize];
        let lo = bf16_rne(FP4_E2M1[(byte & 0x0F) as usize] * block * global);
        let hi = bf16_rne(FP4_E2M1[(byte >> 4) as usize] * block * global);
        o[..2].copy_from_slice(&lo.to_le_bytes());
        o[2..].copy_from_slice(&hi.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
#[path = "modelopt_codec_tests.rs"]
mod tests;
