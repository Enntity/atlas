// SPDX-License-Identifier: AGPL-3.0-only

//! Borrowed EXL3 checkpoint data for native kernels. No I/O or GPU allocation.
//!
//! Initial contract: integer bitrates 1–8, `mul1`, and FP16 sign/scale
//! vectors. This is format bring-up, not an enabled inference backend.

mod checkpoint;
pub mod dense;
pub mod ffn_io;
pub mod logits_io;
pub mod native;
mod reference;
mod store;

pub use checkpoint::Exl3CheckpointRouter;
pub use reference::{reference_linear, reference_reconstruct, reference_rotated_weights};

use anyhow::{Result, ensure};
use half::f16;
use safetensors::{Dtype, tensor::TensorView};

const MUL1_MULTIPLIER: u32 = 0x83dc_d12d;
const TILE_WIDTH: usize = 16;
const HADAMARD_WIDTH: usize = 128;

pub mod tp;

/// Validated views of one checkpoint linear, retaining the original bytes.
///
/// Views may come from different shards. The IORouter owns their backing
/// buffers; validation never uploads, copies, or converts the scale vectors.
#[derive(Debug)]
pub struct Exl3LinearView<'a> {
    trellis: TensorView<'a>,
    input_scale: TensorView<'a>,
    output_scale: TensorView<'a>,
    bits: u8,
    in_features: usize,
    out_features: usize,
}

impl<'a> Exl3LinearView<'a> {
    /// Validate the modern mul1 tensor group against explicit model widths.
    ///
    /// The caller must dispatch the checkpoint's declared format and exclude
    /// unsupported companion tensors (legacy signs, bias, alternate codebook).
    /// Neither average bpw nor original pre-conversion quantization metadata
    /// determines this tensor's bitrate.
    pub fn new(
        trellis: TensorView<'a>,
        input_scale: TensorView<'a>,
        output_scale: TensorView<'a>,
        mul1: TensorView<'a>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        ensure!(
            in_features > 0
                && out_features > 0
                && in_features.is_multiple_of(HADAMARD_WIDTH)
                && out_features.is_multiple_of(HADAMARD_WIDTH),
            "EXL3 widths must be non-zero multiples of {HADAMARD_WIDTH}"
        );
        ensure!(trellis.dtype() == Dtype::I16, "EXL3 trellis must be I16");
        let shape = trellis.shape();
        ensure!(
            shape.len() == 3
                && shape[0] == in_features / TILE_WIDTH
                && shape[1] == out_features / TILE_WIDTH,
            "EXL3 trellis shape {:?} does not match input={in_features}, output={out_features}",
            shape
        );
        ensure!(
            shape[2].is_multiple_of(TILE_WIDTH)
                && (TILE_WIDTH..=8 * TILE_WIDTH).contains(&shape[2]),
            "EXL3 requires integer bitrate 1..=8; trellis depth is {}",
            shape[2]
        );
        let bits = (shape[2] / TILE_WIDTH) as u8;
        ensure!(
            mul1.dtype() == Dtype::I32 && mul1.shape().is_empty(),
            "EXL3 mul1 marker must be an I32 scalar"
        );
        ensure!(
            mul1.data() == MUL1_MULTIPLIER.to_le_bytes(),
            "EXL3 mul1 marker must contain multiplier 0x{MUL1_MULTIPLIER:08X}"
        );
        validate_scale(&input_scale, in_features, "input")?;
        validate_scale(&output_scale, out_features, "output")?;
        Ok(Self {
            trellis,
            input_scale,
            output_scale,
            bits,
            in_features,
            out_features,
        })
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn trellis(&self) -> &TensorView<'a> {
        &self.trellis
    }

    pub fn input_scale(&self) -> &TensorView<'a> {
        &self.input_scale
    }

    pub fn output_scale(&self) -> &TensorView<'a> {
        &self.output_scale
    }

    /// Extract a tile's states in packed stream order, before the tensor-core
    /// layout permutation and the two normalized Hadamard transforms.
    pub fn tile_states(&self, input_tile: usize, output_tile: usize) -> Result<[u16; 256]> {
        let shape = self.trellis.shape();
        ensure!(
            input_tile < shape[0] && output_tile < shape[1],
            "EXL3 tile coordinates are out of bounds"
        );
        let bytes_per_tile = 32 * self.bits as usize;
        let start = (input_tile * shape[1] + output_tile) * bytes_per_tile;
        decode_trellis_tile(
            &self.trellis.data()[start..start + bytes_per_tile],
            self.bits,
        )
    }
}

fn validate_scale(scale: &TensorView<'_>, width: usize, label: &str) -> Result<()> {
    ensure!(
        scale.dtype() == Dtype::F16 && scale.shape() == [width],
        "EXL3 {label} scale must be F16 [{width}]"
    );
    ensure!(
        scale
            .data()
            .chunks_exact(2)
            .all(|pair| { f16::from_le_bytes([pair[0], pair[1]]).is_finite() }),
        "EXL3 {label} scales must be finite"
    );
    Ok(())
}

/// Reference mul1 codebook entry, with unsigned arithmetic and one FP16
/// rounding after the fused multiply-add. The FP16 constants are part of
/// the checkpoint format; replacing them with decimal approximations changes
/// decoded weights. This does not include scales or Hadamard transforms.
pub fn decode_mul1(state: u16) -> f16 {
    let product = u32::from(state).wrapping_mul(MUL1_MULTIPLIER);
    let sum: u16 = product.to_le_bytes().iter().map(|b| u16::from(*b)).sum();
    let value = f16::from_bits(0x6400 + sum).to_f32();
    let inverse = f16::from_bits(0x1eee).to_f32();
    let bias = f16::from_bits(0xc931).to_f32();
    f16::from_f32(value.mul_add(inverse, bias))
}

/// Decode one packed 16×16 tile into circular 16-bit trellis states.
///
/// The on-disk I16 payload is interpreted as pairs forming little-endian u32
/// words, with windows proceeding from each word's most significant bit.
/// States retain packed stream order, not row-major matrix order.
pub fn decode_trellis_tile(bytes: &[u8], bits: u8) -> Result<[u16; 256]> {
    ensure!((1..=8).contains(&bits), "EXL3 tile bitrate must be 1..=8");
    let bits = usize::from(bits);
    ensure!(bytes.len() == 32 * bits, "EXL3 tile byte length mismatch");
    let word_count = 8 * bits;
    let word = |index: usize| {
        let start = (index % word_count) * 4;
        u32::from_le_bytes(bytes[start..start + 4].try_into().expect("four-byte word"))
    };
    Ok(std::array::from_fn(|offset| {
        let end = (offset + 257) * bits;
        let first_word = (end - 16) / 32;
        let last_word = (end - 1) / 32;
        let shift = (last_word + 1) * 32 - end;
        let pair = (u64::from(word(first_word)) << 32) | u64::from(word(last_word));
        (pair >> shift) as u16
    }))
}

#[cfg(test)]
mod tests;
