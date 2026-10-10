// SPDX-License-Identifier: AGPL-3.0-only

//! CPU correctness oracles for modern mul1 EXL3, independently implemented
//! from the pinned format definition. These allocate dense matrices and are
//! unsuitable for production inference. Arithmetic after codebook decoding
//! is F32, without CUDA's intermediate FP16 stores or final output rounding.
//!
//! Format references (upstream commit 151539c77abc7ab7425d30da7a4e8e3c5c154e7b):
//! - [tile coordinates](https://github.com/turboderp-org/exllamav3/blob/151539c77abc7ab7425d30da7a4e8e3c5c154e7b/exllamav3/modules/quant/exl3_lib/quantize.py)
//! - [scale/rotation order](https://github.com/turboderp-org/exllamav3/blob/151539c77abc7ab7425d30da7a4e8e3c5c154e7b/exllamav3/modules/quant/exl3.py)

use super::{Exl3LinearView, HADAMARD_WIDTH, TILE_WIDTH, decode_mul1};
use anyhow::{Result, ensure};
use half::f16;

/// Decode the row-major `[in_features, out_features]` rotated-basis matrix.
/// No Hadamard transforms or sign/scale vectors have been applied.
pub fn reference_rotated_weights(view: &Exl3LinearView<'_>) -> Result<Vec<f32>> {
    let k = view.in_features();
    let n = view.out_features();
    let len = k
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("EXL3 reference matrix size overflow"))?;
    let mut weights = vec![0.0; len];
    for input_tile in 0..k / TILE_WIDTH {
        for output_tile in 0..n / TILE_WIDTH {
            let states = view.tile_states(input_tile, output_tile)?;
            for (stream, state) in states.into_iter().enumerate() {
                // Stream index bits: [column low 3, row low pair,
                // column high, row high, row parity]. Translate those bits
                // to a matrix coordinate without a stored permutation table.
                let row = ((stream >> 3) & 3) * 2 + (stream & 1) + ((stream >> 1) & 1) * 8;
                let col = (stream >> 5) + ((stream >> 2) & 1) * 8;
                weights[(input_tile * TILE_WIDTH + row) * n + output_tile * TILE_WIDTH + col] =
                    decode_mul1(state).to_f32();
            }
        }
    }
    Ok(weights)
}

/// Reconstruct row-major `[in_features, out_features]` original-basis weights.
/// Computes `diag(suh) H_k W_rotated H_n diag(svh)`, with block diagonal
/// normalized H128 on each side and the original FP16 scales decoded to F32.
pub fn reference_reconstruct(view: &Exl3LinearView<'_>) -> Result<Vec<f32>> {
    let k = view.in_features();
    let n = view.out_features();
    let mut weights = reference_rotated_weights(view)?;
    let mut column = [0.0; HADAMARD_WIDTH];
    for input_start in (0..k).step_by(HADAMARD_WIDTH) {
        for output in 0..n {
            for (offset, value) in column.iter_mut().enumerate() {
                *value = weights[(input_start + offset) * n + output];
            }
            hadamard_blocks(&mut column);
            for (offset, value) in column.iter().enumerate() {
                weights[(input_start + offset) * n + output] = *value;
            }
        }
    }
    for (row, values) in weights.chunks_exact_mut(n).enumerate() {
        hadamard_blocks(values);
        for (col, value) in values.iter_mut().enumerate() {
            *value *=
                scale(view.input_scale().data(), row) * scale(view.output_scale().data(), col);
        }
    }
    Ok(weights)
}

/// Apply an EXL3 linear to row-major `[batch, in_features]` F32 inputs.
/// Returns row-major `[batch, out_features]` without a bias or final FP16
/// conversion. This oracle establishes mathematical TP gather/reduce parity;
/// floating point collectives may differ in their summation order.
pub fn reference_linear(
    view: &Exl3LinearView<'_>,
    input: &[f32],
    batch: usize,
) -> Result<Vec<f32>> {
    let k = view.in_features();
    let n = view.out_features();
    ensure!(
        batch.checked_mul(k) == Some(input.len()),
        "EXL3 reference input shape mismatch or overflow"
    );
    let output_len = batch
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("EXL3 reference output size overflow"))?;
    if batch == 0 {
        return Ok(Vec::new());
    }
    let weights = reference_rotated_weights(view)?;
    let mut output = vec![0.0; output_len];
    let mut transformed = vec![0.0; k];
    for (input_row, output_row) in input.chunks_exact(k).zip(output.chunks_exact_mut(n)) {
        for (index, (value, original)) in transformed.iter_mut().zip(input_row).enumerate() {
            *value = *original * scale(view.input_scale().data(), index);
        }
        hadamard_blocks(&mut transformed);
        for (coefficient, weight_row) in transformed.iter().zip(weights.chunks_exact(n)) {
            for (value, weight) in output_row.iter_mut().zip(weight_row) {
                *value += coefficient * weight;
            }
        }
        hadamard_blocks(output_row);
        for (index, value) in output_row.iter_mut().enumerate() {
            *value *= scale(view.output_scale().data(), index);
        }
    }
    Ok(output)
}

fn scale(bytes: &[u8], index: usize) -> f32 {
    f16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]).to_f32()
}

fn hadamard_blocks(values: &mut [f32]) {
    let normalization = 1.0 / (HADAMARD_WIDTH as f32).sqrt();
    for block in values.chunks_exact_mut(HADAMARD_WIDTH) {
        let mut distance = 1;
        while distance < HADAMARD_WIDTH {
            for pair in block.chunks_exact_mut(distance * 2) {
                let (left, right) = pair.split_at_mut(distance);
                for (a, b) in left.iter_mut().zip(right) {
                    let original = *a;
                    *a = original + *b;
                    *b = original - *b;
                }
            }
            distance *= 2;
        }
        for value in block {
            *value *= normalization;
        }
    }
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
