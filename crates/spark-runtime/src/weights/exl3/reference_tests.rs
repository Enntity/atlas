// SPDX-License-Identifier: AGPL-3.0-only

use super::{reference_linear, reference_reconstruct, reference_rotated_weights};
use crate::weights::exl3::{Exl3LinearView, decode_mul1};
use half::f16;
use safetensors::{Dtype, tensor::TensorView};

struct Fixture {
    bits: usize,
    packed: Vec<u8>,
    input: Vec<u8>,
    output: Vec<u8>,
    marker: [u8; 4],
}

impl Fixture {
    fn new(bits: usize) -> Self {
        let packed = (0..8 * 16 * 8 * bits)
            .flat_map(|i| ((i as u32).wrapping_mul(0x9e37_79b9) ^ 0xabcd_5678).to_le_bytes())
            .collect();
        let scales = |width: usize| {
            (0..width)
                .flat_map(|i| f16::from_f32((i as f32 % 7.0 - 3.0) * 0.25).to_le_bytes())
                .collect()
        };
        Self {
            bits,
            packed,
            input: scales(128),
            output: scales(256),
            marker: 0x83dc_d12du32.to_le_bytes(),
        }
    }

    fn view(&self) -> Exl3LinearView<'_> {
        Exl3LinearView::new(
            TensorView::new(Dtype::I16, vec![8, 16, 16 * self.bits], &self.packed).unwrap(),
            TensorView::new(Dtype::F16, vec![128], &self.input).unwrap(),
            TensorView::new(Dtype::F16, vec![256], &self.output).unwrap(),
            TensorView::new(Dtype::I32, vec![], &self.marker).unwrap(),
            128,
            256,
        )
        .unwrap()
    }

    // Read circular bits individually and enumerate matrix coordinates rather
    // than production stream offsets. This catches permutation direction errors.
    fn weight(&self, row: usize, col: usize) -> f32 {
        let tile = row / 16 * 16 + col / 16;
        let r = row % 16;
        let c = col % 16;
        let lane = (c % 8) * 4 + (r % 8) / 2;
        let slot = (c / 8) * 4 + (r / 8) * 2 + r % 2;
        let end = (lane * 8 + slot + 257) * self.bits;
        let mut state = 0u16;
        for position in end - 16..end {
            let position = position % (256 * self.bits);
            let word_offset = tile * 32 * self.bits + position / 32 * 4;
            let word = u32::from_le_bytes(
                self.packed[word_offset..word_offset + 4]
                    .try_into()
                    .unwrap(),
            );
            state = (state << 1) | ((word >> (31 - position % 32)) & 1) as u16;
        }
        decode_mul1(state).to_f32()
    }
}

fn scale(bytes: &[u8], index: usize) -> f64 {
    f64::from(f16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]).to_f32())
}

// Explicit Walsh matrix multiplication in f64, independent of the fast
// butterfly implementation and its intermediate f32 rounding.
fn walsh(input: &[f64]) -> Vec<f64> {
    (0..input.len())
        .map(|i| {
            input
                .iter()
                .enumerate()
                .map(|(j, value)| {
                    let sign = if ((i % 128) & (j % 128)).count_ones().is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    if i / 128 == j / 128 {
                        value * sign / 128f64.sqrt()
                    } else {
                        0.0
                    }
                })
                .sum()
        })
        .collect()
}

#[test]
fn rotated_weights_match_independent_coordinate_decoder_for_all_bitrates() {
    for bits in 1..=8 {
        let fixture = Fixture::new(bits);
        let weights = reference_rotated_weights(&fixture.view()).unwrap();
        for row in 0..128 {
            for col in 0..256 {
                assert_eq!(
                    weights[row * 256 + col],
                    fixture.weight(row, col),
                    "K={bits}, row={row}, col={col}"
                );
            }
        }
    }
}

#[test]
fn batched_linear_matches_dense_walsh_oracle_with_signed_scales() {
    for bits in [1, 4, 5, 6, 8] {
        let fixture = Fixture::new(bits);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 % 13.0 - 6.0) / 8.0).collect();
        let actual = reference_linear(&fixture.view(), &input, 2).unwrap();
        for batch in 0..2 {
            let scaled: Vec<f64> = (0..128)
                .map(|i| f64::from(input[batch * 128 + i]) * scale(&fixture.input, i))
                .collect();
            let transformed = walsh(&scaled);
            let product: Vec<f64> = (0..256)
                .map(|j| {
                    (0..128)
                        .map(|i| transformed[i] * f64::from(fixture.weight(i, j)))
                        .sum()
                })
                .collect();
            let expected = walsh(&product);
            for j in 0..256 {
                let expected = expected[j] * scale(&fixture.output, j);
                assert!(
                    (f64::from(actual[batch * 256 + j]) - expected).abs() < 2e-5,
                    "K={bits}, batch={batch}, col={j}"
                );
            }
        }
    }
}

#[test]
fn reconstructed_weights_match_batched_linear() {
    let fixture = Fixture::new(6);
    let view = fixture.view();
    let weights = reference_reconstruct(&view).unwrap();
    let input: Vec<f32> = (0..256).map(|i| (i % 17) as f32 / 16.0 - 0.5).collect();
    let actual = reference_linear(&view, &input, 2).unwrap();
    for batch in 0..2 {
        for col in 0..256 {
            let expected: f32 = (0..128)
                .map(|row| input[batch * 128 + row] * weights[row * 256 + col])
                .sum();
            assert!((actual[batch * 256 + col] - expected).abs() < 2e-5);
        }
    }
}

#[test]
fn linear_rejects_malformed_batches_and_handles_empty_batch() {
    let fixture = Fixture::new(4);
    let view = fixture.view();
    assert!(reference_linear(&view, &[0.0; 127], 1).is_err());
    assert!(reference_linear(&view, &[0.0; 128], 0).is_err());
    assert!(reference_linear(&view, &[], usize::MAX).is_err());
    assert!(reference_linear(&view, &[], 0).unwrap().is_empty());
}
