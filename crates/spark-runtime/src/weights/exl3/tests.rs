// SPDX-License-Identifier: AGPL-3.0-only

use super::{Exl3LinearView, decode_mul1, decode_trellis_tile};
use safetensors::{Dtype, tensor::TensorView};

struct Fixture {
    trellis: Vec<u8>,
    input: Vec<u8>,
    output: Vec<u8>,
    marker: [u8; 4],
}

impl Fixture {
    fn new(bits: usize) -> Self {
        Self {
            trellis: vec![0x81; 8 * 16 * 16 * bits * 2],
            input: [0x00, 0xbc].repeat(128), // -1.0 F16, preserve its sign
            output: [0x00, 0x3c].repeat(256),
            marker: 0x83dc_d12du32.to_le_bytes(),
        }
    }

    fn view(&self, bits: usize) -> anyhow::Result<Exl3LinearView<'_>> {
        Exl3LinearView::new(
            TensorView::new(Dtype::I16, vec![8, 16, 16 * bits], &self.trellis)?,
            TensorView::new(Dtype::F16, vec![128], &self.input)?,
            TensorView::new(Dtype::F16, vec![256], &self.output)?,
            TensorView::new(Dtype::I32, vec![], &self.marker)?,
            128,
            256,
        )
    }
}

#[test]
fn integer_bitrates_are_per_tensor_and_bytes_are_borrowed() {
    for bits in 1..=8 {
        let fixture = Fixture::new(bits);
        let view = fixture.view(bits).unwrap();
        assert_eq!(view.bits(), bits as u8);
        assert_eq!(view.in_features(), 128);
        assert_eq!(view.out_features(), 256);
        assert_eq!(view.trellis().data().as_ptr(), fixture.trellis.as_ptr());
        assert_eq!(view.input_scale().data(), fixture.input);
        assert_eq!(view.output_scale().data(), fixture.output);
        let last = view.tile_states(7, 15).unwrap();
        assert_eq!(
            last,
            decode_trellis_tile(&fixture.trellis[..32 * bits], bits as u8).unwrap()
        );
        assert!(view.tile_states(8, 0).is_err());
        assert!(view.tile_states(0, 16).is_err());
        assert!(view.tile_states(usize::MAX, usize::MAX).is_err());
    }
}

#[test]
fn rejects_invalid_marker_and_nonfinite_scales() {
    let mut fixture = Fixture::new(4);
    fixture.marker = 1u32.to_le_bytes();
    assert!(fixture.view(4).unwrap_err().to_string().contains("mul1"));
    fixture.marker = 0x83dc_d12du32.to_le_bytes();
    fixture.input[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert!(fixture.view(4).unwrap_err().to_string().contains("finite"));
    fixture.input[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    fixture.output[..2].copy_from_slice(&0x7e00u16.to_le_bytes());
    assert!(fixture.view(4).unwrap_err().to_string().contains("finite"));
}

#[test]
fn rejects_wrong_layout_dtype_and_fractional_bitrate() {
    let fixture = Fixture::new(4);
    let input = || TensorView::new(Dtype::F16, vec![128], &fixture.input).unwrap();
    let output = || TensorView::new(Dtype::F16, vec![256], &fixture.output).unwrap();
    let marker = || TensorView::new(Dtype::I32, vec![], &fixture.marker).unwrap();
    for (dtype, shape, data) in [
        (Dtype::I16, vec![16, 8, 64], fixture.trellis.as_slice()),
        (Dtype::BF16, vec![8, 16, 64], fixture.trellis.as_slice()),
        (
            Dtype::I16,
            vec![8, 16, 56],
            &fixture.trellis[..8 * 16 * 56 * 2],
        ),
        (Dtype::I16, vec![8, 16, 0], &[]),
    ] {
        assert!(
            Exl3LinearView::new(
                TensorView::new(dtype, shape, data).unwrap(),
                input(),
                output(),
                marker(),
                128,
                256,
            )
            .is_err()
        );
    }
    let trellis = || TensorView::new(Dtype::I16, vec![8, 16, 64], &fixture.trellis).unwrap();
    for (k, n) in [(0, 256), (128, 0), (127, 256), (128, 255), (256, 128)] {
        assert!(Exl3LinearView::new(trellis(), input(), output(), marker(), k, n).is_err());
    }
    assert!(
        Exl3LinearView::new(
            trellis(),
            TensorView::new(Dtype::BF16, vec![128], &fixture.input).unwrap(),
            output(),
            marker(),
            128,
            256
        )
        .is_err()
    );
    assert!(
        Exl3LinearView::new(
            trellis(),
            input(),
            output(),
            TensorView::new(Dtype::I32, vec![1], &fixture.marker).unwrap(),
            128,
            256
        )
        .is_err()
    );
}

#[test]
fn mul1_matches_independent_half_precision_vectors() {
    // Generated using Python struct '<e' from unsigned byte sums and the
    // checkpoint codebook constants; final rounding happens only once.
    for (state, expected) in [
        (0, 0xc2e8),
        (1, 0x3921),
        (2, 0xb72d),
        (42, 0xba06),
        (255, 0x3ab3),
        (256, 0xb3db),
        (32767, 0x3fca),
        (32768, 0xbaf2),
        (65535, 0xb936),
    ] {
        assert_eq!(decode_mul1(state).to_bits(), expected, "state {state}");
    }
}

#[test]
fn real_checkpoint_tile_matches_state_and_codebook_vectors() {
    // HF revision 0f46c18259efb8886839b446e25534f9f2e1fb1e, shard 1,
    // layer 0 self_attn.o_proj.trellis, first 192 bytes (K=6).
    let words: [u32; 48] = [
        2157240677, 1449657629, 1979631029, 2431511892, 3501543154, 222258518, 2524586639,
        465157270, 2283102628, 3562875881, 3652764264, 1405298092, 3449716482, 3872972043,
        2348756128, 3659113684, 1123093639, 2798506247, 94359482, 1279809146, 3664506601,
        4269470966, 1324619357, 779522715, 231412142, 2279800780, 876684657, 3137688962,
        2055525822, 1981758634, 4042085484, 2677767227, 1619099075, 1577041579, 390886479,
        1012479549, 2675259065, 17124032, 159727773, 1697629859, 3042652500, 3274998413, 24831143,
        2831963435, 1505588911, 1598163296, 2219464411, 10781224,
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let states = decode_trellis_tile(&bytes, 6).unwrap();
    assert_eq!(
        &states[..8],
        &[35360, 34825, 595, 38113, 14425, 5717, 38233, 22120]
    );
    let values: Vec<u16> = states[..8]
        .iter()
        .map(|s| decode_mul1(*s).to_bits())
        .collect();
    assert_eq!(
        values,
        [
            0xb10a, 0x3cad, 0xbb61, 0xa96e, 0xb28e, 0x385f, 0x3a0d, 0x31ca
        ]
    );
}

#[test]
fn tile_decoder_checks_bounds_and_circular_windows() {
    for bits in 1..=8 {
        assert_eq!(
            decode_trellis_tile(&vec![0; 32 * bits as usize], bits).unwrap(),
            [0; 256]
        );
        assert_eq!(
            decode_trellis_tile(&vec![255; 32 * bits as usize], bits).unwrap(),
            [65535; 256]
        );
        assert!(decode_trellis_tile(&vec![0; 32 * bits as usize - 1], bits).is_err());
    }
    for bits in [0, 9, 255] {
        assert!(decode_trellis_tile(&[], bits).is_err());
    }
}

#[test]
fn all_tile_states_match_an_independent_bit_reader() {
    for bits in 1..=8u8 {
        let words: Vec<u32> = (0..8 * usize::from(bits))
            .map(|i| (i as u32).wrapping_mul(0x9e37_79b9) ^ 0xa3c5_6789)
            .collect();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let states = decode_trellis_tile(&bytes, bits).unwrap();
        // Construct the circular bitstream explicitly, independent of the
        // production pair-of-words extraction and shift calculations.
        let stream: Vec<u16> = words
            .iter()
            .flat_map(|w| (0..32).rev().map(move |shift| ((w >> shift) & 1) as u16))
            .collect();
        for (offset, state) in states.iter().enumerate() {
            let end = (offset + 257) * usize::from(bits);
            let mut expected = 0u16;
            for position in end - 16..end {
                expected = (expected << 1) | stream[position % stream.len()];
            }
            assert_eq!(*state, expected, "K={bits}, offset={offset}");
        }
    }
}

#[test]
fn linear_tile_lookup_uses_input_major_coordinates() {
    let mut fixture = Fixture::new(4);
    let tile_bytes = 32 * 4;
    let start = (3 * 16 + 11) * tile_bytes;
    fixture.trellis[start..start + tile_bytes].fill(255);
    let view = fixture.view(4).unwrap();
    assert_eq!(view.tile_states(3, 11).unwrap(), [65535; 256]);
    let ordinary = decode_trellis_tile(&fixture.trellis[..tile_bytes], 4).unwrap();
    assert_eq!(view.tile_states(3, 10).unwrap(), ordinary);
    assert_eq!(view.tile_states(4, 11).unwrap(), ordinary);
}
