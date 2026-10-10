// SPDX-License-Identifier: AGPL-3.0-only

/// Convert a little-endian IEEE-754 half-precision (F16) tensor byte buffer
/// to BF16 bytes. F16 and BF16 are both 2 bytes/element but have different
/// bit layouts (5-bit vs 8-bit exponent), so the bytes cannot be
/// reinterpreted — each value goes f16 → f32 (exact) → bf16
/// (round-to-nearest-even). Shared by both disk loaders so F16 checkpoints
/// (e.g. centml modelopt W4A4 exports, which ship all unquantized tensors as
/// F16) land in the store as BF16; [`super::WeightDtype`] itself stays closed to
/// store-legal dtypes and F16 can never appear on the RDMA wire.
pub(crate) fn f16_to_bf16_bytes(src: &[u8]) -> Vec<u8> {
    use half::{bf16, f16};
    debug_assert_eq!(src.len() % 2, 0, "F16 tensor byte length must be even");
    let mut out = Vec::with_capacity(src.len());
    for pair in src.chunks_exact(2) {
        let h = f16::from_le_bytes([pair[0], pair[1]]);
        out.extend_from_slice(&bf16::from_f32(h.to_f32()).to_le_bytes());
    }
    out
}

/// Convert checkpoint F16 values exactly to little-endian F32 for a router
/// whose accumulation and parameter policy explicitly requires FP32.
pub(crate) fn f16_to_f32_bytes(src: &[u8]) -> Vec<u8> {
    debug_assert_eq!(src.len() % 2, 0, "F16 tensor byte length must be even");
    src.chunks_exact(2)
        .flat_map(|word| {
            half::f16::from_le_bytes([word[0], word[1]])
                .to_f32()
                .to_le_bytes()
        })
        .collect()
}
