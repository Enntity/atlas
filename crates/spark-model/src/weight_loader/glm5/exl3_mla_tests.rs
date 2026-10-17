// SPDX-License-Identifier: AGPL-3.0-only

use super::kv_view_bytes;
use half::{bf16, f16};

#[test]
fn ordinary_kv_b_preserves_head_order_and_builds_absorption_views() {
    // Two heads, two key channels, three value channels, two latent channels.
    let values: Vec<f32> = (0..20).map(|i| i as f32 - 7.0).collect();
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| f16::from_f32(*v).to_le_bytes())
        .collect();
    let (kv, uk, uv) = kv_view_bytes(&bytes, 2, 2, 3, 2).unwrap();
    let decode = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|p| bf16::from_le_bytes([p[0], p[1]]).to_f32())
            .collect::<Vec<_>>()
    };
    assert_eq!(decode(&kv), values);
    assert_eq!(decode(&uk), [-7., -5., -6., -4., 3., 5., 4., 6.]);
    assert_eq!(
        decode(&uv),
        [-3., -2., -1., 0., 1., 2., 7., 8., 9., 10., 11., 12.]
    );
    assert!(kv_view_bytes(&bytes[..bytes.len() - 2], 2, 2, 3, 2).is_err());
}
