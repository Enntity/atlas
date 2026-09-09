// SPDX-License-Identifier: AGPL-3.0-only
//! Compare the preflight quote with actual constructor allocation receipts.
use super::*;

#[test]
fn private_reserve_matches_actual_cache_and_slab_allocations() {
    for indexed in [false, true] {
        for owners in 2..=4 {
            let gpu = TestGpu::new();
            let head = configured_owner_head(&gpu, true, indexed, Some(owners)).unwrap();
            let shared = [
                head.module.enorm.weight,
                head.module.hnorm.weight,
                head.module.norm.weight,
                head.module.eh_proj.weight,
                head.embed_tokens.weight,
                head.lm_head.weight,
            ];
            let actual: usize = gpu
                .live_allocations()
                .iter()
                .filter(|(ptr, _)| !shared.iter().any(|weight| weight.0 == **ptr))
                .map(|(_, bytes)| *bytes)
                .sum();
            let quote =
                Glm5MtpHead::paired_private_reserve_bytes(&capacity_config(indexed), 2044, owners)
                    .unwrap();
            assert_eq!(quote, actual, "indexed={indexed}, owners={owners}");
            assert_eq!(quote, owners * if indexed { 5_423_104 } else { 4_243_456 });
        }
    }
}

#[test]
fn private_reserve_rejects_unqualified_capacity_context_and_shape() {
    let config = capacity_config(true);
    for owners in [0, 1, 5, usize::MAX] {
        assert!(Glm5MtpHead::paired_private_reserve_bytes(&config, 2044, owners).is_err());
    }
    for context in [1, 2045, usize::MAX] {
        assert!(Glm5MtpHead::paired_private_reserve_bytes(&config, context, 4).is_err());
    }
    let mut invalid = config;
    invalid.ep_rank = 1;
    assert!(Glm5MtpHead::paired_private_reserve_bytes(&invalid, 2044, 4).is_err());
}
