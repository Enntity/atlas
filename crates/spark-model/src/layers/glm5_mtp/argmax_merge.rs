// SPDX-License-Identifier: AGPL-3.0-only
//! Merge the two actual contiguous BF16 vocabulary-shard maxima.
//! No collective, allocation, or ownership authority lives here.

pub(crate) fn merge(v0: f32, i0: u32, v1: f32, i1: u32, shard: u32) -> u32 {
    // The value kernel returns a BF16 maximum above its FP32 floor, or the
    // untouched (-1e30,0) sentinel. BF16 cannot represent that exact floor.
    // Preserve all-invalid fallback0 while valid ties choose the higher ID.
    if v1 > v0 || (v1 == v0 && v1 > -1e30) {
        i1 + shard
    } else {
        i0
    }
}

#[cfg(test)]
mod tests {
    use super::merge;
    const SHARD: u32 = 77428;
    const FLOOR: f32 = -1e30;

    #[test]
    fn finite_equal_maxima_choose_the_higher_global_token() {
        for value in [-7.0, -0.0, 0.0, 7.0] {
            assert_eq!(merge(value, 2047, value, 17, SHARD), SHARD + 17);
        }
    }

    #[test]
    fn unequal_maxima_choose_value_before_index() {
        assert_eq!(merge(7.0, 17, 6.0, SHARD - 1, SHARD), 17);
        assert_eq!(merge(-7.0, 2047, -6.0, 17, SHARD), SHARD + 17);
    }

    #[test]
    fn invalid_shards_preserve_full_kernel_fallback_and_valid_negative() {
        // The CUDA value kernel ignores NaNs and values below its floor;
        // an entirely invalid shard therefore produces (FLOOR,0), not NaN.
        assert_eq!(merge(FLOOR, 0, -7.0, 17, SHARD), SHARD + 17);
        assert_eq!(merge(-7.0, 2047, FLOOR, 0, SHARD), 2047);
        assert_eq!(merge(FLOOR, 0, FLOOR, 0, SHARD), 0);
    }

    #[test]
    fn equal_infinities_choose_the_higher_global_token() {
        assert_eq!(
            merge(f32::INFINITY, 17, f32::INFINITY, 2047, SHARD),
            SHARD + 2047
        );
        assert_eq!(merge(f32::INFINITY, 17, 7.0, 2047, SHARD), 17);
    }

    #[test]
    fn closest_selectable_bf16_tie_is_above_the_unrepresentable_floor() {
        assert_eq!(FLOOR.to_bits(), 0xf149_f2ca);
        assert_ne!(FLOOR.to_bits() & 0xffff, 0);
        let above = f32::from_bits(FLOOR.to_bits() & 0xffff_0000);
        let below = f32::from_bits((FLOOR.to_bits() & 0xffff_0000) + 0x1_0000);
        assert!(above > FLOOR && below < FLOOR);
        assert_eq!(merge(above, 17, above, 2047, SHARD), SHARD + 2047);
    }
}
