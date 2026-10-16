// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

/// Arithmetic profile of the attested vLLM GLM-5.3 run. This is a serving
/// profile, not checkpoint metadata: EXL3_TEMP_ROWS_FUSED was explicitly 32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::layers::glm5_exl3_ffn) struct RoutedPrecisionPolicy {
    pub(in crate::layers::glm5_exl3_ffn) thin_capacity: u32,
    pub(in crate::layers::glm5_exl3_ffn) tile_k: u32,
    pub(in crate::layers::glm5_exl3_ffn) tile_n: u32,
    pub(in crate::layers::glm5_exl3_ffn) slices: u32,
    pub(in crate::layers::glm5_exl3_ffn) half_partial: bool,
    pub(in crate::layers::glm5_exl3_ffn) fat_tile_k: u32,
    pub(in crate::layers::glm5_exl3_ffn) fat_tile_n: u32,
    pub(in crate::layers::glm5_exl3_ffn) fat_slices: u32,
    pub(in crate::layers::glm5_exl3_ffn) fat_half_partial: bool,
}
impl RoutedPrecisionPolicy {
    pub(in crate::layers::glm5_exl3_ffn) const GLM53_TP2_REFERENCE: Self = Self {
        thin_capacity: 32,
        tile_k: 32,
        tile_n: 256,
        slices: 8,
        half_partial: true,
        fat_tile_k: 16,
        fat_tile_n: 128,
        fat_slices: 1,
        fat_half_partial: false,
    };
    pub(super) fn validate(self) -> Result<()> {
        ensure!(
            self.thin_capacity > 0,
            "EXL3 routed thin capacity must be explicit and positive"
        );
        ensure!(
            (self.tile_k, self.tile_n, self.slices, self.half_partial) == (32, 256, 8, true),
            "unsupported EXL3 thin MMA accumulation profile"
        );
        ensure!(
            (
                self.fat_tile_k,
                self.fat_tile_n,
                self.fat_slices,
                self.fat_half_partial
            ) == (16, 128, 1, false),
            "unsupported EXL3 fat MMA accumulation profile"
        );
        Ok(())
    }
    pub(super) fn matmul_args(self) -> [u32; 8] {
        [
            self.tile_k,
            self.tile_n,
            self.slices,
            u32::from(self.half_partial),
            self.fat_tile_k,
            self.fat_tile_n,
            self.fat_slices,
            u32::from(self.fat_half_partial),
        ]
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn precision_capacity_is_explicit_and_zero_is_rejected() {
        assert!(
            RoutedPrecisionPolicy {
                thin_capacity: 0,
                ..RoutedPrecisionPolicy::GLM53_TP2_REFERENCE
            }
            .validate()
            .is_err()
        );
        let policy = RoutedPrecisionPolicy::GLM53_TP2_REFERENCE;
        policy.validate().unwrap();
        assert_eq!(policy.thin_capacity, 32);
        for invalid in [
            RoutedPrecisionPolicy {
                tile_k: 64,
                ..policy
            },
            RoutedPrecisionPolicy {
                tile_n: 128,
                ..policy
            },
            RoutedPrecisionPolicy {
                slices: 4,
                ..policy
            },
            RoutedPrecisionPolicy {
                half_partial: false,
                ..policy
            },
            RoutedPrecisionPolicy {
                fat_tile_k: 32,
                ..policy
            },
            RoutedPrecisionPolicy {
                fat_tile_n: 256,
                ..policy
            },
            RoutedPrecisionPolicy {
                fat_slices: 8,
                ..policy
            },
            RoutedPrecisionPolicy {
                fat_half_partial: true,
                ..policy
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }
    #[test]
    fn reference_launch_binds_distinct_thin_and_fat_accumulation_profiles() {
        assert_eq!(
            RoutedPrecisionPolicy::GLM53_TP2_REFERENCE
                .matmul_args()
                .as_slice(),
            [32, 256, 8, 1, 16, 128, 1, 0]
        );
    }
}
