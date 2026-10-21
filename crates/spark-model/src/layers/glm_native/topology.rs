// SPDX-License-Identifier: AGPL-3.0-only
//! Shared topology boundary for explicit GLM serving modes and native MTP.

/// The target overlaps tensor and expert ranks; the private native predictor
/// is full-sized on rank zero and is not subject to this target boundary.
pub fn supported_parallelism(world: usize, tp: usize, ep: usize) -> bool {
    matches!(world, 2 | 4) && tp == world && ep == world
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_modes_and_native_mtp_admit_pair_and_four_rank_targets() {
        assert!(supported_parallelism(2, 2, 2));
        assert!(supported_parallelism(4, 4, 4));
    }

    #[test]
    fn target_boundary_rejects_partial_or_unsupported_parallel_worlds() {
        for (world, tp, ep) in [
            (0, 0, 0),
            (1, 1, 1),
            (3, 3, 3),
            (8, 8, 8),
            (2, 1, 2),
            (4, 2, 4),
            (4, 4, 2),
            (2, 4, 4),
        ] {
            assert!(!supported_parallelism(world, tp, ep), "{world}/{tp}/{ep}");
        }
    }
}
