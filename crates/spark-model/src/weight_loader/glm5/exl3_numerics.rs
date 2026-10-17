// SPDX-License-Identifier: AGPL-3.0-only
//! Pinned GLM-5.3 TP2 packed-linear arithmetic; independent of kernel tuning.
use anyhow::{Result, bail, ensure};
use spark_runtime::weights::exl3::native::{Exl3CooperativeProfile, Exl3LinearNumerics};

/// Exact original installed cooperative cache, 31 records (EX3ATUNE v1).
/// Entries use source-bound architecture enum5, mul1 cb2, salt version3,
/// actual SMS48. This is a serving arithmetic profile, not checkpoint metadata.
pub const GLM53_PACKED_PROFILE_CACHE_SHA256: &str =
    "d694efa56f4b1283da9d58dbe805ecb4e41e7757ef63a4416eeb4983c46b0067";

pub const GLM53_TP2_HEAD_BITS: u8 = 6;

/// Explicit selector seam for standalone, declared numerical profiles.
pub type Exl3NumericsSelector =
    fn(usize, usize, usize, u8, usize, usize) -> Result<Exl3LinearNumerics>;

const fn p(tile_k: u32, tile_n: u32, slices: u32) -> Option<Exl3CooperativeProfile> {
    Some(Exl3CooperativeProfile {
        tile_k,
        tile_n,
        slices,
        half_partial: false,
    })
}

pub fn glm53_tp2_packed_numerics(
    world: usize,
    k: usize,
    physical_n: usize,
    bits: u8,
    logical_n: usize,
    offset: usize,
) -> Result<Exl3LinearNumerics> {
    ensure!(
        world == 2,
        "Pinned GLM-5.3 packed numerical profile requires TP2; unsupported topology"
    );
    let geometry = (k, physical_n, logical_n, bits);
    let shared_packed = std::env::var("ATLAS_EXL3_SHARED_PACKED").as_deref() == Ok("1");
    if shared_packed && matches!(geometry, (4096, 1024, 2048, 5..=6)) {
        ensure!(
            [0, 1024].contains(&offset),
            "Unsupported packed shared-expert gate|up component offset"
        );
    } else if geometry == (4096, 4096, 12288, 6) {
        ensure!(
            [0, 4096, 8192].contains(&offset),
            "Unsupported GLM-5.3 fused QKV component offset"
        );
    } else {
        ensure!(
            offset == 0,
            "Ordinary GLM-5.3 packed projection requires zero logical offset"
        );
    }
    // Buckets2/4/8/16 follow the actual cache hash. Bucket2 is unreachable for
    // ordinary modules because the pinned wrapper explicitly pads to3 rows.
    let row_profiles = match geometry {
        (4096, 4096, 4096, 6) => [None, p(32, 256, 48), p(32, 256, 48), p(32, 128, 48)],
        (4096, 4096, 12288, 6) => [None, p(16, 512, 48), p(16, 512, 48), p(32, 256, 48)],
        (6144, 4096, 4096, 5) => [None, p(16, 512, 48), p(32, 256, 48), p(32, 256, 48)],
        (4096, 6144, 6144, 5) => [None, p(16, 512, 48), p(16, 512, 48), p(32, 128, 48)],
        (1536, 4096, 4096, 6) | (4096, 1536, 1536, 6) => {
            [None, p(32, 128, 48), p(32, 128, 48), p(32, 128, 48)]
        }
        (1536, 8192, 8192, 6) | (8192, 4096, 4096, 6) => {
            [None, p(16, 512, 48), p(16, 512, 48), p(32, 128, 48)]
        }
        (4096, 512, 512, 6) => [None, p(32, 128, 20), p(32, 128, 16), p(32, 128, 16)],
        // ATLAS_EXL3_SHARED_PACKED=1 only (numerics-changing): the shared expert's
        // gate/up (K4096 -> N1024) and down (K1024 -> N4096) have no reference
        // profile because the reference keeps them BF16. Untuned profiles.
        (4096, 1024, 1024, 5..=6) | (1024, 4096, 4096, 5..=6)
            if std::env::var("ATLAS_EXL3_SHARED_PACKED").as_deref() == Ok("1") =>
        {
            [None, p(32, 128, 48), p(32, 128, 48), p(32, 128, 48)]
        }
        // Shared gate|up as one fused [gate | up] matrix of logical width 2048
        // (one cooperative launch, exl3_coop_gemm2): tile 32x128 gives 16
        // column tiles x 128 k-tiles over 48 slices (~43 tiles per slice,
        // ~3 slices per column), instead of 4 column tiles for 48 slices.
        (4096, 1024, 2048, 5..=6) if shared_packed => {
            [None, p(32, 128, 48), p(32, 128, 48), p(32, 128, 48)]
        }
        (4096, 77440, 77440, GLM53_TP2_HEAD_BITS) => [
            p(32, 128, 48),
            p(32, 128, 48),
            p(32, 128, 48),
            p(32, 128, 48),
        ],
        _ => bail!(
            "Unsupported pinned GLM-5.3 packed geometry K{k}/N{physical_n}/logicalN{logical_n}/bits{bits}"
        ),
    };
    let head = geometry == (4096, 77440, 77440, GLM53_TP2_HEAD_BITS);
    // ATLAS_EXL3_VERIFY_PROFILES=1 (speculative lane, numerics-changing for
    // 5-8 row VERIFY launches only): the padded-rows 5..8 bucket's 16x512
    // profile has a 32x256 alternative (~10-16 us faster per call at r = 5-8
    // in the coop bench) that applies only inside a verify scope
    // (spark_runtime::weights::exl3::native::verify_scope); prefill chunks
    // and serial decode keep the reference profiles.
    let verify_rows_5_8 = (std::env::var("ATLAS_EXL3_VERIFY_PROFILES").as_deref() == Ok("1"))
        .then_some(row_profiles[2])
        .flatten()
        .filter(|p5| {
            (p5.tile_k, p5.tile_n) == (16, 512)
                && physical_n.is_multiple_of(256)
                && logical_n.is_multiple_of(256)
                && offset.is_multiple_of(256)
        })
        .and_then(|p5| p(32, 256, p5.slices));
    Ok(Exl3LinearNumerics {
        logical_n,
        logical_offset: offset,
        bits,
        reference_min_rows: if head { 1 } else { 3 },
        reference_pad: !head,
        row_profiles,
        verify_rows_5_8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_cached_profiles_keep_row_bucket_changes_and_small_kva_slices() {
        let kda = glm53_tp2_packed_numerics(2, 4096, 4096, 6, 12288, 8192).unwrap();
        let p = kda.profile_for_rows(1).unwrap();
        assert_eq!((p.tile_k, p.tile_n, p.slices), (16, 512, 48));
        let p = kda.profile_for_rows(10).unwrap();
        assert_eq!((p.tile_k, p.tile_n, p.slices), (32, 256, 48));
        assert!(kda.row_profiles[0].is_none());
        let kv = glm53_tp2_packed_numerics(2, 4096, 512, 6, 512, 0).unwrap();
        assert_eq!(kv.profile_for_rows(1).unwrap().slices, 20);
        assert_eq!(kv.profile_for_rows(5).unwrap().slices, 16);
        let head = glm53_tp2_packed_numerics(2, 4096, 77440, 6, 77440, 0).unwrap();
        assert!(!head.reference_pad);
        assert_eq!(head.reference_min_rows, 1);
        assert_eq!(kda.reference_min_rows, 3);
        assert_eq!(head.profile_for_rows(128).unwrap().tile_n, 128);
    }
    #[test]
    fn every_declared_profile_resolves_all_reachable_buckets_without_allocation() {
        use spark_runtime::{
            gpu::mock::MockGpuBackend, weights::exl3::native::Exl3LinearWorkspace,
        };
        let gpu = MockGpuBackend::new();
        for (k, n, logical_n, bits) in [
            (4096, 4096, 4096, 6),
            (4096, 4096, 12288, 6),
            (6144, 4096, 4096, 5),
            (4096, 6144, 6144, 5),
            (1536, 4096, 4096, 6),
            (4096, 1536, 1536, 6),
            (1536, 8192, 8192, 6),
            (8192, 4096, 4096, 6),
            (4096, 512, 512, 6),
            (4096, 77440, 77440, GLM53_TP2_HEAD_BITS),
        ] {
            let profile = glm53_tp2_packed_numerics(2, k, n, bits, logical_n, 0).unwrap();
            Exl3LinearWorkspace::preflight_numerics(&gpu, k, n, 128, profile).unwrap();
            for rows in [1, 3, 5, 128] {
                profile.profile_for_rows(rows).unwrap();
            }
        }
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.launch_count(), 0);
    }
    #[test]
    fn unsupported_topology_geometry_bits_and_component_offsets_have_no_fallback() {
        for args in [
            (1, 4096, 77440, 6, 77440, 0),
            (4, 4096, 4096, 6, 4096, 0),
            (2, 128, 128, 4, 128, 0),
            (2, 4096, 77440, 4, 77440, 0),
            (2, 4096, 4096, 6, 12288, 128),
            (2, 4096, 4096, 6, 4096, 128),
        ] {
            assert!(
                glm53_tp2_packed_numerics(args.0, args.1, args.2, args.3, args.4, args.5).is_err()
            );
        }
    }
}
