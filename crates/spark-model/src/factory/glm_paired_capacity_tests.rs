// SPDX-License-Identifier: AGPL-3.0-only
//! Actual factory boundary and appended-head owner capacity, not serving authority.
use super::*;

#[test]
fn actual_factory_validator_requires_explicit_two_to_eight_owners() {
    if isolated("capacity_tests::actual_factory_validator_requires_explicit_two_to_eight_owners") {
        return;
    }
    for rank in 0..2 {
        let c = config(rank);
        let comm = Comm(rank, Arc::new(AtomicUsize::new(0)), None);
        for owners in [2, 3, 4, 5, 6, 7, 8, 1, 9] {
            let result = validate(
                GlmMtpBuildMode::Paired,
                &c,
                1024,
                2044,
                owners,
                MtpQuantization::Bf16,
                true,
                false,
                4,
                KvCacheDtype::Bf16,
                &[KvCacheDtype::Bf16],
                Some(&comm),
                None,
                false,
                false,
            );
            assert_eq!(
                result.is_ok(),
                (2..=8).contains(&owners),
                "actual factory rank={rank} owners={owners}: {result:?}"
            );
        }
    }
}

#[test]
fn actual_factory_head_allocates_exact_explicit_owner_capacity() {
    if isolated("capacity_tests::actual_factory_head_allocates_exact_explicit_owner_capacity") {
        return;
    }
    for rank in 0..2 {
        for owners in 2..=8 {
            let gpu = Gpu::new();
            let embed = DenseWeight {
                weight: gpu.alloc(8 * 8192).unwrap(),
            };
            let head = build_head(
                GlmMtpBuildMode::Paired,
                module(&gpu),
                embed,
                embed,
                None,
                &config(rank),
                &gpu,
                8,
                2044,
                owners,
            )
            .unwrap();
            let handoff = head.glm_pair_repair().unwrap().paired_handoff().unwrap();
            assert_eq!(handoff.owner_capacity(&gpu).unwrap(), owners);
            let mut states: Vec<_> = (0..owners)
                .map(|_| head.alloc_state(&gpu).unwrap())
                .collect();
            assert!(head.alloc_state(&gpu).is_err());
            // A full actual pool remains healthy; capacity is not free-slot count.
            handoff.validate_session(&gpu).unwrap();
            for state in &mut states {
                head.free_state(&gpu, None, state.as_mut()).unwrap();
            }
            let mut replacement = head.alloc_state(&gpu).unwrap();
            head.free_state(&gpu, None, replacement.as_mut()).unwrap();
        }
    }
}

#[test]
fn actual_factory_head_rejects_invalid_selected_capacity_before_kernel_lookup() {
    if isolated(
        "capacity_tests::actual_factory_head_rejects_invalid_selected_capacity_before_kernel_lookup",
    ) {
        return;
    }
    for owners in [1, 9] {
        for mode in [GlmMtpBuildMode::Legacy, GlmMtpBuildMode::Paired] {
            let gpu = Gpu::new();
            let embed = DenseWeight {
                weight: gpu.alloc(8 * 8192).unwrap(),
            };
            let result = build_head(
                mode,
                module(&gpu),
                embed,
                embed,
                None,
                &config(0),
                &gpu,
                8,
                2044,
                owners,
            );
            if mode == GlmMtpBuildMode::Paired {
                let error = result.err().expect("invalid paired capacity must refuse");
                assert!(format!("{error:#}").contains("owner capacity"));
                assert_eq!(gpu.probe.kernels.load(Ordering::Relaxed), 0);
            } else {
                assert!(
                    result
                        .unwrap()
                        .glm_pair_repair()
                        .unwrap()
                        .paired_handoff()
                        .is_none()
                );
            }
        }
    }
}
