// SPDX-License-Identifier: AGPL-3.0-only
//! Actual model cold selection, independent of retained owner capacity.
use super::*;
use crate::model::glm_owner_wire::Mode;

#[test]
fn wider_compute_requires_explicit_validated_selection() {
    if flow::isolated(
        "pair_group_tests::owner_policy::wider_compute_requires_explicit_validated_selection",
    ) {
        return;
    }
    for rank in 0..2 {
        let (mut f, _) = Four::prepare_fixture(Fixture::new_owner_compute(rank));
        assert!(
            !f.f.model
                .glm_paired_execution()
                .unwrap()
                .owner_verification_enabled()
        );
        f.f.gpu.clear();
        f.f.model
            .initialize_glm_owner_verification(Mode::OwnersJoint)
            .expect("actual wider model must accept explicit checked cold mode");
        assert!(
            f.f.model
                .glm_paired_execution()
                .unwrap()
                .owner_verification_enabled()
        );
        assert!(f.f.gpu.trace().is_empty());
        assert!(
            f.f.model
                .initialize_glm_owner_verification(Mode::OwnersJoint)
                .is_err()
        );
        let (mut narrow, _) = Four::prepared(rank);
        assert!(
            narrow
                .f
                .model
                .initialize_glm_owner_verification(Mode::OwnersJoint)
                .is_err()
        );
        assert!(
            !narrow
                .f
                .model
                .glm_paired_execution()
                .unwrap()
                .owner_verification_enabled()
        );
    }
}
