// SPDX-License-Identifier: AGPL-3.0-only
//! Actual normalization dispatch with recorded GPU calls, not CUDA numerics.

use crate::model::glm_c2_test_support::fixture::{CALLER, Event, Fixture};
use crate::traits::Model;

fn check(name: &str, setting: Option<&str>, glm: bool, expected: bool) {
    const CHILD: &str = "ATLAS_GLM_SSM_NORMALIZE_TEST_CHILD";
    if std::env::var(CHILD).as_deref() != Ok(name) {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                &format!("model::trait_impl::meta::glm_ssm_normalization_tests::{name}"),
                "--nocapture",
            ])
            .env(CHILD, name)
            .env_remove("ATLAS_GLM_SSM_NORMALIZE");
        if let Some(value) = setting {
            command.env("ATLAS_GLM_SSM_NORMALIZE", value);
        }
        assert!(command.status().unwrap().success(), "child failed: {name}");
        return;
    }

    for rank in 0..2 {
        let mut fixture = Fixture::new_legacy_ssm(rank);
        if !glm {
            fixture.model.config.model_type = "qwen3_next".into();
        }
        assert!(fixture.model.ssm_pool.num_ssm_layers > 0);
        assert_ne!(fixture.model.ssm_state_norm_kernel.0, 0);
        fixture.gpu.clear();
        fixture
            .model
            .normalize_ssm_states(&fixture.seqs[0], CALLER)
            .unwrap();
        let events = fixture.gpu.trace();
        if expected {
            assert!(
                matches!(events.as_slice(), [Event::Upload(_, 8, CALLER), Event::Kernel(name, _, CALLER)]
                if name == "ssm_state_clamp_norm_fused"),
                "rank {rank}: {events:?}"
            );
        } else {
            assert!(events.is_empty(), "disabled GLM touched GPU: {events:?}");
        }
    }
}

#[test]
fn glm_explicit_zero_skips_normalization_dispatch() {
    check(
        "glm_explicit_zero_skips_normalization_dispatch",
        Some("0"),
        true,
        false,
    );
}

#[test]
fn glm_absent_preserves_normalization_dispatch() {
    check(
        "glm_absent_preserves_normalization_dispatch",
        None,
        true,
        true,
    );
}

#[test]
fn glm_explicit_one_preserves_normalization_dispatch() {
    check(
        "glm_explicit_one_preserves_normalization_dispatch",
        Some("1"),
        true,
        true,
    );
}

#[test]
fn non_glm_ignores_zero_and_preserves_normalization_dispatch() {
    check(
        "non_glm_ignores_zero_and_preserves_normalization_dispatch",
        Some("0"),
        false,
        true,
    );
}
