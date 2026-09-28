// SPDX-License-Identifier: AGPL-3.0-only
//! Server-owned subprocess name and supported environment, not model test-path reuse.
pub(super) fn isolated(full_test_path: &str) -> bool {
    if std::env::var("ATLAS_C2_SERVER_FIXTURE_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", full_test_path, "--nocapture"])
        .env("ATLAS_C2_SERVER_FIXTURE_CHILD", "1")
        .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
        .env("ATLAS_GLM_MTP_REPAIR", "0")
        .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
        .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
        .env("ATLAS_GLM_MTP_ALL_GATHER", "1")
        .env("ATLAS_GLM_MTP_DISTRIBUTED_ARGMAX", "0")
        .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1");
    for key in [
        "ATLAS_NO_MTP_EAGER_DRAFTER",
        "ATLAS_NO_MTP_DRAFTER_CONTEXT",
        "ATLAS_MTP_CARRY_DRAFTER",
        "ATLAS_MTP_ACCEPT_DEBUG",
        "ATLAS_GLM_MTP_FUSED_EH_NORM",
        "ATLAS_GLM_MTP_SERIAL_PREFILL",
        "ATLAS_MTP_CATCHUP",
        "ATLAS_GLM_MTP_PROFILE",
    ] {
        cmd.env_remove(key);
    }
    assert!(
        cmd.status().unwrap().success(),
        "actual server fixture child failed: {full_test_path}"
    );
    true
}
