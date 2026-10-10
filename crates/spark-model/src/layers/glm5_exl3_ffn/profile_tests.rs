// SPDX-License-Identifier: AGPL-3.0-only
use super::profile::{FfnProfile, ProfileMetadata};
use spark_runtime::gpu::mock::MockGpuBackend;

fn metadata() -> ProfileMetadata {
    ProfileMetadata {
        layer: Some(3),
        rows: 5,
        rank: 1,
        world: 2,
        routes: 5,
        role: "routed",
        precision: Some([32, 256, 8, 1, 16, 128, 1, 0]),
        thin_capacity: Some(32),
    }
}
#[test]
fn disabled_profile_adds_no_sync_or_reads_and_preserves_failure() {
    let gpu = MockGpuBackend::new();
    let profile = FfnProfile::new(&gpu, 77, false, metadata());
    assert_eq!(profile.measure("gate", || Ok(42)).unwrap(), 42);
    assert_eq!(
        profile
            .measure::<()>("down", || anyhow::bail!("original launch failure"))
            .unwrap_err()
            .to_string(),
        "original launch failure"
    );
    assert_eq!(gpu.sync_count(), 0);
    assert_eq!(gpu.d2h_blocking_count(), 0);
    assert_eq!(gpu.d2h_async_count(), 0);
}
#[test]
fn enabled_profile_syncs_only_operation_boundaries_and_preserves_failure() {
    let gpu = MockGpuBackend::new();
    let profile = FfnProfile::new(&gpu, 77, true, metadata());
    profile.measure("gate", || Ok(())).unwrap();
    assert_eq!(gpu.sync_count(), 2);
    profile
        .measure::<()>("down", || anyhow::bail!("original launch failure"))
        .unwrap_err();
    assert_eq!(gpu.sync_count(), 3); // no successful completion event for failed launch
    assert_eq!(gpu.d2h_blocking_count(), 0);
    assert_eq!(gpu.d2h_async_count(), 0);
}

#[path = "../../../examples/exl3_moe_scratch_check/protocol.rs"]
mod scratch_gpu_protocol;

#[path = "../../../examples/exl3_moe_scratch_check/guarded.rs"]
mod scratch_gpu_guarded;
