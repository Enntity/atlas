// SPDX-License-Identifier: AGPL-3.0-only

use super::Exl3CheckpointRouter;
use safetensors::{Dtype, tensor::TensorView};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        Self::with_width(128)
    }
    fn with_width(width: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "atlas-exl3-checkpoint-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let fixture = Self(dir);
        fixture.config(
            serde_json::json!({"quantization_config":{"quant_method":"exl3","codebook":"mul1"}}),
        );
        let trellis: Vec<u8> = (0..(width / 16).pow(2) * 16 * 4 * 2)
            .map(|i| (i.wrapping_mul(71) ^ (i >> 8)) as u8)
            .collect();
        let scale: Vec<u8> = (0..width).flat_map(|_| 0x3c00u16.to_le_bytes()).collect();
        let marker = 0x83dc_d12du32.to_le_bytes();
        let a = [
            (
                "linear.trellis",
                TensorView::new(Dtype::I16, vec![width / 16, width / 16, 64], &trellis).unwrap(),
            ),
            (
                "linear.suh",
                TensorView::new(Dtype::F16, vec![width], &scale).unwrap(),
            ),
        ];
        let b = [
            (
                "linear.svh",
                TensorView::new(Dtype::F16, vec![width], &scale).unwrap(),
            ),
            (
                "linear.mul1",
                TensorView::new(Dtype::I32, vec![], &marker).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(a, None, &fixture.0.join("a.safetensors")).unwrap();
        safetensors::serialize_to_file(b, None, &fixture.0.join("b.safetensors")).unwrap();
        fixture.index(serde_json::json!({"weight_map":{
            "linear.trellis":"a.safetensors","linear.suh":"a.safetensors",
            "linear.svh":"b.safetensors","linear.mul1":"b.safetensors"
        }}));
        fixture
    }
    fn config(&self, value: serde_json::Value) {
        std::fs::write(
            self.0.join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
    }
    fn index(&self, value: serde_json::Value) {
        std::fs::write(
            self.0.join("model.safetensors.index.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn indexed_cross_shard_group_preserves_checkpoint_bytes() {
    let f = Fixture::new();
    let router = Exl3CheckpointRouter::open(&f.0).unwrap();
    let linear = router.linear_view("linear", 128, 128).unwrap();
    assert_eq!(linear.bits(), 4);
    assert_eq!(linear.trellis().dtype(), Dtype::I16);
    assert_eq!(linear.input_scale().dtype(), Dtype::F16);
    assert_eq!(linear.output_scale().data()[..2], 0x3c00u16.to_le_bytes());
    assert!(router.linear_view("absent", 128, 128).is_err());
    assert!(router.linear_view("linear", 256, 128).is_err());
}

#[test]
fn missing_companion_is_refused() {
    let f = Fixture::new();
    f.index(serde_json::json!({"weight_map":{"linear.trellis":"a.safetensors"}}));
    let router = Exl3CheckpointRouter::open(&f.0).unwrap();
    assert!(router.linear_view("linear", 128, 128).is_err());
}

#[test]
fn config_uses_converted_format_and_requires_explicit_codebook() {
    let f = Fixture::new();
    for config in [
        serde_json::json!({"quantization_config":{"quant_method":"nvfp4","codebook":"mul1"},"original_quantization_config":{"quant_method":"exl3","codebook":"mul1"}}),
        serde_json::json!({"quantization_config":{"quant_method":"exl3"}}),
        serde_json::json!({"quantization_config":{"quant_method":"exl3","codebook":"other"}}),
        serde_json::json!({}),
    ] {
        f.config(config);
        assert!(Exl3CheckpointRouter::open(&f.0).is_err());
    }
    f.config(serde_json::json!({"quantization_config":{"quant_method":"exl3","codebook":"mul1"},"original_quantization_config":{"quant_method":"nvfp4"}}));
    assert!(Exl3CheckpointRouter::open(&f.0).is_ok());
}

#[test]
fn malformed_indices_and_shard_paths_are_refused() {
    let f = Fixture::new();
    for index in [
        serde_json::json!({}),
        serde_json::json!({"weight_map":{}}),
        serde_json::json!({"weight_map":{"missing":"a.safetensors"}}),
        serde_json::json!({"weight_map":{"linear.trellis":"../a.safetensors"}}),
        serde_json::json!({"weight_map":{"linear.trellis":"/a.safetensors"}}),
        serde_json::json!({"weight_map":{"linear.trellis":"dir/a.safetensors"}}),
        serde_json::json!({"weight_map":{"linear.trellis":"missing.safetensors"}}),
    ] {
        f.index(index);
        assert!(Exl3CheckpointRouter::open(&f.0).is_err());
    }
    f.index(serde_json::json!({"weight_map":{"linear.trellis":"a.safetensors"}}));
    std::fs::write(f.0.join("a.safetensors"), b"invalid").unwrap();
    assert!(Exl3CheckpointRouter::open(&f.0).is_err());
}

#[test]
fn legacy_companions_and_conflicting_dense_weight_are_refused() {
    for suffix in ["su", "sv", "mcg", "scale", "bias", "weight"] {
        let f = Fixture::new();
        let data = [0u8; 2];
        let name = format!("linear.{suffix}");
        safetensors::serialize_to_file(
            [(
                name.as_str(),
                TensorView::new(Dtype::F16, vec![1], &data).unwrap(),
            )],
            None,
            &f.0.join("c.safetensors"),
        )
        .unwrap();
        f.index(serde_json::json!({"weight_map":{
            "linear.trellis":"a.safetensors","linear.suh":"a.safetensors",
            "linear.svh":"b.safetensors","linear.mul1":"b.safetensors",name:"c.safetensors"
        }}));
        let router = Exl3CheckpointRouter::open(&f.0).unwrap();
        assert!(router.linear_view("linear", 128, 128).is_err(), "{suffix}");
    }
}

#[cfg(unix)]
#[test]
fn shard_symlink_cannot_escape_model_directory() {
    let f = Fixture::new();
    let outside = Fixture::new();
    std::fs::remove_file(f.0.join("a.safetensors")).unwrap();
    std::os::unix::fs::symlink(outside.0.join("a.safetensors"), f.0.join("a.safetensors")).unwrap();
    assert!(Exl3CheckpointRouter::open(&f.0).is_err());
}

#[test]
fn cross_shard_checkpoint_uploads_only_native_tp_rank_bytes() {
    use crate::gpu::mock::MockGpuBackend;
    use crate::weights::exl3::tp::{Exl3DeviceShard, ShardAxis, TpShardPlan};
    use atlas_core::scope::ModelResource;

    let f = Fixture::with_width(256);
    let router = Exl3CheckpointRouter::open(&f.0).unwrap();
    let view = router.linear_view("linear", 256, 256).unwrap();
    for axis in [ShardAxis::Input, ShardAxis::Output] {
        for rank in 0..2 {
            let plan = TpShardPlan::new(&view, axis, rank, 2).unwrap();
            let expected = plan.materialize(&view).unwrap();
            let gpu = MockGpuBackend::new();
            let mut device = Exl3DeviceShard::upload(&gpu, &view, &plan).unwrap();
            let [trellis, input, output] = device.buffers().unwrap();
            assert_eq!(gpu.read_alloc(trellis).unwrap(), expected.trellis_bytes());
            assert_eq!(
                gpu.read_alloc(input).unwrap(),
                expected.view().unwrap().input_scale().data()
            );
            assert_eq!(
                gpu.read_alloc(output).unwrap(),
                expected.view().unwrap().output_scale().data()
            );
            assert_eq!(
                expected.trellis_bytes().len() * 2,
                view.trellis().data().len()
            );
            device.release(&gpu).unwrap();
            assert!(device.buffers().is_err());
        }
    }
}

#[test]
fn explicit_cache_eviction_keeps_borrowed_checkpoint_views_valid() {
    let f = Fixture::new();
    let router = Exl3CheckpointRouter::open(&f.0).unwrap();
    let view = router.linear_view("linear", 128, 128).unwrap();
    let before = view.trellis().data().to_vec();
    router.evict_cached_pages().unwrap();
    assert_eq!(view.trellis().data(), before);
    assert_eq!(view.input_scale().data()[..2], 0x3c00u16.to_le_bytes());
    assert_eq!(
        router
            .linear_view("linear", 128, 128)
            .unwrap()
            .trellis()
            .data(),
        before
    );
}

#[cfg(unix)]
#[test]
fn hf_snapshot_shards_allow_only_local_content_addressed_blob_targets() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    let repository = f.0.join("models--test--native");
    let snapshot = repository.join("snapshots").join("a".repeat(40));
    let blobs = repository.join("blobs");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir(&blobs).unwrap();
    for name in ["config.json", "model.safetensors.index.json"] {
        std::fs::rename(f.0.join(name), snapshot.join(name)).unwrap();
    }
    for (name, hash) in [
        ("a.safetensors", "b".repeat(64)),
        ("b.safetensors", "c".repeat(64)),
    ] {
        std::fs::rename(f.0.join(name), blobs.join(&hash)).unwrap();
        symlink(format!("../../blobs/{hash}"), snapshot.join(name)).unwrap();
    }
    let router = Exl3CheckpointRouter::open(&snapshot).unwrap();
    assert_eq!(router.linear_view("linear", 128, 128).unwrap().bits(), 4);
    drop(router);
    let shard = snapshot.join("a.safetensors");
    // Even inside the same repository, sibling snapshots are not blob storage.
    let sibling = repository.join("snapshots").join("d".repeat(40));
    std::fs::create_dir(&sibling).unwrap();
    std::fs::copy(blobs.join("b".repeat(64)), sibling.join("a.safetensors")).unwrap();
    std::fs::remove_file(&shard).unwrap();
    symlink(sibling.join("a.safetensors"), &shard).unwrap();
    assert!(Exl3CheckpointRouter::open(&snapshot).is_err());
    // A file in blobs without an HF LFS hash basename is not trusted.
    std::fs::copy(blobs.join("b".repeat(64)), blobs.join("not-a-content-hash")).unwrap();
    std::fs::remove_file(&shard).unwrap();
    symlink("../../blobs/not-a-content-hash", &shard).unwrap();
    assert!(Exl3CheckpointRouter::open(&snapshot).is_err());
    // A symlinked blobs directory cannot extend the repository trust boundary.
    std::fs::remove_file(&shard).unwrap();
    symlink(format!("../../blobs/{}", "b".repeat(64)), &shard).unwrap();
    let outside = f.0.join("outside-blobs");
    std::fs::rename(&blobs, &outside).unwrap();
    symlink(&outside, &blobs).unwrap();
    assert!(Exl3CheckpointRouter::open(&snapshot).is_err());
}
