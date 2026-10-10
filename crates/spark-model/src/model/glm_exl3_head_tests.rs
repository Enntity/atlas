// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use safetensors::{Dtype, tensor::TensorView};
use spark_runtime::{gpu::mock::MockGpuBackend, weights::exl3::Exl3LinearView};
use std::sync::Mutex;

struct Comm {
    rank: usize,
    world: usize,
    calls: Mutex<Vec<(u64, u64, usize, u64)>>,
}
impl CommBackend for Comm {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world
    }
    fn all_gather_on_stream(&self, send: u64, recv: u64, bytes: usize, stream: u64) -> Result<()> {
        self.calls.lock().unwrap().push((send, recv, bytes, stream));
        Ok(())
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        unreachable!()
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        unreachable!()
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        unreachable!()
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        unreachable!()
    }
    fn barrier(&self) -> Result<()> {
        unreachable!()
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        unreachable!()
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        unreachable!()
    }
}
fn comm(rank: usize, world: usize) -> Comm {
    Comm {
        rank,
        world,
        calls: Mutex::new(vec![]),
    }
}
fn shard(gpu: &MockGpuBackend, world: usize) -> Exl3DeviceShard {
    let bytes = vec![0; 128 * 256 / 2];
    let suh = 0x3c00u16.to_le_bytes().repeat(128);
    let svh = 0x3c00u16.to_le_bytes().repeat(256);
    let marker = 0x83dc_d12du32.to_le_bytes();
    let view = Exl3LinearView::new(
        TensorView::new(Dtype::I16, vec![8, 16, 64], &bytes).unwrap(),
        TensorView::new(Dtype::F16, vec![128], &suh).unwrap(),
        TensorView::new(Dtype::F16, vec![256], &svh).unwrap(),
        TensorView::new(Dtype::I32, vec![], &marker).unwrap(),
        128,
        256,
    )
    .unwrap();
    let plan = TpShardPlan::new(&view, ShardAxis::Output, 0, world).unwrap();
    Exl3DeviceShard::upload(gpu, &view, &plan).unwrap()
}

#[test]
fn native_head_gathers_f32_full_physical_rows_once_on_compute_stream() {
    let gpu = MockGpuBackend::new();
    let mut head =
        Glm5Exl3VocabHead::new(&gpu, shard(&gpu, 2), 250, 3, 77, fixture_numerics()).unwrap();
    assert!(
        gpu.kernel_requests_snapshot()
            .iter()
            .any(|(module, name)| module == "exl3" && name == "exl3_input_hadamard_f16")
    );
    assert!(
        gpu.kernel_requests_snapshot()
            .iter()
            .any(|(module, name)| module == "residual_add" && name == "bf16_to_f32")
    );
    let tp = comm(0, 2);
    head.forward(
        &gpu,
        DevicePtr(0x2000),
        head.logits_buffer(),
        3,
        Some(&tp),
        77,
    )
    .unwrap();
    let calls = tp.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].2, 3 * 128 * 4);
    assert_eq!(calls[0].3, 77);
    assert_eq!(gpu.launch_count(), 5); // BF16 producer, F32 transport, rank-major reorder
    drop(calls);
    head.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn head_invalid_topology_stream_capacity_and_aliases_submit_no_work() {
    let gpu = MockGpuBackend::new();
    let mut head =
        Glm5Exl3VocabHead::new(&gpu, shard(&gpu, 2), 250, 3, 77, fixture_numerics()).unwrap();
    let tp = comm(0, 2);
    assert!(
        head.forward(&gpu, DevicePtr(0x2000), head.logits_buffer(), 1, None, 77)
            .is_err()
    );
    assert!(
        head.forward(
            &gpu,
            DevicePtr(0x2000),
            head.logits_buffer(),
            1,
            Some(&comm(1, 2)),
            77
        )
        .is_err()
    );
    assert!(
        head.forward(
            &gpu,
            DevicePtr(0x2000),
            head.logits_buffer(),
            1,
            Some(&tp),
            78
        )
        .is_err()
    );
    assert!(
        head.forward(
            &gpu,
            DevicePtr(0x2000),
            head.logits_buffer(),
            4,
            Some(&tp),
            77
        )
        .is_err()
    );
    assert!(
        head.forward(
            &gpu,
            head.buffers[0],
            head.logits_buffer(),
            1,
            Some(&tp),
            77
        )
        .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
    assert!(tp.calls.lock().unwrap().is_empty());
    head.forward(&gpu, DevicePtr::NULL, DevicePtr::NULL, 0, Some(&tp), 77)
        .unwrap();
    assert_eq!(gpu.launch_count(), 0);
    gpu.fail_next_free();
    assert!(head.release(&gpu).is_err());
    assert!(
        head.forward(
            &gpu,
            DevicePtr(0x2000),
            head.logits_buffer(),
            1,
            Some(&tp),
            77
        )
        .is_err()
    );
    head.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn head_rejects_active_vocabulary_outside_physical_weights_and_cleans_ownership() {
    let gpu = MockGpuBackend::new();
    assert!(Glm5Exl3VocabHead::new(&gpu, shard(&gpu, 2), 257, 3, 77, fixture_numerics()).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

fn fixture_numerics() -> Exl3LinearNumerics {
    use spark_runtime::weights::exl3::native::Exl3CooperativeProfile;
    Exl3LinearNumerics {
        logical_n: 128,
        logical_offset: 0,
        bits: 4,
        reference_min_rows: 1,
        reference_pad: false,
        row_profiles: [Some(Exl3CooperativeProfile {
            tile_k: 32,
            tile_n: 128,
            slices: 48,
            half_partial: false,
        }); 4],
    }
}

#[path = "trait_impl/prefill_stream_test_fixture.rs"]
mod fixture;

#[test]
fn native_head_resource_requires_configured_prefill_chunk_budget() {
    use crate::traits::Model;
    let mut fixture = fixture::Fixture::new(1, 1, 0);
    assert!(!fixture.model.requires_configured_prefill_chunk_budget());
    let gpu = MockGpuBackend::new();
    let head =
        Glm5Exl3VocabHead::new(&gpu, shard(&gpu, 2), 250, 3, 77, fixture_numerics()).unwrap();
    fixture.model.exl3_vocab_head = Some(head);
    let capped = fixture.model.requires_configured_prefill_chunk_budget();
    fixture
        .model
        .exl3_vocab_head
        .take()
        .unwrap()
        .release(&gpu)
        .unwrap();
    assert!(
        capped,
        "attached native EXL3 resources require the configured issued chunk geometry"
    );
    assert!(!fixture.model.requires_configured_prefill_chunk_budget());
}
