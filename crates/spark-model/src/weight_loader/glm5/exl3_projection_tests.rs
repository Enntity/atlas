// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use safetensors::{Dtype, tensor::TensorView};
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::exl3::{Exl3LinearView, tp::TpShardPlan};
use std::sync::Mutex;

struct Comm {
    rank: usize,
    world: usize,
    fail: bool,
    calls: Mutex<Vec<(u64, usize, u64)>>,
}

impl CommBackend for Comm {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world
    }
    fn all_reduce_async(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        self.calls.lock().unwrap().push((ptr, bytes, stream));
        ensure!(!self.fail, "collective failed");
        Ok(())
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        unreachable!()
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
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

fn projection(gpu: &MockGpuBackend, axis: ShardAxis) -> Glm5Exl3Projection {
    let trellis = vec![0; 256 * 256 / 2];
    let scale = 0x3c00u16.to_le_bytes().repeat(256);
    let marker = 0x83dc_d12du32.to_le_bytes();
    let view = Exl3LinearView::new(
        TensorView::new(Dtype::I16, vec![16, 16, 64], &trellis).unwrap(),
        TensorView::new(Dtype::F16, vec![256], &scale).unwrap(),
        TensorView::new(Dtype::F16, vec![256], &scale).unwrap(),
        TensorView::new(Dtype::I32, vec![], &marker).unwrap(),
        256,
        256,
    )
    .unwrap();
    let plan = TpShardPlan::new(&view, axis, 1, 2).unwrap();
    let shard = Exl3DeviceShard::upload(gpu, &view, &plan).unwrap();
    Glm5Exl3Projection::new(gpu, shard, 3, 77).unwrap()
}

fn comm(rank: usize, world: usize, fail: bool) -> Comm {
    Comm {
        rank,
        world,
        fail,
        calls: Mutex::new(vec![]),
    }
}

#[test]
fn row_parallel_validates_topology_before_launch_and_reduces_once() {
    let gpu = MockGpuBackend::new();
    let mut projection = projection(&gpu, ShardAxis::Input);
    for wrong in [comm(0, 2, false), comm(1, 3, false)] {
        assert!(
            projection
                .forward_bf16(&gpu, DevicePtr(123), DevicePtr(456), 3, Some(&wrong))
                .is_err()
        );
        assert!(wrong.calls.lock().unwrap().is_empty());
    }
    assert!(
        projection
            .forward_bf16(&gpu, DevicePtr(123), DevicePtr(456), 3, None)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
    let correct = comm(1, 2, false);
    assert_eq!(
        projection
            .forward_bf16(&gpu, DevicePtr(123), DevicePtr(456), 3, Some(&correct))
            .unwrap(),
        Exl3OutputSemantics::FullChannels
    );
    assert_eq!(gpu.launch_count(), 3);
    assert_eq!(*correct.calls.lock().unwrap(), vec![(456, 3 * 256 * 2, 77)]);
    projection.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn empty_rows_and_non_reducing_axes_do_not_enter_collectives() {
    for (axis, semantics) in [
        (ShardAxis::Input, Exl3OutputSemantics::FullChannels),
        (ShardAxis::Output, Exl3OutputSemantics::LocalChannels),
        (ShardAxis::Replicated, Exl3OutputSemantics::FullChannels),
    ] {
        let gpu = MockGpuBackend::new();
        let mut projection = projection(&gpu, axis);
        let correct = comm(1, 2, false);
        assert_eq!(
            projection
                .forward_bf16(&gpu, DevicePtr::NULL, DevicePtr::NULL, 0, Some(&correct))
                .unwrap(),
            semantics
        );
        assert_eq!(gpu.launch_count(), 0);
        assert!(correct.calls.lock().unwrap().is_empty());
        if axis != ShardAxis::Input {
            assert_eq!(
                projection
                    .forward_bf16(&gpu, DevicePtr(123), DevicePtr(456), 1, None)
                    .unwrap(),
                semantics
            );
        }
        projection.release(&gpu).unwrap();
    }
}

#[test]
fn collective_failure_propagates_and_teardown_retries_failed_frees() {
    let gpu = MockGpuBackend::new();
    let mut projection = projection(&gpu, ShardAxis::Input);
    let failing = comm(1, 2, true);
    let error = projection
        .forward_bf16(&gpu, DevicePtr(123), DevicePtr(456), 1, Some(&failing))
        .unwrap_err();
    assert!(format!("{error:#}").contains("collective failed"));
    assert_eq!(failing.calls.lock().unwrap().len(), 1);
    gpu.fail_next_free();
    assert!(projection.release(&gpu).is_err());
    assert_eq!(gpu.alloc_count(), 1);
    projection.release(&gpu).unwrap();
    projection.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn local_forward_validates_geometry_stream_and_topology_without_reducing() {
    let gpu = MockGpuBackend::new();
    let mut projection = projection(&gpu, ShardAxis::Input);
    for (k, n, stream, rank, world) in [
        (256, 256, 77, 1, 2),
        (128, 128, 77, 1, 2),
        (128, 256, 78, 1, 2),
        (128, 256, 77, 0, 2),
        (128, 256, 77, 1, 3),
    ] {
        assert!(
            projection
                .validate_local_forward(3, k, n, stream, rank, world)
                .is_err()
        );
    }
    projection
        .validate_local_forward(3, 128, 256, 77, 1, 2)
        .unwrap();
    projection
        .launch_local_bf16(&gpu, DevicePtr(123), DevicePtr(456), 3)
        .unwrap();
    assert_eq!(gpu.launch_count(), 3);
    projection.release(&gpu).unwrap();
    assert!(
        projection
            .launch_local_bf16(&gpu, DevicePtr(123), DevicePtr(456), 1)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 3);
}
