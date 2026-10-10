// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::scope::ModelResource;
use safetensors::{Dtype, tensor::TensorView};
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::exl3::{Exl3LinearView, tp::TpShardPlan};
use std::sync::Mutex;

struct Comm(Mutex<Vec<(u64, usize, u64)>>);
impl CommBackend for Comm {
    fn rank(&self) -> usize {
        1
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        panic!("async expected")
    }
    fn all_reduce_async(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        self.0.lock().unwrap().push((ptr, bytes, stream));
        Ok(())
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        panic!("unexpected")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        panic!("unexpected")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        panic!("unexpected")
    }
    fn barrier(&self) -> Result<()> {
        panic!("unexpected")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        panic!("unexpected")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        panic!("unexpected")
    }
}

fn projection(
    gpu: &MockGpuBackend,
    k: usize,
    n: usize,
    axis: ShardAxis,
    stream: u64,
) -> Glm5Exl3Projection {
    let trellis = vec![0; k / 16 * (n / 16) * 128];
    let input: Vec<u8> = (0..k).flat_map(|_| 0x3c00u16.to_le_bytes()).collect();
    let output: Vec<u8> = (0..n).flat_map(|_| 0x3c00u16.to_le_bytes()).collect();
    let marker = 0x83dc_d12du32.to_le_bytes();
    let view = Exl3LinearView::new(
        TensorView::new(Dtype::I16, vec![k / 16, n / 16, 64], &trellis).unwrap(),
        TensorView::new(Dtype::F16, vec![k], &input).unwrap(),
        TensorView::new(Dtype::F16, vec![n], &output).unwrap(),
        TensorView::new(Dtype::I32, vec![], &marker).unwrap(),
        k,
        n,
    )
    .unwrap();
    let plan = TpShardPlan::new(&view, axis, 1, 2).unwrap();
    let shard =
        spark_runtime::weights::exl3::tp::Exl3DeviceShard::upload(gpu, &view, &plan).unwrap();
    Glm5Exl3Projection::new(gpu, shard, 4, stream).unwrap()
}

fn buffers() -> Exl3FfnBuffers {
    Exl3FfnBuffers {
        input: DevicePtr(100),
        gate: DevicePtr(200),
        up: DevicePtr(300),
        output: DevicePtr(400),
    }
}

#[test]
fn ffn_launches_local_gate_up_activation_then_reduces_down_once() {
    let gpu = MockGpuBackend::new();
    let mut gate = projection(&gpu, 256, 512, ShardAxis::Output, 77);
    let mut up = projection(&gpu, 256, 512, ShardAxis::Output, 77);
    let mut down = projection(&gpu, 512, 256, ShardAxis::Input, 77);
    let ffn = Glm5Exl3Ffn::new(
        &gpu,
        &gate,
        &up,
        &down,
        Exl3FfnActivation::ClampedSiLu { limit: 10.0 },
    )
    .unwrap();
    let comm = Comm(Mutex::new(vec![]));
    assert_eq!(
        ffn.forward_bf16(&gpu, buffers(), 3, Some(&comm)).unwrap(),
        Exl3OutputSemantics::FullChannels
    );
    assert_eq!(gpu.launch_count(), 10);
    assert_eq!(*comm.0.lock().unwrap(), vec![(400, 3 * 256 * 2, 77)]);
    let launches = gpu.launches_snapshot();
    assert_eq!(launches[6].grid, [3, 1, 1]);
    assert_eq!(launches[6].block, [256, 1, 1]);
    gate.release(&gpu).unwrap();
    up.release(&gpu).unwrap();
    down.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn ffn_preflight_rejects_missing_comm_rows_and_aliases_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let mut gate = projection(&gpu, 256, 512, ShardAxis::Output, 77);
    let mut up = projection(&gpu, 256, 512, ShardAxis::Output, 77);
    let mut down = projection(&gpu, 512, 256, ShardAxis::Input, 77);
    let ffn = Glm5Exl3Ffn::new(&gpu, &gate, &up, &down, Exl3FfnActivation::SiLu).unwrap();
    let comm = Comm(Mutex::new(vec![]));
    assert!(ffn.forward_bf16(&gpu, buffers(), 3, None).is_err());
    assert!(ffn.forward_bf16(&gpu, buffers(), 5, Some(&comm)).is_err());
    let mut aliases = buffers();
    aliases.up = aliases.gate;
    assert!(ffn.forward_bf16(&gpu, aliases, 3, Some(&comm)).is_err());
    assert_eq!(gpu.launch_count(), 0);
    assert_eq!(
        ffn.forward_bf16(&gpu, buffers(), 0, Some(&comm)).unwrap(),
        Exl3OutputSemantics::FullChannels
    );
    assert_eq!(gpu.launch_count(), 0);
    assert!(comm.0.lock().unwrap().is_empty());
    gate.release(&gpu).unwrap();
    up.release(&gpu).unwrap();
    down.release(&gpu).unwrap();
}

#[test]
fn ffn_requires_correct_ownership_shapes_streams_and_explicit_valid_activation() {
    let gpu = MockGpuBackend::new();
    let mut gate = projection(&gpu, 256, 512, ShardAxis::Output, 77);
    let mut up = projection(&gpu, 256, 512, ShardAxis::Output, 77);
    let mut down = projection(&gpu, 512, 256, ShardAxis::Input, 77);
    for limit in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            Glm5Exl3Ffn::new(
                &gpu,
                &gate,
                &up,
                &down,
                Exl3FfnActivation::ClampedSiLu { limit }
            )
            .is_err()
        );
    }
    let mut wrong_shape = projection(&gpu, 256, 256, ShardAxis::Output, 77);
    assert!(Glm5Exl3Ffn::new(&gpu, &gate, &wrong_shape, &down, Exl3FfnActivation::SiLu).is_err());
    let mut wrong_stream = projection(&gpu, 256, 512, ShardAxis::Output, 78);
    assert!(Glm5Exl3Ffn::new(&gpu, &gate, &wrong_stream, &down, Exl3FfnActivation::SiLu).is_err());
    let mut wrong_axis = projection(&gpu, 512, 256, ShardAxis::Replicated, 77);
    assert!(Glm5Exl3Ffn::new(&gpu, &gate, &up, &wrong_axis, Exl3FfnActivation::SiLu).is_err());
    assert_eq!(gpu.launch_count(), 0);
    for projection in [
        &mut gate,
        &mut up,
        &mut down,
        &mut wrong_shape,
        &mut wrong_stream,
        &mut wrong_axis,
    ] {
        projection.release(&gpu).unwrap();
    }
}

#[test]
fn ffn_activation_uses_explicit_glm_text_policy() {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    assert_eq!(
        Exl3FfnActivation::from_glm_config(&config).unwrap(),
        Exl3FfnActivation::SiLu
    );
    config.text_swiglu_limit = Some(10.0);
    assert_eq!(
        Exl3FfnActivation::from_glm_config(&config).unwrap(),
        Exl3FfnActivation::ClampedSiLu { limit: 10.0 }
    );
}

#[test]
fn ffn_activation_rejects_wrong_architecture_and_invalid_manual_clamps() {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    assert!(Exl3FfnActivation::from_glm_config(&config).is_err());
    config.model_type = "glm5_next".into();
    for limit in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        config.text_swiglu_limit = Some(limit);
        assert!(
            Exl3FfnActivation::from_glm_config(&config).is_err(),
            "{limit}"
        );
    }
}
