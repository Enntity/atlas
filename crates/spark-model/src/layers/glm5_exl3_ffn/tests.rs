// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::layers::FfnComponent;
use safetensors::{Dtype, tensor::TensorView};
use spark_runtime::{
    buffers::BufferArena,
    weights::exl3::{Exl3LinearView, tp::TpShardPlan},
};
use std::sync::Mutex;
#[allow(clippy::duplicate_mod)]
#[path = "../moe/exl3_kda_test_gpu.rs"]
mod backend;
use backend::Gpu;
struct Comm(Mutex<Vec<(usize, u64)>>);
impl spark_comm::CommBackend for Comm {
    fn rank(&self) -> usize {
        1
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        unreachable!()
    }
    fn all_reduce_async(&self, _: u64, n: usize, s: u64) -> Result<()> {
        self.0.lock().unwrap().push((n, s));
        Ok(())
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
fn shard(gpu: &Gpu, axis: ShardAxis) -> Exl3DeviceShard {
    shard_with_topology(gpu, axis, 1, 2)
}
fn shard_with_topology(gpu: &Gpu, axis: ShardAxis, rank: usize, world: usize) -> Exl3DeviceShard {
    let (k, n) = if axis == ShardAxis::Output {
        (128, 256)
    } else {
        (256, 128)
    };
    let packed = vec![0; k * n / 2];
    let suh = 0x3c00u16.to_le_bytes().repeat(k);
    let svh = 0x3c00u16.to_le_bytes().repeat(n);
    let marker = 0x83dc_d12du32.to_le_bytes();
    let view = Exl3LinearView::new(
        TensorView::new(Dtype::I16, vec![k / 16, n / 16, 64], &packed).unwrap(),
        TensorView::new(Dtype::F16, vec![k], &suh).unwrap(),
        TensorView::new(Dtype::F16, vec![n], &svh).unwrap(),
        TensorView::new(Dtype::I32, vec![], &marker).unwrap(),
        k,
        n,
    )
    .unwrap();
    let plan = TpShardPlan::new(
        &view,
        if world == 1 {
            ShardAxis::Replicated
        } else {
            axis
        },
        rank,
        world,
    )
    .unwrap();
    Exl3DeviceShard::upload(gpu, &view, &plan).unwrap()
}
#[test]
fn actual_component_prefill_has_one_tp_sum_and_preflight_rejects_missing_comm() {
    for routed in [false, true] {
        let gpu = Gpu::new();
        let config = ModelConfig {
            scoring_func: "sigmoid".into(),
            moe_expert_groups: Some(1),
            moe_topk_groups: Some(1),
            routed_scaling_factor: 2.5,
            hidden_size: 128,
            intermediate_size: 256,
            moe_intermediate_size: 256,
            num_experts: 2,
            num_experts_per_tok: 1,
            vocab_size: 128,
            tp_rank: 1,
            tp_world_size: 2,
            ep_rank: 0,
            ep_world_size: 1,
            expert_tp: false,
            ..ModelConfig::qwen3_next_80b_nvfp4()
        };
        let activation = NativeDenseActivation::PreSiLuClampedBf16 { limit: 10.0 };
        let dense = if routed {
            let weights = [
                shard_with_topology(&gpu, ShardAxis::Output, 0, 1),
                shard_with_topology(&gpu, ShardAxis::Output, 0, 1),
                shard_with_topology(&gpu, ShardAxis::Input, 0, 1),
            ];
            NativeDenseFfn::from_shared(
                &gpu,
                shared::SharedBf16Weights::from_global(&gpu, weights, 1, 2, 77).unwrap(),
                3,
                77,
                activation,
            )
            .unwrap()
        } else {
            NativeDenseFfn::new(
                &gpu,
                [
                    shard(&gpu, ShardAxis::Output),
                    shard(&gpu, ShardAxis::Output),
                    shard(&gpu, ShardAxis::Input),
                ],
                3,
                77,
                activation,
                NativeDenseFfnRole::DensePacked,
            )
            .unwrap()
        };
        let native = if routed {
            use spark_runtime::weights::exl3::dense::{Exl3DenseDeviceTensor, Exl3DenseDtype};
            let tables = [ShardAxis::Output, ShardAxis::Output, ShardAxis::Input]
                .into_iter()
                .map(|axis| {
                    tables::ExpertTable::upload(&gpu, (0..2).map(|_| shard(&gpu, axis)).collect())
                        .unwrap()
                })
                .collect();
            let gate = vec![0; 2 * 128 * 2];
            let bias = vec![0; 2 * 4];
            let companions = [(vec![2, 128], gate), (vec![2], bias)]
                .into_iter()
                .map(|(shape, data)| {
                    let rows = shape[0];
                    let router = shape.len() == 2;
                    let view =
                        TensorView::new(if router { Dtype::F16 } else { Dtype::F32 }, shape, &data)
                            .unwrap();
                    Exl3DenseDeviceTensor::upload(
                        &gpu,
                        &view,
                        if router {
                            Exl3DenseDtype::Bf16
                        } else {
                            Exl3DenseDtype::F32
                        },
                        &[0..rows],
                    )
                    .unwrap()
                })
                .collect();
            Glm5Exl3FfnLayer::Moe(Box::new(
                NativeMoeFfn::new(
                    &gpu,
                    &config,
                    dense,
                    tables,
                    companions,
                    3,
                    77,
                    Exl3FfnActivation::ClampedSiLu { limit: 10.0 },
                    super::moe::RoutedPrecisionPolicy::GLM53_TP2_REFERENCE,
                    None,
                )
                .unwrap(),
            ))
        } else {
            Glm5Exl3FfnLayer::Dense(Box::new(dense))
        };
        let mut layer = FfnComponent::Exl3(Box::new(native));
        let layer_allocations = gpu.alloc_count();
        let arena = BufferArena::new(&config, 3, 128, 16, 3, &gpu).unwrap();
        let arena_allocations = gpu.alloc_count() - layer_allocations;
        let dispatch = crate::layers::ops::GemmDispatch::defaults();
        let derived = crate::layers::ops::DerivedWeights::new();
        let levers = crate::layers::ops::ModelLevers::defaults();
        let stats = crate::layers::ops::ModelStats::new();
        let comm = Comm(Mutex::new(vec![]));
        let mut ctx = ForwardContext {
            buffers: &arena,
            gpu: &gpu,
            config: &config,
            dispatch: &dispatch,
            derived: &derived,
            levers: &levers,
            stats: &stats,
            ssm_batch: None,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
            request_id: None,
        };
        let before = gpu.launch_count();
        let before_sync = gpu.sync_count();
        assert!(
            layer
                .forward_prefill(arena.norm_output(), 3, &ctx, 77)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), before);
        ctx.comm = Some(&comm);
        assert!(
            layer
                .forward_prefill(arena.norm_output(), 3, &ctx, 78)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), before);
        layer
            .forward_prefill(arena.norm_output(), 3, &ctx, 77)
            .unwrap();
        assert_eq!(gpu.launch_count() - before, if routed { 14 } else { 10 });
        assert_eq!(gpu.sync_count(), before_sync);
        let router_gemms = gpu.bf16_gemm_requests_snapshot();
        if routed {
            assert_eq!(router_gemms.len(), 3); // router, merged shared gate/up, shared down
            assert_eq!(
                router_gemms[0].numerics,
                spark_runtime::gpu::Bf16GemmNumerics::CublasLt
            );
            assert!(
                router_gemms[1..]
                    .iter()
                    .all(|r| { r.numerics == spark_runtime::gpu::Bf16GemmNumerics::Glm53Classic })
            );
        } else {
            assert!(router_gemms.is_empty()); // packed dense projections use native EXL3
        }
        assert_eq!(*comm.0.lock().unwrap(), vec![(3 * 128 * 2, 77)]);
        layer
            .forward_prefill(arena.norm_output(), 0, &ctx, 77)
            .unwrap();
        assert_eq!(comm.0.lock().unwrap().len(), 1);
        assert_eq!(gpu.d2h_blocking_count(), 0);
        assert_eq!(gpu.d2h_async_count(), 0);
        let directory = std::env::temp_dir().join(format!(
            "atlas-native-ffn-{}-{routed}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("capture-enabled"), b"").unwrap();
        let capture = diagnostics::Exl3FfnRouter::new(directory.clone(), 3, 1, 2).unwrap();
        let FfnComponent::Exl3(native) = &mut layer else {
            unreachable!()
        };
        match native.as_mut() {
            Glm5Exl3FfnLayer::Dense(d) => d.capture = Some(capture),
            Glm5Exl3FfnLayer::Moe(m) => m.capture = Some(capture),
        }
        let input_bytes = 0x3f80u16.to_le_bytes().repeat(3 * 128);
        gpu.copy_h2d(&input_bytes, arena.norm_output()).unwrap();
        ctx.graph_capture = true;
        let before = gpu.launch_count();
        assert!(
            layer
                .forward_prefill(arena.norm_output(), 3, &ctx, 77)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), before);
        ctx.graph_capture = false;
        layer
            .forward_prefill(arena.norm_output(), 3, &ctx, 77)
            .unwrap();
        assert_eq!(comm.0.lock().unwrap().len(), 2);
        assert_eq!(
            std::fs::read(directory.join("ffn-L3-rank1-000000-input-input.bin")).unwrap(),
            input_bytes
        );
        for stage in if routed {
            vec![
                "input",
                "router",
                "gate_up",
                "down_input_hadamard",
                "down_shared",
                "local_output",
                "output_after_tp",
            ]
        } else {
            vec![
                "input",
                "gate_up",
                "activation",
                "local_output",
                "output_after_tp",
            ]
        } {
            let metadata: serde_json::Value = serde_json::from_slice(
                &std::fs::read(directory.join(format!("ffn-L3-rank1-000000-{stage}.json")))
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(metadata["tp_rank"], 1);
            assert_eq!(metadata["stream"], 77);
            if routed && ["gate_up", "down_input_hadamard"].contains(&stage) {
                for tensor in metadata["tensors"].as_array().unwrap() {
                    assert_eq!(tensor["dtype"], "f32-le");
                }
            }
            if routed && stage == "down_shared" {
                assert_eq!(metadata["tensors"][0]["name"], "down_weighted");
                assert_eq!(metadata["tensors"][0]["dtype"], "f32-le");
                assert_eq!(metadata["tensors"][1]["dtype"], "bf16-le");
            }
        }
        if routed {
            let launches = gpu.launch_count();
            let reductions = comm.0.lock().unwrap().clone();
            gpu.fail_next_bf16_gemm();
            assert!(
                layer
                    .forward_prefill(arena.norm_output(), 3, &ctx, 77)
                    .is_err()
            );
            assert_eq!(gpu.launch_count(), launches);
            assert_eq!(*comm.0.lock().unwrap(), reductions);
        }
        std::fs::remove_dir_all(directory).unwrap();
        let (launches, syncs, reductions) = (
            gpu.launch_count(),
            gpu.sync_count(),
            comm.0.lock().unwrap().len(),
        );
        let reads = (gpu.d2h_blocking_count(), gpu.d2h_async_count());
        ctx.profile = true;
        ctx.graph_capture = true;
        assert!(
            layer
                .forward_prefill(arena.norm_output(), 3, &ctx, 77)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), launches);
        assert_eq!(gpu.sync_count(), syncs);
        ctx.graph_capture = false;
        layer
            .forward_prefill(arena.norm_output(), 3, &ctx, 77)
            .unwrap();
        assert_eq!(gpu.launch_count() - launches, if routed { 14 } else { 10 });
        assert_eq!(gpu.sync_count() - syncs, if routed { 26 } else { 10 });
        assert_eq!(comm.0.lock().unwrap().len(), reductions + 1);
        assert_eq!((gpu.d2h_blocking_count(), gpu.d2h_async_count()), reads);
        let syncs = gpu.sync_count();
        layer
            .forward_prefill(arena.norm_output(), 0, &ctx, 77)
            .unwrap();
        assert_eq!(gpu.sync_count(), syncs);
        gpu.fail_next_free();
        assert!(layer.release_exl3(&gpu).is_err());
        assert_eq!(gpu.alloc_count(), arena_allocations + 1);
        assert!(
            layer
                .forward_prefill(arena.norm_output(), 3, &ctx, 77)
                .is_err()
        );
        layer.release_exl3(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), arena_allocations);
    }
}
#[test]
fn expert_table_abi_preserves_local_shards_without_expert_workspaces() {
    let gpu = Gpu::new();
    let shards = (0..2).map(|_| shard(&gpu, ShardAxis::Output)).collect();
    let mut table = tables::ExpertTable::upload(&gpu, shards).unwrap();
    table
        .validate(2, 1, 2, 128, 128, ShardAxis::Output)
        .unwrap();
    let mut bytes = vec![0; 64];
    gpu.copy_d2h(table.ptr, &mut bytes).unwrap();
    for (i, shard) in table.shards.iter().enumerate() {
        let desc = &bytes[i * 32..(i + 1) * 32];
        for (j, ptr) in shard.buffers().unwrap().iter().enumerate() {
            assert_eq!(&desc[j * 8..j * 8 + 8], &ptr.0.to_le_bytes());
        }
        assert_eq!(&desc[24..28], &4u32.to_le_bytes());
        assert_eq!(&desc[28..32], &[0; 4]);
    }
    table.release(&gpu).unwrap();
    assert!(
        table
            .validate(2, 1, 2, 128, 128, ShardAxis::Output)
            .is_err()
    );
    table.release(&gpu).unwrap();
}

#[test]
fn dense_and_shared_policies_require_explicit_input_rounding_and_bf16_activation_boundary() {
    use spark_runtime::weights::exl3::native::Exl3InputPrecision;
    assert_eq!(
        NativeDenseFfnRole::DensePacked
            .packed_input_precision()
            .unwrap(),
        Exl3InputPrecision::F16
    );
    assert!(
        NativeDenseFfnRole::SharedExperts
            .packed_input_precision()
            .is_err()
    );
    assert_eq!(
        NativeDenseActivation::from_text_policy(Exl3FfnActivation::SiLu).unwrap(),
        NativeDenseActivation::SiLuBf16
    );
    assert!(
        NativeDenseActivation::from_text_policy(Exl3FfnActivation::ClampedSiLu { limit: f32::NAN })
            .is_err()
    );
}
