// SPDX-License-Identifier: AGPL-3.0-only

//! Execute the production KDA forward body with native weights and recorded
//! kernel/collective calls. Numeric validation still requires a CUDA device.

use super::*;
use crate::weight_loader::Glm5Exl3Projection;
use atlas_core::{
    config::{LayerType, ModelConfig},
    scope::ModelResource,
};
use safetensors::{Dtype, tensor::TensorView};
#[path = "exl3_kda_test_gpu.rs"]
mod backend;
use backend::Gpu;
use spark_runtime::weights::exl3::{
    Exl3LinearView,
    tp::{Exl3DeviceShard, ShardAxis, TpShardPlan},
};

struct TpComm {
    rank: usize,
    sums: Mutex<Vec<(usize, u64)>>,
}
impl spark_comm::CommBackend for TpComm {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        unreachable!()
    }
    fn all_reduce_async(&self, _: u64, bytes: usize, stream: u64) -> Result<()> {
        self.sums.lock().unwrap().push((bytes, stream));
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

fn projection(gpu: &Gpu, rank: usize, axis: ShardAxis) -> Glm5Projection {
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
    let plan = TpShardPlan::new(&view, axis, rank, 2).unwrap();
    let shard = Exl3DeviceShard::upload(gpu, &view, &plan).unwrap();
    Glm5Projection::Exl3(Glm5Exl3Projection::new(gpu, shard, 3, 77).unwrap())
}

#[test]
fn actual_kda_native_prefill_dispatches_packed_projections_and_sums_once() {
    for rank in 0..2 {
        let gpu = Gpu::new();
        let config = ModelConfig {
            model_type: "glm5_next".into(),
            hidden_size: 128,
            num_hidden_layers: 1,
            layer_types: vec![LayerType::LinearAttention],
            tp_rank: rank,
            tp_world_size: 2,
            ep_rank: 0,
            ep_world_size: 1,
            expert_tp: false,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            intermediate_size: 256,
            moe_intermediate_size: 256,
            shared_expert_intermediate_size: 256,
            num_experts: 2,
            num_experts_per_tok: 1,
            hc_mult: 4,
            hc_sinkhorn_iters: 1,
            ..ModelConfig::qwen3_next_80b_nvfp4()
        };
        let dense = |bytes| DenseWeight {
            weight: gpu.alloc(bytes).unwrap(),
        };
        let site = || HcSiteWeights {
            hc_fn: gpu.alloc(24 * 4 * 128 * 4).unwrap(),
            hc_base: gpu.alloc(24 * 4).unwrap(),
            hc_scale: gpu.alloc(12).unwrap(),
            hc_fn_bf16: DevicePtr::NULL,
            lowrank: None,
        };
        let aux_base = dense(257 * 128 * 2);
        let weights = Glm5KdaWeights {
            q_proj: projection(&gpu, rank, ShardAxis::Output),
            k_proj: projection(&gpu, rank, ShardAxis::Output),
            v_proj: projection(&gpu, rank, ShardAxis::Output),
            o_proj: projection(&gpu, rank, ShardAxis::Input),
            // One merged [B; FA; GA] base, as the EXL3 loader leaves it.
            b_proj: aux_base,
            f_a_proj: DenseWeight {
                weight: aux_base.weight.offset(128 * 2),
            },
            g_a_proj: DenseWeight {
                weight: aux_base.weight.offset(129 * 128 * 2),
            },
            f_b_proj: dense(128 * 128 * 2),
            g_b_proj: dense(128 * 128 * 2),
            conv: dense(3 * 128 * 4 * 2),
            a_log: dense(4),
            dt_bias: dense(128 * 4),
            o_norm: dense(128 * 2),
        };
        let mut layer = Glm5KdaLayer::new(
            dense(128 * 2),
            dense(128 * 2),
            weights,
            FfnComponent::None,
            HcWeights {
                attn: site(),
                ffn: site(),
                head: None,
                hc_mult: 4,
                sinkhorn_iters: 1,
                hc_eps: 1e-6,
                is_first_model_layer: true,
                is_last_model_layer: true,
            },
            0,
            &config,
            &gpu,
            None,
        )
        .unwrap();
        let arena = BufferArena::new(&config, 3, 128, 16, 3, &gpu).unwrap();
        let resources = ContextResources::new();
        let comm = TpComm {
            rank,
            sums: Mutex::new(vec![]),
        };
        let mut ctx = resources.view(&arena, &config, &gpu);
        ctx.comm = Some(&comm);
        let mut state = SsmLayerState {
            h_state: gpu.alloc(128 * 128 * 4).unwrap(),
            conv_state: gpu.alloc(3 * 128 * 4 * 2).unwrap(),
            h_is_f16: false,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: vec![],
            conv_state_intermediates: vec![],
            kda_records: DevicePtr::NULL,
            gdn_commit_qkv: DevicePtr::NULL,
            gdn_commit_gb: DevicePtr::NULL,
            gdn_commit_pending: false,
            h_prefill_stage: None,
            ple: None,
        };
        let mut cache = PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 128,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            3,
            &gpu,
        )
        .unwrap();
        let before = gpu.launch_count();
        assert!(
            layer
                .prefill(
                    arena.hidden_states(),
                    arena.hidden_states(),
                    3,
                    &mut state,
                    &mut cache,
                    0,
                    &mut vec![],
                    &mut vec![],
                    &mut vec![],
                    0,
                    &ctx,
                    78
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), before);
        layer
            .prefill(
                arena.hidden_states(),
                arena.hidden_states(),
                3,
                &mut state,
                &mut cache,
                0,
                &mut vec![],
                &mut vec![],
                &mut vec![],
                0,
                &ctx,
                77,
            )
            .unwrap();
        layer
            .decode(
                arena.hidden_states(),
                arena.hidden_states(),
                &mut state,
                &mut cache,
                3,
                &mut vec![],
                &mut vec![],
                &mut vec![],
                &ctx,
                77,
            )
            .unwrap();
        for name in [
            "exl3_input_hadamard",
            "exl3_mul1_matmul",
            "exl3_output_hadamard",
        ] {
            let handle = gpu.kernel("exl3", name).unwrap().0;
            assert_eq!(
                gpu.launches_snapshot()
                    .iter()
                    .filter(|launch| launch.func == handle)
                    .count(),
                8
            );
        }
        assert_eq!(
            *comm.sums.lock().unwrap(),
            vec![(3 * 128 * 2, 77), (128 * 2, 77)]
        );
        let aux: Vec<_> = gpu
            .bf16_gemm_requests_snapshot()
            .into_iter()
            .filter(|r| r.weight == aux_base.weight)
            .map(|r| (r.rows, r.columns, r.inner, r.numerics, r.stream))
            .collect();
        let numerics = spark_runtime::gpu::Bf16GemmNumerics::Glm53AuxiliaryClassic;
        assert_eq!(
            aux,
            [(3, 257, 128, numerics, 77), (3, 257, 128, numerics, 77)],
            "prefill M3 and padded decode M1 use the merged auxiliary GEMM"
        );
        let allocated = gpu.alloc_count();
        layer.release(&gpu).unwrap();
        // 20 projection resources plus two merged-auxiliary scratch buffers.
        assert_eq!(gpu.alloc_count(), allocated - 22);
        layer.release(&gpu).unwrap();
        let before = gpu.launch_count();
        assert!(
            layer
                .prefill(
                    arena.hidden_states(),
                    arena.hidden_states(),
                    3,
                    &mut state,
                    &mut cache,
                    0,
                    &mut vec![],
                    &mut vec![],
                    &mut vec![],
                    0,
                    &ctx,
                    77
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), before);
    }
}
