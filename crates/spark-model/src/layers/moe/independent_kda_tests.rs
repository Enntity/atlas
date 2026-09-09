// SPDX-License-Identifier: AGPL-3.0-only
//! Real KDA constructor and independent entry; backend records launches only.
use super::*;
use crate::layer::{LayerState, SsmLayerState, TransformerLayer};
use crate::layers::glm5_kda::{Glm5KdaLayer, Glm5KdaWeights, Glm5Projection};
use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights};
use crate::weight_map::{DenseWeight, QuantizedWeight};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

pub(super) fn actual_kda_rows() {
    for rank in 0..2 {
        let gpu = Gpu::new();
        let (_store, mut config, _) = resident_tests::setup(&gpu, rank);
        config.num_hidden_layers = 1;
        config.layer_types = vec![atlas_core::config::LayerType::LinearAttention];
        config.linear_num_key_heads = 32;
        config.linear_num_value_heads = 32;
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;
        config.linear_conv_kernel_dim = 4;
        config.kda_gate_lower_bound = -5.0;
        config.num_attention_heads = 32;
        config.q_lora_rank = 1536;
        config.kv_lora_rank = 512;
        config.qk_nope_head_dim = 256;
        config.qk_rope_head_dim = 0;
        config.v_head_dim = 256;
        config.index_topk = 2048;
        config.hc_mult = 4;
        let dense = |bytes| DenseWeight {
            weight: gpu.alloc(bytes).unwrap(),
        };
        let projection = || Glm5Projection {
            nvfp4: QuantizedWeight {
                weight: gpu.alloc(4096 * 4096 / 2).unwrap(),
                weight_scale: gpu.alloc(4096 * 4096 / 16).unwrap(),
                weight_scale_2: 1.0,
                input_scale: DevicePtr::NULL,
                weight_scale_2_vec: DevicePtr::NULL,
            },
            prefill_nvfp4_t: None,
        };
        let site = || HcSiteWeights {
            hc_fn: gpu.alloc(24 * 4 * 4096 * 4).unwrap(),
            hc_base: gpu.alloc(24 * 4).unwrap(),
            hc_scale: gpu.alloc(12).unwrap(),
        };
        let weights = Glm5KdaWeights {
            q_proj: projection(),
            k_proj: projection(),
            v_proj: projection(),
            o_proj: projection(),
            b_proj: dense(32 * 4096 * 2),
            f_a_proj: dense(128 * 4096 * 2),
            f_b_proj: dense(4096 * 128 * 2),
            g_a_proj: dense(128 * 4096 * 2),
            g_b_proj: dense(4096 * 128 * 2),
            conv: dense(12288 * 4 * 2),
            a_log: dense(32 * 4),
            dt_bias: dense(4096 * 4),
            o_norm: dense(4096 * 2),
        };
        let layer = Glm5KdaLayer::new(
            dense(8192),
            dense(8192),
            weights,
            FfnComponent::None,
            HcWeights {
                attn: site(),
                ffn: site(),
                head: None,
                hc_mult: 4,
                sinkhorn_iters: 1,
                hc_eps: 1e-6,
            },
            0,
            &config,
            &gpu,
        )
        .unwrap();
        let arena = BufferArena::new(&config, 8, 2048, 16, 8, &gpu).unwrap();
        let h = [gpu.alloc(8 * 2097152).unwrap()];
        let conv = [gpu.alloc(8 * 196608).unwrap()];
        let metadata = gpu.alloc(512).unwrap();
        let pool =
            crate::layer::ssm_batch::SsmPoolView::new(&h, &conv, 2097152, 2097152, 196608, 8)
                .unwrap();
        let ids = [7, 0, 6, 1, 5, 2, 4, 3];
        let resources = ContextResources::new();
        let mut levers = ops::ModelLevers::defaults();
        levers.max_decode_seqs = 8;
        let comm = Comm {
            gpu: &gpu,
            rank,
            reductions: Mutex::new(vec![]),
        };
        let conv_kernel = gpu
            .kernel("causal_conv1d", "glm_kda_conv_indexed")
            .unwrap()
            .0;
        let recurrent = gpu.kernel("kda", "glm_kda_recurrent_indexed").unwrap().0;
        let mut cache = PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 512,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            8,
            &gpu,
        )
        .unwrap();
        for rows in 2..=8 {
            let mut states: Vec<Box<dyn LayerState>> = ids[..rows]
                .iter()
                .map(|&i| {
                    Box::new(SsmLayerState {
                        h_state: h[0].offset(i as usize * 2097152),
                        conv_state: conv[0].offset(i as usize * 196608),
                        h_is_f16: false,
                        h_state_checkpoint: None,
                        conv_state_checkpoint: None,
                        h_state_intermediates: vec![],
                        conv_state_intermediates: vec![],
                        h_prefill_stage: None,
                    }) as Box<dyn LayerState>
                })
                .collect();
            let mut refs: Vec<&mut dyn LayerState> = states
                .iter_mut()
                .map(|s| s.as_mut() as &mut dyn LayerState)
                .collect();
            let mut ctx = resources.view(&arena, &config, &gpu);
            ctx.levers = &levers;
            ctx.comm = Some(&comm);
            ctx.ssm_batch = Some(
                crate::layer::ssm_batch::SsmBatchView::new(
                    pool,
                    metadata.offset(256),
                    &ids[..rows],
                )
                .unwrap(),
            );
            ctx.attn_metadata = Some(AttnMetadataDev {
                positions: metadata,
                positions_h: metadata,
                positions_w: metadata,
                slot: metadata.offset(32),
                seq_len: metadata.offset(64),
                block_table: metadata.offset(96),
                max_blocks_per_seq: 1,
                num_seqs: rows as u32,
                seq_slot: metadata,
                moe_row_adapter: DevicePtr::NULL,
            });
            let positions: Vec<_> = (0..rows).map(|i| i * 17).collect();
            let blocks = vec![vec![]; rows];
            gpu.clear();
            layer
                .decode_multi_seq(
                    arena.hidden_states(),
                    arena.hidden_states(),
                    rows,
                    &mut refs,
                    &mut cache,
                    &positions,
                    &blocks,
                    &ctx,
                    91,
                )
                .unwrap();
            let trace = gpu.trace();
            for (kernel, grid) in [
                (conv_kernel, [48, rows as u32, 1]),
                (recurrent, [32, rows as u32, 1]),
            ] {
                let calls: Vec<_> = trace
                    .iter()
                    .filter_map(|e| {
                        if let Event::Launch(k, g, _, _, _, args) = e {
                            (*k == kernel).then_some((*g, args))
                        } else {
                            None
                        }
                    })
                    .collect();
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].0, grid);
                assert!(calls[0].1.contains(&Arg::Ptr(metadata.offset(256))));
            }
            assert!(
                !trace
                    .iter()
                    .any(|e| matches!(e, Event::Alloc(..) | Event::Free(..) | Event::Read(..)))
            );
        }
    }
}
