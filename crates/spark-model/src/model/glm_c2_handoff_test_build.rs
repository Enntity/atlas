// SPDX-License-Identifier: AGPL-3.0-only
//! Shared actual-head fixture construction; default paired ordering is retained.
use super::*;
use crate::traits::Model;

impl Fixture {
    pub fn new(rank: usize) -> Self {
        Self::build(rank, true, 8)
    }

    pub fn new_pair_compute(rank: usize) -> Self {
        Self::build(rank, true, 20)
    }

    pub fn new_legacy(rank: usize) -> Self {
        Self::build(rank, false, 8)
    }

    fn build(rank: usize, paired: bool, target_rows: usize) -> Self {
        assert!(rank < 2);
        let record = Arc::new(Recorder::default());
        let gpu = Box::new(Gpu(record.clone()));
        let mut cfg = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        cfg.model_type = "glm5_next".into();
        cfg.hidden_size = 4096;
        if target_rows == 20 {
            cfg.hc_mult = 4;
        }
        cfg.vocab_size = 8;
        cfg.num_hidden_layers = 1;
        cfg.layer_types = vec![atlas_core::config::LayerType::FullAttention];
        cfg.num_attention_heads = 32;
        cfg.num_key_value_heads = 1;
        cfg.head_dim = 128;
        cfg.kv_lora_rank = 512;
        cfg.qk_rope_head_dim = 0;
        cfg.index_kpool = 0;
        cfg.index_head_dim = 0;
        cfg.index_topk = 2048;
        cfg.intermediate_size = 2048;
        cfg.moe_intermediate_size = 2048;
        cfg.shared_expert_intermediate_size = 2048;
        cfg.num_experts = 2;
        cfg.num_experts_per_tok = 2;
        cfg.linear_num_key_heads = 64;
        cfg.linear_num_value_heads = 64;
        cfg.num_mtp_modules = 0;
        cfg.tp_world_size = 2;
        cfg.ep_world_size = 2;
        cfg.tp_rank = rank;
        cfg.ep_rank = rank;
        cfg.adapter_max_rank = 0;
        let dense = |bytes| DenseWeight {
            weight: gpu.alloc(bytes).unwrap(),
        };
        let embed = dense(8 * ROW_BYTES);
        let final_norm = dense(ROW_BYTES);
        let lm_head = dense(8 * ROW_BYTES);
        for token in 0..8 {
            record.write_span(
                embed.weight.offset(token * ROW_BYTES),
                &vec![token as u8; ROW_BYTES],
            );
        }
        let construct = if paired {
            Glm5MtpHead::new_paired
        } else {
            Glm5MtpHead::new
        };
        let head = Arc::new(
            construct(
                Glm5MtpModule {
                    body: Box::new(Body {
                        record: record.clone(),
                        target: false,
                    }),
                    enorm: dense(ROW_BYTES),
                    hnorm: dense(ROW_BYTES),
                    norm: dense(ROW_BYTES),
                    eh_proj: dense(4096 * 4096 * 4),
                    eh_proj_nvfp4: None,
                },
                embed,
                lm_head,
                None,
                &cfg,
                gpu.as_ref(),
                8,
                2044,
            )
            .unwrap(),
        );
        let buffers =
            spark_runtime::buffers::BufferArena::new(&cfg, target_rows, 2044, 16, 1, gpu.as_ref())
                .unwrap();
        let cache = PagedKvCache::new(
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
            256,
            gpu.as_ref(),
        )
        .unwrap();
        let mut model = TransformerModel::new(
            cfg,
            embed,
            final_norm,
            lm_head,
            None,
            None,
            None,
            vec![Box::new(Body {
                record: record.clone(),
                target: true,
            })],
            buffers,
            cache,
            vec![],
            gpu,
            2044,
            if paired { 2 } else { 1 },
            crate::layers::MtpQuantization::Bf16,
            false,
            false,
            Box::new(spark_runtime::prefix_cache::NoPrefixCaching),
            8,
            Some(Arc::new(Rank(rank, record.clone()))),
            false,
            4,
            None,
            2,
            16,
        )
        .unwrap();
        model.levers.max_decode_seqs = if paired { 2 } else { 1 };
        if target_rows == 20 {
            model.ep_protocol_v2 = true;
        }
        model.levers.drafter.prefill = true;
        model.levers.drafter.carry = false;
        model.dispatch.cublas_gemm = false;
        model.mtp_prefill_hidden = model.gpu.alloc(2044 * ROW_BYTES).unwrap();
        model.mtp_prefill_capacity = 2044;
        model.mtp_hidden_save = model.gpu.alloc(ROW_BYTES).unwrap();
        model.proposer = Some(head.clone());
        let seqs = std::array::from_fn(|slot| {
            if paired {
                let mut seq = model.alloc_sequence().unwrap();
                assert_eq!(seq.slot_idx, slot);
                seq.prompt_len = 4;
                return seq;
            }
            let mut seq = SequenceState::host_only(slot);
            seq.prompt_len = 4;
            seq.layer_states = vec![Box::new(EmptyLayerState)];
            seq.disk_last_offloaded_per_layer = vec![0];
            if slot == 0 {
                seq.proposer_state = Some(head.alloc_state(model.gpu.as_ref()).unwrap());
            }
            seq
        });
        record.clear();
        Self {
            model,
            seqs,
            head,
            gpu: record,
        }
    }
}
