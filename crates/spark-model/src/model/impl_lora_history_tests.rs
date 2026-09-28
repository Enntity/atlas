// SPDX-License-Identifier: AGPL-3.0-only
//! Actual constructor/setter tests; no production-specific test shortcuts.
use super::*;
use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use crate::lora::{AdapterSlot, LoraLayerWeights, LoraWeights};
use crate::weight_map::DenseWeight;
use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig, PeftAdapterConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use std::sync::atomic::{AtomicU64, AtomicUsize};

struct Unadaptable;
impl TransformerLayer for Unadaptable {
    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }
    fn decode(
        &self,
        _: DevicePtr,
        _: DevicePtr,
        _: &mut dyn LayerState,
        _: &mut PagedKvCache,
        _: usize,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &ForwardContext,
        _: u64,
    ) -> Result<()> {
        unreachable!("setter-only fixture never decodes")
    }
}

fn model() -> TransformerModel {
    let gpu = Box::new(MockGpuBackend::new());
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.model_type = "qwen2".into();
    cfg.hidden_size = 128;
    cfg.vocab_size = 8;
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![LayerType::FullAttention];
    cfg.num_attention_heads = 1;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 128;
    cfg.intermediate_size = 128;
    cfg.moe_intermediate_size = 128;
    cfg.shared_expert_intermediate_size = 128;
    cfg.num_experts = 2;
    cfg.num_experts_per_tok = 1;
    cfg.linear_num_key_heads = 0;
    cfg.linear_num_value_heads = 0;
    cfg.num_mtp_modules = 0;
    let buffers =
        spark_runtime::buffers::BufferArena::new(&cfg, 5, 64, 16, 1, gpu.as_ref()).unwrap();
    let kv = PagedKvCache::new(
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
        4,
        gpu.as_ref(),
    )
    .unwrap();
    let dense = DenseWeight {
        weight: gpu.alloc(8192).unwrap(),
    };
    TransformerModel::new(
        cfg,
        dense,
        dense,
        dense,
        None,
        None,
        None,
        vec![Box::new(Unadaptable)],
        buffers,
        kv,
        vec![],
        gpu,
        64,
        1,
        crate::layers::MtpQuantization::Bf16,
        false,
        false,
        Box::new(spark_runtime::prefix_cache::NoPrefixCaching),
        8,
        None,
        false,
        4,
        None,
        1,
        16,
    )
    .unwrap()
}

fn adapter(fail_install: bool) -> LoraWeights {
    let config = PeftAdapterConfig {
        r: 4,
        lora_alpha: 8.0,
        target_modules: vec!["k_proj".into()],
        use_rslora: false,
        layers_to_transform: None,
        trainable_token_indices: vec![],
        modules_to_save: vec![],
        lora_embedding: false,
    };
    LoraWeights {
        name: "history".into(),
        adapter_config: config.clone(),
        max_rank: 4,
        max_loras: 1,
        pool: DevicePtr::NULL,
        pool_bytes: 0,
        expert_pool: None,
        expert_pool_bytes: 0,
        slots: vec![AdapterSlot {
            name: "history".into(),
            adapter_config: config,
            layers: if fail_install {
                vec![Some(LoraLayerWeights::empty(0))]
            } else {
                vec![]
            },
            generation: 0,
        }],
        active: 0,
        tables: Default::default(),
        scale_table: DevicePtr::NULL,
        ref_counts: vec![AtomicUsize::new(0)],
        pinned: 1,
        last_used: vec![AtomicU64::new(0)],
        lru_tick: AtomicU64::new(0),
        overlay_raw: vec![],
    }
}

#[test]
fn actual_lora_setter_records_success_and_failure_before_none_detach() {
    for fail_install in [false, true] {
        let mut model = model();
        assert!(!model.lora_install_attempted);
        model
            .hidden_trace_adapter_ownership()
            .ensure_absent()
            .unwrap();
        model.set_lora_weights(None).unwrap();
        assert!(!model.lora_install_attempted);
        let result = model.set_lora_weights(Some(adapter(fail_install)));
        assert_eq!(result.is_err(), fail_install);
        assert!(
            model.lora_install_attempted,
            "any actual Some setter attempt must poison trace ownership"
        );
        model.set_lora_weights(None).unwrap();
        assert!(model.lora.is_none());
        assert_eq!(model.moe_lora_route(-1), crate::layer::MoeLoraRoute::Fold);
        assert!(
            model.lora_install_attempted,
            "detach cannot prove old layer fields were cleared"
        );
        assert!(
            model
                .hidden_trace_adapter_ownership()
                .ensure_absent()
                .is_err()
        );
    }
}
