// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn test_empty_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(EmptyLayerState);
    assert!(state.as_any().downcast_ref::<EmptyLayerState>().is_some());
    assert!(state.as_any().downcast_ref::<SsmLayerState>().is_none());
}

#[test]
fn test_ssm_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        kda_records: spark_runtime::gpu::DevicePtr::NULL,
        conv_state_intermediates: Vec::new(),
        gdn_commit_qkv: DevicePtr(0),
        gdn_commit_gb: DevicePtr(0),
        gdn_commit_pending: false,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    });
    let ssm = state.as_any().downcast_ref::<SsmLayerState>().unwrap();
    assert_eq!(ssm.h_state.0, 0x1000);
    assert_eq!(ssm.conv_state.0, 0x2000);
}

#[test]
fn test_ssm_layer_state_mut() {
    let mut state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        kda_records: spark_runtime::gpu::DevicePtr::NULL,
        conv_state_intermediates: Vec::new(),
        gdn_commit_qkv: DevicePtr(0),
        gdn_commit_gb: DevicePtr(0),
        gdn_commit_pending: false,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    });
    let ssm = state.as_any_mut().downcast_mut::<SsmLayerState>().unwrap();
    ssm.h_state = DevicePtr(0x3000);
    assert_eq!(ssm.h_state.0, 0x3000);
}

#[test]
fn test_forward_context_lifetime() {
    use spark_runtime::gpu::mock::MockGpuBackend;

    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let buffers = BufferArena::new(&config, 1, 4096, 16, 1, &gpu).unwrap();

    // A test can now state the dispatch it wants instead of mutating the
    // process environment — the point of carrying it rather than caching it.
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        ssm_batch: None,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        gpu: &gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
        request_id: None,
    };

    assert_eq!(ctx.config.hidden_size, 2048);
    assert_eq!(ctx.config.num_hidden_layers, 48);
}

/// Request binding must survive every derived-context construction shape in
/// production unchanged (same `Arc<str>`, not a re-mint): the glm5_mtp /
/// deepseek_v4_mtp full-copy re-wraps (`request_id: ctx.request_id.clone()`),
/// the impl_b* profiling re-wraps, and the `..*parent` spread sites
/// (qwen3_attention multi_seq) that must override the non-Copy field
/// explicitly.
#[test]
fn request_id_binding_survives_derived_context_construction() {
    use spark_runtime::gpu::mock::MockGpuBackend;

    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let buffers = BufferArena::new(&config, 1, 4096, 16, 1, &gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let binding = std::sync::Arc::<str>::from("req-derived");
    let ctx = ForwardContext {
        ssm_batch: None,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        gpu: &gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
        request_id: Some(binding.clone()),
    };

    // glm5_mtp / deepseek_v4_mtp style: full-copy re-wrap with explicit
    // request_id clone.
    let mtp_ctx = ForwardContext {
        ssm_batch: None,
        dispatch: ctx.dispatch,
        derived: ctx.derived,
        levers: ctx.levers,
        stats: ctx.stats,
        buffers: ctx.buffers,
        gpu: ctx.gpu,
        config: ctx.config,
        attn_metadata: None,
        profile: ctx.profile,
        comm: ctx.comm,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: ctx.token_ids,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        request_id: ctx.request_id.clone(),
    };

    // qwen3_attention multi_seq style: spread re-wrap, where the non-Copy
    // fields (request_id, ssm_batch, midchunk_capture) must be overridden
    // explicitly or the borrow fails.
    let parent = &mtp_ctx;
    let row_ctx = ForwardContext {
        request_id: parent.request_id.clone(),
        ssm_batch: None,
        midchunk_capture: None,
        graph_capture: true,
        ..*parent
    };

    for (name, derived_ctx) in [("mtp_ctx", &mtp_ctx), ("row_ctx", &row_ctx)] {
        let bound = derived_ctx
            .request_id
            .as_ref()
            .unwrap_or_else(|| panic!("{name} must keep the request-id binding"));
        assert_eq!(&**bound, "req-derived", "{name} binding value");
        assert!(
            std::sync::Arc::ptr_eq(&binding, bound),
            "{name} must carry the same Arc<str>, not a re-mint"
        );
    }
    assert!(std::sync::Arc::ptr_eq(
        mtp_ctx.request_id.as_ref().unwrap(),
        row_ctx.request_id.as_ref().unwrap()
    ));
}

/// `crate::traits::single_request_id` binds a batch only when exactly one
/// sequence owns it; mixed or empty batches stay unbound (assumption A8).
#[test]
fn single_request_id_binds_exactly_one_sequence() {
    use crate::traits::{SequenceState, single_request_id};

    let mut one = SequenceState::host_only(0);
    one.request_id = Some(std::sync::Arc::from("req-a"));
    let mut two = SequenceState::host_only(1);
    two.request_id = Some(std::sync::Arc::from("req-b"));

    assert_eq!(
        single_request_id(&[&mut one]),
        Some(std::sync::Arc::from("req-a"))
    );
    assert_eq!(single_request_id(&[&mut one, &mut two]), None);
    assert_eq!(single_request_id(&[]), None);
    let mut unbound = SequenceState::host_only(2);
    assert_eq!(single_request_id(&[&mut unbound]), None);
}
