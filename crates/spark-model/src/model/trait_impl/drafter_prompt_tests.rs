// SPDX-License-Identifier: AGPL-3.0-only
//! Actual model ownership and eager error funnel, with an injected proposer error.
use super::*;
use crate::layer::{EmptyLayerState, LayerState, TransformerLayer};
use crate::layers::glm5_mtp::hidden_trace;
use crate::layers::{glm5_mtp::Glm5MtpHead, ops};
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::glm5::Glm5MtpModule;
use crate::weight_map::DenseWeight;
use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn child(name: &str) -> bool {
    if std::env::var("ATLAS_PROMPT_OWNER_TEST_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("model::trait_impl::drafter_prefill::prompt_tests::{name}"),
            "--nocapture",
        ])
        .env("ATLAS_PROMPT_OWNER_TEST_CHILD", "1")
        .env("ATLAS_GLM_MTP_REPAIR", "1")
        .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
        .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
        .env("ATLAS_MTP_SPEC_THINK", "1")
        .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1")
        .env_remove("ATLAS_GLM_MTP_HIDDEN_TRACE")
        .status()
        .unwrap();
    assert!(status.success());
    true
}

struct Body;
impl TransformerLayer for Body {
    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }
    fn prefill(
        &self,
        _: DevicePtr,
        _: DevicePtr,
        _: usize,
        _: &mut dyn LayerState,
        _: &mut PagedKvCache,
        _: usize,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: usize,
        _: &ForwardContext,
        _: u64,
    ) -> Result<()> {
        Ok(())
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
        anyhow::bail!("owner fixture never executes target")
    }
}
struct Rank;
impl spark_comm::CommBackend for Rank {
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        Ok(())
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        Ok(())
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        Ok(())
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        Ok(())
    }
    fn barrier(&self) -> Result<()> {
        Ok(())
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        2
    }
}
struct Failing {
    head: Glm5MtpHead,
    calls: Arc<AtomicUsize>,
}
impl DraftProposer for Failing {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        self.head.alloc_state(gpu)
    }
    fn drafter_rows(&self, s: &mut dyn ProposerState) -> usize {
        self.head.drafter_rows(s)
    }
    fn prefill_drafter(
        &self,
        _: &[u32],
        _: DevicePtr,
        _: &mut dyn ProposerState,
        _: &ForwardContext,
        _stream: u64,
    ) -> Result<usize> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        anyhow::bail!("injected original primer failure")
    }
    fn propose(
        &self,
        _: u32,
        _: DevicePtr,
        _: usize,
        _: usize,
        _: &mut dyn ProposerState,
        _: &ForwardContext,
        _: u64,
        _: Option<DevicePtr>,
        _: Option<&[i32]>,
        _: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("never proposes")
    }
    fn after_verify(&self, _: usize, _: &mut dyn ProposerState, _: u64) -> Result<()> {
        Ok(())
    }
}
fn fixture(p: usize, enabled: bool) -> (TransformerModel, SequenceState, Arc<AtomicUsize>) {
    let gpu = Box::new(MockGpuBackend::new());
    let mut cfg = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    cfg.model_type = "glm5_next".into();
    cfg.hidden_size = 4096;
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
    cfg.linear_num_key_heads = 0;
    cfg.linear_num_value_heads = 0;
    cfg.num_mtp_modules = 0;
    cfg.tp_world_size = 2;
    cfg.ep_world_size = 2;
    cfg.adapter_max_rank = 0;
    let buffers =
        spark_runtime::buffers::BufferArena::new(&cfg, 5, 2044, 16, 1, gpu.as_ref()).unwrap();
    let kv = PagedKvCache::new(
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
        128,
        gpu.as_ref(),
    )
    .unwrap();
    let dense = DenseWeight {
        weight: gpu.alloc(8 * 8192).unwrap(),
    };
    let head = Glm5MtpHead::new(
        Glm5MtpModule {
            body: Box::new(Body),
            enorm: dense,
            hnorm: dense,
            eh_proj: dense,
            eh_proj_nvfp4: None,
            norm: dense,
        },
        dense,
        dense,
        None,
        &cfg,
        gpu.as_ref(),
        8,
        2044,
    )
    .unwrap();
    let mut model = TransformerModel::new(
        cfg,
        dense,
        dense,
        dense,
        None,
        None,
        None,
        vec![Box::new(Body)],
        buffers,
        kv,
        vec![],
        gpu,
        2044,
        1,
        crate::layers::MtpQuantization::Bf16,
        false,
        false,
        Box::new(spark_runtime::prefix_cache::NoPrefixCaching),
        8,
        Some(Arc::new(Rank)),
        false,
        4,
        None,
        1,
        16,
    )
    .unwrap();
    model.levers.max_decode_seqs = 1;
    model.levers.drafter.prefill = true;
    model.levers.drafter.carry = false;
    model.mtp_prefill_hidden = model.gpu.alloc(p.max(4) * 8192).unwrap();
    model.mtp_prefill_capacity = p.max(4);
    model.mtp_prefill_capture_gen.store(11, Ordering::Relaxed);
    model.mtp_prefill_capture_len.store(p, Ordering::Relaxed);
    let calls = Arc::new(AtomicUsize::new(0));
    model.proposer = Some(Arc::new(Failing {
        head,
        calls: calls.clone(),
    }));
    let mut seq = SequenceState::host_only(0);
    seq.prompt_len = p;
    seq.seq_len = p;
    seq.tokens = vec![1; p];
    seq.mtp_capture_gen = 11;
    let mut state = model
        .proposer
        .as_ref()
        .unwrap()
        .alloc_state(model.gpu.as_ref())
        .unwrap();
    hidden_trace::fixture_set_enabled(state.as_mut(), enabled);
    seq.proposer_state = Some(state);
    (model, seq, calls)
}

#[test]
fn actual_eager_funnel_propagates_selected_failure_but_preserves_legacy_fallback() {
    if child("actual_eager_funnel_propagates_selected_failure_but_preserves_legacy_fallback") {
        return;
    }
    for (p, enabled, selected) in [
        (2, true, true),
        (148, true, true),
        (256, true, true),
        (148, false, false),
        (257, true, false),
        (1984, true, false),
    ] {
        let (model, mut seq, calls) = fixture(p, enabled);
        let result = model.try_eager_drafter_prefill(&mut seq, true, 37);
        assert_eq!(result.is_err(), selected, "P{p} enabled{enabled}");
        if selected {
            assert!(
                format!("{:#}", result.unwrap_err()).contains("injected original primer failure")
            );
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        if selected {
            assert!(model.try_eager_drafter_prefill(&mut seq, true, 37).is_err());
            assert_eq!(calls.load(Ordering::Relaxed), 1);
        }
    }
}

#[test]
fn actual_live_capture_owner_rejects_stale_partial_missing_and_profile_before_proposer() {
    if child("actual_live_capture_owner_rejects_stale_partial_missing_and_profile_before_proposer")
    {
        return;
    }
    for fault in 0..12 {
        let (mut model, mut seq, calls) = fixture(148, true);
        match fault {
            0 => model.mtp_prefill_capture_gen.store(12, Ordering::Relaxed),
            1 => model.mtp_prefill_capture_len.store(147, Ordering::Relaxed),
            2 => model.mtp_prefill_capacity = 147,
            3 => model.mtp_prefill_hidden = DevicePtr::NULL,
            4 => {
                seq.tokens.pop();
            }
            5 => seq.seq_len = 149,
            6 => seq.cached_prefix_tokens = 1,
            7 => model.config.adapter_max_rank = 1,
            8 => model.comm = None,
            9 => model.proposer = None,
            10 => model.mtp_prefill_hidden = DevicePtr(u64::MAX - 1),
            _ => {
                let pool = Arc::get_mut(&mut model.ssm_pool).unwrap();
                pool.has_mtp = true;
                pool.mtp_slots = 0;
            }
        }
        assert!(
            model.try_eager_drafter_prefill(&mut seq, true, 37).is_err(),
            "fault{fault}"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        // Fixing a selected model fault must not silently grant a fresh attempt.
        model.mtp_prefill_capture_gen.store(11, Ordering::Relaxed);
        model.mtp_prefill_capture_len.store(148, Ordering::Relaxed);
        assert!(model.try_eager_drafter_prefill(&mut seq, true, 37).is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn actual_midchunk_and_disabled_missing_owner_are_inert() {
    if child("actual_midchunk_and_disabled_missing_owner_are_inert") {
        return;
    }
    let (mut model, mut seq, calls) = fixture(148, true);
    model.mtp_prefill_hidden = DevicePtr::NULL;
    assert!(model.try_eager_drafter_prefill(&mut seq, false, 37).is_ok());
    hidden_trace::fixture_set_enabled(seq.proposer_state.as_mut().unwrap().as_mut(), false);
    assert!(model.try_eager_drafter_prefill(&mut seq, true, 37).is_ok());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn actual_prefill_policy_failure_is_spent_before_proposer() {
    let name = "model::trait_impl::drafter_prefill::prompt_tests::actual_prefill_policy_failure_is_spent_before_proposer";
    if std::env::var("ATLAS_PROMPT_POLICY_TEST_CHILD").as_deref() != Ok("1") {
        for (key, value) in [
            ("ATLAS_GLM_MTP_REPAIR", "0"),
            ("ATLAS_GLM_MTP_BATCHED_PREFILL", "0"),
            ("ATLAS_GLM_MTP_SERIAL_PREFILL", "1"),
        ] {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env("ATLAS_PROMPT_POLICY_TEST_CHILD", "1")
                .env("ATLAS_GLM_MTP_REPAIR", "1")
                .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
                .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
                .env("ATLAS_MTP_SPEC_THINK", "1")
                .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1")
                .env_remove("ATLAS_GLM_MTP_HIDDEN_TRACE")
                .env(key, value)
                .status()
                .unwrap();
            assert!(status.success());
        }
        return;
    }
    let (model, mut seq, calls) = fixture(148, true);
    assert!(model.try_eager_drafter_prefill(&mut seq, true, 37).is_err());
    assert!(
        format!(
            "{:#}",
            model
                .try_eager_drafter_prefill(&mut seq, true, 37)
                .unwrap_err()
        )
        .contains("already spent")
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn actual_model_prefill_chunk_propagates_selected_error_and_preserves_flag_off() {
    use crate::traits::Model;
    if child("actual_model_prefill_chunk_propagates_selected_error_and_preserves_flag_off") {
        return;
    }
    for enabled in [false, true] {
        let (model, mut seq, calls) = fixture(4, enabled);
        seq.tokens.clear();
        seq.seq_len = 0;
        seq.mtp_capture_gen = 0;
        seq.layer_states = vec![Box::new(EmptyLayerState)];
        model.mtp_prefill_capture_len.store(0, Ordering::Relaxed);
        let result = model.prefill_chunk(&[1, 2, 3, 4], &mut seq, 0, 4, true, 37);
        assert_eq!(
            result.is_err(),
            enabled,
            "actual Model wrapper enabled={enabled}: {result:?}"
        );
        if enabled {
            assert!(
                format!("{:#}", result.unwrap_err()).contains("injected original primer failure")
            );
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(seq.tokens, [1, 2, 3, 4]);
        assert_eq!(seq.seq_len, 4);
        assert_eq!(seq.mtp_capture_gen, 12);
        assert_eq!(model.mtp_prefill_capture_len.load(Ordering::Relaxed), 4);
    }
}
