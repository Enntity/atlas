// SPDX-License-Identifier: AGPL-3.0-only
//! Real factory/head/target constructor calls, not native numerical validation.
use super::*;
use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use crate::model::TransformerModel;
use crate::speculative::DraftProposer;
use crate::traits::Model;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheConfig, PagedKvCache};
use spark_runtime::prefix_cache::NoPrefixCaching;
use spark_runtime::weights::WeightStore;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
#[path = "glm_paired_test_gpu.rs"]
mod recording;
use recording::Gpu;

#[path = "glm_paired_capacity_tests.rs"]
mod capacity_tests;

fn isolated(name: &str) -> bool {
    if std::env::var("ATLAS_PAIRED_FACTORY_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "--exact",
        &format!("factory::glm_paired::tests::{name}"),
        "--nocapture",
    ])
    .env("ATLAS_PAIRED_FACTORY_CHILD", "1")
    .env("ATLAS_EP_PROTOCOL", "v2")
    .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
    .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
    .env("ATLAS_GLM_MTP_REPAIR", "0")
    .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
    .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1");
    for key in [
        "ATLAS_NO_MTP_EAGER_DRAFTER",
        "ATLAS_NO_MTP_DRAFTER_CONTEXT",
        "ATLAS_MTP_CARRY_DRAFTER",
        "ATLAS_MTP_ACCEPT_DEBUG",
        "ATLAS_GLM_MTP_FUSED_EH_NORM",
        "ATLAS_GLM_MTP_SERIAL_PREFILL",
        "ATLAS_MTP_CATCHUP",
        "ATLAS_GLM_MTP_PROFILE",
    ] {
        cmd.env_remove(key);
    }
    assert!(
        cmd.status().unwrap().success(),
        "factory child {name} failed"
    );
    true
}

fn config(rank: usize) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.vocab_size = 8;
    c.num_hidden_layers = 0;
    c.layer_types = vec![];
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_kpool = 0;
    c.index_head_dim = 0;
    c.index_topk = 2048;
    c.intermediate_size = 32;
    c.moe_intermediate_size = 32;
    c.shared_expert_intermediate_size = 32;
    c.num_experts = 2;
    c.num_experts_per_tok = 2;
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    c.tp_rank = rank;
    c.ep_rank = rank;
    c.num_mtp_modules = 1;
    c.adapter_max_rank = 0;
    c
}
struct Comm(usize, Arc<AtomicUsize>, Option<(Arc<AtomicUsize>, usize)>);
impl Drop for Comm {
    fn drop(&mut self) {
        self.1.fetch_add(1, Ordering::Relaxed);
    }
}
impl spark_comm::CommBackend for Comm {
    fn register_buffer(&self, _: u64, _: usize) -> Result<u64> {
        if let Some((calls, fail_at)) = &self.2 {
            let ordinal = calls.fetch_add(1, Ordering::Relaxed) + 1;
            anyhow::ensure!(
                ordinal != *fail_at,
                "injected registration failure {ordinal}"
            );
        }
        Ok(0)
    }
    fn rank(&self) -> usize {
        self.0
    }
    fn world_size(&self) -> usize {
        2
    }
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
}
struct Body;
impl TransformerLayer for Body {
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
        anyhow::bail!("construction never executes body")
    }
}
fn module(gpu: &dyn GpuBackend) -> Glm5MtpModule {
    let dense = || DenseWeight {
        weight: gpu.alloc(8192).unwrap(),
    };
    Glm5MtpModule {
        body: Box::new(Body),
        enorm: dense(),
        hnorm: dense(),
        norm: dense(),
        // No numerical EH call in this constructor test, so only owned bytes matter.
        eh_proj: dense(),
        eh_proj_nvfp4: None,
    }
}

#[test]
fn actual_factory_head_selects_paired_without_changing_legacy() {
    if isolated("actual_factory_head_selects_paired_without_changing_legacy") {
        return;
    }
    for rank in 0..2 {
        for mode in [GlmMtpBuildMode::Legacy, GlmMtpBuildMode::Paired] {
            let gpu = Gpu::new();
            let embed = DenseWeight {
                weight: gpu.alloc(8 * 8192).unwrap(),
            };
            let head = build_head(
                mode,
                module(&gpu),
                embed,
                embed,
                None,
                &config(rank),
                &gpu,
                8,
                2044,
                2,
            )
            .unwrap();
            assert_eq!(
                head.glm_pair_repair().unwrap().paired_handoff().is_some(),
                mode == GlmMtpBuildMode::Paired
            );
        }
    }
}

#[test]
fn actual_factory_loader_error_retains_only_selected_backend() {
    if isolated("actual_factory_loader_error_retains_only_selected_backend") {
        return;
    }
    for mode in [GlmMtpBuildMode::Legacy, GlmMtpBuildMode::Paired] {
        let gpu = Gpu::new();
        let probe = gpu.probe.clone();
        let comm_drops = Arc::new(AtomicUsize::new(0));
        let result = crate::factory::build_model(
            config(0),
            WeightStore::empty(),
            Box::new(gpu),
            5,
            16,
            2044,
            2,
            MtpQuantization::Bf16,
            true,
            Box::new(NoPrefixCaching),
            8,
            Some(Arc::new(Comm(0, comm_drops.clone(), None))),
            false,
            4,
            KvCacheDtype::Bf16,
            1 << 20,
            0.9,
            2,
            vec![],
            0,
            None,
            None,
            None,
            None,
            None,
            mode,
        );
        let error = result.err().expect("empty actual checkpoint must fail");
        assert!(
            format!("{error:#}").contains("not found"),
            "actual loader control: {error:#}"
        );
        assert!(probe.kernels.load(Ordering::Relaxed) > 0);
        let expected = usize::from(mode == GlmMtpBuildMode::Legacy);
        assert_eq!(
            (
                probe.drops.load(Ordering::Relaxed),
                probe.sweeps.load(Ordering::Relaxed),
                comm_drops.load(Ordering::Relaxed)
            ),
            (expected, expected, expected)
        );
    }
}

#[test]
fn actual_target_constructor_error_retains_only_selected_backend() {
    if isolated("actual_target_constructor_error_retains_only_selected_backend") {
        return;
    }
    for selected in [false, true] {
        let gpu = Gpu::new();
        let probe = gpu.probe.clone();
        let comm_drops = Arc::new(AtomicUsize::new(0));
        let result = target(gpu, Comm(0, comm_drops.clone(), None), selected, true);
        assert!(
            format!("{:#}", result.err().expect("actual first kernel fails"))
                .contains("injected actual constructor kernel")
        );
        let expected = usize::from(!selected);
        assert_eq!(
            (
                probe.drops.load(Ordering::Relaxed),
                probe.sweeps.load(Ordering::Relaxed),
                comm_drops.load(Ordering::Relaxed)
            ),
            (expected, expected, expected)
        );
    }
}

#[test]
fn actual_paired_capability_requires_the_inherited_rank_without_health_io() {
    if isolated("actual_paired_capability_requires_the_inherited_rank_without_health_io") {
        return;
    }
    use crate::model::glm_c2_test_support::{fixture::Fixture, wire::Wire};
    for rank in 0..2 {
        let mut f = Fixture::new(rank);
        let wire = Wire::install(&mut f, rank);
        f.gpu.clear();
        let cap = f.model.glm_paired_execution().unwrap();
        cap.validate_session_rank(rank as u8).unwrap();
        assert!(cap.validate_session_rank((1 - rank) as u8).is_err());
        assert!(cap.validate_session_rank(2).is_err());
        assert!(wire.packets().is_empty());
        assert!(f.gpu.trace().is_empty());
    }
}

fn target(gpu: Gpu, comm: Comm, selected: bool, fail_first: bool) -> Result<TransformerModel> {
    let c = config(0);
    let buffers = BufferArena::new(&c, 5, 32, 16, 2, &gpu)?;
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
        4,
        &gpu,
    )?;
    let w = DenseWeight {
        weight: gpu.alloc(8192)?,
    };
    gpu.probe.fail_kernel.store(fail_first, Ordering::Relaxed);
    TransformerModel::new_with_cold_retention(
        c,
        w,
        w,
        w,
        None,
        None,
        None,
        vec![],
        buffers,
        kv,
        vec![],
        Box::new(gpu),
        32,
        2,
        MtpQuantization::Bf16,
        true,
        true,
        Box::new(NoPrefixCaching),
        8,
        Some(Arc::new(comm)),
        false,
        4,
        None,
        2,
        0,
        selected,
    )
}

#[test]
fn actual_assembled_target_is_retained_on_appended_head_error() {
    if isolated("actual_assembled_target_is_retained_on_appended_head_error") {
        return;
    }
    for mode in [GlmMtpBuildMode::Legacy, GlmMtpBuildMode::Paired] {
        let gpu = Gpu::new();
        let probe = gpu.probe.clone();
        let comm_drops = Arc::new(AtomicUsize::new(0));
        let module = module(&gpu);
        let w = DenseWeight {
            weight: gpu.alloc(8 * 8192).unwrap(),
        };
        let model = target(
            gpu,
            Comm(0, comm_drops.clone(), None),
            mode == GlmMtpBuildMode::Paired,
            false,
        )
        .unwrap();
        probe.fail_kernel.store(true, Ordering::Relaxed);
        let result = install_head(model, mode, Some(module), w, w, None, 8, 32, 2, true);
        assert!(
            format!("{:#}", result.err().expect("actual head kernel fails"))
                .contains("injected actual constructor kernel")
        );
        let expected = usize::from(mode == GlmMtpBuildMode::Legacy);
        assert_eq!(
            (
                probe.drops.load(Ordering::Relaxed),
                probe.sweeps.load(Ordering::Relaxed),
                comm_drops.load(Ordering::Relaxed),
                probe.pinned_frees.load(Ordering::Relaxed)
            ),
            (expected, expected, expected, expected)
        );
    }
}

#[test]
fn actual_constructor_registration_failure_stops_only_selected_construction() {
    if isolated("actual_constructor_registration_failure_stops_only_selected_construction") {
        return;
    }
    for selected in [false, true] {
        for fail_at in [1, 2] {
            let gpu = Gpu::new();
            let probe = gpu.probe.clone();
            let comm_drops = Arc::new(AtomicUsize::new(0));
            let calls = Arc::new(AtomicUsize::new(0));
            let result = target(
                gpu,
                Comm(0, comm_drops.clone(), Some((calls.clone(), fail_at))),
                selected,
                false,
            );
            if selected {
                assert!(
                    format!(
                        "{:#}",
                        result
                            .err()
                            .expect("selected registration failure must propagate")
                    )
                    .contains("injected registration failure")
                );
            } else {
                drop(result.expect("legacy registration remains optional"));
            }
            assert_eq!(
                calls.load(Ordering::Relaxed),
                if selected { fail_at } else { 2 }
            );
            let expected = usize::from(!selected);
            assert_eq!(
                (
                    probe.drops.load(Ordering::Relaxed),
                    probe.sweeps.load(Ordering::Relaxed),
                    comm_drops.load(Ordering::Relaxed)
                ),
                (expected, expected, expected)
            );
        }
    }
}

#[test]
fn missing_actual_module_refuses_paired_without_legacy_fallback() {
    if isolated("missing_actual_module_refuses_paired_without_legacy_fallback") {
        return;
    }
    for mode in [GlmMtpBuildMode::Legacy, GlmMtpBuildMode::Paired] {
        let gpu = Gpu::new();
        let probe = gpu.probe.clone();
        let comm_drops = Arc::new(AtomicUsize::new(0));
        let w = DenseWeight {
            weight: gpu.alloc(8192).unwrap(),
        };
        let model = target(
            gpu,
            Comm(0, comm_drops.clone(), None),
            mode == GlmMtpBuildMode::Paired,
            false,
        )
        .unwrap();
        let before = probe.kernels.load(Ordering::Relaxed);
        let result = install_head(model, mode, None, w, w, None, 8, 32, 2, true);
        if mode == GlmMtpBuildMode::Paired {
            assert!(
                format!("{:#}", result.err().expect("actual module required"))
                    .contains("actual appended GLM module")
            );
        } else {
            drop(result.expect("legacy absent module remains optional"));
        }
        assert_eq!(probe.kernels.load(Ordering::Relaxed), before);
        let expected = usize::from(mode == GlmMtpBuildMode::Legacy);
        assert_eq!(
            (
                probe.drops.load(Ordering::Relaxed),
                probe.pinned_frees.load(Ordering::Relaxed),
                comm_drops.load(Ordering::Relaxed)
            ),
            (expected, expected, expected)
        );
    }
}
