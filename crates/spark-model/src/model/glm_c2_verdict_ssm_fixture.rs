// SPDX-License-Identifier: AGPL-3.0-only
//! Actual tiny SSM pool and recurrent numerical boundary for ownership tests.
use super::*;

struct RecurrentBoundary;
impl TransformerLayer for RecurrentBoundary {
    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        anyhow::bail!("use actual slot pool")
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
        anyhow::bail!("fixture requires real K5 wrapper")
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
        anyhow::bail!("install tiny recurrent pool after actual producer preparation")
    }
    fn decode_batched(
        &self,
        _: DevicePtr,
        _: DevicePtr,
        rows: usize,
        state: &mut dyn LayerState,
        _: &mut PagedKvCache,
        base: usize,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(rows == 5, "real K5 expected");
        let state = state.as_any_mut().downcast_mut::<SsmLayerState>().unwrap();
        for row in 0..rows {
            let h = (base + row + 0x40) as u8;
            let c = (base + row + 0x60) as u8;
            ctx.gpu.memset_async(state.h_state, h, 16, stream)?;
            ctx.gpu.memset_async(state.conv_state, c, 48, stream)?;
            if row < 4 {
                ctx.gpu
                    .memset_async(state.h_state_intermediates[row], h, 16, stream)?;
                ctx.gpu
                    .memset_async(state.conv_state_intermediates[row], c, 48, stream)?;
            }
        }
        Ok(())
    }
}

pub(super) fn isolated(name: &str) -> bool {
    if std::env::var("ATLAS_C2_SSM_TEST_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("model::glm_c2_handoff_tests::verdict_ssm_tests::{name}"),
            "--nocapture",
        ])
        .env("ATLAS_C2_SSM_TEST_CHILD", "1")
        .env("ATLAS_C2_VERDICT_TEST_CHILD", "1")
        .env("ATLAS_EP_PROTOCOL", "v2")
        .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
        .env("ATLAS_GLM_MTP_REPAIR", "0")
        .env("ATLAS_GLM_MTP_DISTRIBUTED", "1")
        .env("ATLAS_GLM_MTP_ALL_GATHER", "1")
        .env("ATLAS_GLM_MTP_DISTRIBUTED_ARGMAX", "0")
        .env("ATLAS_GLM_MTP_BATCHED_PREFILL", "1")
        .env("ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE", "1")
        .env_remove("ATLAS_GLM_MTP_FUSED_EH_NORM")
        .env_remove("ATLAS_NO_MTP_EAGER_DRAFTER")
        .env_remove("ATLAS_MTP_CARRY_DRAFTER")
        .env_remove("ATLAS_MTP_CATCHUP")
        .status()
        .unwrap();
    assert!(status.success());
    true
}

pub(super) fn prepared(rank: usize) -> (Fixture, [flow::History; 2]) {
    let (mut f, history) = flow::prepare(rank, [0, 1]);
    let cfg = &mut f.model.config;
    cfg.num_hidden_layers = 2;
    cfg.layer_types
        .push(atlas_core::config::LayerType::LinearAttention);
    cfg.linear_num_key_heads = 1;
    cfg.linear_num_value_heads = 1;
    cfg.linear_key_head_dim = 2;
    cfg.linear_value_head_dim = 2;
    cfg.linear_conv_kernel_dim = 2;
    cfg.mamba_num_heads = 0;
    cfg.mamba_head_dim = 0;
    let pool = Arc::new(
        SsmStatePool::new(
            cfg,
            2,
            true,
            5,
            4,
            false,
            crate::ssm_reserve::SsmRollbackMode::Snapshot,
            f.model.gpu.as_ref(),
        )
        .unwrap(),
    );
    assert_eq!(
        (pool.num_ssm_layers, pool.h_stored_bytes, pool.conv_bytes),
        (1, 16, 48)
    );
    f.model.layers.push(Box::new(RecurrentBoundary));
    for owner in 0..2 {
        assert!(pool.verify_draft_capacity(owner) >= 4);
        let guard = pool.claim_guarded().unwrap();
        assert_eq!(guard.idx(), Some(owner));
        f.seqs[owner].ssm_slot = Some(guard);
        f.seqs[owner].disk_last_offloaded_per_layer.push(0);
        f.seqs[owner].layer_states.push(Box::new(SsmLayerState {
            h_state: pool.h_state(0, owner),
            conv_state: pool.conv_state(0, owner),
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: (0..4)
                .map(|row| pool.h_intermediate(0, owner, row))
                .collect(),
            kda_records: spark_runtime::gpu::DevicePtr::NULL,
            conv_state_intermediates: (0..5)
                .map(|row| pool.conv_intermediate(0, owner, row))
                .collect(),
            h_is_f16: false,
            h_prefill_stage: None,
            ple: None,
        }));
        f.gpu
            .write_span(pool.conv_intermediate(0, owner, 4), &[0xd7; 48]);
    }
    f.model.ssm_pool = pool;
    f.gpu.clear();
    (f, history)
}
