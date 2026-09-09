// SPDX-License-Identifier: AGPL-3.0-only
//! Actual arena byte copies and immutable preflight, not attention numerics.
use super::*;
use crate::layer::{EmptyLayerState, MoeLoraRoute, glm_pair_verify::GlmPairLayerInput};
use crate::layers::ops;
use atlas_core::config::ModelConfig;
use spark_runtime::{
    buffers::BufferArena,
    gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend},
};

struct Comm;
impl spark_comm::CommBackend for Comm {
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected collective")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected collective")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected collective")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        anyhow::bail!("unexpected collective")
    }
    fn barrier(&self) -> Result<()> {
        anyhow::bail!("unexpected collective")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected receive")
    }
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        2
    }
}

fn arena(rows: usize, run: impl FnOnce(&ForwardContext, &MockGpuBackend)) {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.hc_mult = 4;
    config.tp_world_size = 2;
    config.ep_world_size = 2;
    config.tp_rank = 0;
    config.ep_rank = 0;
    let gpu = MockGpuBackend::new();
    let buffers = BufferArena::new(&config, rows, 2048, 16, rows, &gpu).unwrap();
    let dispatch = ops::GemmDispatch::defaults();
    let derived = ops::DerivedWeights::new();
    let levers = ops::ModelLevers::defaults();
    let stats = ops::ModelStats::new();
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
        comm: Some(&Comm),
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    run(&ctx, &gpu);
}

fn read(gpu: &MockGpuBackend, ptr: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut out = vec![0; bytes];
    gpu.copy_d2h(ptr, &mut out).unwrap();
    out
}

#[test]
fn explicit_shape_and_actual_arena_capacity() {
    for invalid in [0, 1, 2, 9, usize::MAX] {
        assert!(GlmOwnerBatchShape::new(invalid).is_err());
    }
    for owners in 3..=8 {
        let shape = GlmOwnerBatchShape::new(owners).unwrap();
        assert_eq!((shape.owners(), shape.rows()), (owners, owners * 5));
        arena(shape.rows() * 2 - 1, |ctx, gpu| {
            assert!(GlmOwnerBatchWorkspace::new(ctx, shape).is_err());
            assert_eq!(gpu.d2d_count(), 0);
        });
        arena(shape.rows() * 2, |ctx, gpu| {
            let workspace = GlmOwnerBatchWorkspace::new(ctx, shape)
                .expect("actual exact-size owner arena must be accepted");
            assert_eq!(workspace.shape(), shape);
            assert_eq!(workspace.scratch.owner_count(), owners);
            workspace
                .scratch
                .validate_context(ctx, gpu.default_stream())
                .unwrap();
            assert_eq!(gpu.d2d_count(), 0);
        });
    }
}

#[test]
fn four_eight_owner_saved_tails_pack_and_restore_last_index() {
    for count in [4, 8] {
        arena(count * 10, |ctx, gpu| {
            let last = count - 1;
            let mut w =
                GlmOwnerBatchWorkspace::new(ctx, GlmOwnerBatchShape::new(count).unwrap()).unwrap();
            let w = &mut w.scratch;
            let b = ctx.buffers;
            let stream = gpu.default_stream();
            let spans = [
                (b.norm_output(), 40960),
                (b.hc_streams(), 327680),
                (b.hc_post(), 80),
                (b.hc_comb(), 320),
            ];
            for (ptr, bytes) in [
                (b.norm_output(), b.sizes().norm_output),
                (b.hc_streams(), b.sizes().hc_streams),
                (b.hc_post(), b.sizes().hc_post),
                (b.hc_comb(), b.sizes().hc_comb),
            ] {
                gpu.memset(ptr, 0xe7, bytes).unwrap();
            }
            w.restore_highway(last, stream).unwrap();
            assert_eq!(gpu.d2d_count(), 0, "first layer has no saved highway");
            for owner in 0..count {
                for (kind, &(ptr, bytes)) in spans.iter().enumerate() {
                    gpu.memset(ptr, (owner * 16 + kind + 1) as u8, bytes)
                        .unwrap();
                }
                w.save_attention(owner, stream).unwrap();
            }
            w.pack_norms(stream).unwrap();
            for owner in 0..count {
                let expected = vec![(owner * 16 + 1) as u8; 40960];
                assert_eq!(
                    read(gpu, b.norm_output().offset(owner * 40960), 40960),
                    expected
                );
                assert_eq!(
                    read(gpu, b.norm_output().offset((count + owner) * 40960), 40960),
                    expected
                );
            }
            let packed = read(gpu, b.norm_output(), count * 5 * 8192);
            for &(ptr, bytes) in &spans[1..] {
                gpu.memset(ptr, 0x99, bytes).unwrap();
            }
            w.restore_ffn(last, false, stream).unwrap();
            assert_eq!(read(gpu, b.norm_output(), count * 5 * 8192), packed);
            for (kind, &(ptr, bytes)) in spans.iter().enumerate().skip(1) {
                assert_eq!(
                    read(gpu, ptr, bytes),
                    vec![(last * 16 + kind + 1) as u8; bytes]
                );
                assert_eq!(
                    read(gpu, ptr.offset(count * bytes), bytes),
                    vec![(last * 16 + kind + 1) as u8; bytes]
                );
                assert_eq!(
                    read(gpu, ptr.offset((count + 1) * bytes), bytes),
                    vec![0xe7; bytes],
                    "tail canary"
                );
            }
            gpu.memset(b.hc_streams(), 0x71, 327680).unwrap();
            w.save_highway(last, stream).unwrap();
            w.finish_layer();
            gpu.memset(b.hc_streams(), 0x98, 327680).unwrap();
            w.restore_highway(last, stream).unwrap();
            assert_eq!(read(gpu, b.hc_streams(), 327680), vec![0x71; 327680]);
            let copies = gpu.d2d_count();
            assert!(w.save_attention(count, stream).is_err());
            assert!(w.restore_ffn(count, true, stream).is_err());
            assert!(w.restore_highway(count, stream).is_err());
            assert_eq!(gpu.d2d_count(), copies);
        });
    }
}

#[test]
fn all_owner_contexts_and_rows_validate_before_copy() {
    for count in [4, 8] {
        arena(count * 10, |ctx, gpu| {
            let w =
                GlmOwnerBatchWorkspace::new(ctx, GlmOwnerBatchShape::new(count).unwrap()).unwrap();
            let w = &w.scratch;
            let positions = [0, 1, 2, 3, 4];
            let mut states: Vec<_> = (0..count).map(|_| EmptyLayerState).collect();
            let mut owners: Vec<_> = states
                .iter_mut()
                .enumerate()
                .map(|(index, state)| GlmPairLayerInput {
                    hidden: ctx.buffers.hidden_states().offset(index * 40960),
                    state,
                    positions: &positions,
                    block_table: &[],
                })
                .collect();
            let stream = gpu.default_stream();
            let contexts = vec![ctx; count];
            w.begin_layer(0, &owners, &contexts, stream).unwrap();
            owners[count - 1].hidden = owners[0].hidden;
            assert!(w.begin_layer(0, &owners, &contexts, stream).is_err());
            owners[count - 1].hidden = ctx.buffers.hidden_states().offset((count - 1) * 40960);
            let wrong = ForwardContext {
                buffers: ctx.buffers,
                gpu: ctx.gpu,
                config: ctx.config,
                dispatch: ctx.dispatch,
                derived: ctx.derived,
                levers: ctx.levers,
                stats: ctx.stats,
                attn_metadata: None,
                ssm_batch: None,
                profile: false,
                comm: ctx.comm,
                graph_capture: true,
                gdn_exact_replay: false,
                token_ids: None,
                routed_lora_layers: None,
                midchunk_capture: None,
                moe_lora_route: MoeLoraRoute::Fold,
            };
            let mut wrong_contexts = contexts.clone();
            wrong_contexts[count - 1] = &wrong;
            assert!(w.begin_layer(0, &owners, &wrong_contexts, stream).is_err());
            assert!(w.begin_layer(1, &owners, &contexts, stream).is_err());
            assert!(
                w.begin_layer(0, &owners[..count - 1], &contexts, stream)
                    .is_err()
            );
            assert!(
                w.begin_layer(0, &owners, &contexts[..count - 1], stream)
                    .is_err()
            );
            assert_eq!(gpu.d2d_count(), 0);
        });
    }
}
