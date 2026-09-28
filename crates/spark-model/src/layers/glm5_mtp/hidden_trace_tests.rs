// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::speculative::glm_pair_plan::{BootstrapInput, Limits, Profile};
#[path = "hidden_trace_test_gpu.rs"]
mod support;
use support::{Event, fixture};

#[path = "hidden_trace_boundary_tests.rs"]
mod boundary_tests;
#[path = "hidden_trace_kv_hook_tests.rs"]
mod kv_hook_tests;
#[path = "hidden_trace_ownership_tests.rs"]
mod ownership_tests;
#[path = "hidden_trace_post_eh_tests.rs"]
mod post_eh_tests;
#[path = "hidden_trace_request_tests.rs"]
mod request_tests;

fn no_adapter_owners() -> AdapterOwnership {
    AdapterOwnership {
        pool: false,
        overlays: false,
        rotatable: false,
        install_attempted: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn arm_prepared(
    seq: &mut SequenceState,
    token: u32,
    position: usize,
    drafts: usize,
    saved: DevicePtr,
    hidden_row: usize,
    grammar: bool,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    super::arm_prepared(
        seq,
        token,
        position,
        drafts,
        saved,
        hidden_row,
        grammar,
        ctx,
        stream,
        no_adapter_owners,
    )
}

fn sequence(
    head: &Glm5MtpHead,
    ctx: &ForwardContext,
    gpu: &support::TraceGpu,
    generation: u64,
) -> SequenceState {
    let mut state = head.alloc_state_inner(ctx.gpu).unwrap();
    prepared_with_source(head, ctx, gpu, &mut state, generation, 3);
    let mut seq = SequenceState::host_only(0);
    seq.tokens = vec![1, 2, 3];
    seq.seq_len = 3;
    seq.prompt_len = 2;
    seq.mtp_capture_gen = generation;
    seq.proposer_state = Some(Box::new(state));
    seq
}
fn prepared_with_source(
    head: &Glm5MtpHead,
    ctx: &ForwardContext,
    gpu: &support::TraceGpu,
    state: &mut Glm5MtpProposerState,
    generation: u64,
    position: usize,
) {
    prepared(state, generation, position);
    if state.hidden_trace.enabled && (2..=256).contains(&(position - 1)) {
        let setup_ctx = ForwardContext {
            gpu: &gpu.inner,
            midchunk_capture: None,
            ..*ctx
        };
        super::prompt::fixture_observe(head, state, generation, position - 1, &setup_ctx).unwrap();
    }
}
fn prepared(state: &mut Glm5MtpProposerState, generation: u64, position: usize) {
    let limits = Limits::new(
        Profile {
            sequences: 1,
            drafts: 4,
            continuous: true,
            grammar: false,
            adaptive_depth: false,
            catchup: false,
            carry: false,
            prefix_reuse: false,
        },
        2044,
        2048,
        2044,
    )
    .unwrap();
    let finish = limits
        .bootstrap(BootstrapInput {
            generation,
            capture_generation: generation,
            prompt_tokens: position - 1,
            target_position: position,
            token_rows: position,
            normalized_hidden_rows: position - 1,
            cached_rows: 0,
        })
        .unwrap();
    state.seq_len = finish.state().cache_rows();
    state.repair = repair_state::RepairPhase::Proposed(
        limits
            .propose(finish.state(), generation, position, state.seq_len, 4)
            .unwrap(),
    );
}

#[test]
fn actual_forward_one_traces_input_then_post_norm_before_vocabulary_on_both_ranks() {
    for rank in 0..2 {
        fixture(rank, |head, ctx, gpu, saved| {
            let mut seq = sequence(head, ctx, gpu, 1);
            arm_prepared(&mut seq, 3, 3, 4, saved, 0, false, ctx, 7).unwrap();
            gpu.events.lock().clear();
            let state = seq
                .proposer_state
                .as_mut()
                .unwrap()
                .as_any_mut()
                .downcast_mut::<Glm5MtpProposerState>()
                .unwrap();
            let draft = head
                .forward_one(3, saved, 3, 0, state, ctx, 7, None)
                .unwrap();
            assert_eq!(draft, 7);
            let events = gpu.events.lock();
            assert_eq!(
                events.first(),
                Some(&Event::Read(saved, ROW_BYTES, 7)),
                "actual input hook must precede body"
            );
            let body = events.iter().position(|e| *e == Event::Body).unwrap();
            let post_eh = events
                .iter()
                .position(|e| *e == Event::Read(ctx.buffers.hidden_states(), ROW_BYTES, 7))
                .expect("actual post-EH hook must read full row before body");
            assert!(post_eh < body);
            assert!(matches!(&events[post_eh-1], Event::Kernel(102,p)
                if p[2] == ctx.buffers.hidden_states()));
            let final_read = events
                .iter()
                .position(|e| *e == Event::Read(ctx.buffers.norm_output(), ROW_BYTES, 7))
                .expect("actual final hook");
            assert!(body < final_read);
            assert!(
                matches!(&events[final_read-1],Event::Kernel(101,p) if p[2]==ctx.buffers.norm_output())
            );
            assert!(
                matches!(&events[final_read+1],Event::Kernel(102,p) if p[0]==ctx.buffers.norm_output())
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::Read(_, ROW_BYTES, _)))
                    .count(),
                3
            );
        });
    }
}
