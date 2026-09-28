// SPDX-License-Identifier: AGPL-3.0-only

use super::test_model::Target;
use super::*;
use crate::scheduler::{sched_ctx::SchedCtx, test_support::test_seq};

pub(super) fn context(s: &SchedCtx) -> LogitsContext<'_> {
    let sampling = crate::scheduler::logit_processors::SamplingLevers {
        fast_greedy_chat: true,
        fast_greedy_grammar: true,
        dflash_masked_verify: false,
        force_temp_zero: false,
        mtp_verify_sample: true,
        ..Default::default()
    };
    LogitsContext {
        glm_tool_boundary: None,
        scratch: &s.scratch,
        dumps: &s.dumps,
        stats: s.stats.clone(),
        watchdog: s.watchdog,
        boundary_mask: None,
        mid_word_mask: None,
        sampling,
        timing: s.timing.clone(),
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    }
}
pub(super) fn active(history: Vec<u32>) -> ActiveSeq {
    let (mut a, _) = test_seq(history, 100, None, 10);
    a.lz_penalty = 0.0;
    a.min_tokens = 0;
    a.repetition_penalty = 2.0;
    a
}
pub(super) fn target() -> Target {
    let mut rows = vec![vec![-8.0; 8]; 8];
    rows[0][7] = 16.0;
    rows[1][6] = 16.0;
    for row in &mut rows[2..7] {
        row[1] = 4.0;
        row[2] = 3.0;
    }
    rows[7][5] = 16.0;
    Target::new(&rows, false)
}

#[test]
fn legacy_penalty_selected_nonraw_k5_uses_exact_owned_rows() {
    let model = target();
    let sched = SchedCtx::for_test();
    let mut a = active(vec![1]);
    assert_eq!(
        verify_pick_all_with_pipeline(&model, &[1; 5], &mut a, &context(&sched), 2),
        vec![2; 5]
    );
    assert_eq!(model.reads(), vec![(model.base.offset(32), 80)]);
    assert_eq!(a.output_tokens, vec![1]);
    model.gpu.clear(1);
    assert_eq!(
        verify_pick_all_with_pipeline(&model, &[1; 5], &mut a, &context(&sched), 2),
        vec![1; 5]
    );
    assert_eq!(model.reads(), vec![(model.base.offset(32), 80)]);
}

#[test]
fn legacy_failed_probe_recovers_via_real_full_pipeline() {
    let model = target();
    let sched = SchedCtx::for_test();
    let mut a = active(vec![3]);
    let ctx = context(&sched);
    assert_eq!(
        verify_pick_all_with_pipeline(&model, &[1; 5], &mut a, &ctx, 2),
        vec![1; 5]
    );
    assert_eq!(
        model.reads(),
        (2..7)
            .map(|r| (model.base.offset((r * 8 + 1) * 2), 2))
            .collect::<Vec<_>>()
    );
    model.gpu.clear(1);
    assert_eq!(
        verify_pick_all_with_pipeline(&model, &[1; 5], &mut a, &ctx, 2),
        vec![1; 5]
    );
    assert_eq!(
        model.reads(),
        vec![(model.base.offset(34), 2), (model.base.offset(32), 80)]
    );
    // An unrelated mask is intentionally bypassed by the existing fast path.
    // After probe failure the real full pipeline must run and may change picks.
    model.gpu.clear(1);
    let mut ctx = context(&sched);
    ctx.think_end_token = Some(1);
    a.think_ended = true;
    assert_eq!(
        verify_pick_all_with_pipeline(&model, &[1; 5], &mut a, &ctx, 2),
        vec![2; 5]
    );
    assert_eq!(
        model.reads(),
        vec![(model.base.offset(34), 2), (model.base.offset(32), 80)]
    );
}

#[test]
fn legacy_neutral_and_empty_are_no_io() {
    let model = target();
    let sched = SchedCtx::for_test();
    let mut a = active(vec![1]);
    a.repetition_penalty = 1.0;
    model.gpu.clear(1);
    assert_eq!(
        verify_pick_all_with_pipeline(&model, &[1; 5], &mut a, &context(&sched), 2),
        vec![1; 5]
    );
    assert!(verify_pick_all_with_pipeline(&model, &[], &mut a, &context(&sched), 2).is_empty());
    assert!(model.reads().is_empty());
}
