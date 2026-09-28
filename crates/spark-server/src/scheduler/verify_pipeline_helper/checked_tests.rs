// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    tests::{active, context, target},
    *,
};
use crate::scheduler::sched_ctx::SchedCtx;

#[test]
fn checked_actual_backend_full_copy_error_is_not_raw_success() {
    let model = target();
    let sched = SchedCtx::for_test();
    let mut a = active(vec![1]);
    let ctx = context(&sched);
    assert_eq!(
        verify_pick_all_with_pipeline(&model, &[1; 5], &mut a, &ctx, 2),
        vec![2; 5]
    );
    let control = model.reads();
    assert_eq!(control, vec![(model.base.offset(32), 80)]);
    model.gpu.clear(1);
    let err = verify_pick_all_with_pipeline_checked(&model, &[1; 5], &mut a, &ctx, 2).unwrap_err();
    assert!(err.to_string().contains("injected backend logits copy 1"));
    assert_eq!(model.reads(), control);
    assert_eq!(a.output_tokens, vec![1]);
}

#[test]
fn checked_actual_probe_error_never_falls_through() {
    for masked in [false, true] {
        let model = target();
        let sched = SchedCtx::for_test();
        let mut a = active(vec![3]);
        let mut ctx = context(&sched);
        ctx.sampling.dflash_masked_verify = masked;
        ctx.sampling.fast_masked = true;
        assert_eq!(
            verify_pick_all_with_pipeline(&model, &[1; 5], &mut a, &ctx, 2),
            vec![1; 5]
        );
        let control = model.reads();
        assert_eq!(control.len(), 5);
        for ordinal in 1..=5 {
            model.gpu.clear(ordinal);
            let err = verify_pick_all_with_pipeline_checked(&model, &[1; 5], &mut a, &ctx, 2)
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains(&format!("injected backend logits copy {ordinal}"))
            );
            assert_eq!(model.reads(), control[..ordinal]);
        }
    }
}

#[test]
fn checked_penalty_and_seeded_sampling_reuse_real_pipeline() {
    let model = target();
    let sched = SchedCtx::for_test();
    let ctx = context(&sched);
    for temperature in [0.0, 0.7] {
        for seed in [11, 29, 31] {
            let mut a = active(vec![1]);
            a.temperature = temperature;
            a.seed = Some(seed);
            model.gpu.clear(0);
            let legacy = verify_pick_all_with_pipeline(&model, &[1; 5], &mut a, &ctx, 2);
            let reads = model.reads();
            let mut b = active(vec![1]);
            b.temperature = temperature;
            b.seed = Some(seed);
            model.gpu.clear(0);
            let checked =
                verify_pick_all_with_pipeline_checked(&model, &[1; 5], &mut b, &ctx, 2).unwrap();
            assert_eq!(checked, legacy);
            assert_eq!(model.reads(), reads);
            assert_eq!(a.output_tokens, b.output_tokens);
            if temperature == 0.0 {
                assert_eq!(checked, vec![2; 5]);
            }
        }
    }
}

#[test]
fn checked_neutral_and_empty_keep_zero_io() {
    let model = target();
    let sched = SchedCtx::for_test();
    let mut a = active(vec![1]);
    a.repetition_penalty = 1.0;
    model.gpu.clear(1);
    assert_eq!(
        verify_pick_all_with_pipeline_checked(&model, &[1; 5], &mut a, &context(&sched), 2)
            .unwrap(),
        vec![1; 5]
    );
    assert!(
        verify_pick_all_with_pipeline_checked(&model, &[], &mut a, &context(&sched), 2)
            .unwrap()
            .is_empty()
    );
    assert!(model.reads().is_empty());
}

#[test]
fn checked_actual_grammar_is_refused_even_for_empty_without_mutation() {
    let model = target();
    let sched = SchedCtx::for_test();
    let vocab: Vec<String> = (0..128u8).map(|b| (b as char).to_string()).collect();
    let mut engine = crate::grammar::GrammarEngine::new(&vocab, &[]).unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    let mut a = active(vec![1]);
    let mut grammar = crate::grammar::GrammarState::new(&compiled, vocab.len()).unwrap();
    assert!(grammar.accept_token(b'{' as u32));
    let before = grammar.num_history_steps();
    a.grammar_state = Some(grammar);
    model.gpu.clear(1);
    for raw in [&[][..], &[1; 5][..]] {
        let err = verify_pick_all_with_pipeline_checked(&model, raw, &mut a, &context(&sched), 2)
            .unwrap_err();
        assert!(err.to_string().contains("grammarless"));
        assert_eq!(
            a.grammar_state.as_ref().unwrap().num_history_steps(),
            before
        );
        assert!(model.reads().is_empty());
    }
}

#[test]
fn actual_probe_formats_and_valid_nonpositive_are_not_io_errors() {
    use super::test_model::Target;
    for fp32 in [false, true] {
        for value in [0.0, -1.0, f32::NAN, 4.0] {
            let model = Target::new(&[vec![0.0; 8], vec![value; 8]], fp32);
            let expected = value > 0.0;
            assert_eq!(
                crate::scheduler::fast_greedy::logit_is_positive(&model, model.base, 1, 8, 2),
                expected
            );
            let width = if fp32 { 4 } else { 2 };
            assert_eq!(model.reads(), vec![(model.base.offset(10 * width), width)]);
            model.gpu.clear(0);
            assert_eq!(
                crate::scheduler::fast_greedy::logit_is_positive_checked(
                    &model, model.base, 1, 8, 2
                )
                .unwrap(),
                expected
            );
            model.gpu.clear(1);
            assert!(!crate::scheduler::fast_greedy::logit_is_positive(
                &model, model.base, 1, 8, 2
            ));
            model.gpu.clear(1);
            assert!(
                crate::scheduler::fast_greedy::logit_is_positive_checked(
                    &model, model.base, 1, 8, 2
                )
                .unwrap_err()
                .to_string()
                .contains("injected backend logits copy 1")
            );
        }
    }
}

#[test]
fn legacy_real_grammar_fast_and_full_paths_restore_history() {
    use super::test_model::Target;
    let sched = SchedCtx::for_test();
    let vocab: Vec<String> = (0..128u8).map(|b| (b as char).to_string()).collect();
    let mut engine = crate::grammar::GrammarEngine::new(&vocab, &[]).unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    let raw = [b'{', b'"', b'a', b'"', b':'].map(u32::from);
    let mut rows = vec![vec![-8.0; 128]; 7];
    for (row, &token) in rows[2..].iter_mut().zip(raw.iter()) {
        row[token as usize] = 4.0;
    }
    let model = Target::new(&rows, false);
    for fast in [true, false] {
        let mut a = active(vec![]);
        a.repetition_penalty = 1.0;
        a.grammar_state = Some(crate::grammar::GrammarState::new(&compiled, vocab.len()).unwrap());
        let mut ctx = context(&sched);
        ctx.sampling.fast_greedy_grammar = fast;
        let before = a.grammar_state.as_ref().unwrap().num_history_steps();
        model.gpu.clear(0);
        assert_eq!(
            verify_pick_all_with_pipeline(&model, &raw, &mut a, &ctx, 2),
            raw
        );
        assert_eq!(
            a.grammar_state.as_ref().unwrap().num_history_steps(),
            before
        );
        assert_eq!(
            model.reads(),
            if fast {
                vec![]
            } else {
                vec![(model.base.offset(512), 1280)]
            }
        );
    }
}
