// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Model scalar boundaries; no native arithmetic or terminal-driver claim.
use super::{sample_token_with_grammar, sample_token_with_grammar_checked};
use crate::scheduler::{
    logit_processors::SamplingLevers,
    sample_step::{PositionKind, penalty_history_scope, penalty_params_for},
    test_support::test_owned_seq,
    types::ActiveSeq,
};
use spark_model::{
    model::{
        TransformerModel,
        glm_c2_test_support::{Event, Fixture, Observer, Snapshot},
    },
    traits::Model,
};
use spark_runtime::gpu::DevicePtr;

#[path = "glm_c2_fixture_test_process.rs"]
mod process;

fn isolated(name: &str) -> bool {
    process::isolated(&format!(
        "scheduler::sample_step::checked_scalar::tests::{name}"
    ))
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Neutral,
    Reduce,
    History,
    Blocked,
    Suppress,
}

struct Harness {
    model: TransformerModel,
    active: [ActiveSeq; 2],
    observer: Observer,
    logits: DevicePtr,
    owner: usize,
    levers: SamplingLevers,
}
struct Saved {
    tokens: [Vec<u32>; 2],
    lengths: [usize; 2],
    cursors: [usize; 2],
    snapshots: [Snapshot; 2],
    outputs: [Vec<u32>; 2],
}
impl Harness {
    fn new(rank: usize, owner: usize) -> Self {
        let mut fixture = Fixture::paired(rank);
        fixture.deterministic_logits(true);
        let (model, sequences) = fixture.parts_mut();
        let prompts = [vec![1, 2, 3, 4], vec![6, 5, 4, 3, 2, 1]];
        for i in 0..2 {
            sequences[i].prompt_len = prompts[i].len();
            model.prefill(&prompts[i], &mut sequences[i], 37).unwrap();
        }
        // The sampled owner's real producer is last; never read peer-overwritten logits.
        model
            .decode(5 + (1 - owner) as u32, &mut sequences[1 - owner], 37)
            .unwrap();
        let logits = model
            .decode(5 + owner as u32, &mut sequences[owner], 37)
            .unwrap();
        let (model, sequences, observer) = fixture.into_parts();
        let active = sequences.map(|seq| {
            let (mut a, _) = test_owned_seq(seq, Vec::new(), 32, None);
            a.lz_penalty = 0.0;
            a.dry_multiplier = 0.0;
            a.min_tokens = 0;
            a.finished = false;
            a
        });
        observer.clear();
        Self {
            model,
            active,
            observer,
            logits,
            owner,
            levers: SamplingLevers {
                fast_greedy_grammar: true,
                force_temp_zero: false,
                mtp_minp: true,
                ..Default::default()
            },
        }
    }
    fn raw(&self) -> u32 {
        let raw = self.model.argmax_on_device(self.logits, 0).unwrap();
        assert!(raw < 8);
        self.observer.clear();
        raw
    }
    fn configure(&mut self, kind: Kind, raw: u32) -> Vec<u32> {
        let a = &mut self.active[self.owner];
        a.repetition_penalty = if matches!(kind, Kind::Neutral) {
            1.0
        } else {
            2.0
        };
        a.presence_penalty = if matches!(kind, Kind::Blocked) {
            -0.25
        } else {
            0.0
        };
        a.output_tokens = if matches!(kind, Kind::History | Kind::Blocked) {
            vec![raw]
        } else {
            vec![]
        };
        if matches!(kind, Kind::Suppress) {
            vec![raw]
        } else {
            vec![]
        }
    }
    fn sample(&mut self, checked: bool, suppress: &[u32]) -> anyhow::Result<u32> {
        let a = &mut self.active[self.owner];
        let penalties = penalty_params_for(a, PositionKind::Verify, 0.0, None, Vec::new());
        let history = penalty_history_scope(&a.output_tokens, a.tool_call_end_token).to_vec();
        let sample = if checked {
            sample_token_with_grammar_checked
        } else {
            sample_token_with_grammar
        };
        sample(
            &self.model,
            self.logits,
            0.0,
            0,
            1.0,
            suppress,
            a.grammar_state.as_mut(),
            &penalties,
            &history,
            &self.levers,
        )
    }
    fn save(&self) -> Saved {
        let cursors = [0, 1].map(|i| {
            self.observer
                .private_cursor(&self.model, &self.active[i].seq)
                .unwrap()
        });
        Saved {
            tokens: [0, 1].map(|i| self.active[i].seq.tokens.clone()),
            lengths: [0, 1].map(|i| self.active[i].seq.seq_len),
            outputs: [0, 1].map(|i| self.active[i].output_tokens.clone()),
            snapshots: [0, 1].map(|i| {
                self.observer
                    .snapshot(&self.model, &self.active[i].seq, cursors[i])
                    .unwrap()
            }),
            cursors,
        }
    }
    fn unchanged(&self, saved: &Saved) {
        for i in 0..2 {
            assert_eq!(self.active[i].seq.tokens, saved.tokens[i]);
            assert_eq!(self.active[i].seq.seq_len, saved.lengths[i]);
            assert_eq!(self.active[i].output_tokens, saved.outputs[i]);
            assert_eq!(
                self.observer
                    .private_cursor(&self.model, &self.active[i].seq)
                    .unwrap(),
                saved.cursors[i]
            );
            assert_eq!(
                self.observer.read_snapshot(&saved.snapshots[i]).unwrap(),
                saved.snapshots[i].initial()
            );
        }
    }
    fn close(mut self) {
        // Test cleanup after returned local sampler errors, not a serving recovery policy.
        self.observer.clear();
        for a in &mut self.active {
            self.model.free_sequence(&mut a.seq).unwrap();
        }
        drop(self.active);
        drop(self.observer);
        self.model.teardown().unwrap();
    }
}

fn reads(events: &[Event]) -> Vec<usize> {
    events
        .iter()
        .filter_map(|e| {
            if let Event::Read(n, 7) = e {
                Some(*n)
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn checked_and_legacy_greedy_paths_share_actual_model_bytes() {
    if isolated("checked_and_legacy_greedy_paths_share_actual_model_bytes") {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let mut h = Harness::new(rank, owner);
            let raw = h.raw();
            for (kind, expected_reads) in [
                (Kind::Neutral, vec![4]),
                (Kind::Reduce, vec![4, 2]),
                (Kind::History, vec![4, 16]),
                (Kind::Blocked, vec![16]),
                (Kind::Suppress, vec![16]),
            ] {
                let suppress = h.configure(kind, raw);
                let saved = h.save();
                h.observer.clear();
                let legacy = h.sample(false, &suppress).unwrap();
                let control = h.observer.events();
                assert_eq!(reads(&control), expected_reads, "actual {kind:?}");
                h.unchanged(&saved);
                h.observer.clear();
                let checked = h.sample(true, &suppress).unwrap();
                assert_eq!(checked, legacy, "actual {kind:?}");
                assert_eq!(h.observer.events(), control);
                h.unchanged(&saved);
            }
            h.close();
        }
    }
}

#[test]
fn checked_reduce_probe_error_is_not_legacy_full_read_success() {
    if isolated("checked_reduce_probe_error_is_not_legacy_full_read_success") {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            let mut control = Harness::new(rank, owner);
            let raw = control.raw();
            control.configure(Kind::Reduce, raw);
            control.sample(false, &[]).unwrap();
            let events = control.observer.events();
            assert_eq!(reads(&events), vec![4, 2]);
            let ordinal = events.iter().position(|e| *e == Event::Read(2, 7)).unwrap() + 1;
            control.close();
            for checked in [false, true] {
                let mut h = Harness::new(rank, owner);
                let raw = h.raw();
                h.configure(Kind::Reduce, raw);
                let saved = h.save();
                h.observer.fail_at(ordinal);
                let result = h.sample(checked, &[]);
                let actual = h.observer.events();
                assert_eq!(actual[ordinal - 1], Event::Read(2, 7));
                assert_eq!(&actual[..ordinal], &events[..ordinal]);
                if checked {
                    let error =
                        result.expect_err("checked probe failure must not become raw/full success");
                    assert!(format!("{error:#}").contains("injected fixture operation failure"));
                    assert_eq!(
                        actual.len(),
                        ordinal,
                        "no later full read after probe error"
                    );
                } else {
                    assert!(result.unwrap() < 8);
                    assert_eq!(reads(&actual), vec![4, 2, 16]);
                    assert_eq!(actual.len(), ordinal + 1, "one legacy full-copy recovery");
                }
                h.unchanged(&saved);
                h.close();
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    ArgmaxKernel,
    ArgmaxRead,
    FullRead,
}
#[test]
fn actual_argmax_and_full_read_faults_stop_at_failed_boundary() {
    if isolated("actual_argmax_and_full_read_faults_stop_at_failed_boundary") {
        return;
    }
    for rank in 0..2 {
        for owner in 0..2 {
            for fault in [Fault::ArgmaxKernel, Fault::ArgmaxRead, Fault::FullRead] {
                let kind = if matches!(fault, Fault::FullRead) {
                    Kind::History
                } else {
                    Kind::Reduce
                };
                let mut control = Harness::new(rank, owner);
                let raw = control.raw();
                control.configure(kind, raw);
                control.sample(false, &[]).unwrap();
                let events = control.observer.events();
                let ordinal = events
                    .iter()
                    .position(|e| match fault {
                        Fault::ArgmaxKernel => {
                            matches!(e,Event::Kernel(name,7) if name=="argmax_bf16")
                        }
                        Fault::ArgmaxRead => *e == Event::Read(4, 7),
                        Fault::FullRead => *e == Event::Read(16, 7),
                    })
                    .expect("real successful scalar boundary")
                    + 1;
                control.close();
                for checked in [false, true] {
                    let mut h = Harness::new(rank, owner);
                    let raw = h.raw();
                    h.configure(kind, raw);
                    let saved = h.save();
                    h.observer.fail_at(ordinal);
                    let error = h
                        .sample(checked, &[])
                        .expect_err("actual scalar failure must propagate");
                    assert!(format!("{error:#}").contains("injected fixture operation failure"));
                    assert_eq!(h.observer.events(), events[..ordinal], "actual {fault:?}");
                    h.unchanged(&saved);
                    h.close();
                }
            }
        }
    }
}

#[test]
fn checked_grammar_refuses_before_actual_argmax() {
    if isolated("checked_grammar_refuses_before_actual_argmax") {
        return;
    }
    let vocabulary = ["{", "}", " ", "\"", "a", ":", ",", "0"].map(str::to_owned);
    let mut engine = crate::grammar::GrammarEngine::new(&vocabulary, &[]).unwrap();
    let grammar = engine.compile_json_grammar().unwrap();
    for rank in 0..2 {
        for owner in 0..2 {
            let mut h = Harness::new(rank, owner);
            let raw = h.raw();
            h.configure(Kind::Neutral, raw);
            h.active[owner].grammar_state =
                Some(crate::grammar::GrammarState::new(&grammar, 8).unwrap());
            let before = h.active[owner]
                .grammar_state
                .as_ref()
                .unwrap()
                .num_history_steps();
            let saved = h.save();
            let legacy = h.sample(false, &[]).unwrap();
            assert!(
                h.active[owner]
                    .grammar_state
                    .as_ref()
                    .unwrap()
                    .is_token_allowed(legacy)
            );
            assert_eq!(
                h.active[owner]
                    .grammar_state
                    .as_ref()
                    .unwrap()
                    .num_history_steps(),
                before
            );
            h.unchanged(&saved);
            h.observer.clear();
            let error = h
                .sample(true, &[])
                .expect_err("checked scalar must refuse actual grammar");
            assert!(format!("{error:#}").contains("grammar"));
            assert!(h.observer.events().is_empty());
            assert_eq!(
                h.active[owner]
                    .grammar_state
                    .as_ref()
                    .unwrap()
                    .num_history_steps(),
                before
            );
            h.unchanged(&saved);
            h.close();
        }
    }
}
