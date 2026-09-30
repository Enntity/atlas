// SPDX-License-Identifier: AGPL-3.0-only

//! A chunk the cache covers completely: which chunk that is, and on the real
//! `prefill_chunk` that the switch changes nothing but the zero and embed of
//! those chunks. Then the worker's real chunk handler fed the prompt as a
//! delta, and the trace switch.

// The real model and recording layer of `prefill_stream_tests`.
#[allow(clippy::duplicate_mod)]
#[path = "../prefill_stream_test_fixture.rs"]
mod fixture;

use super::fully_cached;
use crate::layer::EmptyLayerState;
use crate::model::warm_turn::prompt_hash;
use crate::traits::{Model, SequenceState};
use fixture::*;

#[test]
fn a_chunk_is_cached_when_a_skip_covers_all_of_it_and_it_is_not_the_last() {
    // Chunk [8, 16) against restore depths around it.
    for (skip_to, want) in [(0, false), (8, false), (15, false), (16, true), (40, true)] {
        assert_eq!(fully_cached(true, skip_to, 8, 8, false), want, "{skip_to}");
        // The last chunk always runs its final row.
        assert!(!fully_cached(true, skip_to, 8, 8, true));
        // A prefix hit without a restore recomputes from token 0.
        assert!(!fully_cached(false, skip_to, 8, 8, false));
    }
    // Chunk 0.
    assert!(fully_cached(true, 8, 0, 8, false));
    assert!(!fully_cached(true, 7, 0, 8, false));
}

/// Re-run `name` in a child process with the Marconi restore floor lifted:
/// the fixture's prompts are tens of tokens. `true` in the parent.
fn in_child(name: &str) -> bool {
    const CHILD: &str = "ATLAS_WARM_TEST_CHILD";
    if std::env::var(CHILD).as_deref() == Ok("1") {
        return false;
    }
    let path = module_path!().split_once("::").unwrap().1;
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &format!("{path}::{name}"), "--nocapture"])
        .env(CHILD, "1")
        .env("ATLAS_MARCONI_MIN_TOKENS", "0")
        // As served: the prefill checkpoints are the only restore points.
        .env("ATLAS_MARCONI_PREFILL_ONLY", "1")
        .env_remove("ATLAS_GLM_DET_TRACE")
        .env_remove("ATLAS_NO_TAIL_SPLIT")
        .env_remove("ATLAS_MARCONI_EXACT")
        .env_remove("ATLAS_SSM_SAVE_DUMP")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    // A filter that matches nothing also exits 0: require the one test.
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed"),
        "child failed: {name}\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

const SENTINEL: [u8; 8] = [0xAB; 8];

/// Run the chunks of `tokens` (8-row chunks, the last one final) on `seq`.
/// Returns, per chunk, whether it zeroed the arena: a sentinel written to
/// the residual buffer before the chunk is gone after it.
fn run_chunks(f: &Fixture, seq: &mut SequenceState, tokens: &[u32]) -> Vec<bool> {
    let residual = f.model.buffers.residual();
    (0..tokens.len())
        .step_by(8)
        .map(|start| {
            f.model.gpu.copy_h2d(&SENTINEL, residual).unwrap();
            let last = start + 8 >= tokens.len();
            f.model
                .prefill_chunk(
                    tokens,
                    seq,
                    start,
                    tokens.len().min(start + 8) - start,
                    last,
                    CALLER,
                )
                .unwrap();
            let mut now = [0u8; 8];
            f.model.gpu.copy_d2h(residual, &mut now).unwrap();
            now == [0u8; 8]
        })
        .collect()
}

/// What a turn leaves: the layer passes, the sequence, the zeroed chunks.
type Turn = (Vec<Event>, (usize, usize, usize, usize, usize), Vec<bool>);

/// A cold 24-token turn, then the 28-token next turn of the conversation on
/// a fresh sequence (the pool's last free block), with the switch `on`.
/// Returns the second turn.
fn warm_turn(tp: usize, ep: usize, rank: usize, on: bool) -> Turn {
    let mut f = Fixture::with_tail_split(tp, ep, rank);
    f.disable_capture();
    f.model.warm.skip_cached = on;
    let tokens: Vec<u32> = (1..=28).collect();
    // Cold: [0,8), then [8,24) split at the tail cut 16, where the
    // checkpoint the next turn restores is saved.
    let mut first = std::mem::replace(&mut f.seq, SequenceState::host_only(0));
    f.model
        .prefill_chunk(&tokens[..24], &mut first, 0, 8, false, CALLER)
        .unwrap();
    f.model
        .prefill_chunk(&tokens[..24], &mut first, 8, 16, true, CALLER)
        .unwrap();
    let cold = f.events().len();
    assert_eq!(cold, 3, "three passes of 8 rows");

    let mut seq = SequenceState::host_only(0);
    seq.layer_states = vec![Box::new(EmptyLayerState)];
    seq.disk_last_offloaded_per_layer = vec![0];
    seq.proposer_state = first.proposer_state.take();
    let zeroed = run_chunks(&f, &mut seq, &tokens);
    assert_eq!(seq.tokens, tokens);
    let state = (
        seq.seq_len,
        seq.block_table.len(),
        seq.cached_prefix_tokens,
        seq.marconi_skip_to,
        seq.kv_valid_tokens,
    );
    (f.events()[cold..].to_vec(), state, zeroed)
}

/// The next turn matches the 24 cached tokens and restores at 16: chunks
/// [0,8) and [8,16) compute nothing, [16,24) replays and [24,28) is new.
/// With the switch a multi-rank world leaves the first two unzeroed and runs
/// the same two passes to the same sequence state; a single rank, which
/// zeroes once per request, is unchanged.
#[test]
fn actual_cached_chunks_skip_the_zero_and_nothing_else_changes() {
    if in_child("actual_cached_chunks_skip_the_zero_and_nothing_else_changes") {
        return;
    }
    for (tp, ep, rank) in [(2, 2, 0), (2, 2, 1), (2, 1, 0), (2, 1, 1), (1, 1, 0)] {
        let ctx = format!("TP{tp}/EP{ep}/rank{rank}");
        let (off, on) = (
            warm_turn(tp, ep, rank, false),
            warm_turn(tp, ep, rank, true),
        );
        let passes = &off.0;
        assert!(
            matches!(passes[..], [Event::Target(8, _), Event::Target(4, _)]),
            "{ctx}: {passes:?}"
        );
        assert_eq!(off.1, (28, 7, 24, 16, 28), "{ctx}");
        assert_eq!((&on.0, on.1), (passes, off.1), "{ctx}");
        if tp > 1 {
            assert_eq!(off.2, [true, true, true, true], "{ctx}");
            assert_eq!(on.2, [false, false, true, true], "{ctx}");
        } else {
            assert_eq!(off.2, [true, false, false, false], "{ctx}");
            assert_eq!(on.2, off.2, "{ctx}");
        }
    }
}

/// The cold turn has no chunk to skip: with the switch every chunk still
/// zeroes, and the lookup moving ahead of the zero changes no pass.
#[test]
fn actual_cold_chunks_all_zero_with_the_switch() {
    let tokens: Vec<u32> = (1..=24).collect();
    let cold = |on: bool| {
        let mut f = Fixture::with_tail_split(2, 2, 0);
        f.disable_capture();
        f.model.warm.skip_cached = on;
        let mut seq = std::mem::replace(&mut f.seq, SequenceState::host_only(0));
        let zeroed = run_chunks(&f, &mut seq, &tokens);
        (f.events(), seq.seq_len, seq.block_table.len(), zeroed)
    };
    let (off, on) = (cold(false), cold(true));
    assert_eq!(off.3, [true, true, true]);
    assert_eq!(on, off);
    assert_eq!(off.0.len(), 3);
}

/// The worker's real 0xFFFFFFF0 handler, driven through `ep_worker_step` by
/// the head's words. With the prompt delta the first chunk command carries
/// the whole cold prompt after its announce words (slot, base slot, shared
/// tokens, hash) and the second chunk of the same prompt carries no token;
/// either way the worker reads every word and runs the same passes to the
/// same sequence.
#[test]
fn actual_worker_chunks_run_the_same_from_the_prompt_delta() {
    let tokens: Vec<u32> = (1..=24).collect();
    let hash = prompt_hash(&tokens);
    let run = |delta: bool| {
        let mut f = Fixture::with_tail_split(2, 2, 1);
        f.disable_capture();
        f.model.warm.prompt_delta = delta;
        let v2_slot: &[u32] = if f.model.ep_protocol_v2 { &[0] } else { &[] };
        let prompt = |start: u32| match (delta, start) {
            (false, _) => tokens.clone(),
            (true, 0) => [&[0, 0, 0, hash][..], &tokens[..]].concat(),
            (true, _) => vec![0, 0, 24, hash],
        };
        let mut slots = [Some(std::mem::replace(
            &mut f.seq,
            SequenceState::host_only(0),
        ))];
        for (len, start) in [(8, 0), (16, 8)] {
            f.script(&[v2_slot, &[0xFFFF_FFF0, len, start, 24], &prompt(start)].concat());
            assert!(f.model.ep_worker_step(&mut slots).unwrap());
            assert_eq!(f.unread(), 0, "delta {delta}: command words left unread");
        }
        let s = slots[0].take().unwrap();
        (f.events(), s.tokens, s.seq_len, s.block_table.len())
    };
    let (bulk, delta) = (run(false), run(true));
    assert_eq!(bulk.0.len(), 3);
    assert_eq!((&bulk.1, bulk.2, bulk.3), (&tokens, 24, 6));
    assert_eq!(delta, bulk);
}

/// `ATLAS_GLM_WARM_TRACE` syncs the stream at every step of a chunk and logs
/// the request's line after the last one: the passes, the sequence and the
/// zeroed chunks are those of a run without it.
#[test]
fn actual_trace_changes_no_pass() {
    let tokens: Vec<u32> = (1..=24).collect();
    let cold = |trace: bool| {
        let mut f = Fixture::with_tail_split(2, 2, 0);
        f.disable_capture();
        f.model.warm.trace = trace;
        let mut seq = std::mem::replace(&mut f.seq, SequenceState::host_only(0));
        let zeroed = run_chunks(&f, &mut seq, &tokens);
        (f.events(), seq.seq_len, seq.block_table.len(), zeroed)
    };
    assert_eq!(cold(true), cold(false));
}
