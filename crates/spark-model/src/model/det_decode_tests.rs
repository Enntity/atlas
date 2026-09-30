// SPDX-License-Identifier: AGPL-3.0-only

//! The model-side decode determinism taps on the small real model of the
//! prefill stream fixture: what each logs, and that they are inert when off.

#[allow(dead_code)]
#[path = "trait_impl/prefill_stream_test_fixture.rs"]
mod fixture;

use crate::det_trace::{decode, hash_bytes, take_lines};
use crate::layer::SsmLayerState;
use crate::layers::DflashProposerState;
use fixture::*;
use spark_runtime::gpu::DevicePtr;

/// Re-run `name` in a child process with `ATLAS_GLM_DET_TRACE_DECODE` set to
/// `level`; `true` in the parent once the child passed.
fn isolated(name: &str, level: Option<&str>) -> bool {
    if std::env::var("ATLAS_DET_DECODE_TEST_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let path = module_path!().split_once("::").unwrap().1;
    let mut child = std::process::Command::new(std::env::current_exe().unwrap());
    for name in ["", "_DECODE", "_STAGES", "_STEPS", "_REQUESTS"] {
        child.env_remove(format!("ATLAS_GLM_DET_TRACE{name}"));
    }
    let output = child
        .envs(level.map(|level| ("ATLAS_GLM_DET_TRACE_DECODE", level)))
        .env_remove("ATLAS_GLM_VERIFY_VOCAB_SPLIT")
        .env("ATLAS_DET_DECODE_TEST_CHILD", "1")
        .args(["--exact", &format!("{path}::{name}"), "--nocapture"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed"),
        "child failed: {name}\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 + i / 251 + seed) as u8).collect()
}

/// What the taps read, with known bytes behind every device pointer.
struct Staged {
    f: Fixture,
    h: Vec<u8>,
    conv: Vec<u8>,
    stack: (DevicePtr, Vec<u8>),
    ctx: Vec<u8>,
    hidden: Vec<u8>,
    logits: Vec<u8>,
}

fn staged() -> Staged {
    // Rank 1 of the pair, one KDA layer with pooled state.
    let mut f = Fixture::with_tail_split(2, 2, 1);
    let model = &f.model;
    let put = |bytes: &[u8], ptr: DevicePtr| model.gpu.copy_h2d(bytes, ptr).unwrap();
    let pool = &model.ssm_pool;
    let (h_state, conv_state) = (pool.h_state(0, 0), pool.conv_state(0, 0));
    let h = pattern(pool.h_stored_bytes, 1);
    let conv = pattern(pool.conv_bytes, 2);
    put(&h, h_state);
    put(&conv, conv_state);
    let stack = (model.gpu.alloc(16).unwrap(), pattern(16, 3));
    put(&stack.1, stack.0);
    let ctx = pattern(32, 4);
    let ctx_hidden_acc = model.gpu.alloc(64).unwrap();
    put(&ctx, ctx_hidden_acc);
    let hidden = pattern(3 * model.config.hidden_size * 2, 5);
    put(&hidden, model.buffers.hidden_states());
    // Three rows of eight BF16 logits: 1.0 everywhere, then one 3.0 per row
    // and a 2.5 in the last row.
    let mut logits: Vec<u8> = [0x80u8, 0x3f].repeat(24);
    for (row, column, value) in [
        (0, 1, 0x4040u16),
        (1, 7, 0x4040),
        (2, 0, 0x4040),
        (2, 5, 0x4020),
    ] {
        logits[(row * 8 + column) * 2..][..2].copy_from_slice(&value.to_le_bytes());
    }
    put(&logits, model.buffers.logits());
    f.seq.layer_states = vec![Box::new(SsmLayerState {
        h_state,
        conv_state,
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        kda_records: DevicePtr::NULL,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    })];
    f.seq.proposer_state = Some(Box::new(DflashProposerState {
        block_table: Vec::new(),
        seq_len: 0,
        last_num_drafted: 0,
        prefill_done: true,
        ctx_hidden_acc,
        ctx_len: 2,
        last_num_accepted: 1,
        skip_next_decode_append: true,
        max_ctx_len: 4,
        ctx_slot_bytes: 16,
        lane_id: 0,
        lifecycle: None,
        block_table_dev: None,
        ctx_count_drafter: 2,
        max_ctx_count_drafter: 16,
        ctx_committed: 1,
        ctx_positions: vec![22, 23],
        end_floor: 30,
    }));
    (f.seq.cached_prefix_tokens, f.seq.marconi_skip_to) = (16, 12);
    (f.seq.kv_valid_tokens, f.seq.seq_len) = (20, 24);
    Staged {
        f,
        h,
        conv,
        stack,
        ctx,
        hidden,
        logits,
    }
}

/// One request's taps in forward order: prefill done, a propose, a verify of
/// three rows, the commit of two.
fn run(s: &mut Staged) {
    let (model, seq) = (&s.f.model, &mut s.f.seq);
    crate::det_trace::begin_request(seq.slot_idx);
    model.det_decode_pre(seq, 24, DEFAULT);
    let at = model.det_propose_enter(seq, 11, 24, 7, Some(s.stack.0));
    for drafts in [[5, 6], [8, 9]] {
        if let Some(at) = at {
            let state = seq.proposer_state.as_deref().unwrap();
            model.det_propose_done(at, seq.slot_idx, state, &drafts);
        }
    }
    {
        let _scope = model.det_verify_enter(&[11, 5, 6], seq, DEFAULT);
        model.det_verify_layer_out(3, DEFAULT);
        model.det_verify_done(&[5, 9, 2], DEFAULT);
    }
    seq.seq_len = 26;
    model.det_committed(seq, 2, 3);
}

#[test]
fn decode_taps_log_each_stage_in_forward_order() {
    if isolated("decode_taps_log_each_stage_in_forward_order", Some("1")) {
        return;
    }
    let mut s = staged();
    run(&mut s);
    let words =
        |values: &[u32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let line = |t: u32, p: u32, layer: u32, stage: &str, n: usize, bytes: &[u8], v: &str| {
        let v = if v.is_empty() {
            String::new()
        } else {
            format!(" v={v}")
        };
        let (b, h) = (bytes.len(), hash_bytes(bytes));
        format!("DETD r=1 q=1 t={t} p={p} L={layer} s={stage} r0=0 n={n} b={b} h={h:016x}{v}")
    };
    let values = |t, p, layer, stage, v: &[u32]| {
        let text: Vec<String> = v.iter().map(u32::to_string).collect();
        line(t, p, layer, stage, v.len(), &words(v), &text.join(","))
    };
    assert_eq!(
        take_lines(),
        [
            values(0, 24, 1, "pre", &[0, 24, 16, 12, 20, 2]),
            line(0, 24, 1, "kda_h", 1, &s.h, ""),
            line(0, 24, 1, "kda_conv", 1, &s.conv, ""),
            values(1, 24, 0, "d_in", &[11, 24, 7, 2, 1, 2, 0, 1, 1, 0, 30]),
            line(1, 24, 0, "d_hid", 1, &s.stack.1, ""),
            line(1, 24, 0, "d_ctx0", 2, &s.ctx, "2"),
            line(1, 24, 0, "d_pos", 2, &words(&[22, 23]), "2,22,23"),
            values(1, 24, 0, "d_out", &[5, 6]),
            // Only the request's first propose hashes the whole context.
            line(1, 24, 0, "d_pos", 2, &words(&[22, 23]), "2,22,23"),
            values(1, 24, 0, "d_out", &[8, 9]),
            values(1, 24, 0, "tok", &[11, 5, 6]),
            line(1, 24, 0, "emb", 3, &s.hidden, ""),
            line(1, 24, 1, "final", 3, &s.hidden, ""),
            line(1, 24, 1, "logits", 3, &s.logits, "2.0000,2.0000,0.5000"),
            values(1, 24, 1, "top1", &[5, 9, 2]),
            values(1, 26, 1, "acc", &[2, 3]),
            // The recurrent state after a commit (`st_h`) must be named.
            line(1, 26, 1, "st_conv", 1, &s.conv, ""),
        ]
    );
    // The commit ended step 1; no layer ran and no scope is left behind.
    assert_eq!(decode::step_at(1, 0, 26).unwrap().step, Some(2));
    assert!(decode::scope_at().is_none() && s.f.events().is_empty());
}

#[test]
fn decode_taps_are_inert_when_unset() {
    if isolated("decode_taps_are_inert_when_unset", None) {
        return;
    }
    let mut s = staged();
    s.f.order();
    run(&mut s);
    assert!(take_lines().is_empty());
    // Not one synchronize, wait or layer call.
    assert!(s.f.order().is_empty() && s.f.events().is_empty());
}
