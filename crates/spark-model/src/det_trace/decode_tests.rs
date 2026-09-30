// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::det_trace::{hash_bytes, on_stream, take_lines};
use spark_runtime::gpu::mock::MockGpuBackend;

const STEP: At = At {
    rank: 0,
    request: 21,
    chunk_start: 57,
    layer: 3,
    step: Some(4),
};

/// Re-run `name` in a child process with exactly `env` of the tracer's
/// variables set; `true` in the parent once the child passed.
fn isolated(name: &str, env: &[(&str, &str)]) -> bool {
    if std::env::var("ATLAS_DET_DECODE_TEST_CHILD").as_deref() == Ok("1") {
        return false;
    }
    let path = module_path!().split_once("::").unwrap().1;
    let mut child = std::process::Command::new(std::env::current_exe().unwrap());
    for name in ["", "_DECODE", "_STAGES", "_STEPS", "_REQUESTS"] {
        child.env_remove(format!("ATLAS_GLM_DET_TRACE{name}"));
    }
    let output = child
        .envs(env.iter().copied())
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

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn ranges_parse_and_fall_back_to_everything() {
    let all = (0, u64::MAX);
    for (text, want) in [
        (None, all),
        (Some(""), all),
        (Some("junk"), all),
        (Some("3-x"), all),
        (Some("7"), (7, 7)),
        (Some("3-9"), (3, 9)),
        (Some(" 3 - 9 "), (3, 9)),
        (Some("5-"), (5, u64::MAX)),
        (Some("-5"), (0, 5)),
    ] {
        assert_eq!(parse_range(text), want, "{text:?}");
    }
    assert!(within((3, 9), 3) && within((3, 9), 9));
    assert!(!within((3, 9), 2) && !within((3, 9), 10));
}

#[test]
fn default_stages_are_the_documented_set_and_a_list_replaces_them() {
    for stage in [
        "pre", "d_out", "tok", "in", "sel", "attn_red", "ffn", "moe", "acc",
    ] {
        assert!(stage_listed(None, stage), "{stage}");
    }
    // The prefill-only stages and the large per-step hashes must be named.
    for stage in [
        "attn", "kv_lat", "rt_ids", "out", "x_qkv", "x_d_ctx", "st_h", "plogits",
    ] {
        assert!(!stage_listed(None, stage), "{stage}");
    }
    assert!(stage_listed(Some("out,x_d_ctx"), "x_d_ctx"));
    assert!(!stage_listed(Some("out,x_d_ctx"), "in"));
    // A prefill chunk keeps the prefill tracer's selection.
    let prefill = At { step: None, ..STEP };
    assert!(selected(&prefill, None, "attn") && !selected(&prefill, None, "x_qkv"));
    assert!(selected(&STEP, None, "in") && !selected(&STEP, None, "attn"));
}

#[test]
fn line_format_is_fixed_and_lists_the_values() {
    if isolated("line_format_is_fixed_and_lists_the_values", &[]) {
        return;
    }
    values(STEP, "tok", 0, &[5, 70000, 9]);
    hashed(STEP, "d_pos", (0, 2), &words(&[3, 4]), "2,3,4");
    values(STEP, "attn", 0, &[1]); // not a default stage
    let tok = hash_bytes(&words(&[5, 70000, 9]));
    let pos = hash_bytes(&words(&[3, 4]));
    assert_eq!(
        take_lines(),
        [
            format!("DETD r=0 q=21 t=4 p=57 L=3 s=tok r0=0 n=3 b=12 h={tok:016x} v=5,70000,9"),
            format!("DETD r=0 q=21 t=4 p=57 L=3 s=d_pos r0=0 n=2 b=8 h={pos:016x} v=2,3,4"),
        ]
    );
}

#[test]
fn spans_hash_the_concatenation_once_synchronized() {
    if isolated("spans_hash_the_concatenation_once_synchronized", &[]) {
        return;
    }
    let gpu = MockGpuBackend::new();
    let data: Vec<u8> = (0..96).collect();
    let ptr = gpu.alloc(data.len()).unwrap();
    gpu.copy_h2d(&data, ptr).unwrap();
    // Two layers' states with a gap between them.
    let parts = [(ptr, 32), (ptr.offset(64), 32)];
    spans(STEP, &gpu, 0, "kda_h", 2, &parts, "");
    spans(STEP, &gpu, 0, "kda_conv", 0, &[], "");
    spans(STEP, &gpu, 0, "x_d_ctx", 2, &parts, "2");
    let want = hash_bytes(&[&data[..32], &data[64..]].concat());
    assert_eq!(
        take_lines(),
        [format!(
            "DETD r=0 q=21 t=4 p=57 L=3 s=kda_h r0=0 n=2 b=64 h={want:016x}"
        )]
    );
    assert_eq!(gpu.sync_count(), 1);
    // A span the device cannot serve is reported, not hashed.
    spans(STEP, &gpu, 0, "kda_h", 1, &[(DevicePtr(0x10), 16)], "");
    assert!(take_lines()[0].ends_with("s=kda_h r0=0 n=1 b=16 h=ERR"));
}

#[test]
fn logits_hash_the_rows_and_report_each_top2_margin() {
    if isolated("logits_hash_the_rows_and_report_each_top2_margin", &[]) {
        return;
    }
    assert_eq!(top2_margin(&bf16(&[1.0, 4.0, 3.5, -2.0])), 0.5);
    assert_eq!(top2_margin(&bf16(&[2.0, 2.0, 1.0])), 0.0);
    assert_eq!(top2_margin(&bf16(&[-1.0, -3.0])), 2.0);
    assert_eq!(top2_margin(&bf16(&[7.0])), f32::INFINITY);
    // Two rows of four columns; this rank computed columns 2..4 of each.
    let gpu = MockGpuBackend::new();
    let data = bf16(&[9.0, 9.0, 1.0, 4.0, 9.0, 9.0, 8.0, 7.75]);
    let ptr = gpu.alloc(data.len()).unwrap();
    gpu.copy_h2d(&data, ptr).unwrap();
    logits(STEP, &gpu, 0, &[(ptr.offset(4), 4), (ptr.offset(12), 4)]);
    let want = hash_bytes(&[&data[4..8], &data[12..]].concat());
    assert_eq!(
        take_lines(),
        [format!(
            "DETD r=0 q=21 t=4 p=57 L=3 s=logits r0=0 n=2 b=8 h={want:016x} v=3.0000,0.2500"
        )]
    );
}

#[test]
fn off_is_silent_and_numbers_nothing() {
    if isolated("off_is_silent_and_numbers_nothing", &[]) {
        return;
    }
    assert!(!on() && !eager() && !wanted("tok"));
    assert!(!crate::det_trace::begin_request(2));
    assert_eq!(SLOT_REQUEST[2].load(Ordering::Relaxed), 0);
    assert!(pre_at(0, 2, 10).is_none() && step_at(0, 2, 10).is_none());
    assert!(mute().is_none() && scope_at().is_none());
    end_step(2);
    assert_eq!(SLOT_STEP[2].load(Ordering::Relaxed), 0);
    serial(0, 2, 10, 99);
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(64).unwrap();
    on_stream(&gpu, 0).tap("in", ptr, (0, 1), 64);
    assert!(take_lines().is_empty());
    assert_eq!((gpu.sync_count(), gpu.d2h_blocking_count()), (0, 0));
}

#[test]
fn steps_are_numbered_per_request_and_scopes_route_the_layer_taps() {
    let name = "steps_are_numbered_per_request_and_scopes_route_the_layer_taps";
    if isolated(name, &[("ATLAS_GLM_DET_TRACE_DECODE", "1")]) {
        return;
    }
    assert!(on() && !eager() && !crate::det_trace::on());
    // Decode tracing numbers the request but never asks for a recompute, and
    // leaves the prefill chunk scope off.
    assert!(!crate::det_trace::begin_request(5));
    assert!(crate::det_trace::enter(1, 5, 0).is_none());
    let pre = pre_at(1, 5, 40).unwrap();
    assert_eq!((pre.request, pre.step, pre.chunk_start), (1, Some(0), 40));
    let gpu = MockGpuBackend::new();
    let data: Vec<u8> = (0..64).collect();
    let ptr = gpu.alloc(data.len()).unwrap();
    gpu.copy_h2d(&data, ptr).unwrap();
    let hash = hash_bytes(&data);
    let det = on_stream(&gpu, 0);

    assert!(first_propose(5) && !first_propose(5));
    let step = step_at(1, 5, 40).unwrap();
    assert_eq!(step.step, Some(1));
    {
        let _scope = enter(step);
        assert_eq!(scope_at(), Some(step));
        crate::det_trace::set_layer(7);
        det.tap("in", ptr, (0, 2), 32);
        det.tap("attn", ptr, (0, 2), 32); // a prefill stage: named only
        {
            let _mute = mute();
            det.tap("in", ptr, (0, 2), 32); // inside a captured run
        }
        det.tap("moe", ptr, (0, 2), 32);
    }
    assert!(scope_at().is_none());
    det.tap("in", ptr, (0, 2), 32); // outside any scope
    // The same step until it is committed; then the next one.
    assert_eq!(step_at(1, 5, 40).unwrap().step, Some(1));
    serial(1, 5, 43, 77);
    end_step(5);
    assert_eq!(step_at(1, 5, 43).unwrap().step, Some(2));
    let dec = hash_bytes(&77u32.to_le_bytes());
    assert_eq!(
        take_lines(),
        [
            format!("DETD r=1 q=1 t=1 p=40 L=7 s=in r0=0 n=2 b=64 h={hash:016x}"),
            format!("DETD r=1 q=1 t=1 p=40 L=7 s=moe r0=0 n=2 b=64 h={hash:016x}"),
            format!("DETD r=1 q=1 t=1 p=43 L=0 s=dec r0=43 n=1 b=4 h={dec:016x} v=77"),
        ]
    );
    // The slot's next request starts over.
    crate::det_trace::begin_request(5);
    let next = step_at(1, 5, 9).unwrap();
    assert_eq!((next.request, next.step), (2, Some(1)));
    assert!(first_propose(5));
}

#[test]
fn step_request_and_stage_filters_bound_the_trace() {
    let env = [
        ("ATLAS_GLM_DET_TRACE_DECODE", "2"),
        ("ATLAS_GLM_DET_TRACE_STEPS", "2-3"),
        ("ATLAS_GLM_DET_TRACE_REQUESTS", "2-"),
        ("ATLAS_GLM_DET_TRACE_STAGES", "tok,out"),
    ];
    if isolated("step_request_and_stage_filters_bound_the_trace", &env) {
        return;
    }
    assert!(on() && eager() && wanted("out") && !wanted("in"));
    crate::det_trace::begin_request(0);
    // Request 1 is outside the request range: only its `pre` line logs.
    assert!(!traced(&pre_at(0, 0, 8).unwrap()) && step_at(0, 0, 8).is_none());
    crate::det_trace::begin_request(0);
    assert!(traced(&pre_at(0, 0, 8).unwrap()), "step 0 always logs");
    let traced: Vec<bool> = (1..=4)
        .map(|_| {
            let traced = step_at(0, 0, 8).is_some();
            end_step(0);
            traced
        })
        .collect();
    assert_eq!(traced, [false, true, true, false]);
    values(STEP, "tok", 0, &[1]);
    values(STEP, "top1", 0, &[1]);
    assert_eq!(take_lines().len(), 1);
}
