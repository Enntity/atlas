// SPDX-License-Identifier: AGPL-3.0-only

//! Decode determinism tracer (`ATLAS_GLM_DET_TRACE_DECODE`), a debugging aid.
//!
//! Greedy speculative decode of one prompt can differ between requests of
//! the same server start. This logs one line per (request, verify step,
//! layer, stage) with the prefill tracer's hash, in forward order on every
//! rank, so two requests diff down to the first stage that differs:
//!
//! ```text
//! DETD r=<rank> q=<request> t=<step> p=<position> L=<layer> s=<stage> r0=<first row> n=<rows> b=<bytes> h=<hash>[ v=<values>]
//! ```
//!
//! * `q` is the prefill tracer's request ordinal. `t` counts the request's
//!   verify steps from 1 (mirrored on both ranks; 0 is the state decode
//!   starts from). `p` is the sequence length when the line was logged: the
//!   step's first row for the drafter and verify stages, the committed
//!   length for `acc` and the state after it. A line's key is
//!   `(r, q, t, L, s, r0)`.
//! * `v` lists small host values; `h` then hashes them as little-endian
//!   32-bit words (for `logits` it hashes the logits, `v` is a summary).
//! * Step 0, once per request after its prefill: `pre` (`v` = sequence slot,
//!   prompt tokens = first decode position, tokens the prefix cache served,
//!   tokens the restored KDA snapshot covered, KV-valid tokens, drafter
//!   context rows) and `kda_h`, `kda_conv` (every KDA layer's recurrent and
//!   conv state handed to decode, one hash each).
//! * Each step, head rank, the propose that made the step's drafts: `d_in`
//!   (`v` = last token, position, drafts asked, context rows, rows already
//!   in the drafter KV, drafter KV rows, drafter sequence length, rows the
//!   last verify accepted, append-skip flag, lane, min_tokens floor),
//!   `d_hid` (the captured target hidden stack row handed over), `d_logit`
//!   (the drafter's logits, what its confidences derive from), `d_ctx0` (the
//!   whole context accumulator after the request's first traced propose),
//!   `d_pos` (its positions; `v` = rows, first, last), `d_out` (the drafts).
//! * Each step, every rank, the verify forward: `tok` (input ids, `n` = the
//!   width), `emb`; per layer `in`, on MLA layers one `sel` per sparse
//!   piece, `attn_red`, then the FFN output: `ffn` on KDA layers, `moe_red`
//!   and `moe` wherever the prefill MoE serves the rows (the MLA layers); at
//!   `L = layers` `final`, `logits` (`v` = each row's top-2 margin; under
//!   the vocabulary split the hash and the margin cover this rank's half),
//!   `top1` (argmax ids), `acc` (`v` = rows committed = accepted drafts + 1,
//!   the width) and `st_conv` (the KDA conv state after the commit).
//! * `dec`: a draftless decode of one token (`v`) at row `r0`, labelled with
//!   the step it precedes. Its layers are not traced.
//!
//! `ATLAS_GLM_DET_TRACE_STAGES=a,b` keeps only the named stages, and is the
//! only way to get the rest: every prefill stage of the shared layer code
//! (`attn`, `kv_lat`, `moe_in`, `rt_ids`, `rt_w`, `moe_local`, `moe_sh`, the
//! `x_` ones), `out` (the mHC highway after a layer), `st_h` (the KDA
//! recurrent state after every commit: every layer's, so 100+ MB a step) and
//! `x_d_ctx` (the context accumulator at every propose).
//! `ATLAS_GLM_DET_TRACE_STEPS=a-b` and `ATLAS_GLM_DET_TRACE_REQUESTS=a-b`
//! bound the steps (step 0 always logs) and the requests (every request
//! still logs `pre`, which reads nothing from the device).
//!
//! The tracer only reads: it synchronizes streams and copies tensors to the
//! host, so timing differs, but no path, cache decision or kernel changes.
//! Its `v` values carry token ids, so the log then holds the generated text.
//! Captured regions (a whole-step verify graph, a piecewise KDA run) are
//! muted rather than forced eager, so their layers log nothing; `=2` keeps
//! those regions eager to trace them, which is a different execution path.
//! Only the per-sequence verify traces its forward: an owner-batched or
//! fused-chunk verify logs the drafter stages, `acc` and the state.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{At, CURRENT, SEGMENT, SLOT_REQUEST, SLOTS, Scope};

/// Stages logged when `ATLAS_GLM_DET_TRACE_STAGES` is unset.
const DEFAULT_STAGES: [&str; 23] = [
    "pre", "kda_h", "kda_conv", "dec", "d_in", "d_hid", "d_ctx0", "d_pos", "d_logit", "d_out",
    "tok", "emb", "in", "sel", "attn_red", "ffn", "moe_red", "moe", "final", "logits", "top1",
    "acc", "st_conv",
];

/// Verify steps the slot's request has committed.
static SLOT_STEP: [AtomicU32; SLOTS] = [const { AtomicU32::new(0) }; SLOTS];
/// Whether the slot's request has proposed.
static SLOT_PROPOSED: [AtomicBool; SLOTS] = [const { AtomicBool::new(false) }; SLOTS];

thread_local! {
    /// Depth of the captured regions this thread is inside.
    static MUTED: Cell<u32> = const { Cell::new(0) };
}

/// Whether the decode tracer is enabled.
#[inline]
pub fn on() -> bool {
    super::levels().decode != 0
}

/// Whether captured verify regions must stay eager so their layers trace.
#[inline]
pub fn eager() -> bool {
    super::levels().decode == 2
}

/// An inclusive `a-b` range (`a`, `a-` and `-b` allowed); anything else is
/// the full range.
fn parse_range(value: Option<&str>) -> (u64, u64) {
    let all = (0, u64::MAX);
    let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return all;
    };
    let (lo, hi) = value.split_once('-').unwrap_or((value, value));
    let bound = |text: &str, open: u64| match text.trim() {
        "" => Some(open),
        text => text.parse().ok(),
    };
    match (bound(lo, 0), bound(hi, u64::MAX)) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => all,
    }
}

fn within((lo, hi): (u64, u64), value: u64) -> bool {
    (lo..=hi).contains(&value)
}

fn env_range(name: &str, cell: &'static OnceLock<(u64, u64)>) -> (u64, u64) {
    *cell.get_or_init(|| parse_range(std::env::var(name).ok().as_deref()))
}

/// Whether `stage` passes an `ATLAS_GLM_DET_TRACE_STAGES` list; without a
/// list, whether it is a default decode stage.
fn stage_listed(list: Option<&str>, stage: &str) -> bool {
    match list {
        Some(_) => super::stage_listed(list, stage),
        None => DEFAULT_STAGES.contains(&stage),
    }
}

/// Whether a tap of `stage` in the scope `at` logs (`super::current`).
pub(super) fn selected(at: &At, list: Option<&str>, stage: &str) -> bool {
    match at.step {
        None => super::stage_listed(list, stage),
        Some(_) => MUTED.with(Cell::get) == 0 && stage_listed(list, stage),
    }
}

/// Whether `stage` is logged at all (skip the work of preparing it if not).
pub fn wanted(stage: &str) -> bool {
    on() && stage_listed(super::stage_list(), stage)
}

/// A request starts on `slot` (`super::begin_request`).
pub(super) fn begin_request(slot: usize) {
    SLOT_STEP[slot % SLOTS].store(0, Ordering::Relaxed);
    SLOT_PROPOSED[slot % SLOTS].store(false, Ordering::Relaxed);
}

/// Whether `request` is inside `ATLAS_GLM_DET_TRACE_REQUESTS`.
fn request_traced(request: u64) -> bool {
    static REQUESTS: OnceLock<(u64, u64)> = OnceLock::new();
    within(
        env_range("ATLAS_GLM_DET_TRACE_REQUESTS", &REQUESTS),
        request,
    )
}

/// Whether the request of `at` is traced beyond its `pre` line.
pub fn traced(at: &At) -> bool {
    request_traced(at.request)
}

/// Where the lines of slot `slot`'s request sit before its first step, at
/// sequence length `position`. `None` when the tracer is off.
pub fn pre_at(rank: usize, slot: usize, position: usize) -> Option<At> {
    on().then(|| At {
        rank,
        request: SLOT_REQUEST[slot % SLOTS].load(Ordering::Relaxed),
        chunk_start: position,
        layer: 0,
        step: Some(0),
    })
}

/// Where the lines of the step slot `slot`'s request is in sit. `None` when
/// the step is not traced.
pub fn step_at(rank: usize, slot: usize, position: usize) -> Option<At> {
    static STEPS: OnceLock<(u64, u64)> = OnceLock::new();
    let at = pre_at(rank, slot, position)?;
    let step = SLOT_STEP[slot % SLOTS].load(Ordering::Relaxed) + 1;
    let steps = env_range("ATLAS_GLM_DET_TRACE_STEPS", &STEPS);
    (traced(&at) && within(steps, step.into())).then_some(At {
        step: Some(step),
        ..at
    })
}

/// The verify step of `slot`'s request is committed: the next one starts.
pub fn end_step(slot: usize) {
    if on() {
        SLOT_STEP[slot % SLOTS].fetch_add(1, Ordering::Relaxed);
    }
}

/// Whether this is the first propose of `slot`'s request.
pub fn first_propose(slot: usize) -> bool {
    !SLOT_PROPOSED[slot % SLOTS].swap(true, Ordering::Relaxed)
}

/// Trace this thread's taps as `at` until the scope drops.
pub fn enter(at: At) -> Scope {
    Scope(CURRENT.with(|c| c.replace(Some(at))))
}

/// The decode scope this thread is in, if any.
pub fn scope_at() -> Option<At> {
    if !on() {
        return None;
    }
    CURRENT.with(Cell::get).filter(|at| at.step.is_some())
}

/// Silences this thread's decode-scope taps until dropped.
pub struct Mute(());

impl Drop for Mute {
    fn drop(&mut self) {
        MUTED.with(|m| m.set(m.get() - 1));
    }
}

/// The region that follows is captured into (or replayed from) a CUDA graph,
/// where a tap's synchronize is illegal and would log only on eager passes.
pub fn mute() -> Option<Mute> {
    on().then(|| {
        MUTED.with(|m| m.set(m.get() + 1));
        Mute(())
    })
}

fn line(at: At, stage: &str, rows: (usize, usize), bytes: usize, hash: Option<u64>, v: &str) {
    let line = super::format_line(at, stage, rows, bytes, hash);
    super::emit(if v.is_empty() {
        line
    } else {
        format!("{line} v={v}")
    });
}

fn join<T: ToString>(values: impl IntoIterator<Item = T>) -> String {
    let values: Vec<String> = values.into_iter().map(|v| v.to_string()).collect();
    values.join(",")
}

/// Log host `bytes` as `rows` of `stage`; `v` is any summary of them.
pub fn hashed(at: At, stage: &str, rows: (usize, usize), bytes: &[u8], v: &str) {
    if stage_listed(super::stage_list(), stage) {
        let hash = super::hash_bytes(bytes);
        line(at, stage, rows, bytes.len(), Some(hash), v);
    }
}

/// Log host `values` as `stage` from row `row0`.
pub fn values(at: At, stage: &str, row0: usize, values: &[u32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    hashed(at, stage, (row0, values.len()), &bytes, &join(values));
}

/// Log one hash over the device `spans` in order, read after `stream`, as
/// `rows` rows of `stage`; `v` is any summary of them.
pub fn spans(
    at: At,
    gpu: &dyn GpuBackend,
    stream: u64,
    stage: &str,
    rows: usize,
    spans: &[(DevicePtr, usize)],
    v: &str,
) {
    let bytes = spans.iter().map(|span| span.1).sum();
    if bytes == 0 || !stage_listed(super::stage_list(), stage) {
        return;
    }
    let hash = super::spans_hash(gpu, stream, spans, SEGMENT);
    line(at, stage, (0, rows), bytes, hash, v);
}

/// The gap between the two largest of the BF16 values in `row`.
fn top2_margin(row: &[u8]) -> f32 {
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for pair in row.chunks_exact(2) {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
        if value > best {
            (best, second) = (value, best);
        } else if value > second {
            second = value;
        }
    }
    best - second
}

/// Log the BF16 logits rows at `rows` (each the columns this rank computed)
/// as `logits`, with every row's top-2 margin.
pub fn logits(at: At, gpu: &dyn GpuBackend, stream: u64, rows: &[(DevicePtr, usize)]) {
    if !stage_listed(super::stage_list(), "logits") {
        return;
    }
    let mut hasher = super::Hasher::new();
    let mut margins = Vec::with_capacity(rows.len());
    let mut host = Vec::new();
    let read = gpu.synchronize(stream).is_ok()
        && rows.iter().all(|&(ptr, bytes)| {
            host.resize(bytes, 0);
            let read = gpu.copy_d2h(ptr, &mut host).is_ok();
            hasher.update(&host);
            margins.push(format!("{:.4}", top2_margin(&host)));
            read
        });
    let bytes = rows.iter().map(|row| row.1).sum();
    let hash = read.then(|| hasher.finish());
    line(at, "logits", (0, rows.len()), bytes, hash, &join(margins));
}

/// A draftless decode of `token` at row `position` of `slot`'s request.
pub fn serial(rank: usize, slot: usize, position: usize, token: u32) {
    if let Some(at) = step_at(rank, slot, position) {
        values(at, "dec", position, &[token]);
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
