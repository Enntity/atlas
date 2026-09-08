# Checked verify selection for the paired serial driver

Status: implemented and source-frozen, CPU behavior/regression and non-test
checks passed, awaiting final independent approval/root commit. No C2 caller,
activation, native execution, or throughput claim belongs to this slice.

## Purpose and current behavior

The future serialized paired driver must consume the token chosen by the actual
penalty/processor/sampling pipeline, not treat raw verifier argmax as semantic
authority. It also must distinguish a failed logits read from successful token
selection before issuing a verdict or another rank-wide transaction.

`verify_pipeline_helper.rs::verify_pick_all_with_pipeline` currently returns raw
IDs if its full BF16 logits read fails. Earlier fast paths use
`fast_greedy::logit_is_positive`, whose 2/4-byte read failures become `false`;
the caller can then fall through to a successful full read and choose a processed
token. These are distinct legacy behaviors and both remain unchanged.

## Interface and shared implementation

Keep the existing public helper signature and all existing callers unchanged:

```rust
verify_pick_all_with_pipeline(model, raw, active, context, row_base) -> Vec<u32>
```

Add the sibling entrypoint, with the same argument types:

```rust
verify_pick_all_with_pipeline_checked(model, raw, active, context, row_base)
    -> anyhow::Result<Vec<u32>>
```

The checked entrypoint is explicitly **grammarless only**. Reject an actual
`active.grammar_state.is_some()` before any logits I/O or state mutation, even
for an empty span. This matches Partition B's admitted profile. It does not
claim checked grammar support. Legacy grammar advance/rollback remains literal
and reachable only through the unchanged legacy entrypoint.

Both wrappers use one private all-position driver with an explicit internal
`CopyFailurePolicy::{LegacyFallback, Propagate}`. Move, do not duplicate, the
existing driver. Preserve its fast-path eligibility/order, BF16 full-row format,
row-base addressing, timing, and existing GPU-versus-host tie behavior. Do not
force all checked requests onto the full-read path merely to avoid probe errors.

The existing per-position `verify_pick_with_pipeline` remains the sole source
of penalty construction, processor stages, forced tokens, per-position seeded
sampling, and host argmax. No copied sampler or penalty math.

Introduce a Result-returning one-logit probe next to the existing boolean probe
in `fast_greedy.rs`. Share its actual byte read and BF16/FP32 interpretation;
keep the old boolean API as its legacy error-to-false wrapper. The all-position
driver and masked-chat fast path use the explicit policy to distinguish:

- Legacy probe error: false and existing next-path processing; it must not turn
  directly into raw-ID fallback.
- Checked probe error: return the actual error immediately, before another probe,
  full-row copy, or any downstream selection work.
- Legacy full-row copy error: return original raw IDs, as today.
- Checked full-row copy error: return the actual error, no raw-ID success.

No new Model trait method, GPU kernel, device allocation, stream operation,
scheduler policy, request field, or active paired caller is introduced. The
existing Model logits-copy abstraction remains the byte-ownership boundary;
this slice does not pretend it exposes a full device-capacity proof.

## Exclusive file ownership

Only these existing production files may change:

- `crates/spark-server/src/scheduler/verify_pipeline_helper.rs`
- `crates/spark-server/src/scheduler/verify_pipeline_helper/fast_masked.rs`
- `crates/spark-server/src/scheduler/fast_greedy.rs`

New private children under `verify_pipeline_helper/`:

- `selection.rs`: moved shared all-position driver and explicit failure policy.
- `selection_io.rs`: small policy shared by the driver and masked fast path;
  keeps that dependency explicit and separate from the selection algorithm.
- `tests.rs`, `checked_tests.rs`, and `test_model.rs`: legacy characterization,
  checked behavior/regression tests, and the owned-byte Model/backend fixture.

This plan is the sole documentation file in scope. No model, runtime, factory,
worker protocol, or existing scheduler caller changes. Root additionally approved
`crates/spark-server/Cargo.toml`: mirror the model crate's existing runtime
dev-dependency (`default-features = false`, `features = ["test-utils"]`). This
exposes the owned-byte backend in ordinary server tests without requiring a
special test command, changing normal production features, or gating tests off.

## Actual behavioral gates

First characterize the unchanged legacy public entrypoint. The test Model's
`copy_logits_to_host` delegates to an owned-memory GPU backend/recording wrapper;
injection fails that actual copy boundary, not a substitute selection oracle.
Record exact source pointers, byte lengths, and operation ordinals. No CUDA or
numerical-kernel correctness is claimed by this host-only fixture.

1. Full-row legacy read failure returns raw IDs. A small-probe failure followed
   by a successful full read returns the actual processed result, not necessarily
   raw IDs. No error is injected before a successful positive control proves the
   intended path and exact copy shape.
2. Add the checked API initially delegating old behavior, then execute assertions
   expecting errors from actual full-read and small-probe failures. Preserve
   these runtime assertion failures as behavioral RED, not compile failures.
3. Implement policy handling; strict failure is terminal for the call, with no
   later read. The corresponding legacy observations remain unchanged.
4. Non-raw positive control: exact BF16 logits with raw token 1 at 4.0 and token
   2 at 3.0, output history containing token 1, repetition penalty 2.0, and other
   unrelated processors neutral. The actual pipeline must select token 2. Use
   five rows and a nonzero row base; distinguish surrounding rows. Checked and
   legacy successful IDs and backend read traces must agree.
5. Exercise neutral zero-I/O shortcuts, reduce-only immunity probes, history
   membership avoiding an unnecessary probe, masked-chat mode, and empty spans.
   Probe positivity still treats valid zero/NaN as ineligible, not I/O errors.
   Retain BF16/FP32 probe interpretation while full verify rows remain BF16.
6. Seeded temperature-positive sampling parity uses the existing sampler and
   actual request seed/position offsets. No test replacement sampler.
7. An actual grammar state passed to the new checked entrypoint is rejected
   without reads or matcher-history changes. Legacy grammar tests/regressions
   continue to exercise the original path; no new checked grammar promise.

Strict selection success does not authorize a transaction by itself. The future
paired driver must still validate returned cardinality/token bounds and follow
its pre-issue versus post-issue failure policy. That caller is outside this slice.

## CPU build and freeze protocol

The server's `init_gpu_backend` exists only under `cuda` or `metal`; plain
`--no-default-features` cannot compile this target. Minimum Linux controller
feature selection is:

```text
cargo test --no-default-features --features cuda -p spark-server --bin spark -j4 <filter>
```

Use root-provided persistent CPU target/link libraries, `ATLAS_SKIP_BUILD=1`,
and `CUDARC_CUDA_VERSION=13000`; do not initialize CUDA or invent driver symbol
implementations to obtain a pass. The initial server link required the real
installed cuBLASLt library and original NVIDIA cuda-driver-dev 13.0.96-1 stub;
their provenance is recorded in the campaign, and this is not GPU qualification.
Root must grant the exclusive Cargo/source
window before baseline, RED, focused GREEN, and full server regressions. No
parallel model graph/source edits while recording exact source receipts.

Preserve raw commands, terminal logs, baseline and actual RED/GREEN distinctions
under the campaign directory chosen with root. Run non-test server check,
format/diff/SPDX/line-cap checks, then freeze exact touched-file hashes for
independent review and root commit. Report any unrelated baseline failures
accurately; do not weaken tests or feature configuration to hide them.

## Development qualification (2026-09-08)

Raw commands, link provenance, source hashes, and logs are under
`/home/abc/storage/models/atlas-campaigns/20260908/glm-c2-checked-selection/`.

- Legacy characterization: 5 passed (3 new actual-copy tests, 2 existing argmax
  tests). Behavioral RED then executed 3 expected assertion failures, with
  8 other tests passing: checked full-copy error, checked probe error, and actual
  grammar rejection before an empty span.
- Focused implementation GREEN: 12 passed. Final full server rerun includes
  those tests and passed 2,359 with 12 ignored, in 23.00 seconds, using isolated
  command environment `NO_COLOR=` and `RUST_TEST_THREADS=1`.
- The first full run under inherited `NO_COLOR=1` and parallel execution had
  2,355 passes and 4 unrelated TUI failures (3 color assertions and a shared-log
  rendering race). Its failure receipt remains preserved; no TUI source changed.
- Strict Clippy stopped on 10 existing model-dependency style errors. A scoped
  server `--no-deps --tests` retry, after fixing the new test initializer style,
  stops only on the existing `serve_phases/preflight.rs` items-after-test-module
  lint. Neither invocation is a lint pass; no lint levels were reduced.
- All changed/new Rust files are below 500 lines; formatting, scoped diff check,
  and source license-header checks passed. The per-position pipeline text is
  byte-identical to the preserved RED source.
- Final non-test CUDA-feature server check passed in 17.42 seconds.

These are development-source receipts: the server suite links the simultaneously
frozen B1 model checkpoint recorded in `model-development.sha256`, rather than
claiming an exact committed-tree result. Root will qualify separately committed
slices afterward. No checked serving caller, C2 activation, GPU numerical proof,
or throughput improvement is claimed.
