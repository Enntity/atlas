# Preserve the actual target stream through eager drafter prefill

2026-09-08. Root-approved bounded correctness fix, separate from the v4 source
diagnostic and resident B-tile integration. Source audit confirms a missing
ordering dependency in the v24 source2da770dc. Its contribution to the observed
rank0 KV variation still requires native evidence; no speedup is assumed.

## Reachable defect

The head scheduler creates a non-blocking prefill stream and passes it through
`scheduler/prefill_a_step.rs` to `Model::prefill_chunk`. The concrete dispatch
in `model/trait_impl/prefill_b.rs` selects the default stream for the multi-rank
protocol. Target layers, final hidden capture and final projection use that
effective stream. The Model wrapper then passes its original caller stream to
`try_eager_drafter_prefill`. The primer reads the capture and reuses the same
BufferArena on the other stream without a producer-completion dependency.

The metadata synchronization in prefill_b is BEFORE the target layers. Later
diagnostic synchronizations are conditional, not a production ordering contract.
The cold no-prefix-cache finalizer and asynchronous LM head do not join the
producer. The scheduler's prefill-to-default event occurs AFTER the wrapper
returns: it is too late and in the opposite direction for the capture consumer.
The worker passes default stream through both operations, explaining the
asymmetric exposure. Shared cuBLASLt workspace is another potentially overlapping
resource; this audit does not prove a specific bad read or GEMM result.

Full `prefill_dispatch` unconditionally uses default stream, even on one rank.
Chunked and two-phase dispatch use default only for the multi-rank protocol.
Preserve those existing policies; do not conflate them with the separately
excluded mixed-decode stream mismatch.

## Minimal implementation

Resolve the already existing effective stream at each of the three actual
Model prefill wrappers BEFORE invoking dispatch and eager consumption:

- Full prefill: default stream.
- Chunked prefill: default when `multi_rank_protocol_active`, otherwise caller.
- Two-phase prefill: the same multi-rank rule as its existing dispatch.

Pass that same stream through both calls. Keep existing inner checks as
defensive behavior, without adding global synchronization, event allocation,
new flags, numerical kernels, scheduler commands or trait changes. Do not
change the mixed wrapper in this slice. Existing nested/fallback dispatch must
remain safe and an already completed eager primer remains a no-op.

This fixes the ordering contract with diagnostics OFF too. Do not hide it
inside source probing, and do not use a diagnostic copy as a synchronization
fix. Keep the source diagnostic's selected-error propagation separate.

## Actual behavioral RED/GREEN

Add private tests under the real Model implementation. Construct an actual
TransformerModel, actual capture allocation and SequenceState with a recording
GPU/backend, a recording target layer and a recording proposer. Invoke public
Model methods with different caller/default stream identities. Record target
execution, real capture-normalization launch and proposer consumption; do not
test only a detached stream selector or directly call the eager helper.

Before the fix, execute a test demonstrating target/capture on default7 and
primer on caller37. After the fix require ordered target/capture/primer on7.
Cover EP2 ranks0/1, pure TP2 protocol, full prefill, chunked prefill and the
actual no-SSM two-phase fallback. Single-rank chunked caller37 must stay37;
single-rank full prefill must follow its existing default7 policy. Nonfinal
chunks must not consume, failed dispatch must not consume, and both diagnostic
off and selected diagnostic error propagation must remain covered.

Tests use a recording backend for stream/order evidence, not simulated model
numerical correctness. Preserve existing ownership/capture/repair tests. Run
focused regression, full model CPU suite, non-test check, formatting, SPDX and
file-size gates. The test author cannot independently approve this fix; obtain
a separate source review and freeze exact hashes before committing.

## Native sequencing and interpretation

Complete and commit the v4 diagnostic independently first. Then run RED/GREEN
and commit this fix. Do not deploy the known unordered handoff just to obtain
another reproduction. First v25 native build uses both reviewed commits and
unchanged CUDA189db87e, with the otherwise unchanged v24 recipe and safety caps.
The six-request v4 probe and health checks follow the revised v25 native plan.

Compare source/token/immediate/composed/later-prefix hashes across ranks and
requests, retaining v24 as the pre-fix observation. If differences remain,
continue from their measured boundary without reverting a proven ordering
repair. If diagnostic/health gates pass, separately qualify trace-OFF148/256
C1 with one warmup/three measured requests and fresh-repeat confirmation.
Only full-wall cap-complete results can satisfy30 C1; no diagnostic timing or
assumed acceptance improvement qualifies. Preserve both nodes' memory guards,
exclusive GPU use, graceful stop and rollback images throughout.

## Implemented CPU evidence

The source diagnostic is separately committed asaa2b7e16. The stream fix changes
only the three real wrappers in trait_impl/mod.rs; the two new private test
files are164 and380 lines. The existing903-line trait module remains on the
repository's explicit file-size allow-list; no new exemption was added.

Receipts are under atlas-campaigns/20260908/eager-prefill-stream-fix:

- `behavior-red.log`: actual runtime failures in three tests, including
  chunked target/capture on7 and primer on37. Two negative-path tests passed.
  The separate initial missing-fixture-method compile failure is not RED proof.
- `focused-green.log`: all five tests pass after the wrapper fix,0.05s. The
  matrix covers12 multi-rank entry combinations, single-rank policy, two-chunk
  consumption, target/capture errors and disabled capture. A separate compiler
  check caught the intentionally shadowed full-prefill parameter; it is now
  named `_stream`, with no lint suppression.
- `full-green.log`:917/917 PASS34.07s, including the committed diagnostic's
  actual public-caller selected-error/ordinary-fallback tests. No unreferenced
  resident-reader drafts are compiled by this gate.
- `lib-check.log`: non-test library check PASS4.05s.
- `format-check.log` and `diff-check.log`: workspace format and whitespace
  checks pass; all three touched Rust files have the required SPDX header.
- `clippy.log`: FAIL at four existing spark-runtime Metal stub argument-count
  errors before the model check; no Clippy or full-workspace CI pass claimed.

These tests prove the dispatch/stream contract, not GPU numerical equality or
throughput. Root native gates and independent exact-source approval remain
required before this becomes a serving recommendation.
