# Explicit idle-command receive classification

Status: implemented and CPU-gated; independent final hash review/native idle
validation pending. Root approved this complete partition before code. The
retirement author explicitly transferred Cargo/the integrated module graph;
no overlapping Cargo or compiled-source edits occurred.

## Observed boundary and scope

The worker enters `Model::ep_worker_step` immediately after its ready message.
Its first synchronous NCCL broadcast can wait for a future head command.
`NcclBackend::broadcast` starts its timer before enqueue and checks it after
`cuStreamSynchronize`: normal command-idle time is included. The v25 worker's
376.9-second warning matches ready-to-first-command wall time. This is a
post-completion duration check, not an interrupting watchdog. Its sticky
unhealthy latch is not currently consumed by HTTP health or the worker loop.
The existing receipt remains health-failed; this change cannot relabel it.

Classify exactly the first word of each outer worker command receive as an
idle wait. In v1 this is the command; in v2 it is the sequence-ID preamble.
The second v2 word, all arguments, token payloads, acceptance words, head-side
sends, barriers and other collectives retain their existing behavior. Do not
infer the exemption from message size, rank, elapsed time, or command value.

No flags, changed threshold, reconnect, health-endpoint wiring, asynchronous
watchdog, model numerical path, GLM/factory change, allocation, or extra GPU
synchronization is in scope. The idle wait can still block on a genuinely
broken transport; addressing an interrupting watchdog is a separate design.

## Explicit interface and production execution

Add an object-safe `CommBackend::receive_idle_command_word(ptr)` method with
fixed four-byte width and fixed root zero. It is receiver-only: reject rank
zero, single-rank/out-of-range rank, and null or unaligned word pointers before
I/O. The compatible default delegates to the existing `broadcast(ptr, 4, 0)`;
other backends need not implement an exemption. The NCCL override selects the
private `IdleCommand` classification. Ordinary `broadcast` always selects
`TimedPayload`; no public generic timing-policy parameter is introduced.

Use one small production broadcast runner for both NCCL entries. A private,
monomorphized operations interface supplies clock start/elapsed, actual launch,
stream completion, slow-duration reporting/latching, and async-error query.
Its real NCCL adapter retains the same communicator snapshot, Uint8 datatype,
root, in-place send/receive pointer, byte count and legacy stream. The runner's
ordering is start -> launch/check -> synchronize/check -> elapsed -> optional
slow-duration latch -> async-error query. `IdleCommand` omits only the duration
warning/latch. It must not clear an existing latch, suppress NCCL/CUDA errors,
skip the async-error query, or alter the existing return contract (the bool
from the async-error check remains diagnostic, as today). Early launch/sync
errors retain today's short-circuit behavior. Keep the threshold comparison
exactly `elapsed.as_secs() >= 30` for timed broadcasts.

The runner and its CPU recording operations are compiled without NCCL for
unit tests; the real adapter is compiled with NCCL. This is a production
execution seam, not a test-only replacement policy or simulated numerical
backend. The test clock avoids actual 30-second waits. No GPU success claim
comes from these CPU tests.

In the actual `TransformerModel::ep_recv_seq_and_cmd`, read the first word via
a private idle-word receiver. For v2, read the second word via unchanged
`ep_broadcast_u32`; for v1, return the first word as command and sequence zero.
The new private receiver keeps the existing broadcast -> default-stream
synchronize -> four-byte D2H -> little-endian decode chain. It does not change
`ep_broadcast_u32` or any payload sender/receiver call site. Worker entry,
dispatch, shutdown and error handling remain unchanged.

## Exact ownership

- `crates/spark-comm/src/lib.rs`: documented trait entry, private runner
  registration and default-method tests (split tests into a child if needed).
- New `crates/spark-comm/src/broadcast.rs` and `broadcast_tests.rs`: private
  production runner/receiver validation and actual-chain CPU tests.
- `crates/spark-comm/src/nccl_backend/comm_impl.rs`: delegate broadcast and
  implement the explicit receiver override.
- New `crates/spark-comm/src/nccl_backend/broadcast.rs` plus its registration
  in `nccl_backend.rs`: real operations adapter; clarify only the existing
  duration-check documentation, not reconnect behavior.
- `crates/spark-model/src/model/impl_a2.rs`: only the actual outer receive seam,
  private idle receiver and private test-child declaration.
- New `crates/spark-model/src/model/impl_a2/idle_command_tests.rs` and bounded
  fixture child if needed: actual model/protocol entry tests.
- This plan and persistent receipts; no other source ownership.

All new Rust files carry SPDX and remain at most 500 lines. No changes to
the retirement author's model modules, factory, GLM or shared test fixtures.

## Actual behavioral TDD

First introduce the explicit method and characterization runner preserving
the old timed behavior, while the outer receive still uses ordinary broadcast.
Run assertions to obtain two genuine executable RED receipts: actual protocol
entry chooses timed rather than idle for its first word, and the extracted
production runner still latches an idle duration above threshold. Compile
errors are recorded separately and never called behavioral RED. Then connect
the explicit entry and classification, retaining all characterization tests.

Production-runner tests cover both classifications at 29.999s, 30s and 376.9s;
exact operation order and arguments; already-unhealthy state remains sticky;
launch failure, synchronization failure, async query failure/reported async
error retain old propagation/latch behavior; no query after launch/sync error;
and invalid explicit receivers perform no operations. A timed four-byte
rank-one broadcast remains timed, proving there is no size/rank heuristic.
Default-method compatibility must call the underlying ordinary broadcast once
and preserve its error. CPU tests never create a live NCCL communicator.

Use a real small `TransformerModel` and a recording `CommBackend` plus owned
mock GPU memory. Drive the actual `Model::ep_worker_step` shutdown entry for
v1 and v2 and repeated worker steps, asserting exactly one idle selection per
outer step. Drive `ep_recv_seq_and_cmd` with nonzero sequence/command values,
then actual `ep_broadcast_u32` and token payload helpers to prove follow-ons
remain timed; drive actual head sequence/command sends to prove both are timed.
Verify the exact four-byte pointer, little-endian result, GPU synchronize/D2H
ordering and no allocation during receive. Inject idle failure, second-word
failure, GPU synchronize failure and D2H failure; assert exact short-circuit
and returned errors. These exercise production entry points, not a rewritten
test protocol. No full numerical decode is needed to classify its preamble.

## Gates and handoff

After explicit Cargo handoff, preserve RED/GREEN and command environment under
`atlas-campaigns/20260908/idle-command-receive/`. Run focused spark-comm tests
without default features, focused actual model protocol tests, complete
no-default model and comm library suites, and non-test checks. Compile-check
the NCCL-feature adapter with `ATLAS_SKIP_BUILD=1`; do not invent CUDA entry
points or execute GPU code. Run changed-file formatting, SPDX and size checks,
and record any pre-existing unrelated lint blocker without lowering lints.
Full shared-tree counts must be labeled as such if retirement WIP is present.

Freeze exact hashes and obtain independent source review before root commit.
Root alone rebuilds and validates on stopped/healthy nodes. Native acceptance
must include a deliberate idle interval beyond 30 seconds followed by an
immediate small valid request and clean shutdown, with no new unhealthy log,
plus normal v1/v2 protocol behavior as applicable. A separate timed-payload
failure need not be manufactured on physical Sparks: CPU fault injection
proves unchanged duration classification without risking a transport stall.
This partition makes no throughput or recovery claim.

## CPU evidence (2026-09-08)

Persistent raw receipts:
`/home/abc/storage/models/atlas-campaigns/20260908/idle-command-receive/`.
The exact command environment and receipt descriptions are in `commands.md`
there. No CUDA calls, live NCCL communicator, node operation or numerical
emulation was used for these CPU gates.

- `comm-red.log`: actual production runner 6/7 passing, intended assertion
  failure because an explicit idle wait at 30,000ms still marked unhealthy.
- `model-red.log`: actual model entry 1/5 passing, four intended failures
  exposing timed-first-word selection (including unselected idle failure);
  the unchanged head-send characterization passed.
- `comm-green.log`: 7/7 communication tests pass; `model-green.log`: all five
  actual worker/protocol test groups pass, including both versions, prefill
  headers/bulk and K2/K3/K4/generic verify argument paths.
- `full-suite.log`: 7/7 communication and 940/940 model tests pass, model
  runtime 33.94s. This is a shared-workspace CPU receipt on base `6a486364`
  plus this partition, not a claim that a future commit itself was tested.
  The other author's remaining reader files were unreferenced and uncompiled.
- `lib-check.log`: non-test no-default model/comm check passes (5.19s).
- `nccl-check.log`: real NCCL-feature adapter compile-check passes (0.40s).
- `comm-clippy.log`: strict NCCL-feature comm/tests clippy passes (0.48s).
- `model-clippy.log`: blocked by four unchanged no-CUDA runtime stub
  `too_many_arguments` errors; no lint was relaxed and this is not a pass.

The runner recording backend verifies short-circuit and sticky-latch behavior
at its async-check hook. Distinct NCCL query-failure and reported-async-error
branches remain in the unchanged real `check_async_error` and are verified by
source review, not separate live NCCL fault injection. No broader async health
or transport-recovery claim is inferred. Root's separate
`glm_ep_idle_native_plan.md` defines the required deliberate 35-second idle and
normal-workload/fresh-confirmation gates before native acceptance.
