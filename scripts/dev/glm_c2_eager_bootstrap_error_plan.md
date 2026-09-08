# Selected eager bootstrap: stop before graph cleanup on body error

Status: root-approved implementation complete and controller CPU-gated;
independent final review and root commit/postcommit qualification remain pending.
No native work or selected serving activation. Allocation/retirement belongs to
the model author's separate selected-ownership partition.

## Concrete current boundary

The selected outer `Model::decode` calls `paired_before_decode`, then the actual
`decode_dispatch`, then `paired_after_decode`. The latter restores the temporarily
taken proposer owner before propagating a failed producer Result and quarantines
its lease. Its caller can subsequently use T1/T2 terminal containment.

However `model/trait_impl/decode_a.rs:344` unconditionally calls
`gpu.abort_capture_if_active(stream)` after `decode_forward_body` fails, including
when `use_graphs` and `capture_active` are false. The native implementation in
`spark-runtime/src/cuda_backend/gpu_impl_graph.rs:45` always invokes
`cuStreamEndCapture`, ignores its error, and may call `cuGraphDestroy`. Thus the
comment's eager no-op is not an actual no-driver-call contract. An outer terminal
handler is too late to prevent it.

The first selected control is explicitly eager. Current capability reachability
is broader: `use_graphs` is enabled under EP by either `ATLAS_EP_GRAPHS` or
`ATLAS_GDN_DECODE_GRAPH`, unless another resolved suppression wins. The actual
`suppress_graphs` field is initialized from calibration and
`ATLAS_DEBUG_NO_GRAPH=1`, and can later change during calibration. Checking only
`stream_is_capturing` before a call does not prevent a new capture inside it.

## Proposed bounded implementation

Resolve `paired_handoff().is_some()` once inside the actual scalar decode path.
Use this real sealed-capability presence, never a model-name or environment-only
selector, for exactly two changes:

1. Require `!selected_paired` in the existing `use_graphs` expression. This
   makes the selected scalar bootstrap explicitly eager even if unrelated graph
   flags were inherited. It does not change ordinary C1/C2/C3/C4 or legacy GLM
   graph selection, does not add a serving/admission flag, and does not support
   selected graph capture/replay. Future selected graph support needs its own
   reviewed completion/error contract.
2. On a selected `decode_forward_body` Err, return that same error without
   `abort_capture_if_active`. Retain the existing ordinary branch verbatim.
   The selected outer wrapper still marks/restores the actual lease, with no
   replacement/free/cleanup. Do not add a synchronize, retry, fallback or global
   GPU error hook here.

Do not change target compute, successful norm/bonus copy/completion, stream7
selection, public E1 admission, Model traits or generic graph/backend behavior.
T2's real admission still requires the complete eager launch/profile contract;
these two guards close the scalar producer itself, not all protocol paths.

## Exact file ownership and tests

- `crates/spark-model/src/model/trait_impl/decode_a.rs`: the two selected-only
  conditions above, no refactor of ordinary graph logic.
- `crates/spark-model/src/model/glm_c2_handoff_test_fixture.rs`: narrowly add
  observable `BeginCapture`, `EndCapture`, `AbortCapture`, `LaunchGraph` events and backend
  overrides. Preserve numerical sentinels, allocation/normal event ordinals,
  streams and existing failure semantics. Abort is a void observation, not an
  invented successful driver operation. End-capture keeps the existing mock's
  null graph behavior; no CUDA graph numerical claim.
- New `crates/spark-model/src/model/glm_c2_eager_bootstrap_error_tests.rs`, nested
  beneath the existing handoff tests; <=500 lines.
- `crates/spark-model/src/model/glm_c2_handoff_tests.rs`: its one private child
  declaration only. Existing `Fixture::new_legacy` supplies an actual unpaired
  constructor for the ordinary control; do not remove a paired proposer merely
  to fabricate a legacy fixture.

Tests use actual `Model::prefill` then actual scalar `Model::decode`, the real
paired head/lease and existing owned-byte recorder. Separate subprocess cases
control graph environment before construction; no parent `set_var` or global
test reset. Cover both ranks and both failed owners:

1. Successful eager control derives the exact target-body event and its ordinal,
   checks normalized bootstrap bonus detachment and same-default-stream order.
2. Inject that target-body failure. Assert the injected event was reached once,
   immediate event suffix contains no abort/end/begin capture, synchronization,
   bonus copy or further target work, and the actual error reaches `Model::decode`.
   Later successful sync must not allow the same lease to retry. Peer owned
   tail/bonus/private KV and block identities remain unchanged.
3. Root-required behavioral RED: without the production changes, the real body
   fault is followed by `AbortCapture(DEFAULT)`. Preserve this raw assertion
   failure separately from any fixture compilation failures.
4. Graph flags off and both graph-enabling flags on: selected successful decode
   remains eager, with zero begin/end/replay calls. The old selected branch
   under enabling flags must produce a genuine observed capture RED before fix.
   Disable unrelated sparse/debug suppression in that child so the control
   actually exercises the old graph selector, rather than relying on a zero
   event count from an already-suppressed path.
5. Actual legacy fixture, graph flags off: the same injected body failure still
   calls the ordinary abort helper once; successful ordinary decode is unchanged.
   Existing legacy graph behavior is not reimplemented or exempted from tests.

Keep existing handoff producer/decode-fault and transport regressions. Root or
the assigned Cargo owner runs focused behavioral RED/GREEN, full model suite,
non-test check, formatting/SPDX/cap and independent review. Freeze source during
every build. No native process/driver fault injection; root alone owns later
healthy eager correctness and memory/stop gates.

## Remaining inventory is not hidden by this fix

Prefill uses borrowed model arena/layer state and host metadata; paired primer
uses the actual fixed private reserve and restores its proposer Box before Err.
Target fresh-block fill may fail before attaching its raw block ID to the
sequence; the cache refcount remains allocated rather than being automatically
recycled. Track that separately from unsafe RAII release.

`!prefix_cache.is_active()` does not imply `ssm_snapshots.is_enabled()==false`:
the snapshot region is independently constructed from `ssm_cache_slots`.
Current optional full/chunk finalizers can swallow snapshot write/event errors.
Initial selected admission must exclude that actual region, or a separately
reviewed selected bypass/strict path is required. NVIDIA midchunk capture is
already excluded by its `cfg!(atlas_scale)` guard.

Optional cuBLAS EH uses a process-lived workspace; its temporary planning
descriptors are host library objects, not arena/device owners. The runtime
destroys those descriptors before checking Matmul's status. This is not a
device-buffer RAII free, but must not be described as zero native calls after
the Matmul error. No cuBLAS/runtime ownership redesign is included here.

## Implementation receipts

Persistent controller campaign:
`/home/abc/storage/models/atlas-campaigns/20260908/glm-c2-eager-bootstrap/`.
The exact `cpu-command.sh` uses the existing offline/no-default-features model
test configuration, persistent CPU target and unit link libraries. No CUDA
numerical backend, nodes or native execution is involved.

- `red-source.sha256` + `behavioral-red-attempt.log`: actual compiled runtime
  RED, not a compilation failure. Legacy control passed all four flag profiles;
  two selected tests failed. The injected target event was followed by
  `AbortCapture(7)`; an otherwise successful selected bootstrap under EP graphs
  reached `BeginCapture(7)` and `EndCapture(7)` before owned bonus publication.
- `focused-green.log`: all three new tests passed after the two selected guards.
- `full-handoff-green.log`: initial entire handoff regression96/96 passed.
- `final-handoff-green.log`: exact final Rust source, including root's correction
  of the inaccurate old no-op comment,96/96 passed in3.64s. This includes the
  new selected and real legacy controls plus existing producer, acceptance,
  cleanup, transport and reuse tests. It is not a full-model-suite claim.
- `final-nontest-check.log`: model library check passed in4.42s.
- `final-fmt.log`, `final-diff-check.log`, `final-spdx.log`, `final-cap.log`:
  clean checks; all four Rust paths remain under500 lines.

The fixture's graph helpers are event observers, not a graph executor. An agreed
default-false `capture_handles` control allows later actual-capture publication
tests to request unique nonzero handles from the existing handle allocator;
the default remains null and the new tests keep that default. No destroy-graph
ledger or fake replay arithmetic is added in this slice.

The tests execute nine isolated profile children: one selected fault matrix,
four selected-success graph-flag combinations, and four actual-legacy controls.
Selected cases run both ranks and both owner choices; legacy controls use the
real legacy constructor and its only supported owner0. Fresh successful traces
determine every injected ordinal. The four selected faults preserve peer slab,
private K/V bytes, block IDs and token/cursor state and reject a later retry
without operations after a separately successful sync. These are host ownership
and call-order observations, not GPU numerical or hardware-safety proofs.

Root owns full-model and exact-tip postcommit qualification and all subsequent
healthy native gates. Selected C2 admission, supervision and retirement closure
remain separate requirements.
