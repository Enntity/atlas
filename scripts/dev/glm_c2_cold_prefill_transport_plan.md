# Checked paired cold F0 transport

2026-09-09, root-authorized bounded Model transport. No scheduler admission,
factory, T2 authority, supervisor or native action. Ordinary F0 remains literal.

## Exact scope

- `speculative/glm_paired_execution.rs`: sealed `validate_cold_prefill(seq,tokens)`
  and `cold_prefill(seq,tokens)->Result<DevicePtr>`; no generic Model trait change.
- `model/glm_c2_predispatch.rs`: actual implementation delegates shared helpers.
- `model/glm_c2_transport.rs`: register a private small
  `glm_c2_cold_prefill_transport.rs` child for sender/receiver/bounds.
- `model/impl_a2.rs`: selected-only F0 dispatch before unchanged ordinary body.
- `model/glm_c2_handoff_tests.rs`: one new `glm_c2_cold_prefill_transport_tests.rs`.
- Root-approved `model/glm_c2_handoff_worker_tests.rs`: inspect the complete error
  chain in the existing six-receive failure assertion, preserving all checks.
- Root-approved `model/glm_c2_test_wire.rs` narrowly opt-in cold
  root0/root1 zero-prefix agreement and root transcript; default unchanged.

## Ordering and bounds

Head validates actual rank0 `paired_wire_profile`, bounded complete prompt and
existing immutable `paired_prefill_preflight` BEFORE first header. Enforce
2..=1024, actual arena/hidden-capture capacities, checked token-byte extent within
actual scratch. This is the initial transport envelope, not all-context support.
The actual private owner/slot/cold state/token values/default stream checks remain
in the existing preflight; validation does not reserve or publish anything.

Send existing v2 slot/F0, P,0,P, full tokens. Then call actual
`Model::prefill_chunk(tokens,seq,0,P,true,default_stream)` on BOTH ranks.
Do not call head full `prefill`: unlike chunked prefill it lacks worker's two
unconditional rooted prefix-agreement broadcasts. Preserve those collectives.
Real target capture, mandatory P-1 primer and owned H[P-1] publication remain
in existing Model wrappers; no synthetic handoff or extra copies.

Worker's selected helper checks actual rank1 profile, receives fixed metadata,
requires start0/chunk==full and valid extent BEFORE allocating a token receive
vector. Receive tokens, run the same existing immutable preflight and actual
complete-chunk wrapper. Skip old `normalize_ssm_states` only for selected F0;
the selected producer owns canonical state and old normalize refuses it.

Every head error from first header onward and worker helper error poisons the
existing session via `paired_transport_error`. No fallback/free/F1/retry here.
Preamble/slot-resolution errors precede this helper and remain T2's whole-worker
scope. Returning Err is not no-unwind process containment.

## Focused TDD

Non-sending new-entry scaffold supplies actual behavioral RED. Existing real
Fixture/Wire drives immutable repeated validation, invalid cold/shape/token/rank
zero-wire refusal, both owner orders and actual head-to-worker complete F0 replay
with matching root0/root1 prefix negotiation. Compare real token/cursor/tail/KV
owners and continue through already-tested scalar bootstrap if bounded.
Control-derived first-header/payload/target/primer/tail-completion fault sites,
plus worker malformed metadata/receive fault, prove terminal refusal and no
later target work. No forged Ready/new mock framework/all25/full suite.
Model-focused offline CPU wrapper, focused handoff regressions/check/fmt/lint;
source hashes and exact intermediate failures retained. Native remains root-only.

## Implemented CPU checkpoint

Actual scaffold RED: two runtime failures at the new immutable validator and
selected worker entry (not a historical unsafe sender experiment). Final focused
6/6 PASS includes 30 actual issued-fault injections, five invalid worker metadata
cases, both owner orders and both-root packet replay. Handoff regressions 134/134
PASS (3.82s), non-test check PASS (5.20s), fmt/diff/SPDX/new-file caps PASS.
Clippy stopped at four unchanged runtime stub `too_many_arguments` errors; no
lint relaxation. No full model/server suite or native qualification in this slice.

Intermediate controls are retained: one six-token prompt initially kept the
fixture's predeclared length4; a cold unbound lease can legitimately bind either
valid slot, so the invalid-slot test now uses2; the old worker fault assertion
needed full-chain formatting after adding terminal context. Production ownership
checks were not relaxed. The local Wire models fixed-zero cold prefix responses,
not concurrent NCCL agreement or target numerical correctness.

Residual end-to-end boundary: scheduler cold admission/first-token selection and
driver/checked-retirement hookup remain absent, as do actual T2 registration and
supervisor-backed authority. Preamble/slot failures remain outside the local F0
latch. Returning Err does not stop callee destructors or make native failure safe.
