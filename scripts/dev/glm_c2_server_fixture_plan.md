# Actual paired Model fixture for server tests

2026-09-09. Root approved this bounded implementation after aligned allocation
closure and explicitly transferred the engine source/Cargo window. This is a B2 testing
prerequisite, not a selected scheduler, admission or native activation change.

## Existing boundary and intended reuse

`spark-model` currently registers `model/glm_c2_handoff_tests.rs` only under
`cfg(test)`. Its fixture constructs the actual TransformerModel and actual
Glm5MtpHead::new_paired, with genuine Model-allocated SequenceStates. Server
unit tests compile spark-model as a dependency, without that dependency's
`cfg(test)`. Importing its unit-test module or copying a fake Model would not
exercise the actual sealed capability.

Keep production `GlmPairedExecution::sealed`, the crate-private paired head
constructor, and all private lease/publication fields unchanged. Reuse the
existing fixture constructor and byte backend once; do not implement Model or
the sealed capability on a new test adapter. No factory switch or Ready setter.

## Feature, registration and exact source ownership

- `crates/spark-model/Cargo.toml`: add non-default
  `glm-c2-test-utils = ["spark-runtime/test-utils"]`. No CUDA default change,
  version change, new external dependency or feature on normal model builds.
- `crates/spark-server/Cargo.toml`: add the same existing spark-model path as a
  dev-dependency, `default-features = false`, with that feature. Leave its
  production dependency and production feature forwarding untouched. Existing
  runtime test-utils and approved CPU link helpers remain sufficient.
- `spark-model/src/model/mod.rs`: register one internal support module under
  `cfg(any(test, feature = "glm-c2-test-utils"))`; expose its narrow public
  facade only with the explicit feature. The model module is already public;
  no new crate-root production exports are necessary.
- New `model/glm_c2_test_support.rs` and bounded children own the facade and
  shared fixture registration. Move the registration of
  `glm_c2_handoff_test_fixture.rs` out of `glm_c2_handoff_tests.rs`; that test
  parent aliases the same shared module. Preserve existing test names and
  subprocess paths. Do not compile the whole handoff test tree under a feature.
- Reuse `glm_c2_handoff_test_fixture.rs`, its constructor child
  `glm_c2_handoff_test_build.rs`, and `glm_c2_verdict_test_numerics.rs` by their
  existing paths. Only nominal helper visibility needed by sibling model tests
  becomes crate-private; none becomes an unguarded external API. Keep one
  constructor implementation, not two module registrations of the same fixture.
- Extract the reusable Wire implementation from
  `model/glm_c2_transport_test_fixture.rs` into one internal support child.
  Keep its flow-dependent `bootstrapped`, `worker` and `same_private` helpers
  in the unit-test tree with aliases to the shared core. Recorded local packet
  replay is not an actual concurrent collective implementation.
- The two inspection methods currently inside
  `layers/glm5_mtp/paired_capacity_tests.rs` (`paired_test_free_blocks` and
  `paired_test_kv_rows`) move literally into a small internal inspection child
  registered from `paired.rs` with the same test-or-feature gate. Do not feature
  enable the capacity test suite or export raw paired internals. Preserve live
  owner validation before canonical pointer discovery.
- Server owns a new bounded scheduler test child, registered only under
  `cfg(test)`, and a narrow helper in existing `scheduler/test_support.rs` if
  needed to move a genuine SequenceState into the existing ActiveSeq builder.
  Do not duplicate the large ActiveSeq initializer. The temporary host-only
  placeholder has no device owner and must not replace a live state implicitly.

Root assigns this entire mechanical shared-fixture move to one author after the
aligned-allocation freeze. Other agents may write only unreferenced test children
until the module graph is released. Keep every new Rust file <=500 lines, with
SPDX; no global formatting over another author's working tree.

## Minimum facade and ownership

The public feature-only facade constructs rank 0 or 1 with the existing fixed
bounded fixture geometry. It may provide an explicit legacy constructor control.
It returns/moves the actual model and genuine sequences, plus an observer holding
the recorder and actual head Arc; it does not proxy Model calls. Server tests
must obtain `Model::glm_paired_execution()` from that actual model.

Expose only the controls required by the first integration tests:

- move each initial SequenceState once, so ActiveSeq owns the real target guard
  and private lease; later allocation uses actual Model::alloc_sequence;
- event/packet readout, clear, deterministic sentinel-logit mode, and bounded
  ordinal fault injection through the existing recorder;
- install the shared Wire on the fixture's actual model before moving it out;
  queue exact worker packets, inspect recorded packets and assert queue drained;
- immutable canonical KV/slab snapshots for a supplied live sequence, cursor,
  and free-count observations. Read addresses only through the existing checked
  getters. If an error test needs post-revocation byte comparison, snapshot its
  exact bounded addresses while valid; do not rediscover authority afterward.

Approved setup refinement: `Fixture::parts_mut` temporarily borrows the actual
model and sequence array so the server test itself calls cold prefill/bootstrap
before installing the existing E1/F5-only Wire and consuming `into_parts`.
This adds no producer callback, sequence clone or hidden driver. The local wire
smoke does not claim worker F0/cold-prefix transport coverage. Snapshots expose
bytes only; saved addresses remain opaque, foreign-observer reads and released
backing refuse, and a saved read after retirement never grants resume authority.

No generic arbitrary-pointer write/export, mutable Pool/lease accessor,
public production config mutation, fake Ready injection or test-only bypass of
profile checks. Add a further control only when an actual test needs it and root
reviews the corresponding boundary. Observer lifetimes do not authorize cleanup
after Model teardown; tests drain real owners before explicit teardown and drop
inspection handles in the required order. No new automatic cleanup promise.

For future bootstrap/K5/checked-selection/emission tests this supplies the real
model, request owners, logits buffer and event transcript. Existing public Model
operations perform prefill/decode, capability proposal/verify, acceptance record,
trim/commit and retirement. The facade must not implement that driver sequence
on the server's behalf: otherwise an ordering test could bypass the actual
scheduler entry it is supposed to qualify.

## Actual tests and RED before exposing the seam

1. Add one server unit test importing the planned facade; record its genuine
   missing-feature/export compile RED separately from runtime behavior evidence.
2. After the mechanical move, run an actual server-side smoke test in an
   isolated child with the existing supported paired environment. Create real
   owners, call actual Model prefill and scalar bootstrap, obtain the non-None
   sealed capability, and invoke `validate_propose` then `propose` for four
   drafts. Verify the exact selected E1 packet and actual body/byte writes.
   Execute both owners/ranks with replayed local wire as appropriate. The test
   must fail if the actual model returns None or if its capability call is
   omitted. The new feature/export compile RED is the TDD evidence for this
   mechanical exposure; do not disable an existing correct production capability
   to manufacture a reproduced bug. Any optional omission sensitivity check is
   explicitly mutation evidence, not runtime bug RED. Preserve the actual
   capability smoke as GREEN evidence. Do not substitute a driver mock or
   prepare private state through a hidden setter.
3. Exercise actual K5 plus the existing checked-selection helper using the
   model's real sentinel-backed logits buffer. Require five selections and
   unchanged `ActiveSeq.output_tokens` plus canonical `seq.tokens`/`seq.seq_len`
   during speculative selection. Do not require the entire ActiveSeq to remain
   unchanged: checked selection intentionally shares mutable legacy pipeline
   counters and their existing semantics.
   Complete accepted-count/rollback/record/trim/commit using public Model calls
   before retirement; no abandoned Verification or fabricated accepted receipt.
   This checks the reusable seam, not a not-yet-written B2 scheduler driver.
4. Keep future driver emission-order tests outside this slice. They will use
   the same observer and real response sink to assert commit before EOS/cap
   emission, no unnecessary E1, and no subsequent work after failure at the
   actual selected driver entry. Their actual ordering runtime RED must come
   from that future driver work; do not claim it from this fixture smoke or an
   optional mutation check.
5. Re-run the existing model handoff suite to prove the registration move keeps
   actual allocator, transport, fault and inspection tests on the same fixture.
   The model's current `flow::isolated` hardcodes model test names; server tests
   require their own exact child name/environment setup, not that helper.

## Compile and qualification gates

Use the coordinated persistent CPU target and previously approved link helpers;
no synthetic runtime functions, native CUDA execution or node work. Record
actual commands, features and source hashes. Mandatory separate invocations:

- feature-off `cargo check -p spark-model --no-default-features --lib --offline`
  and a tiny external compile probe proving the facade import is unavailable;
- feature-on model library check with `--features glm-c2-test-utils`; a positive
  external compile probe imports the facade without relying on `cfg(test)`;
- feature-off ordinary server binary check with its established CUDA feature
  and link environment; no dev-only facade is referenced by serving source;
- server focused tests with the existing `--features cuda` command and the
  test-only dev-dependency, then full serial server/model suites, non-test
  checks, scoped formatting/SPDX/size checks and truthful clippy status.

The small external probes use isolated temporary manifests outside the engine
workspace and the existing path dependency; no committed Cargo.lock/version
drift. Negative missing-export compile results are not numerical/runtime REDs.
Check feature selection in separate invocations because Cargo feature unification
in workspace tests intentionally makes dev-dependency features available. This
feature is a build-time testing seam, not a security boundary against someone
choosing to compile it into a binary. Default/native recipes never enable it.

Freeze exact source and receipts for independent review before root commits.
This fixture models bounded byte writes and local packet replay. It cannot
qualify real MLA/KDA arithmetic, TP reduction, concurrent NCCL ordering, driver
completion, selected supervision/T2 fatal containment, performance or admission.
