# Bounded GLM MTP hidden trace

Status: root-approved plan implemented; independent source review approved.
This is a default-off
diagnostic, not a throughput optimization or evidence of an uninitialized cache.
Root alone owns native builds, deployment, node/GPU operations and commits.

## Observation and scope

Root launcher follow-up: add a strict default-off `GLM_MTP_HIDDEN_TRACE` switch,
forward it to both ranks, and reject enabling it without the existing accepted-
pair repair profile before any external command. CPU tests evaluate only the
launcher's pure validation prefix and separately inspect both forwarding sites;
native container environments still require root verification. The target's
verifier graphs may remain enabled; the actual proposer stream must be eager.

The phase6 first-eight K5 ledger found later draft disagreements at matching
position/seed while target decisions agreed for matching causal token prefixes.
It did not establish identical target hidden inputs or identical proposer KV.
Capture exact representation boundaries to distinguish these possibilities.

Restrict enabled diagnostics to native GLM, hidden4096, TP2/EP2 distributed
proposer, C1, exactly four drafts, accepted-pair repair, cold/prefill-only
capture, no carry/catchup/prefix reuse, no grammar/adapters/adaptive depth.
Reject invalid explicit flag values and unsupported profiles before diagnostic
I/O; preserve the ordinary disabled path without copies, allocations, capture
queries, counters, or output changes. No target scheduler/precision changes.

## Files and call sites

- New `layers/glm5_mtp/hidden_trace.rs` and focused test/support children,
  each Rust file below500 lines: strict flag/profile, request-owned budget,
  raw-byte digest, stream-ordered snapshot, fixed log records.
- Minimal `glm5_mtp.rs` hooks: request-owned trace field in proposer state,
  initialization and free/reset; input capture at `forward_one` entry before
  `forward_body_one`; final capture after module.norm before vocabulary GEMV.
- Minimal `model/impl_b3.rs` boundary: arm trace after successful
  `prepare_glm_mtp_repair`, immediately before `proposer.propose`. Obtain the
  actual SequenceState slot/capture generation, target position/seed and saved
  hidden-row metadata there. No cross-model DraftProposer trait expansion.
- Add model dependency on existing workspace `sha2`, if needed, rather than
  reusing `hidden_fingerprint` (which hides failed copies as a zero hash).

## Request ownership and attempt bound

Diagnostic state belongs to Glm5MtpProposerState, never a global counter.
The wrapper supplies the real sequence slot and nonzero capture generation;
accepted-pair repair has already checked sequence ownership against the model's
capture generation. New request generation resets the budget even when the
proposer state object is reused. Same-generation calls never reset it, and
stale/regressed generations reject. Allocation/free also initialize/reset it.
Keep request identity distinct from the per-request proposal-attempt ordinal.

Spend one of eight attempt slots before diagnostic I/O. A failed copy or failed
proposal cannot restore the slot. Each armed attempt permits four ordered draft
steps only; no primer/repair-body calls consume or emit draft trace records.
Disabled/exhausted attempts are inert. Correlate to the existing verifier ledger
by request window, slot, position, seed and draft list, not ordinal alone.

## Exact bytes, lifetime and ordering

At each draft step read exactly8192 bytes of raw BF16 conditioning hidden before
the body can overwrite its source. For draft0 this is mtp_hidden_save AFTER
repair (target post-final-norm); draft1..3 use the previous draft's post-module-
norm norm_output. Then read exactly8192 bytes from final_hidden immediately
after module.norm and before vocabulary projection. One reusable8192-byte host
array suffices; total readback16KiB/step,64KiB/attempt,512KiB/request/rank.

Use copy_d2h_on_stream on the actual proposal stream; reject context capture
and actual stream capture before any diagnostic read. Verify exact hidden width,
nonnull/aligned/address-safe source and known final-output owner/capacity. Hash
raw bytes with SHA256, including signed zero; do not normalize or repair data.
Errors propagate explicitly, never producing a valid-looking zero digest.
No GPU allocations, full vocabulary readback, callbacks inside capture, new
collectives or numeric changes. Logging adds synchronization: no timed claims.

Records include rank, slot/generation, attempt, position, seed, draft index,
input token, pre/post private seq_len, relevant repair/cursor metadata, both
digests and produced token. Physical block-table metadata may be logged as
metadata only, never interpreted as a content-equivalence proof. Optionally
reuse the already-read16-byte distributed argmax pairs without extra GPU work.

Equal input digests and different final digests localize divergence to proposer
computation/private state; they do not prove equal KV or identify a reset bug.
Equal final digests with different local maxima localize later. Different first
input digests show differing target conditioning before the proposer.

## TDD and validation

1. Actual hook tests first fail with diagnostics absent. Drive the real GLM
   forward_one/propose boundary using a recording GPU/body fixture, not only
   pure planner tests. Verify exact input/final pointers,8192-byte lengths,
   stream and call order relative to body/norm/vocabulary, unchanged outputs.
2. Use the same production wrapper/request-arming helper on a reused state and
   real SequenceState metadata: same generation exhausts at eight attempts;
   fresh request capture generation resets; stale identity rejects; failed I/O
   spends its attempt; free/reset is idempotent. No claim that alloc alone proves
   the actual request boundary.
3. Disabled and exhausted profiles have zero added backend operations; reject
   capture (context and actual stream), malformed pointers/size and unsupported
   C1/repair policy before diagnostic I/O. Test copy failure distinctly from a
   legitimate all-zero row, all bytes affecting the digest, and last-byte change.
4. Serial scoped tests and full model CPU suite, fmt/file-size/SPDX; independent
   review and freeze before root commits/builds. Preserve RED/GREEN receipts in
   the persistent20260908 campaign directory. No Cargo overlap with other agents.
5. Root native diagnostic: same matched request/flags, both-rank first-eight
   coverage, quality/caps, no full-model sanitizer. Then disable trace for every
   throughput run; keep all results and do not infer causality from hashes alone.

## Implemented ownership evidence and CPU receipts

The real request boundary is not the proposer allocation. The existing
`model/trait_impl/drafter_prefill.rs::try_mtp_prefill_capture_from` assigns
`seq.mtp_capture_gen` on cold chunk zero and preserves it for contiguous chunks.
`model/glm_mtp_repair.rs::glm_repair_input` binds that stamp to the model capture
generation; the actual prepared `ProposalPlan` preserves it. A read-only
`ProposalPlan::generation()` accessor (root-approved) lets the trace reject a
different sequence generation after repair without duplicating planner state.
`impl_a2.rs::EP_CMD_GLM_MTP_PROPOSE` and the head both reach the same
`impl_b3.rs::run_mtp_propose_inner` post-repair arming hook. The trace budget is
stored in that actual request's `Glm5MtpProposerState`, with no global counter.
The two ranks' local generation numbers are ownership stamps, not proof that
their complete model/request histories or private KV contents are identical.

Persistent receipts:
`/home/abc/storage/models/atlas-campaigns/20260908/glm-hidden-trace/`.

- `hook-red.log`: actual `forward_one` first-read assertion failed with the
  input/final hooks wired but readback absent. This is a behavior RED, not a
  compiler failure. `hook-green.log` passes with exact stream-ordered snapshots.
- `generation-red.log`: actual SequenceState/prepared-plan generation mismatch
  was wrongly accepted; `focused-green.log` passes after the immutable plan
  generation binding was added.
- `boundaries-green.log`: 11/11 focused tests. The real `forward_one`/`propose`
  hooks exercise both ranks, exact pointer/byte/stream and norm/vocabulary order,
  off/on output and event equality, eight attempts, reused SequenceState reset,
  stale ownership refusal, input/body/final-copy failure budgets, capture and
  unsupported profile refusal, raw signed-zero/last-byte hashing and errors.
- `full-model-cpu.log`: 835/835 no-default-feature model tests pass, including
  the prior824 tests. `fmt.log` and `git diff --check` pass.
- Independent review caught one strict flag edge: a non-Unicode explicit value
  must not be treated as an absent environment variable. `non-unicode-red.log`
  records the failing production parser behavior; the final parser distinguishes
  NotPresent from NotUnicode without mutating process environment in tests.
  `full-model-cpu-final.log` passes836/836, including all12 focused trace tests;
  `fmt-final.log` and final diff check pass. Independent source review approved
  that frozen production slice (`hidden_trace.rs` SHA256
  `be2b80045ff27d86249db69d9a44e39fbef7deb7fc0e5fcffa1775775b376819`).
- `license.log`:2403 valid,0 invalid,14 ignored. New Rust children are all below
  500 lines. Cargo.lock changes only the model's existing workspace sha2
  dependency edge; no dependency version changed.
- Strict clippy is **not clean**: `clippy.log` stops at four existing runtime
  stub errors; scoped `clippy-model.log` reports seven existing model lint
  errors (including the unchanged MTP divisibility check). No new trace lint
  was reported. These unrelated files were not changed or lint levels lowered.

The fixture uses host-backed `MockGpuBackend` storage and deterministic sentinel
kernel/body behavior. It proves production hook placement, bounds, ownership,
failure propagation and output non-interference, **not CUDA numerical parity**.
No native build, GPU allocation, node run, performance result or explanation of
the observed acceptance variation is claimed for this diagnostic yet.

## Native v21 rejection and bounded ownership correction

The first native v21 diagnostic request rejected on both ranks at the trace's
no-adapter profile guard, before any hidden snapshots. This is a failed
diagnostic gate, not a model throughput result. The exact base-model route was
incorrectly assumed to be `Skip`: production
`lora::resolve_moe_lora_route(-1,-1,false)` returns inert `Fold`, and
`model/impl_lora.rs::moe_lora_route` derives that `false` from `self.lora=None`.
The constructor also initializes the decode route to `Fold`. The CPU fixture
hardcoded `Skip`, so its prior pass did not cover this live representation.

Root approved the following bounded correction before implementation: supply
live model LoRA pool, token-overlay and rotation ownership from `impl_b3` to the
trace arming helper. Evaluate the ownership reader only after the existing
disabled/exhausted returns. Require all three absent, retain adapter_max_rank0,
base sequence adapter id/slot and absent routed-layer guards, and reject
`Refuse` unconditionally. Only that proven no-owner profile may use inert
`Fold` or `Skip`; do not permit arbitrary Fold based on config alone and do not
change the underlying model's LoRA routing semantics.

First record a behavioral RED using the real route resolver's no-pool Fold in
the actual trace arming test. Then cover no-owner Fold/Skip success on both
ranks, pool/overlay/rotation/config/active-sequence refusal before I/O, and
disabled/exhausted ownership-reader non-evaluation. Update the shared fixture to
derive its route through the production resolver. Run focused/full controller
CPU tests and independent review before root commits or builds again. This
fix changes only diagnostic eligibility; the native diagnostic still needs a
successful clean rerun before any hidden-divergence conclusion.

Root additionally authorized one sticky model lifecycle bit to close the
detachment/error history gap: `set_lora_weights(None)` does not clear every
installed layer field, and a failed Some install may mutate layers before the
pool owner is stored. Initialize `lora_install_attempted=false` in the actual
constructor, set it before any fallible Some installation work, and never clear
it on None. Include it in the lazy ownership proof. Actual setter tests cover
both successful and failed Some attempts followed by detach. This adds one
inert host boolean to all models; it changes no LoRA arithmetic, routing,
allocation, or ordinary serving admission. No layer-trait expansion is needed.

Final correction receipts under `atlas-campaigns/20260908/`:

- `hidden-trace-route-red.log`: actual resolver no-pool Fold rejected by the
  old trace guard (behavior RED).
- `hidden-trace-history-red.log`: actual model constructor/setter did not retain
  the attempted-install history (behavior RED, not a compiler failure).
- `hidden-trace-route-green.log`:13/13 focused tests pass after live ownership
  binding. `hidden-trace-ownership-full-final.log`:841/841 model CPU tests pass,
  including actual setter success and failed install followed by None detach,
  all owner/history/config/sequence/Refuse negatives and inert lazy reader when
  disabled/exhausted. An intermediate test-only non-Copy context construction
  compiler error is retained in `hidden-trace-ownership-full-green.log`; it is
  not labeled a behavioral RED or passing receipt.
- `hidden-trace-ownership-fmt.log` and final diff check pass. New Rust files are
  below500 lines; existing `impl_lora.rs`495 and `impl_b3.rs`497 remain below cap.
- Independent review approved the frozen correction. Production trace SHA256:
  `65b2a7df1858e865dc70b6a5421662ee9a1adb8c9fe46459043e105764af443e`.

The native v21 failure remains preserved in `v21-hidden-rejected-rank0.log` and
`v21-hidden-rejected-rank1.log`. This CPU-corrected source has not yet passed a
new native diagnostic request; root must rebuild and verify that separately.

## Post-EH boundary extension (root-approved; CPU implementation complete)

The subsequent v22 native diagnostic succeeded: six requests, 192 complete
records per rank. Exact counts and earliest keys are preserved separately in
`glm-mtp-hidden-trace-v22-results.md`. At every observed step0, identical input
token/position and raw input hash can lead to different final hidden hashes.
Rank0 varies across repeated requests; rank1 is substantially stable, with the
documented generation3 downstream exception. This establishes a boundary to
investigate, not an explanation or a speed result.

### Read-only audit negatives and remaining uncertainty

- `weight_loader/glm5/mtp.rs` constructs a full TP1/EP1 body. Although
  `forward_body_one` passes the target config in its runtime context,
  `load_mla_layer(..., force_dimension_overrides=true)` installs full head
  overrides. `attention_forward` passes those dimensions through Q expansion,
  Q absorption, paged attention, V extraction and O projection. The simple
  partly-unwritten-full-head hypothesis is not supported by this path.
- Body attention TP reduction and MoE EP reduction require `comm=Some`; the
  draft body supplies None. Ordinary single-row MoE uses the supplied stream,
  not the prefill shared-expert auxiliary-stream path. No concurrent scratch
  ownership violation was identified; this is not a native race proof.
- The body explicitly zeros its 8192-byte residual row. Module final RMSNorm
  uses hidden4096 and distinct arena allocations. The deployed 1088-row arena
  exceeds these single-row full-head scratch demands; this does not establish
  eligibility of a differently sized future arena.
- Head and worker use common repair/proposer code. Prompt capture normalizes
  every target row into generation-owned storage. Bootstrap/accepted repair
  write private KV through batched projections, whereas a draft uses GEMV.
  The v22 current-input hash does not cover those prompt rows or private KV;
  numerical parity against a candidate KV writer does not establish equal
  canonical prefixes across requests or ranks.
- Semantic-index maintenance executes, but sparse selection returns None at
  these positions below2048. That semantic history is not selected by the
  observed dense-attention path. No reset or broad config change is justified.

### Exact production hook and ownership

Extend only the already enabled hidden trace, without a new flag or ordinary
serving behavior change. Pass the existing optional `StepTrace` from
`forward_one` into `forward_body_one`; the legacy prompt-body caller passes
None. Invoke the new hook immediately after the EH projection finishes being
queued into `ctx.buffers.hidden_states()` and before private-cache allocation,
metadata upload, residual initialization or body decode. This is the complete
post-EH BF16[4096] row, not a sample or the later normalized row.

The hook returns immediately unless the existing record represents step0.
Thus only step0 of each already admitted first-eight proposal attempts per
actual request gets one additional 8192-byte `copy_d2h_on_stream`, on the same
proposal stream, followed by raw SHA256. No new request counters, ownership
readers, allocations on device, GPU kernels, collectives or scheduler hooks.
Off/exhausted records remain None; later steps perform no added capture query,
readback or digest. An existing admitted attempt is already spent before the
input copy; post-EH copy failure propagates and cannot restore its budget.

Validate exact hidden width, pointer equality to the live hidden_states owner,
capacity at least8192, nonnull/aligned/address-safe span, eager context and
actual stream before the new copy. Reject duplicate post-EH capture. Require a
successful step0 post-EH capture before final capture/emission, so a missing
hook cannot produce an apparently complete extended record. Existing profile,
generation, step ordering and no-adapter/history checks remain unchanged.

Readback totals become24576 bytes for step0,16384 for each later step,
73728 bytes/attempt and589824 bytes/request/rank. Each individual step stays
strictly below64KiB, using the existing bounded snapshot workspace. A full
148-row private K+V prefix would instead require303104 bytes; sampling fewer
rows would not prove equal KV. This is why the complete post-EH row is the
smallest next discriminating boundary. Equal post-EH hashes still do not prove
equal weights, private KV, intermediate arithmetic or freedom from races.

### Wire schema and backward-compatible analyzer

Keep one existing trace record per completed step and its original identity,
cursor, hashes, token and pair fields. New producers append `trace_version=2`
and `post_eh_sha256`: exactly64 lowercase hex characters on step0 and literal
`None` on steps1..3. Both fields are mandatory on every version2 record.
Old records with neither field remain explicitly recognized as legacy version1;
one missing field, unknown version, invalid digest or an unexpected later-step
digest rejects. No separate event stream or record-count expansion.

Update the strict Python analyzer and tests together. Require a consistent
schema within each selected request/rank and across its two rank selectors;
allow explicitly declared legacy and extended requests in the same manifest.
Keep complete-file ownership/count validation unchanged. Comparisons retain
version, digest availability and both digest values; `post_eh_equal` is null
unless both digests exist, never fabricated true for absent evidence.
At matching conditioning tuples, differing available post-EH digests identify
`post_eh_difference` before the existing final-hidden classification. Equal
post-EH and different final hashes localize only to the remaining body/private-
state/final-norm region. Legacy/extended comparisons explicitly report missing
post-EH evidence; they may report an observed downstream difference but must
not claim post-EH agreement. Existing v22 logs must still parse and retain all
384 rows and prior raw comparison counts.

### TDD, review and controlled handoff

1. After root reads/approves this addendum, coordinate exclusive controller
   Cargo with the other author. No node/build/native/GPU work by this agent.
2. Add an actual forward_one/propose regression first: both ranks must read
   input, then exact post-EH hidden_states, then final norm_output, with EH
   kernel preceding the new copy and the body following it. Record a behavior
   RED against the old hooks, not a compile failure. The recording fixture
   gives EH a distinct sentinel to prove the hashed bytes belong to this
   boundary, including a last-byte mutation.
3. Preserve off/on identical tokens and non-diagnostic backend event order.
   Check four-step proposals add exactly one8192-byte read; first8 attempts,
   failure-spent budget, reused generation reset, off/exhausted no extra
   backend operations, later-step inertness, duplicate/missing hook refusal,
   owner/capacity/capture refusal and post-EH copy failure before body execution.
   Existing input/body/final failure tests retain their meaning with updated
   expected read counts, not weakened assertions.
4. Python TDD covers legacy success, extended success, malformed/missing/version
   mismatch, step0-only availability, missing-evidence reporting and earliest
   post-EH classification. Reanalyze the frozen complete v22 files and compare
   all prior integrity and raw-difference counts; do not overwrite old receipts.
5. Focused/full controller CPU tests, formatting, file-size/SPDX and Python
   suite; preserve behavioral RED/GREEN receipts under the persistent20260908
   campaign. Independent source review and frozen hashes before root commit.
   CPU fixtures establish hook/ownership behavior, not CUDA numerical parity.
6. Root alone decides and runs a clean native diagnostic after approval, checks
   schema and32 rows/request/rank plus8 post-EH hashes, then disables tracing
   for performance work. No broad reset fix or additional KV/body probe is
   included in this slice.

Implementation receipts in `atlas-campaigns/20260908/`:

- `hidden-trace-post-eh-hook-red.log`: actual forward_one test fails because
  the full8192-byte post-EH read is absent, before production changes.
  `hidden-trace-post-eh-focused-green.log`:16/16 focused tests pass afterward.
- `hidden-trace-post-eh-analysis-red.log`: old strict parser rejects version2
  valid records; `hidden-trace-post-eh-analysis-green.log`:15/15 pass with the
  extension. Final `hidden-trace-post-eh-analysis-final.log`:16/16, including
  explicitly declared legacy/extended requests and missing-boundary evidence.
- `hidden-trace-post-eh-full-green.log`:859/859 complete model CPU tests pass,
  including three new tests for actual projection sentinel/last byte on both
  ranks, post-EH copy failure consuming all8 attempts then remaining inert,
  new request reset, missing/duplicate/owner/capacity/capture negatives and
  later-step zero added backend operations. The intermediate expanded-test
  receipt contains a test-only borrowed-buffer lifetime compile error; it is
  preserved as such, not called a behavioral RED or passing receipt.
  `hidden-trace-post-eh-full-final.log` repeats859/859 after ensuring the
  existing final-owner/cursor negatives first establish valid post-EH proof;
  those tests therefore still independently exercise their original guards.
- `v22-hidden-post-eh-analyzer-regression.json` is a fresh analysis of the
  same lossless logs using the original all6-selector manifest. The companion
  `.log` verifies complete JSON equality with the old evidence after removing
  only the four new version/availability/digest comparison fields:384 rows,
  every original count, key, classification, hash and metadata value preserved.
  The original v22 evidence and native logs were not overwritten.
- `hidden-trace-post-eh-fmt.log`: scoped rustfmt check passes; only owned files
  were formatted. `hidden-trace-post-eh-license.log` records the full header
  scan:2413 valid,0 invalid,14 ignored. New trace Rust test file is162 lines;
  trace module372, support301.
  Final scoped `hidden-trace-post-eh-fmt-final.log` and owned-file diff check pass.

Root independently reviewed the production/analyzer diff and hook/failure
tests without a source blocker. Final hash handoff precedes commit and any
native build. No new native measurement or explanation of the divergence is
claimed by these CPU tests.
