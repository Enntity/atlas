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
