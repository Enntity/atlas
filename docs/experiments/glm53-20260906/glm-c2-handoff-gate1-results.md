# GLM C2 request-owned handoff: Gate 1

2026-09-08. Frozen source independently reviewed; all 1,003 CPU model tests
passed. This is an unactivated infrastructure milestone, not a serving-image
or speculative-concurrency performance qualification.

## Scope

Gate 1 supplies an explicit, unactivated paired-head constructor and real
per-request draft-cache reserves. Each request retains its own normalized
prompt tail and bootstrap hidden row before another producer overwrites shared
scratch. The first proposal repairs the missing shifted prompt pair and reads
the request's owned bonus, rather than the last request's global hidden row.
The existing single-request constructor remains the factory default.

Two requests require distinct state, even when their operations execute
serially. With prompt length P, eager priming writes P-1 shifted pairs. The
missing pair consumes H[P-1]; the first proposal instead starts from H[P].
Those are separate positions and separate retained rows. Simply enabling two
requests behind the old global hidden-state interface would not preserve this
contract.

The staged path accepts only its checked cold, complete, single-owner producer
profile, with at least two prompt tokens. It is not a general concurrent-prefill implementation. Selected
verification verdicts remain unsupported until Gate 2 implements accepted-row
staging and subsequent repair. Public head/worker C2 speculation admission is
still closed. There are no scheduler, launcher, CUDA arithmetic, or serving
image changes in this slice.

## Allocation contract

At context 2044 with four drafts, each owner reserves 128 physical private-KV
blocks from the actual allocator. The two owners cannot borrow each other's
reserved capacity. Reserved capacity is not initialized cache length.

| Allocation | Two-owner bytes |
|---|---:|
| BF16 NoPE512 private K and V, 256 blocks |8,388,608|
| BF16 pooled index keys, kpool4/head-dim128 |262,144|
| Raw sparse-index key/gate tails |2,097,152|
| Six BF16[4096] hidden rows per owner |98,304|
| Total indexed private cache and handoff slab |10,846,208|

This excludes model weights, target KV/SSM pools, existing scratch and host
metadata. The six rows are prompt tail, four reserved accepted rows and bonus;
Gate 1 does not yet publish the accepted rows. Sparse allocation qualification
does not establish sparse attention numerical correctness.

## Evidence boundaries

The complete CPU `spark-model --lib --no-default-features` suite passed
1,003/1,003 in 81.33 seconds. Focused receipts passed 27 ownership/producer
tests and 75 GLM-head tests. Non-test library checking, scoped formatting,
SPDX first-line checks and `git diff --check` passed. This used the controller's
existing CPU CUDA shim; none of these tests ran a model on the Sparks.

CI is not claimed green: clippy stops at four inherited `too_many_arguments`
errors in runtime Metal stubs. The existing non-allowlisted `glm5_mtp.rs`
file-size violation remains (882 lines versus 902 at the base commit); all new
Rust files are at most 499 lines. The arithmetic bodies were not moved merely
to hide that inherited failure. Whole-workspace CI and native validation are
separate obligations.

Tests exercise the actual TransformerModel producer and cleanup wrappers,
actual Glm5MtpHead construction, allocation, KV writer and first proposal.
The recording backend substitutes numerical kernels and collectives with
recognizable sentinels; it is not a logit or KDA numerical oracle. Physical KV
bytes, peer hidden rows, block ownership, stream ordering and failure behavior
are inspected independently of event counts.

Failure tests cover actual target/capture writers, detached-tail and bootstrap
bonus copies, immediate completion, corrupted mutable block views, legacy
writer-entry rejection, reverse slot reuse, and model close. A selected
zeroing/completion failure now leaves both its private reserve and its target
SSM slot unavailable; only the healthy peer's slots can be reclaimed. These
are host-backed fault injections, not deliberate faults on the Sparks.

Worker coverage uses actual EP-v2 F0 and ordinary bootstrap-token dispatch,
plus refusal of C2 E1 before its payload. A successful internal proposal under
both rank configurations is not a successful public paired-worker E1.
The tiny real SSM cleanup pool proves allocator/cleanup behavior, not the
production four-draft KDA intermediate capacity. That remains a native
activation prerequisite.

Model close assumes drained, sequential host lifetime. Tests observe exclusion
of Model-path owner release and bulk sweep after failed completion; they do
not emulate CUDA backend destruction. The existing CUDA backend Drop performs
its own final allocation sweep. Gate 1 does not change that contract or prove
native unknown-completion recovery. Likewise, failed construction publishes no
head or lease but does not offer transactional allocation rollback: the build
must be abandoned, not retried on the same backend. Later activation must review
both failure policies explicitly.

Campaign logs retain failed implementation and fixture attempts. In particular,
`producer-cleanup-green.log` ends with five passes and eleven failures despite
its filename. The earlier writer-fault control incorrectly expected a GLM
capture copy; GLM writes capture through RMSNorm. Only the corrected control
reaches the actual target/capture failure assertions. Filenames and compile
success alone are not qualification evidence.

Frozen 34-file source manifest SHA256:
`1a876119a3e627875c5fe0344ce9b6f13257e783781f020b79793a05c2911f6f`.
Controller receipts are under `atlas-campaigns/20260908/glm-c2-handoff/`.
They are source-hashed development receipts, not dirty-tree serving gate
records. The result document is outside that source manifest.

## Performance status and next gate

No speculative C2 model benchmark has run. The qualified warm serving baseline
remains approximately 28.6–28.7 tok/s for the separate optimized C1 MTP4
profile, 19.0 aggregate for nonspeculative C2, and 47.2–47.4 aggregate for
nonspeculative C4. Neither the 30 C1 nor 60 C4 full-wall target has been met.

Next is request-owned verdict staging and repair for all 25 pairs of acceptance
counts (0..4 per owner), followed by the serialized slot-addressed C2 control.
Only then should shared ten-row verification be compared with that control.
The qualified standalone ten-row kernels are candidates, not measured model
throughput improvements. See the
[implementation plan](../../../scripts/dev/glm_c2_speculation_plan.md) and
[segmented verification audit](../../../scripts/dev/glm_c2_segmented_verify_audit.md).
