# GLM dual-Spark C2 speculation: ownership before batched execution

2026-09-08. User approved finishing the resident-kernel infrastructure, then
implementing C2 speculation and extending validation across C1/C2/C3/C4.
This plan does not authorize bypassing existing C1 guards or running unfinished
code on the Sparks. Root exclusively owns native builds/model/GPU execution.

## Baseline and reference

Frozen v26 engine e40a9066, CUDA189db87e, binary
`a29c261991b94309a2eb4193f3354a429cb60f77d851604b62fe1f3ea5538cbd`.
Warm literal148/256, temperature0/seed1, normal stop/repetition rules, one
warmup and three measured waves per width. First nonspeculative active4 matrix
full-wall medians: C1 13.478, C2 19.067, C3 35.037, C4 47.440. Separate repaired
C1 MTP4 profile:28.699/28.618 initial/fresh. A matrix restart repeat is in flight;
no speculative concurrency benchmark has run. Engine changes are NOT in v26.

The user's comparison is MiaAI-Lab's GLM dual-Spark recipe:
https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark
Its README specifies MTP4, Marlin, FP8 KV, TP2, eager, max8; reports C2 31–37.
Exact prompt/output/timing formulas are not supplied with that table. It is a
capability reference, not a matched numerical oracle or promised Atlas rate.

## Existing seams and missing contracts

Source inspection at257937a4, excluding resident reader WIP:

- `SequenceState` owns tokens, target KV/block map and SSM states/slot.
- `Glm5MtpProposerState` owns private block table/cursor and repair phase.
- `Glm5MtpHead::new` sizes the shared private KV allocator for roughly one
  maximum-length sequence, not two independent full prefixes.
- `TransformerModel` owns one prompt capture/generation, one bonus hidden and
  last-hidden index; `glm_mtp_repair` reuses prompt capture as accepted-row staging.
- The eager primer consumes P-1 shifted prompt pairs, but bootstrap still needs
  the request's terminal prompt hidden. The existing global capture-generation
  guard is intentional; do not weaken it to admit another owner's bytes.
- `verify_dflash_step` and worker F5 implement one temporal K5 transaction.
  EP v2 already addresses an individual slot. E1 proposer execution remains C1.
- Generic `verify_e` excludes EP and accepts per-request widths2..4, not [5,5].
  GLM KDA lacks the required segmented multi-request verifier. Ordinary C2
  decode and temporal K2 are not two independent five-row causal segments.
- C2 nonspeculative FFNs are scalar in both actual KDA and MLA mHC callers.
  This is a separate opportunity; an older batch2 experiment was neutral.

## Prerequisite: finish the existing resident reader slice

Close the independent review's bind boundary with direct sealed GU span alias
checks and exact retained-key identity checks once at arena binding. Exercise
empty/foreign/replaced stores and real ownership/teardown. Preserve Legacy and
all default engine behavior. Freeze, independently review, commit only the
reviewed manifest, and archive source/CPU receipts before transferring Cargo.
Do not mix resident layout activation into the first C2 speculation experiment.

## Partition A: actual per-request handoff and capacity, no C2 admission

Create one authoritative GLM request-owned handoff for normalized prompt tail,
accepted verification rows (at most4), bonus hidden, absolute positions and
generation. Bind it to the actual proposer state/backend and retire it with that
state. Readers cannot manufacture ownership from a row number or global stamp.

Use six fixed BF16[4096] rows per slot: one prompt tail, four accepted rows,
and one bonus. Two slots need98,304 bytes before guards/metadata, not two full
prompt captures. Reserve this model-owned storage outside decode/verify steps,
with generation-tagged leases retired through actual proposer/sequence cleanup;
no per-step GPU allocation, D2H ownership checks, new streams or implicit
cross-stream lifetime assumptions. Determine the exact capacity from actual
consumers, not a guessed two-entry hidden stash. A complete accepted-row set is
required in addition to the bonus. Do not alias the generic verify stash without
proving every producer/consumer and its exclusion for this lane.

The source audit confirms mandatory eager P-1 priming plus retained H[P-1]
can be the minimal bootstrap representation: the missing pair is (x[P],H[P-1]),
whereas the first proposal seed is (x[P+1],H[P]) after one target bootstrap
decode. These are different rows and must not be interchanged. Introduce a
sealed absolute-position normalized-hidden view and an eager-only bootstrap
planner requiring exactly cached_rows=P-1. Leave the existing C1 full-capture
planner unchanged; never pass a tail pointer as a whole-prompt span.
The existing model-global prompt allocation may remain temporary producer
scratch only if consumed before another request can overwrite it. Any selected
primer error must propagate, not degrade to a silently unprimed speculative lane.
Publish the tail only after successful primer and ordered tail copy. Once
detached, a later global capture generation must not revoke the first request's
owned tail. Preserve both the historical capture identity and current slot lease.
Eager priming alone does not prove safe interleaved prompt chunks: first C2
admission must either serialize capture ownership through all chunks or restrict
to cold single-chunk prompts with a checked pre-work rejection of larger ones.
No accidental partial request execution followed by a late unsupported refusal.

Prove separate private-KV capacity for both full prefixes plus speculative/full-
accept reserve. Checked arithmetic, actual allocated blocks and per-request
ownership are required; no overcommit. Keep the max-context-plus-draft2048 bound
and fixed MTP4 initially. Derive extra memory explicitly; do not raise host or
container budgets. No new C2 factory selection until the actual hooks pass.

TDD must exercise real capture -> eager primer -> bootstrap -> verify verdict
-> repair -> repropose -> retire hooks for alternating owners, not only a host
state-machine imitation. Distinct recognizable hidden/KV contents on both ranks;
all25 acceptance pairs0..4 x0..4; interleaved prefills, unequal histories,
swapped active order, stale generation, foreign owner, slot reuse, source
overwrite, failure-before-work and partial-copy failure. Current C1 paths and
non-GLM defaults retain their existing coverage. No native speed claim.
Model wrapper tests must call the actual paired head/worker bootstrap and
recorded-verdict hooks. Private state transitions alone do not close this gate.
On any partial writer/copy failure, mark the affected lease non-resumable;
another valid owner must retain its bytes and generation. GPU completion and
reuse follow the existing effective default-stream contract, not a new assumed
cross-stream fence.

Split implementation into two internal gates, both with C2 admission OFF:

1. Sealed normalized-hidden views, fixed two-slot slab/lease lifecycle and paired
   capture/primer/bootstrap hooks. A completion receipt binds successful writes
   to exact slot, current lease generation, historical capture identity,
   absolute position and representation. This is not an arbitrary caller-made
   pointer/row/count bundle. Use the actual effective stream and completion
   rules before publishing reuse authority.
2. Paired verdict staging, E1 consumption and all25 acceptance combinations.
   Replace the selected worker E1 reread through global
   `save_hidden_for_mtp(hidden_row)` with the same owned view as the head; another
   request may already have overwritten that global row. Zero acceptance copies
   no accepted rows but must retain its bonus; full acceptance retains four
   accepted rows and a distinct bonus.

Retirement invalidates the lease before unrelated fallible cleanup can return.
Only ordered successful completion allows slot reuse; a failed GPU completion
keeps that lease unavailable. Cover primer-only completion, cancellation before
E1, cancellation after verdict/before repair and model teardown. Free the
model-owned slab once; never free it from a request or return another request's
private blocks. Actual stale same-slot/new-generation readers must refuse.

## Partition B: serialized slot-addressed C2 correctness control

Only after A's frozen source/CPU review, introduce an explicit bounded C2
capability/admission contract at all model/server/launcher boundaries. Do not
widen an environment guard in isolation. Initially two cold text requests,
fixed four drafts, no grammar/adaptive depth/carry/catchup/prefix reuse, TP2/EP2,
EP v2, BF16 target/private KV, context<=2044, active/admitted cap2, existing
memory limits. Live occupancy is1..2: first admission and C2->C1 drain remain
in the C2-capable request-owned handoff regime until each lease is retired.
Never switch a survivor to incompatible legacy C1 global capture/bonus state.
The known-good legacy C1 launch remains a separate capability/profile, alongside
the nonspeculative C1..4 profile. If speculation must stop at a request boundary,
the selected regime must explicitly handle terminal cleanup or defined ordinary
decode continuation; no silent uncovered gap followed by repair resumption.
Initial native admission is cold single-chunk text only (prompt must fit the
resolved1024-token prefill chunk). Reject larger/prefix-restored/interleaved
capture shapes before EP allocation or target work. This is an explicit first
scope, not general2044-token prompt support; retain the full2044 context/private
capacity for generation and test capacity boundaries without broadening admission.

Reuse complete per-slot F5/E1 transactions in a deterministic order, including
accept count, target token rollback, SSM commit, request-owned hidden retention
and private-cache repair. Both ranks must make identical ordered collectives.
Cancellation cannot skip the rest of an issued transaction. Retire only its
owner; EP v2 must not compact surviving slots. Validate K5 intermediate capacity
for both actual SSM slots, not merely configured max batch size.

This serial control proves concurrent request correctness; it is NOT a fused
target batch or a promised throughput improvement. Verify real scheduler and
worker paths with cancellation/EOS/output-cap at every transaction boundary.

## Partition C: shared verification work across two sessions

Implement an explicit two-segment [5,5] execution contract (10 target rows).
Batch weight-bearing work while preserving per-segment MLA causal positions,
KV writes/index metadata, independent KDA recurrence and intermediate states,
GLM mHC and original routing/quantization arithmetic. Preserve all accepted
hidden rows before any proposer can overwrite shared scratch.

The EP protocol must carry checked identities/generations/positions and both
verdicts, with actual head/worker execution tests. Do not reuse independent-row
decode metadata or Qwen-specific generic verifier eligibility as GLM proof.
Start with eager numerical reference against the serialized control, then
fixed-address graph capture/replay and mixed-acceptance/drain validation.
Batched proposer execution is a separate subsequent optimization, not silently
implied by batched target verification. No arbitrary C3/C4 speculative admission.

## Native gates and rollout

Root prepares a bounded plan and complete automatic stop/receipt runner before
each native image. No model/standalone GPU/build overlap;114GiB container limits,
4096MiB guard, >=4GiB host MemAvailable, KV overcommit0, swapspace0. No resets,
clock changes, full-model sanitizer or forced kill as routine recovery.

For each activation: distinct prompts and per-request histories; complete
normalized hidden/private KV/SSM agreement against the sequential reference;
asymmetric acceptance, cancellation/recovery, slot reuse, eager/replayed graphs;
full-rank logs, no health/CUDA faults and clean stops. Then warm fixed148/256
one warmup+three measured waves, fresh repeat, both full-wall and decode-window
rates, output counts and identities. Continue nonspeculative C1/C2/C3/C4
regressions and separate optimized C1 checks. Extend speculative widths only
after C2 correctness and memory qualification, never by multiplying flags.
