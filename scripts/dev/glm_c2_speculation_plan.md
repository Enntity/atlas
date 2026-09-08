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
C1 MTP4 profile:28.699/28.618 initial/fresh. The completed matrix restart repeat
is C1 13.468, C2 18.976, C3 34.828, C4 47.222 (baseline doc539f7a9a).
No speculative concurrency benchmark has run. Engine changes are NOT in v26.

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

### Partition A concrete implementation amendment (for root review)

Keep all server/worker C1 admission checks unchanged. Use the existing GLM-only
`GlmPairRepair` capability to expose an optional request-handoff capability;
the default is absent. Do not add a generic Model/DraftProposer mode flag or
SequenceState hidden-pointer fields. Existing factory construction continues
to call the legacy head constructor and therefore allocates no paired slab.

The actual Glm5MtpHead owns an optional private paired pool, initialized only
by an explicit staged paired constructor (no factory caller in A). The pool
owns the98304-byte slab and two generation-tagged leases. Glm5MtpProposerState
holds an optional unforgeable lease token minted by that head's allocation
path; a lease binds the head/backend, physical slab slot, sequence slot once
known, generation, historical capture stamp and absolute normalized positions.
No raw arbitrary-row constructor is exposed. Private normalized views are
issued only after successful actual producer copies and completion, and are
revalidated against the current live lease before a consumer reads them.

The paired constructor allocates actual BF16 private KV capacity for two
independent full prefixes: per-slot `ceil((context_tokens+4)/16)` blocks,
with `context_tokens+4<=2048`. At context2044 this is128 blocks/slot,256 total,
compared with128 for the qualified legacy max_seq_len2044. For NoPE512 BF16 K
and V this is4194304 extra KV bytes plus98304 slab bytes before any separately derived sparse
index allocation. Use the existing cache-plan arithmetic for exact optional
index bytes, not a guessed budget. Claim each lease's full block reserve through
the actual PagedKvCache allocator, outside propose/verify steps, so one request
cannot consume its peer's promised capacity. The existing legacy allocation
path and block-count formula remain unchanged.

Owned children under `layers/glm5_mtp/` implement the pool, sealed views,
paired primer/bootstrap, verdict/repair and tests. Add only an optional pool
field to the actual head and optional lease token to its actual state; factor
the constructor into a private child if necessary for file limits. Existing
`prefill_kv_batched` and `write_kv_rows` remain the production arithmetic SSOT.
Add a separate eager-tail bootstrap method to the existing host pair planner;
it requires exactly P-1 cached rows and a one-row tail identity, and never
relaxes the existing full-capture bootstrap contract.

Model children under `model/glm_c2_handoff*` mint capture/target/verdict source
receipts from the actual model-owned producer buffers and sequence metadata.
The actual eager-primer wrapper delegates selected paired states before the
legacy best-effort path. Head propose and worker E1 use the same owned bootstrap
or bonus view; selected worker E1 must not call the old global hidden-row reread.
The actual recorded-verdict hook stages accepted rows0..a and distinct bonus
row a before another producer can reuse normalized scratch. Existing C1 repair
functions remain the fallback when the paired capability is absent.

The actual scalar target decode wrapper must validate the selected bootstrap
phase before target work and publish H[P] immediately after successful decode,
while it still owns the normalized producer row and exact SequenceState. Head
and worker both use this wrapper. A delayed read from global hidden-save at
first propose cannot establish H[P] ownership and is not permitted. The legacy
2048-context constructor has129 blocks; that is not the qualified2044 baseline.

Wrap the outer Model::decode return so graph/profile early returns also publish
their producer-time row. Selected retirement records failures from the existing
zero_slot/synchronize branches even though Legacy logs and continues there;
a later successful free_state sync cannot erase that lease quarantine. The
reserved block table is checked against its sealed per-slot allocation receipt,
not trusted as arbitrary mutable state. Reserved capacity is not initialized KV
length: test actual128-entry metadata bounds while the logical cursor is0/P-1.

Retire a selected lease at the beginning of actual sequence cleanup, before
unrelated fallible cleanup; finish reuse only after successful completion and
private block return. Actual head free_state handles primer-only and pending
states. Model teardown invalidates paired leases before freeing the slab once;
it uses the existing model-owned proposer and backend, not a second allocation
owner or allocator API. Failed completion leaves the lease unavailable, and
failed partial copies cannot damage another lease's identity or bytes.

Gate1 tests use a real TransformerModel, actual capture and eager-primer
wrappers, actual Glm5MtpHead KV writer and head/worker bootstrap hooks, with a
recording backend/body only at the numerical kernel boundary. Gate2 extends
the same fixture through actual verdict, E1 repair and reproposal hooks for
all25 acceptance pairs. The independent test author owns one unreferenced
cross-owner negative child once this fixture's interface is fixed. These tests
do not call a test-only Ready/view constructor or lift C2 admission guards.

### Gate 1 commit boundary and evidence limits

Gate 1 freezes construction, cold complete producer preflight, detached prompt
tail, producer-time bootstrap hidden, owned first proposal, and terminal
retirement/teardown. Selected verdict recording and `after_verify` deliberately
refuse until Gate 2 supplies owned accepted-row staging and consumption. No
factory caller or admission guard is enabled in this commit.

Actual worker coverage is EP-v2 F0 plus ordinary bootstrap-token dispatch on
both slots, and C2 E1 refusal before payload. Actual first owned proposal is
tested through `run_mtp_propose_inner` on both rank configurations. Successful
public paired-worker E1 belongs to B's explicit capability/protocol activation;
changing a paired fixture's configured width to one to pass a C1 guard would
not prove that path and is prohibited. The tiny real SSM cleanup pool proves
slot zeroing/cleanup behavior only, not K5 or KDA numerical capacity. Later B
must inspect both actual slot capacities after construction with EP-v2 sizing.

Close assumes drained, sequential host lifetime, as required by actual
`Model::teardown(&mut self)`: there are no concurrently entering host producers
with live borrowed model contexts. It is not a general concurrent external
capability-close barrier. Before synchronization, all leases become terminal;
any close error blocks every later model-owner release and bulk sweep through
`release_pools`, including repeats. No allocator/backend-Drop contract changes
are made. The mock sweep spy proves Model-path invocation exclusion, not native
completion or eventual backend destruction: current CUDA backend Drop invokes
its own sweep independently. That terminal native error policy remains a B
activation review item, not a solved allocator guarantee.

Failed paired construction publishes no head or lease, but is not transactional:
the existing cache constructor/kernel lookup can leave allocations in the
backend ledger. The future B caller must abandon the entire build/backend on
error, never retry construction on that partially initialized backend. Focused
fault tests cover V allocation, index-values, index-tail, required kernel lookup,
and slab allocation, preserving all six input-weight owners and checking exact
remaining ledger entries before explicit test-owner cleanup. Native backend
Drop remains the existing final reclamation path; no rollback promise is made.
At the actual 4x128 BF16 sparse geometry, 256 blocks own 10,747,904 cache bytes
(K/V 8,388,608; pooled index 262,144; raw K/gate tail 2,097,152), plus the
98,304-byte slab. Both slab allocation and source validation include all these
actual cache spans; zero-length BF16 scale storage is not an allocation.

Selected Model prefill requires at least two prompt tokens and rejects
unsupported cold/partial/prefix/capture profiles
before target work. Once an admitted producer starts, target/capture/primer
failure must quarantine its lease before returning. Default legacy behavior is
unchanged. Gate 1's actual-arena bound is not B's admission: B still rejects
outside the resolved 1024-token cold single chunk before EP allocation.
Actual cold `prompt_len=0` is tested. An exact predeclared prompt length is also
accepted by the staged Model boundary; current scheduler/worker allocation does
not predeclare it. Mixed and batched producer APIs are explicitly unavailable
while selected in Gate 1, rather than implicitly reaching unsupported writers.

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

Before sending a slot-addressed F5/E1 command, the head must run the applicable
host-only transaction preflight. Gate 2's Model wrapper checks alone occur
after the current scheduler broadcasts F5; a head-only refusal at that point
could leave the worker entering unmatched collectives. Construction/admission
checks do not replace per-step issued-token, position and remaining-capacity
checks. A failure after command issuance needs an explicit paired-rank abort or
completed transaction policy, never local request removal followed by serving
the peer. This protocol integration is not proved by CPU Model-only tests.

Preserve the existing post-logit-processor token selection (including repetition
penalties). Raw verifier argmax is not bonus-token authority. Bind the actual
selected next seed to the owner and target position at the scheduler/E1
boundary, and have the worker consume that same selected token with its owned
bonus hidden row. Gate 2 seals the resulting next `[seed, four drafts]` inputs;
it neither authenticates the external sampler decision nor forces raw sampling.

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
