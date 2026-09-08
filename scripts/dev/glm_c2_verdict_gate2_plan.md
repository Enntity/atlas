# GLM paired handoff Gate 2: owned K5 verdict and continuation

2026-09-08. Root reviewed the complete plan and approved bounded TDD
implementation, including terminal selected-producer state after post-claim or
incomplete F5 failures. This is CPU unactivated scope, not a native recovery
policy. Implement producer claim/record first, then trim/commit/continuation;
keep factory and admission guards closed. Gate 1 is committed as `bd14c289`. Its frozen 34 source
hashes still match; the post-commit CPU model suite passed 1003/1003 in 81.81s
(`glm-c2-handoff/postcommit-model-cpu.log` in the 20260908 campaign).

This completes Partition A of `glm_c2_speculation_plan.md`, not admission B.
No factory caller, environment selection, scheduler concurrency change, public
C2 E1 success, CUDA change, native build or benchmark is included. Legacy C1
and non-GLM behavior remain unchanged. Root owns commits and all native work.

## Actual source contract

- `model/trait_impl/mod.rs::decode_verify_graphed_kgamma` is the actual outer
  K5 producer boundary. `verify_d.rs` performs target embedding, block allocation,
  metadata, eager/replayed target execution, final normalization and existing
  argmax readback, then appends the five input tokens to the target sequence.
- `scheduler/verify_dflash_step.rs` rolls target tokens back, records the verdict,
  then later calls `commit_accepted_prefix`, global hidden save, proposer trim
  and reproposal. Early EOS/output termination can intervene. Worker F5 in
  `model/impl_a2.rs` uses record -> trim -> commit. Record therefore does NOT
  certify completed target SSM commit.
- Worker F5 currently derives `verify_base` only when the legacy repair flag is
  enabled. Selected paired ownership must work with that flag OFF. An arbitrary
  caller-supplied base/token list plus current global normalized scratch cannot
  establish which actual verification produced those rows.
- `ProposalPlan::finish` already defines the accepted-pair arithmetic. Reuse it,
  the existing `write_kv_rows`, and the actual four-step proposer bodies; do not
  introduce another repair formula or copied numerical implementation.

## Chosen boundary: one short-lived verification-output lease

Use the existing private paired Pool for one host-only active producer lease.
It is not a public Ready capability or a new GPU owner. The drained, serialized
host-lifetime contract from Gate 1 remains: no concurrent entering producer and
teardown calls. Do not hold a Pool mutex across backend callbacks or body work.

After each successful actual first/re-proposal, seal its exact verification
inputs `[seed token, four returned draft IDs]`, absolute base B, checked
monotonic proposal attempt, pool identity, slot and lease generation. Publish
this only when all four actual draft steps succeed. A valid equal-length list
from the other owner is not interchangeable, even at the same base.
Root approved binding the exact committed host token prefix in the same Slot:
one authoritative Vec, at most context_tokens*4 =8176 host bytes per owner at
2044, reused between proposals where possible. Check it at begin/publish/record;
do not duplicate it in the active lease. The no full-prompt-copy promise below
means no prompt-hidden/GPU copy, not absence of this small host identity receipt.

The actual outer K5 wrapper claims the sole active output lease BEFORE target
work. Its private record binds that issued proposal, actual model/backend,
normalized output pointer/capacity, effective default stream, owner and base.
After successful dispatch it validates the actual five-token append and cursor
advance, and changes InFlight to Produced before returning. There is no new
GPU copy or allocation at this publication: normalized output remains scratch
temporarily protected by the active lease. Raw returned argmax IDs are not
semantic bonus authority: the actual scheduler can apply repetition/DRY or
other selection processing after verification. Root approved preserving the
existing valid caller-selected seed contract in this unactivated Gate 2 slice.
The owned hidden row, accepted count and position are authenticated here;
real post-pipeline selected-token ownership and worker matching are a mandatory
B integration gate. Do not silently force raw sampling or disable penalties.

While InFlight/Produced, another selected prefill, scalar decode, proposal or
verification must refuse before backend work. This includes a peer request and
a duplicate same-owner producer. The matching record consumer is the only
operation that may detach these verification rows. The unsupported fixed-width,
batched and fused verifier wrappers refuse selected paired calls before work;
initial support is the actual single-owner K5 gamma wrapper (and the existing
DFlash default delegation to that wrapper), not arbitrary verifier aliases.
Existing mixed/batched producer refusals and legacy raw repair/prefill guards
stay closed. No successful public C2 E1 guard is widened.

This is smaller than a general scratch-version scheduler: do not snapshot all
five rows at verify return, add a second hidden stash, or add generation checks
against the current global prompt capture. Detached owners retain historical
capture identity plus their live pool/slot generation and proposal attempt.

## Selected K5 preflight and target metadata

Before claiming/writing, require the actual paired profile, live communicator
and rank, BF16 target/private caches, no adapters or active stream capture,
exact issued K5 tokens, target position B, checked context/reserve bounds,
actual five-row arena extents and actual slot draft capacity at least four.
For a real SSM pool, test both slots; configuration width is not a capacity
receipt. Replay is distinct from active capture and is not rejected merely for
being replay. No selected generic fallback may run an untracked producer.
Validate each actual FP32 GLM SSM state's live H/conv and first four H/five
allocated conv snapshot addresses against its claimed slot's real pool before
K5 AND before commit. GLM K5 writes only the first four H/conv snapshots;
full acceptance keeps canonical state and does not read the fifth conv entry.
The selected check also verifies the actual SlotGuard's pool Arc identity via
one read-only `belongs_to` helper, not only its numeric slot index. Root approved
this helper without changing allocation/claim/release behavior.

Add the actual `!prefix_cache.is_active()` requirement to selected producer
preflight. Cold sequence fields do not prevent a later peer-prefix lookup hit.
B must repeat this at construction/admission and reject before EP allocation;
Gate 2 does not enable prefix reuse or enlarge cold prompt admission.

The selected path must allocate any required target blocks through the existing
allocator, then prove every K5 logical row has a valid physical block BEFORE
embedding/target launches or metadata upload. Check the actual live block map,
physical bounds, block-table capacity, positions and signed/narrowed arithmetic.
Do not treat `physical_block_for(...).unwrap_or(0)` as a safe selected fallback.
Use a selected-only host preflight/allocation branch and reuse its checked
mapping for the existing uploads; preserve Legacy allocation/event order.
No new allocator or separate target KV owner is introduced.

Validate the actual gamma metadata layout: scratch base +32768, positions +0,
slot IDs +128 when used, physical slots +256, sequence lengths +512, block
tables +768. For K5 the checked last extent is
`32768 + 768 + 5 * max_blocks_per_seq * 4`, together with all smaller fixed
regions. Compare against actual scratch capacity, not a comment's dense-size
estimate. Missing mapping, undersized arena or unsupported width fails closed.
The complete retained historical attention prefix is checked, not just the
five write destinations; no truncation by `.take(max_blocks)` may mask a short
or invalid selected map. Actual HSS/sliding representation is refused here.

## Verdict detachment, then separate SSM acknowledgement

The selected `record_glm_mtp_verified_impl` consumes exactly the matching
Produced lease after target token rollback. Check actual target cursor
`B+a+1`, committed token prefix, exact issued five inputs, base, a in 0..4,
private speculative cursor `B+3`, live identity and one-time attempt. The
existing base/tokens arguments are consistency checks, not minted authority.

Each slot keeps its existing six BF16[4096] rows (8192 bytes each): row 0 prompt
tail, rows 1..4 accepted repair rows, row 5 bonus. On the actual default stream:

- If a>0, copy normalized rows `[0,a)` to owned slab rows `[1,1+a)`.
- Copy normalized row a to owned slab row 5, including a=0.
- Complete these copies before publishing Pending and releasing the active
  scratch lease. At most two D2D copies and `(a+1)*8192 <= 40960` bytes; no new
  GPU allocation, full-vocabulary read or full-prompt copy.

`ProposalPlan::finish` remains the arithmetic authority. Its generation input
comes from the consumed owned receipt, never the later global capture atomic.
Do not relax the legacy planner's global-capture checks for legacy callers.
The Pending record retains the exact five tokens, accepted count, owned spans,
positions and two independent acknowledgements: trim seen and target commit
queued. Head and worker have different acknowledgement order; both orders must
be valid, while duplicate/wrong-count acknowledgements refuse before work.

Wrap actual `Model::commit_accepted_prefix` with selected pre/post validation:
require this Pending owner and `(num_committed,k)=(a+1,5)`; mark queued only
after the existing dispatch succeeds. Full acceptance is the dispatch's valid
no-copy case. Partial acceptance uses its actual per-slot H/conv intermediate
copies and secondary event, not new replacement SSM math.

Before the next selected recurrent consumer, enqueue the existing default-stream
wait on the secondary event and require both acknowledgements. This is GPU
ordering, not proof of CPU-completed SSM state. A failed copy/event/wait poisons
the owner. Recording a verdict alone never authorizes repair/reproposal.
The existing sealed RequestData carries this Model's actual secondary-event
handle; the owned head repair itself enqueues the wait, so a raw capability
call cannot bypass a Model-only fence. No generic Model/factory field is added.

## Actual private repair and next proposal

For verification base B and accepted draft count a, use these exact relations:

| Boundary | Private logical cache rows | Target position |
|---|---:|---:|
| Before four-draft proposal | B-1 | B |
| After four actual private steps | B+3 | B |
| After accepted-pair repair | B+a | B+a+1 |

Keep the already-written seed pair at private row B-1. For a>0 call the existing
KV writer at private rows `[B,B+a)` with tokens `[1,1+a)` and the detached
normalized rows `[0,a)` now held in slab rows 1..a. For a=0 perform no accepted
row copy or KV write; logically discard the rejected suffix and retain the
seed. For a=4 the fourth repaired pair extends one row beyond the prior
four-step speculative cache. Actual reserved capacity must cover that write.

Only after repair completion publish the new canonical private cursor and next
proposal plan. The next caller seed/bonus token must be valid and at actual
target position B+a+1; it is the caller-selected token, not necessarily raw
argmax at row a. Its normalized input is exclusively owned slab row 5.
Call the unchanged actual four-step proposer, then seal its returned draft IDs
for the next K5. Neither global prompt capture, normalized scratch, last-hidden
index nor `mtp_hidden_save` is a repair/proposal source. Do not recheck current
global capture generation after detachment.

## Failure, cancellation and lifecycle policy

Preflight mismatch before claiming the producer lease performs no numerical
work and leaves an otherwise healthy owner intact. Once target verification or
verdict detachment begins, any failure consumes that transaction and quarantines
its owner. Proposal/repair failures retain Gate 1's terminal owner behavior.

For the bounded serialized protocol, propose a terminal active-transaction
error policy: failed InFlight/Produced completion is not silently cleared to let
a peer overwrite uncertain scratch. Block subsequent selected producers until
model abort/close; peer owned bytes and lease identity must remain unchanged,
but do not promise continued peer scheduling after an issued F5 failure.
Root approved this policy for the CPU unactivated slice. No retries on partially
written state and no cancellation that skips the remainder of an issued F5.
After successful detachment the scratch lease is released; retirement of a
Pending owner can discard its owned rows without affecting another owner.

Actual cleanup must clear only its matching completed producer metadata and
invalidate live attempts before reuse; stale same-slot generations never regain
authority. Gate 1's failed SSM-slot quarantine, close-before-release and explicit
backend-Drop limitation remain unchanged. No generic allocator/free rewrite.
EOS/output-cap scheduler transaction completion and native fatal-error policy
remain required B integration gates, not claimed solved by Model-only tests.
Root approved two necessary selected cleanup extensions: wait on the actual
secondary event before target zero/release, and disarm (without release) the
actual SlotGuard even if initial paired retirement returns an error, so Drop
cannot reopen the quarantined physical slot. Fixed paired slots also refuse
legacy compaction before any copy or slot mutation.

Model teardown passes its actual secondary stream to the existing paired
close capability. Close first invalidates/marks closed, joins default and then
distinct secondary streams, and only then frees the slab or permits other model
owners to be freed. A failed record_event can leave secondary writes absent
from the event history; an old event wait is not sufficient. Either join failure
is sticky and forbids Model-path release/sweep/retry. Native backend Drop's
independent sweep remains explicitly outside this CPU recovery proof.

## Approved file/API partition

- Private head children `paired_verify.rs`, `paired_verdict.rs` and
  `paired_repair.rs`: issued inputs, active lease, detachment, owned repair.
  Extend `paired.rs`, `paired_bootstrap.rs`, `paired_lifecycle.rs`/`paired_close.rs`
  narrowly for their metadata transitions; adapt actual `after_verify` in
  `glm5_mtp.rs`. Do not move inherited numerical bodies merely for file size.
- Extend the existing optional GLM `GlmPairedHandoff` interface in
  `speculative/glm_repair.rs` with private-construction receipt operations.
  No generic Model trait/factory/SequenceState ownership fields or new public
  row-pointer constructors. Existing absent capability remains an exact no-op.
- New `model/glm_c2_verification.rs` child plus narrow `glm_c2_handoff.rs`,
  `glm_mtp_repair.rs` and `trait_impl/mod.rs` hooks: actual producer claim/result,
  selected record/commit and unsupported-writer refusal. `verify_d.rs` owns
  selected checked target allocation/mapping before launch, delegated to the new
  verification child to keep the existing producer file at 500 lines.
- `model/impl_a2.rs` F5: save actual selected base before verification independent
  of legacy REPAIR, use the same actual wrapper/record/trim/commit, and poison an
  already-issued producer on accepted-count receive/validation failure. Preserve
  all legacy command payloads and the public C2 E1 refusal.
- New test children under `model/glm_c2_*`; reuse Gate 1's actual model/head
  fixture, split numerical-boundary fixture helpers if needed to stay <=500.
  One author owns module graph/Cargo; independent test children get explicit
  separate ownership only after the production/fixture interface is stable.

## Required actual-path TDD gates

Capture runnable behavioral RED before each fix, separately from compile or
invalid-control failures. Positive controls must execute actual Model wrappers
and head KV writers; no test-only ownership constructor or painted normalized
buffer followed only by a record call.

1. Actual first/re-proposal seals its real returned tokens. Actual outer K5
   rejects wrong issued tokens/base/owner, wrong width, missing blocks/capacity,
   active prefix cache and capture before target work. Begin/return faults have
   matching actual event controls. Successful verification uses the real target
   `decode_multi_seq` default/target loop and actual final normalization.
2. All 25 `(a_A,a_B)` pairs, both ranks and both owner orders: real cold prime,
   bootstrap, first proposal, actual K5, rollback, record, actual trim/commit,
   owned repair and reproposal. Use unequal histories and distinct per-row,
   per-owner, per-cycle bytes. Assert physical K/V prefix, seed, repaired rows,
   rejected suffix exclusion and bonus sources, not only cursors or Copy events.
3. Extend the numerical-boundary private `Body::decode` fixture to write the
   actual addressed speculative private K/V rows. It currently only records
   Body events; unwritten sentinels cannot prove keep_seed or full acceptance.
   No production math changes. Sparse byte correctness is not claimed unless
   its numerical fixture is implemented; retain real sparse capacity/alias tests.
4. Continue for multiple rounds with alternating zero/full/asymmetric acceptance,
   swapped order, another actual producer overwriting global scratch after
   detachment, and one retired/reused slot while its peer continues. A current
   global capture-generation change must not revoke detached rows.
   Include at least one valid selected bonus deliberately different from the
   raw target argmax, then verify its actual returned proposal seals that seed.
5. During actual verify-to-record, peer prefill/decode/proposal/K5 attempts refuse
   before events. Wrong/duplicate record, changed prefix, stale attempt and
   same-address/new-pool or same-slot/new-generation must not consume a valid
   owner's receipt. Cover post-claim failure and active-transaction cancellation.
6. Real tiny SSM pool/claimed slots with EP-v2 sizing set BEFORE construction:
   assert both actual capacities >=4, exercise actual H/conv intermediate commit
   copies for all a and the secondary-event wait before the next consumer.
   Test both trim/commit orders, omitted/duplicate/wrong acknowledgements and
   injected copy/event/wait errors. This proves ownership/order, not KDA numerics.
7. Actual EP-v2 worker F5 with REPAIR=0, all 25 two-slot acceptance pairs and
   repeated transactions: scripted receives drive the actual K5 wrapper and record ->
   trim -> commit. Assert the same owned bytes/base as head-side hooks. C2 E1
   still refuses before payload; next owned consumption is exercised through
   actual `run_mtp_propose_inner`, not a paired fixture masquerading as C1.
8. Faults at actual target/normalization, accepted D2D, bonus D2D, completion,
   accepted-pair writer and reproposal body boundaries. Control-derived event
   identities, sticky quarantine, peer byte preservation and actual retirement
   are mandatory. Preserve Legacy C1 and all Gate 1 regressions.

Freeze source/manifest, independently review, run focused tests, full CPU model
suite, non-test check and scoped hygiene; report inherited lint/file-cap failures
explicitly rather than claiming CI-green. Root commits only reviewed source.
No CPU fixture result proves native graph replay, numerical KDA/MLA correctness,
collective agreement, C2 admission or throughput. Those remain B's serialized
native control before any two-segment/M10 optimization.

## CPU implementation receipts (20260908 campaign)

Raw logs are under `glm-c2-verdict/`. Genuine runtime REDs are retained in
`producer-red.log`, `continuation-red.log`, `escape-prefix-red.log`, `ssm-red.log`
and `foreign-guard-red-fault-coverage.log`. Compile-only failures and invalid
assertion/control attempts remain separately named; they are not behavioral REDs.
The last foreign-guard RED also contains 34 passing tests, including the initial
44 fault injections. Later appended faults are characterization GREEN, not
retroactively claimed RED coverage.

`final-focused.log`: 69/69 actual handoff tests passed in 3.21s. Coverage includes
100 head acceptance-pair cases (25 x 2 rank configurations x 2 owner orders),
50 actual rank-1 worker F5 pair cases, continued unequal histories, a non-raw
selected bonus, real tiny SSM commit/ordering, and 80 control-derived injections
(44 active verifier/detachment, 12 worker accepted receive/read/count, 24 private
repair/reproposal). The two-rank head fixture is not a two-rank collective proof.

After relocating only the new selected host branch out of `verify_d.rs`,
`full-model-cpu.log` passed 1045/1045 in 82.35s. A subsequent one-line safety
comment combination keeps that producer file at its original 500 lines;
no executable code changed after the full run. `lib-check.log` passed in 9.01s.
Workspace format, scoped SPDX and whitespace checks pass. Every new Rust file
is below 500 lines. The inherited non-allowlisted `glm5_mtp.rs` is 885 lines
(committed Gate 1 baseline 882); other changed oversized files are allowlisted.
`clippy.log` stops at the same four inherited `spark-runtime` Metal-stub
too-many-arguments errors, before model lint coverage. This is not CI-green,
native numerical validation, admission completion, or a throughput result.
