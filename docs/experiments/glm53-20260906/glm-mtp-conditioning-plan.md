# GLM native MTP: restore the target-conditioning contract

Status: slice A (GLM normalized target-hidden save/stash) independently
reviewed, committed and native-tested. Root's matched LRU coding receipt
(148 input / 256 output tokens) measured median full-wall throughput
24.280 versus 18.516 tok/s, approximately +31%, with all outputs reaching
the cap. This is the hidden-handoff result, not accepted-cache repair.
The earlier v13 diagnostic chat failure and v14 thinking-EOS delivery issue
are separate lifecycle observations; the latter has its own reviewed fix.
Slice B has a CPU-only pair-planning foundation; accepted cache repair is
not integrated. Scope is the existing bounded
C1 native-MTP lane, not concurrent speculation or long context.

Slice A is committed as `a35ad139`. CPU receipt: behavior-preserving extracted
raw-copy helper produced
RED with 2 failures (GLM save and stash bytes), while non-GLM preservation and
overflow tests passed. Changing only the GLM source to the already-computed
normalized buffer produced GREEN, 4/4 focused tests. The actual save and
stash paths invoke this tested D2D helper. MockGpu performs real byte copies;
tests check non-uniform weighted RMSNorm reference rows, row permutations,
stash survival after source overwrite, both rank-local fixtures, and no new
kernel, allocation, or synchronization. These are CPU transfer-contract tests,
not a GPU RMSNorm or model acceptance oracle.

Slice B pure planning foundation is now isolated in
`crates/spark-model/src/speculative/glm_pair_plan.rs` with sibling tests and
one module export. It has no serving caller, allocation, or wire change.
The initial tests failed to compile before implementation (missing module),
then 9/9 focused tests passed. The final combined model suite passed 751/751,
including all nine latest planner tests; receipt:
`slice-b-foundation-model-tests.log` in the root phase6 receipt directory.
Limits require an explicitly fixed continuous,
grammarless C1/four-draft profile without adaptation, catch-up, carry or
prefix reuse, and preserve the context-plus-four <=2048 bound. Bootstrap
and verified/discarded proposal transitions return private immutable state
and checked `PairWrite` source/destination spans. Caller-provided generation,
position and capacity assertions are not live-state or wire authentication.

Intended next callers, still not implemented:

- GLM proposer bootstrap prepares the missing shifted prompt tail after the
  existing eager primer, consuming owned post-norm prompt rows. Its write
  span can include the whole prefix if the primer is lazy.
- GLM proposal entry records `ProposalPlan` before its four transient writes;
  the actual K5 head and F5 worker verified-commit boundaries bind the same
  acceptance metadata into a pending `FinishPlan`. Confidence discard is an
  explicit separate event, never inferred from `after_verify(0)`.
- The next existing E1 invokes `run_mtp_propose_inner`: after preserving the
  bonus hidden but before any scratch-using primer/proposer work, snapshot
  the plan's accepted normalized rows in checked owned staging and call
  the extracted GLM KV-row writer with `tokens[token_start..]`, hidden offset,
  `cache_start`, and `rows`. Publish planned state only after successful
  writes. No new E1 fields or collective is needed; ended/cancelled requests
  may free their cache without executing an unused next-proposal repair.

Before returning a proposal plan, the pure planner checks the possible
full-accept fifth cache row and all four accepted-hidden staging rows, not
only the four transient proposal writes. Finish retains defensive checks.
Future runtime admission must bind those capacities to actual allocations;
an unexpected error after an issued forward is not permission to continue
with stale cache state.

## Proposed slice B runtime patch, not authorized by this document

Rollout name proposed: `GLM_MTP_REPAIR=0|1` in the dual-Spark launcher,
forwarded as `ATLAS_GLM_MTP_REPAIR` to both ranks, default off. This is a
checked implementation A/B gate, not an independent tuning policy. The actual
v15 native snapshot `4a5` contains the reviewed handoff/EOS fixes and excludes
both uncommitted slice B foundations for attribution. Repair would be a
separate v16 experiment after source and CPU review.

### Exact ownership and hook proposal

The KV extraction author owns `layers/glm5_mtp.rs` and its `glm5_mtp/`
submodules until the helper freezes. A subsequent explicitly assigned layer
patch adds request-local `RepairPhase` to `Glm5MtpProposerState`, with phases
`PromptCapture`, `Canonical(PairState)`, `Proposed(ProposalPlan)` and
`PendingVerified(FinishPlan)`. The layer exposes narrow checked bootstrap,
record-verdict, discard and apply-repair methods; the KV-row executor itself
continues to own neither state cursor updates nor scheduler policy.

The model integration author would own a new small
`model/glm_mtp_repair.rs` adapter plus the following minimal call sites:

1. `model/impl_b3.rs::run_mtp_propose_inner`: prepare bootstrap or consume
   pending repair **before** legacy `ensure_drafter_context`/catch-up and
   proposer scratch work. Repair mode bypasses those legacy context paths;
   off mode calls them unchanged. Before the first transient draft write,
   create/store the checked `ProposalPlan` using actual private cache rows.
2. `traits/model.rs` plus the thin `model/trait_impl/mod.rs` bridge: one
   optional family-specific `record_glm_mtp_verified` operation, no wire
   message. Its arguments include the actual verify base, width and accepted
   count; the model adapter binds the request, generation and live row spans.
   Default behavior for unrelated models is the existing no-op contract.
3. `spark-server/src/scheduler/verify_dflash_step.rs`: call that operation
   immediately after its K5 target token/length rollback and **before** any
   emission can terminate the request. This is an actual verified event,
   unlike the overloaded `after_verify(0)`. Errors finish this request.
4. `model/impl_a2.rs` F5 worker branch: record the same event after receiving
   accepted count and target rollback, before its legacy proposer trim.
   Both ranks retain the original F5 and E1 payloads unchanged.
5. The GLM layer's legacy `after_verify` in repair mode only acknowledges a
   matching already-recorded verified event; it must not mutate the old
   transient length or invent a verified event. Unverified cancellation can
   release the state; an explicit discard method uses `DiscardUnverified`.
   Unexpected trim without a recorded event fails closed in this restricted
   lane. Non-repair behavior is byte-for-byte unchanged.

This hook placement avoids a subtle terminal asymmetry: the worker receives
acceptance before the head emits tokens, so an EOS on the head must not be
the only thing deciding whether a verified descriptor exists. An ended
request may simply free the private cache on each rank; no repair collective
is needed and no E1 need be sent. The next live E1 already calls
`save_hidden_for_mtp(hidden_row)` before common proposal entry; require that
row to equal the pending plan's bonus index and its position to match the
planned committed target position.

### No new persistent staging allocation

Do **not** use `verify_hidden_stash` without checking construction:
`model/impl_a1.rs:394` allocates it only when the constructor already has a
proposer. GLM's external proposer is installed later (`factory/build.rs`),
so that stash is NULL in this lane. This corrects the earlier tentative
stash suggestion; the pure planner never assumes a particular allocation.

Use the already allocated `mtp_prefill_hidden` capture instead, only after
bootstrap has consumed the complete prompt including its missing tail pair.
The buffer is allocated through `has_mtp`, which includes external GLM.
Validate nonnull pointer, checked `capacity * H * 2` bytes, at least four
rows, and this request's capture generation before transition. Mark the
phase `Canonical` and invalidate the legacy capture-length/range metadata
when changing ownership from prompt capture to repair staging. A stale
`ensure_drafter_context` must never reinterpret repaired rows as prompt
hidden states; simply relying on the primer's nonempty-state fast return is
not an ownership guarantee. Cold prefill of the next request installs a
fresh generation and restarts prompt capture normally.

At E1, the original K5 `norm_output` still has all five target rows. Validate
that source span and snapshot only the a accepted repair rows (0..a), at
most four, into the repurposed capture allocation. The bonus row a already
has its independent `mtp_hidden_save` copy. Pass the writer a DeviceSpan of
**exactly a staged rows**, not five; `VerifiedCommit.normalized_hidden_rows`
describes the original live K5 source before staging, not its smaller copy.
There is no write for a=0: retain the seed and advance canonical metadata.
Only after all writer calls succeed may the layer publish new private KV
length and clear pending state. Never overwrite target hidden sources before
their copy, and never restart a failed repair as ordinary serial decoding.

### Single-source eligibility before load and before proposal

The rollout parser and pure profile resolver should be one helper shared by
`main_modules/serve_phases/preflight.rs`, the model adapter, and launcher
tests. Require GLM, native MTP, TP2/EP2, active/admitted C1, BF16 KV,
four drafts, context-plus-four <=2048, batched primer on/serial primer off,
distributed proposer on, `MTP_GATE_FORCE=1`, MTP during thinking enabled,
depth adaptation off, DFlash adaptive suspension off, and draft-confidence
discard disabled. Gate forcing only disables the measured-throughput gate;
it does not disable watchdogs, cancellation, output/context limits, or EOS.

Reject grammar/constrained requests for the initial lane before bootstrap
GPU work, and reject a changed depth or uncovered serial position at model
entry. Require no prefix caching (`ServeArgs.enable_prefix_caching=false`,
its existing default), no SSM prefix snapshot reuse, no drafter carry and no
generic catch-up/refeed. Runtime bootstrap additionally proves owned full
prompt capture and exact P-1 (eager) or zero (lazy) private rows; do not
silently accept missing prefix coverage. Both rank launch environments and
startup-derived capacities must agree before requests are admitted.

The model adapter must validate head-side descriptor/position/capacities
before sending E1, then validate the worker's received fields before its
local repair. These are homogeneous fixed-lane checks, not a new rank
rendezvous protocol. Unexpected device failures remain fatal request/worker
errors under the existing error machinery, not a license to continue or
silently fall back. Root must review both-rank failure handling at the exact
new hook sites before this design is enabled.

### Integration tests after the isolated foundations

Use real model adapter methods with MockGpu/checked writer seams: eager and
lazy bootstrap; two rank-local allocations and identical accepted metadata;
all a=0..4 across repeated cycles; seed retention; full-accept extra row;
original K5 source -> a-row staging -> KV write with the bonus untouched;
capture-generation change across requests; terminal EOS/cancel before E1;
explicit unverified discard; stale E1 hidden row/position; unsupported
profile/request rejection before any copy or broadcast; and injected writer
failure never publishing a canonical cursor. Off-mode tests must prove the
existing primer/proposal/trim sequence is unchanged.

Only after these CPU gates and independent source review should root run a
stopped-model bounded cache oracle and memcheck, then the matched continuous
C1/K5 baseline versus repair. Keep identical gate-force settings in both
arms so the persistent gate's periodic serial measurement is not a confound.

## Why this is the next C1 candidate

The late historical receipts in [the dual-Spark ledger](../../glm53-dual-spark.md)
put four-draft proposal near 10.7–11 ms and K5 target verification near
94–96 ms. Split vocabulary, compact distributed argmax, mixed-precision draft
head, NVFP4 draft EH/O projections, and target verification graphs already exist.
Eliminating proposal altogether cannot improve that approximately 106-ms sum
by 25%. Conditioning repair instead targets useful tokens per verification.
Its gain is unmeasured: the generic 27B accepted-refeed history in
`crates/spark-model/src/speculative.rs` reported only a small improvement on a
different model, and repairing historical KV cannot remove approximation
inside the next autoregressive draft chain.

For illustration, if a measured cycle currently commits 2.5 tokens in 106 ms,
adding 2 ms of repair needs at least 3.18 tokens/cycle for a 25% gain. That is
a testable acceptance target, not a forecast. Current matched coding must be
measured; the historical repetitive 1K/256 result above 30 tok/s is not that
workload.

## Primary reference and two distinct local gaps

Official vLLM reference is pinned at
`6865e67f0be02d53694517f6f71d7fb96492792d`, not the public benchmark binary.

- Its [GLM target](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/model.py#L658)
  returns final-normalized hidden rows. The [GLM MTP layer](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/mtp.py#L77)
  applies its separate learned hidden normalization and recycles its own
  post-shared-head-norm output for subsequent drafts. The current main MTP file
  was also inspected and has the same relevant contract.
- Its [proposer first pass](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/spec_decode/llm_base_proposer.py#L506)
  processes retained target hidden rows before choosing the last row to sample;
  [input preparation](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/spec_decode/llm_base_proposer.py#L756)
  shifts token IDs relative to those hidden rows. This refreshes accepted
  target-conditioned drafter cache entries instead of retaining speculative
  conditioning indefinitely.

SGLang main's [GLM NextN class](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/glm5_next_nextn.py)
inherits DeepSeek NextN forwarding and overrides loading. It is not an
independent GLM-specific mathematical oracle. Its public
[TP8 embedding issue](https://github.com/sgl-project/sglang/issues/37548) also
argues against treating an unpinned SGLang image as an automatic correctness
reference. Do not import either engine's hardware assumptions into SM121.

### A. First-draft hidden representation mismatch

`model/trait_impl/drafter_prefill.rs:149` explicitly final-normalizes GLM
prompt rows. However `model/trait_impl/speculative.rs:279`
(`save_hidden_for_mtp_dispatch`) copies `buffers.hidden_states()` for every
model. `trait_impl/mod.rs:461` calls that exact dispatcher; the existing raw
hidden comment describes another family's contract, not GLM's.
`verify_d.rs:372` separately writes final-normalized K5 rows to `norm_output`.
Those arena accessors are distinct. Both head and EP worker call the same
save dispatcher before invoking the GLM proposer.

First proposed slice: a pure, tested model-specific hidden-source selection
used by the actual save operation. GLM uses its already computed final-norm
row; all other models preserve raw input. Audit save-from-stash consistently,
but do not enable batched GLM MTP. Preserve the last-selected-row bookkeeping.
No additional RMSNorm, normalization-weight changes, or vocabulary changes.
Test the source choice with a non-uniform learned final norm so an accidental
second generic RMS normalization cannot conceal the distinction.

### B. Accepted cache entries and the bootstrap gap

Local evidence, under `crates/spark-model/src/` unless noted:

- `layers/glm5_mtp.rs:884`: one true target hidden initializes the first
  draft, then the loop recycles draft hidden rows.
- `layers/glm5_mtp.rs:919`: `after_verify` only subtracts rejected draft
  count from private KV length. There is no accepted-row regeneration.
- `model/impl_b3.rs:118`: generic catch-up requires a tracked pair key.
  GLM inherits `DraftProposer::last_pair_key -> None` and the no-op catch-up;
  neither existing generic flag supplies the missing GLM implementation.
- `model/trait_impl/speculative.rs:103`: initial context consumes only the
  prompt prefix; GLM's primer writes `prompt_len - 1` shifted pairs.
- `spark-server/src/scheduler/mtp_step.rs:188`: native MTP bootstrap consumes
  the first generated token before the first proposal. Its corresponding
  shifted prompt-tail pair is absent from that primer.
- K5 uses `scheduler/verify_dflash_step.rs`, not a separate K5 scheduler.
  It saves only verify row `accepted`, trims, then proposes. Worker F5 in
  `model/impl_a2.rs:608` trims and commits before later E1 proposal. Neither
  path refreshes the earlier accepted rows.

The ignored `target_hidden_stack` argument is **not evidence of a bug**:
its documented meaning is DFlash's multiple-layer feature concatenation,
not a K-row target verification history. Do not repurpose that pointer.

## Exact row convention for tests and implementation

Let `x[j]` be sequence token j; `H[j]` is the target's post-final-norm hidden
after consuming it. A canonical GLM MTP KV pair at key j is
`(embedding(x[j+1]), H[j])`. GLM NoPE does not make missing pairs harmless:
attention still reads the historical KV rows.

At proposal entry the target has consumed x[0..L), the pending input is x[L],
and its supplied hidden is H[L-1]. Let R be private committed KV row count.
An n-draft proposal writes R..R+n from these inputs:

| Private row | Input token | Hidden source |
|---|---|---|
| R | x[L] | H[L-1], true target |
| R+i, 1 <= i < n | draft i | recycled MTP output |

Verification consumes `[x[L], draft1, ..., draftn]`, producing target rows
`H[L+i]` at verify row i. Accepting a drafts commits a+1 target input rows;
the next pending token is the correction/bonus predicted by row a.
The existing trim retains only a private rows, not a+1. Thus zero acceptance
discards even the true seed; full acceptance lacks the final accepted token's
pair, and partial acceptance keeps some draft-conditioned historical rows.

The proposed canonical commit is:

1. Keep seed pair R, which already used H[L-1].
2. Rewrite/append a pairs at R+1..R+a+1: input draft i with **verify hidden
   row i-1**, for i=1..a. No correction token is included in this repair.
3. Set committed private count R+a+1. The next proposal appends the
   correction/bonus with verify hidden row a at that next row.

At a=0 this retains one seed, writes no repair pairs, and next proposal
appends the correction. At a=n repair includes row R+n, which was never
written by the prior n-draft proposal. Capacity checks must include it.

Prompt P tokens produce P-1 original primer pairs. Before first proposal,
bootstrap has consumed x[P]. Append pair `(x[P], H[P-1])` once, using the
last captured prompt hidden while it is still owned. Then R=P and L=P+1,
so canonical `R=L-1` holds. An eager primer may already have consumed the
original prompt: extending a token slice in the first-propose caller alone
is insufficient because the primer currently rejects nonempty state.

## Implementation shape after review

Keep the two repairs separate for attribution: normalized handoff first,
then accepted/prompt-tail KV repair. No live route is authorized by this plan.

Introduce a small pure checked GLM pair-span planner with explicit logical
position, private row base, prior proposal width, accepted count, token span,
hidden representation/row count, and cache/arena capacities. Separate a
verification commit from **discard unverified proposal**: generic
`after_verify(0)` is also called by draft-confidence rejection, where there
is no committed seed or target verification result. Merely changing trim to
`accepted+1` is incorrect.

Build the repair on the existing NoPE KV-only projection chain in
`layers/glm5_mtp.rs:625` and
`layers/qwen3_attention/prefill/cache_skip_mla.rs:42`: arbitrary checked
destination rows instead of initial-row-only population; no attention output,
O projection, or MoE for historical pairs. Keep the established BF16 primer
projection semantics for the first correctness prototype. It differs from
the NVFP4 EH decode path, so compare against a frozen BF16 KV-only oracle,
not an assertion of bit identity to quantized decode.

Hidden rows must be consumed before the shared arena is mutated. A repair
cannot read `norm_output` in place while its own normalization/projector
overwrites that buffer. Stage at most five H=4096 BF16 rows in an explicitly
checked existing nonaliasing span, or consume from an already owned small
stash; never invent a lifetime for the DFlash stack. Save the bonus hidden
independently before repair. Stage all sources before any destination write.

Both EP ranks have the same verify inputs, accepted count and local target
hidden rows. Perform deterministic repair at the corresponding verified
commit boundary on both ranks, before E1 proposal/scratch reuse, not solely
inside the rank-0 scheduler. Preserve F5/E1 wire bytes. For n=1..3 also audit
F2/F3/F4 before enabling the same repair there. Initially pin n=4 for the
measured lane if narrower parity is not yet implemented; reject unsupported
runtime depth changes rather than silently changing collective ordering.

An EOS/cancel branch may skip next proposal and free state: do not repair
only one rank then let the other wait for an unissued collective. Repair
itself should have no collective. Cache state ownership/freeing and target
SSM commit stay unchanged. Serial adaptive gaps require explicit span
coverage; do not pretend a canonical cache if intermediate hidden rows were
never captured. The initial prototype may require a fixed continuously
speculative profile, with this restriction checked before any mutation.

## TDD and promotion gates

1. Source/MockGpu tests first fail on raw GLM hidden save; verify exact source
   addresses for rows 0/1/4, distinct raw/norm sentinels, both rank callers,
   non-GLM unchanged, and no normalization twice.
2. Pure pair-plan tests use labeled tokens/hiddens: P=1/2, eager and lazy
   primer, bootstrap pair exactly once, n=4 with a=0/1/2/3/4, repeated cycles,
   partial then full, all rejection, block15/16 boundary, capacity and checked
   arithmetic overflow, stale generation, missing hidden rows, discard versus
   commit, EOS/cancel without next proposal, and adaptive-depth rejection.
3. Mock state-machine tests exercise real head/worker commit seams and assert
   identical plans, row order, preserved bonus hidden, and error before
   allocation/write. Test draft-state release and a fresh second request.
4. Root-only bounded GPU oracle: frozen labeled BF16 KV-only chain; compare
   every touched and untouched K/V row plus guards, both ranks; repeated
   acceptance/rollback sequences and graph verifier replay. This prototype
   does not capture the new repair in a graph.
5. Matched coding A/B, each slice separately: same prompt/output/temperature,
   checkpoint, precision, context, watchdog and stop policy. Record full wall
   plus post-first-token rates, actual completion count/finish reason,
   proposal/verify/repair timings, acceptance histogram and conditional p1–p4.
   A first short run must remain valid before longer cap-complete receipts.

Keep existing C1, TP2/EP2, BF16 KV and `max_seq_len + num_drafts <= 2048`
guards; no added model weights or SSM slots, no KV-overcommit relaxation,
and root's >=4 GiB measured free-memory check before full-model experiments.
Fail promotion if repair cost outweighs measured acceptance gain. No claim
that this alone reaches 30 C1 tok/s until matched coding receipts prove it.
