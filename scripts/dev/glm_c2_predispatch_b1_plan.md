# GLM paired B1: shared prewire validation and owned command transport

2026-09-08. Root read the full plan and approved checkpoint 1 implementation.
Checkpoint 2 is approved only after checkpoint 1 behavioral RED/GREEN and root
source review. Coordinate model edits with the selection author's Cargo window.
Partition A Gate 2 is committed as `f8a0bdb9`; its post-commit CPU suite passed
1045/1045. Root closed its archive before this plan. Root approved the design
direction, API and eight-word selected wire format below under that sequencing.

B1 remains unactivated: no factory caller, environment flag, admission change,
selected scheduler dispatch, image, native run, benchmark or fatal-error recovery
claim. Root owns commits and native work. This is the smallest reusable transport
boundary for B2's later serialized C2 driver, not a generic batched verifier.

## Exact current seams

- `scheduler/verify_dflash_step.rs` sends slot/F5, width, and tokens before
  `Model::decode_verify_dflash` reaches actual paired validation. Its first
  `sync_secondary` also precedes that validation. A local predictable refusal
  can therefore leave a worker executing an already-issued transaction.
- `model/glm_c2_verification.rs::paired_before_verify` validates the real model,
  SSM bindings and existing history, then `paired_begin_verify` claims scratch.
  `paired_allocate_target` allocates missing target blocks later. Exhaustion can
  currently occur after claim; available capacity must be checked before wire.
- `model/trait_impl/speculative.rs::run_mtp_propose_multi_dispatch` calls the
  legacy repair validator, then emits E1. With REPAIR=0 that validator is a
  no-op, not selected ownership validation. Its distributed branch is C1-only
  and saves/communicates a global hidden-row index.
- Worker E1 in `model/impl_a2.rs` checks the legacy C1 guard, receives four
  words, validates legacy repair, copies global normalized hidden, then proposes.
  Worker F5 already calls the actual gamma/record/trim/commit path with the
  selected base independent of REPAIR. Neither command proves remote readiness.

## One optional Model capability

Add one object-safe, GLM-specific, sealed execution interface in a new
`speculative/glm_paired_execution.rs`. The only implementation is the actual
`TransformerModel`; its Model accessor returns Some exactly when its actual
proposer supplies `paired_handoff()`. Invalid selected configuration returns an
error from validation, never None/fallback. Other models and legacy C1 inherit
None without backend work. Do not expose the private Pool or arbitrary row spans.

Proposed public signatures (names may follow repository style, semantics fixed):

```rust
fn Model::glm_paired_execution(&self) -> Option<&dyn GlmPairedExecution>;

trait GlmPairedExecution: Sealed + Send + Sync {
    fn validate_verify(&self, seq: &SequenceState, tokens: &[u32]) -> Result<()>;
    fn validate_propose(&self, seq: &SequenceState, seed: u32,
        position: usize, drafts: usize, grammar: Option<&[i32]>) -> Result<()>;
    fn verify(&self, seq: &mut SequenceState, tokens: &[u32]) -> Result<Vec<u32>>;
    fn propose(&self, seq: &mut SequenceState, seed: u32,
        position: usize, drafts: usize, grammar: Option<&[i32]>) -> Result<Vec<u32>>;
}
```

This is the only new cross-model trait seam. It is necessary because the server
owns `dyn Model`, not a concrete TransformerModel or its private paired head.
It is not a Ready value, lease constructor, block reservation, or admission
certificate. Successful validation returns only `()` and can be repeated.
Execution always repeats the same validation immediately before the first wire
operation; the actual producer/writer repeats its local validation before claim.
No server production call is added in B1. B2 will call capability.verify instead
of separately issuing generic F5 and then calling gamma.

The preflight contract is no state changes, GPU allocation/free/copy/kernel,
event record/wait/synchronize, communicator call, or external I/O. Existing
read-only backend `stream_is_capturing` remains an explicit exception: actual
capture cannot be inferred from a cached flag. Host locks and temporary host
collections used by existing KV validators are allowed; do not claim zero host
allocation or change those validators merely to optimize this stage.

## Shared validation, not parallel formulas

Split the checks from the existing Model/Pool transitions without changing
numerical bodies or duplicating arithmetic:

1. The Model's immutable verify checker reuses `paired_profile`, actual
   `paired_ssm_bindings`, exact arena/span checks, `paired_input`, and the real
   target-map validator. It borrows `seq.proposer_state.as_ref()`; it never
   takes/restores it, claims an active producer, or changes phase/attempt.
2. Extract the existing Pool begin-verify checks into a private immutable
   candidate function returning the existing small Verification value. Both
   preflight and actual begin use it. Actual begin checks and assigns under
   the same Pool lock; preflight discards the candidate. Prefix, issued IDs,
   generation, owner, position and scratch checks remain the same authority.
3. For E1, extract private read-only bootstrap and Pending proposal plans from
   `paired_bootstrap.rs`/`paired_repair.rs`. Reuse Limits, ProposalPlan,
   `pending_input`, actual acknowledgements, bonus/tail views and canonical
   block checks. Execution consumes the same derived plan before marking
   writing/Failed; no new publicly constructible state or duplicate repair math.
4. E1 preflight also validates the exact KV-only writer plan (including row
   base and actual block extents) through the same `KvRowsPlan::new` preparation
   used by `write_kv_rows`, plus the actual proposed four-step extent/metadata
   requirements. The private cache is already fully reserved by A. It must not
   allocate private blocks at preflight or manufacture another reserve.
5. Both Model E1 validation and actual selected proposal check the same real
   SSM binding/capacity and target budget for the following fixed K5. Grammar,
   non-four width, stale position, active capture, adapters, prefix caching and
   unsupported HSS remain fail-closed. No current-global-capture recheck is
   added for detached owners.

## Actual target block budget before wire

Create one private dense K5 budget checker, used by immutable preflight and
`paired_allocate_target` before calling the existing allocator:

- derive checked end `base + 5` and needed blocks from the actual cache block
  size; retain the existing 16-token dense restriction and context/2048 bounds;
- validate the complete currently retained historical map and actual SSM slot;
- require needed blocks fit the actual metadata table capacity, not only the
  current block-table length;
- derive additional blocks from needed minus already owned blocks, saturating
  only that nonnegative difference; require it <= `cache.num_free_blocks()`;
- require actual prefix cache inactive and HSS/sliding/disk representation off,
  so no unaccounted eviction/slide path is part of the successful proof.

This holds the actual target-cache lock but does not pop the free list, increase
refs, fill/poison GPU rows or append to the sequence. It prevents predictable
exhaustion under the serialized no-interleaving contract; it does not reserve
capacity or guarantee a later backend operation cannot fail. Actual allocation
rechecks the same budget and then uses `ensure_blocks_through_decode` unchanged.
For E1 the checked base is the next proposal's actual position. A peer command
cannot be interposed between preflight and dispatch by the eventual B2 driver.

The head's count is not proof of the worker's count. Worker revalidation is
mandatory before local target/private compute; any mismatch after wire is an
issued-command failure requiring B2's fatal policy, not a safe local fallback.

## Selected transport: fixed, explicitly versioned E1

Root selected a versioned payload instead of treating global hidden_row as
ownership. The actual EP-v2 preamble remains `[sequence_slot, E1]`. For an
actual paired head only, E1 then carries **eight u32 words / 32 bytes**:

| Word | Meaning |
|---:|---|
| 0 | selected E1 payload version, exactly 1 |
| 1, 2 | live paired lease generation, low/high halves of u64 |
| 3, 4 | next actual proposal attempt, low/high halves of u64 |
| 5 | actual target position, checked u32 conversion |
| 6 | draft count, exactly 4 |
| 7 | caller-selected valid seed token |

Generation and next attempt are derived from the same validated local Pool
plan, never supplied by the caller or truncated to u32. Bootstrap expects
attempt 1; a completed Pending verdict expects the next checked attempt.
The phase and owned bonus position are validated locally, not encoded as a
global row. No addresses or process-local Arc identities go on the wire.

The worker selects this format only from its actual paired capability, checks
version/fixed width/owner preamble plus its own generation, next attempt,
position and complete proposal plan, and then calls the real owned proposal
without `save_hidden_for_mtp`. It checks the returned draft count is four.
This requires both ranks to be constructed in the same selected mode; there is
no format autodetection or compatibility fallback after a header has been sent.

The legacy E1 branch retains its C1 guard, exact four-word payload
`[token, position, drafts, hidden_row]`, save-hidden operation and event order.
The selected capability's E1 branch is a deliberate new staged internal path,
not removal of that legacy guard. Existing A tests asserting that *all* paired
E1 stops before payload must be updated to this new staged contract, with
legacy C1/C2 generic-branch regression checks retained. This does not add a
factory caller or make the ordinary scheduler select paired execution.

Selected F5 keeps the existing preamble, width 5 and five issued token IDs.
Capability.verify validates before the preamble, emits those exact existing
messages, then calls the actual gamma wrapper. Worker F5 reuses immutable
selected validation after receiving inputs and before its first local wait/
compute; legacy F5 ordering stays unchanged. The existing actual producer
receipt still authenticates normalized outputs and later record/commit.

## Selected seed and failure boundaries

B1 accepts a valid caller-selected seed at the exact owned bonus position,
including a seed different from raw target argmax. Both the head execution
and worker consume the exact seed carried by the selected payload; their next
actual proposals seal `[seed, four returned draft IDs]` as in A. The packet
binds local owner/attempt/position, not the correctness of the head's sampler.
B2 must supply the actual checked post-pipeline token and retain its exact
head/worker transaction binding; this plan does not modify sampling or call
the legacy pipeline's raw fallback a successful selection.

Before the first header, validation errors leave all owners healthy and emit
zero wire messages. Starting with the first header attempt, any transport or
execution error marks the selected session terminal at the existing private
Pool boundary and refuses subsequent selected commands. In particular, a
failed header can precede begin-verify, so terminal state cannot depend only
on an already-active Verification. This is CPU fail-closed bookkeeping, not
native recovery: B2 must arm its no-unwind fatal guard **before** that first
header and have an exact-peer external supervisor. No panic-catching cleanup,
retry, F1 substitution, graceful server shutdown, native injected fault or
backend-Drop safety claim is introduced in B1.

## Small implementation/test partition

Production paths: new `speculative/glm_paired_execution.rs` and
`model/glm_c2_predispatch.rs`/`glm_c2_transport.rs`; narrow declarations in
`speculative.rs`, `traits/model.rs`, `trait_impl/mod.rs` and
`glm_c2_handoff.rs`; shared-check refactors in `glm_c2_verification.rs`, private
paired verify/bootstrap/repair children, `kv_rows.rs` plan preparation and the
existing GLM-only handoff trait. Worker E1/F5 delegates in `impl_a2.rs` stay
small; put selected parsing/execution in the new transport child. No server,
factory, allocator, SSM layout or kernel source changes are required by B1.

Two implementation checkpoints, each with actual behavioral RED before fixes:

1. Immutable validation and shared budget/plans. Actual constructed heads,
   real primed/bootstrapped/proposed states, both ranks/owners. Repeated good
   and bad validation preserves slab/KV bytes, sequence/maps, pool free counts,
   attempt/phase and subsequent real execution. Exhaust the actual target
   free list at a block-crossing K5; require failure before any command/backend
   event, then restore capacity and run the original operation. Cover the
   existing full-history, missing-map, max-table, SSM foreign guard/state,
   stale-generation, changed-prefix, wrong-issued-token, active-producer,
   capture/prefix/HSS, missing acknowledgement and context-end negatives.
2. Actual capability transport and actual worker dispatch. Record the head's
   real command payloads, feed those bytes to the rank-1 worker, and execute
   real bootstrap plus all25 accepted pairs/continuation using the existing
   numerical-boundary fixtures (no painted normalized output or fabricated
   lease). Include unequal histories, non-raw seed and retired/reused slots.
   Mutate each selected E1 identity/version/position/width/seed field; invalid
   local headers refuse before private compute, while valid received seed is
   the observed EH token and next sealed proposal seed. Field-valid but altered
   seed is not falsely diagnosed as a sampler violation; B2 owns that proof.
   Every predictable head refusal has zero recorded command messages. Inject
   each actual header/payload failure and representative body failure after
   dispatch; terminal state blocks both owners, with no retry/fallback claim.

Keep packet/fixture helpers in bounded test-only children. Production authority
comes from actual `new_paired` plus real Model methods, never test-only state
constructors. Tests through `dyn Model` must use the actual capability; do not
replace it with an always-pass policy mock to claim shared validation. Retain
all A/legacy regressions and exact legacy transport traces. CPU evidence is
not collective agreement, sampler integration, native fault containment or
performance. Freeze/review/full CPU gates precede root's separate B1 commit.

## Checkpoint 1 implementation receipt

Shared immutable validation is implemented and frozen for root review before
wire implementation. The actual `Model` accessor exposes the sealed capability;
its transport methods still return an explicit not-enabled error. No command
format, worker dispatch, server call, factory or admission code changed yet.

`glm-c2-predispatch/checkpoint1-red.log` executed six failing tests. Four reached
the new validation stubs; two exercised existing operations: target exhaustion
after K5 claim poisoned a healthy owner, and Pending reproposal proceeded with
no target capacity for its following K5. The fixed shared path passed all 75
actual handoff tests in 2.62s (`checkpoint1-green-attempt.log`) and the non-test
model check in 5.49s (`checkpoint1-lib-check.log`). No full-suite or native claim
is made for this intermediate checkpoint. Server selection changes were being
authored concurrently but are outside the model-only command and this manifest.

Target-map checks retain the existing SequenceState/allocator construction
contract. They validate extent, geometry and history but are not an independent
allocation-provenance seal against arbitrary substitution of another live
sequence's numerically valid map. Prefix reuse, HSS, restore and compaction are
refused; no new target-map ownership abstraction is introduced at this checkpoint.
