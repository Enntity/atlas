# EP execution codec and rank binding: next CPU-only slice

Status: legacy codec portion implemented and CPU-tested after the source frozen
for v12 (`0c625fad`); see the implementation record below. Rank binding, acceptance
simulation and framed transport remain proposals. No worker route, negotiation,
GPU allocation, deployment change or throughput claim accompanies this work.

## Choice

Implement **bounded legacy-transcript encoding/decoding plus a pure rank-snapshot
binder and two-rank acceptance simulator** next. Keep stage A's sequential
lowering; do not regroup decode work across a prefill or silently convert its
singleton decode substeps into a batched forward. A draft framed successor is
specified below, but its codec/bootstrap is a separate implementation gate.

This is smaller and more useful now than inserting an unnegotiated V3 command:
it pins actual broadcast boundaries, tests local state agreement and exposes
the missing lifetime authority without changing a single transmitted byte.
It does **not** make today's v1/v2 protocol generation-safe or add a real
cross-rank validation rendezvous.

Stage A is implemented in
[`traits/execution_plan.rs`](../../../crates/spark-model/src/traits/execution_plan.rs).
Its immutable `ValidatedStepPlan` checks caller declarations, canonical full
prompt payloads, positions, budgets and sequential result-consumption obligations.
It deliberately does not check live positions, lifetime generations or replay.
The module export is `traits.rs`, not the earlier roadmap's `traits/mod.rs`.
This proposal advances [the mixed-execution plan](ep-mixed-execution-plan.md)
without enabling its later GPU stages. The pinned vLLM scheduler-output reference
there informs the separation of scheduled intent and worker state; no upstream
wire format or source code is being copied.

## Exact existing protocol and identity constraints

All model paths below are relative to `crates/spark-model/src/`.

| Current operation | Ordered transport calls / local effects |
| --- | --- |
| v1 singleton decode | `U32(token)`; implicit wire slot0; then decode |
| v2 singleton decode | `U32(slot)`, `U32(token)`; then decode |
| v1/v2 prefill | Optional v2 `U32(slot)`, `U32(0xFFFFFFF0)`, `U32(chunk_len)`, `U32(chunk_start)`, `U32(full_prompt_len)`, **one bulk** full-prompt broadcast; then prefill and worker SSM normalization |
| v2 independent decode | `U32(0)`, `U32(0xFFFFFFE0)`, `U32(N)`, **one bulk** ordered slot list, **one bulk** ordered token list; then batched compute |
| slot replacement | Optional v2 `U32(slot)`, `U32(0xFFFFFFF1)`; worker frees old occupant and allocates fresh state |
| shutdown | Optional v2 slot preamble, `U32(0xFFFFFFFF)`; slot ignored |

These are a transcript of separate collectives, **not** interchangeable with a
flat concatenated frame. `model/impl_a2.rs:369` uses little-endian scalar words;
its bulk routine at152 sends native-memory u32 bytes, which are little-endian
on the current Spark hosts. A compatibility codec must encode explicitly LE and
state that legacy bulk interoperability assumes a little-endian host. Never
reinterpret arbitrary Rust structs or `usize` values as wire bytes.

The same file's `ep_worker_step_impl` at409 dispatches allocation before ordinary
commands, and `ep_worker_decode_batch` at665 validates slot bounds/duplicates
after receiving both arrays. The final `token => decode(...)` arm at642 means
an unknown opcode can become a token on an old worker. The command list comment
is not a safe token-range definition: E0/E1 are intercepted despite lying in its
documented broad token interval. Codec token validation must use the actual
vocabulary bound and reject reserved command collisions.

Important identity facts:

- Head commands use `SequenceState.slot_idx as u32`. The worker indexes its
  `slots[wire_slot]`; under v2 replacement it explicitly checks that the newly
  allocated SSM slot equals the requested wire slot (`impl_a2.rs:445`).
- Worker startup preallocates **every** slot in order, before first prefill
  (`spark-server/src/main_modules/serve_phases/build.rs:227`). First use need not
  be preceded by F1. F1 is also sent after head finish/error retirement, not
  merely when a new request arrives (`scheduler/lifecycle.rs:218,244`).
- Thus current legacy logical addressing is coupled to equal SSM slot numbering
  on both ranks. Different local KV block IDs are legitimate; different local
  SSM slot IDs require a later mapped protocol/ownership adapter. Do not claim
  the present v2 implementation already supports that independence.
- There is no wire session, step counter, generation or prompt revision. No
  pure codec can reject a stale-generation legacy frame by inspecting its bytes.
  Head sampling also means a worker cannot independently predict the next
  decode input token; agreement comes from the authoritative transmitted token.

## Proposed files and small APIs

Keep each source and sibling test module below500 lines; no scheduler or
`impl_a2.rs` call-site changes in this slice. Export pure modules alongside
stage A through `traits.rs` to avoid creating a device-dependent model executor.

- `traits/ep_execution_codec.rs`: `LegacyDialect::{V1,V2}`, explicit
  `LegacyWireLimits`, private validated `LegacyTranscript`, and checked
  `encode_legacy(&ValidatedStepPlan, dialect, limits)` / staged decode APIs.
  Payloads may borrow the plan's immutable token slices. Do not allocate another
  full prompt just to construct an encoding transcript.
- `traits/ep_execution_binding.rs`: explicit `RankSnapshot`, borrowed
  `LiveRequestSnapshot`, private `BoundLegacyStep`, and
  `bind_legacy(&ValidatedStepPlan, &RankSnapshot) -> Result<BoundLegacyStep>`.
  Return ordered indices/obligations, not raw device pointers or mutable state.
- Sibling CPU tests: byte-and-boundary goldens, malformed input, binding and a
  small deterministic two-rank acceptance state machine. The state machine is
  test infrastructure, not a transport abstraction installed into the worker.

The transcript should distinguish `BroadcastU32`, `BroadcastWords`, local
compute and local postcompute obligations. `NormalizeSsm` and `ConsumeLogits`
must **not** acquire invented legacy opcodes. Normalize follows worker prefill;
head-side consumption of decode/final-prefill logits must finish before the next
arena-reusing compute. A bare retained logits pointer is not consumption.

Encoding the supported plan emits only singleton decode and prefill operations
in their original order. Separate compatibility fixtures cover the existing E0
batch shape; they do not grant permission to emit it from `LegacySerial` plans.
V1 accepts only its singleton slot0 plan. V2 validates every addressed slot.
The new non-speculative codec rejects verification/MTP messages; preserve their
existing source untouched and test that F2/F3/F4/F5/E1 cannot be mistaken for
decode tokens. Allocation/shutdown are separate lifecycle/control fixtures,
not synthesized from stage A compute work.

## Bounds before allocation or transfer

`LegacyWireLimits` is caller-resolved, never read from env and never supplied
by the untrusted message. Require explicit positive slot/vocabulary/control-byte
limits, exact dialect, maximum full prompt, maximum chunk, maximum batch rows,
and the **actual local bulk staging byte capacity**. Resolve the first proposed
profile to at most4 resident requests, context2048, one prefill chunk up to64
tokens plus at most3 independent decode requests; preserve other deployed
profiles by not connecting this codec to them. A 1MiB absolute control ceiling
is an upper bound, not permission to use an arena smaller than that.

Use a staged parser: scalar command/preamble first, then its fixed scalar header,
then a `ValidatedPayloadShape` specifying exact bulk lengths. Only that private
shape can request bounded storage or accept a bulk slice. For F0 validate
nonzero chunk/full lengths, checked start+count, end<=full length, all u32/usize
conversions and full_len*4 against both profile and staging bounds **before**
receiving/allocating the full prompt. For E0 validate N in the admitted range
before either array; validate each array's exact count, all IDs, uniqueness and
tokens before producing a ready transcript. Reject zero work, extra scalars,
truncation, trailing bulk words, unsupported commands and reserved-token IDs.

This specifically avoids copying existing hazards into the new parser:
`ep_worker_decode_batch` currently constructs `vec![0;n]` before semantic N
validation; the prefill receiver similarly constructs `vec![0;full_len]` before
the bulk routine checks device scratch. Its `n*4` arithmetic is also not the new
codec's checked-arithmetic authority. None of these live paths is changed here.

For the pure codec, use borrowed input slices / a cursor and `try_reserve` only
after the shape check. Tests prove that an invalid header never reaches the
payload-storage phase. This does not establish safe network reception: a future
transport must obtain the shape before it allocates or schedules payload I/O.

## Rank binding and lifetime authority

The snapshot explicitly provides session identity, next expected step, local
registry epoch, immutable resolved execution profile, storage/capacity facts and
ordered live request records. A record holds logical key, phase, current computed
position, full-prompt identity/revision for prefills, and legacy local slot.
No defaults synthesize missing generations or assume all allocated worker slots
already represent admitted requests.

Binding checks exact session/next-step, supported nonspeculative text profile,
local key and generation, non-retiring/non-cancelled phase, expected position
against computed position, prefill full-prompt/revision and bounds, and capacity.
Use a borrowed canonical prompt slice for exact prompt comparison in CPU tests;
a revision number alone is not evidence that bytes agree. Head-side decode
binding can additionally validate its queued input token; worker-side binding
checks the transmitted token rather than comparing against its stale prior
input token. Legacy slot equality is required; KV block numbers stay rank-local.

The binding result's authority must be named **snapshot-validated, legacy
ordered only**. Its generation/session fields are external host declarations,
not values reconstructed from legacy bytes. Do not expose it as `WireVerified`
or claim replay protection. A later actual registry owner must issue generation
increments at admitted-lifetime replacement and refuse wraparound; F1's
preallocated idle slot reset is not by itself that admitted-lifetime transition.

Within the CPU model, freeze the borrowed registry while a bound step is ready,
or invalidate it on any epoch change. A completed/cancelled/reused slot after
planning must fail rebinding before execution. Reserve only in the simulator;
do not reserve GPU KV or mutate SSM state from a validator. Runtime binding later
must reuse `glm_cache_plan` and `ssm_indexed_decode` checks against real guarded
state, not trust these caller-provided CPU capacity facts as a hardware proof.

## Two-rank acceptance model and future transport boundary

CPU interpreter states: `HeaderChecked -> PayloadChecked -> Bound -> Agreed ->
Consumed -> Completed`, with `Rejected` and `Poisoned` terminal states. A rejection
is a value contributed by either rank, not an early return that silently lets
its peer proceed. Both ranks compare exact canonical logical transcripts and
resolved execution contracts, excluding rank-local physical KV IDs/addresses.
Both-ready plus exact agreement permits simulated execution; otherwise neither
executes. Every test records zero compute effects on rejection.

Completion requires all ordered postcompute obligations, including final-prefill
logits consumption. Only then may the next step/registry epoch advance. A failure
after simulated state mutation poisons the session: no retry of a partially
executed step. Cancellation after both ranks agree is not permission for just
one rank to skip the compute collective; resolve subsequent retirement in order.

**Legacy cannot install this rendezvous unilaterally.** Its peer already expects
the next command/payload/compute collective, and does not receive generations
or rejection votes. This slice models the required schedule but never calls
NCCL. It proves logical agreement/rejection, not actual per-layer collective
kind/count/dtype equivalence; that requires the later matched executor review.

For a separately authorized framed successor, pin the following draft before
writing transport code:

- Fixed64-byte LE header: magic8, version u16, header-size u16, flags u32,
  session u64, step u64, profile-id u32, work-count u32, token-count u64,
  body-bytes u64, reserved u64. Unknown bits/version/profile fail closed.
- Fixed64-byte work descriptors: kind u32, wire-slot u32, generation u64,
  expected-position u64, payload-offset/count u64 each, prompt-revision u64,
  chunk-start/count u64 each. Unused decode fields must be zero. Payload is
  canonical u32 token words. Derive total bytes with checked arithmetic.
- The negotiated session contract owns model/checkpoint/geometry/TP-EP/KV/SSM
  format and profile limits. A frame cannot loosen those limits. Do not add a
  hash dependency or treat a short non-cryptographic digest as exact agreement;
  the CPU model compares exact canonical contracts/transcripts.
- Matched upgraded peers first exchange the fixed header and a fixed-size
  header-acceptance status. On rejection neither allocates/transfers payload.
  Then bounded payload, full parse/binding, fixed-size acceptance status, and
  only then compute. Oversized input is rejected, not drained into a huge buffer.
  Session/step identity must be part of status matching; disconnect uses
  coordinated teardown, not waiting forever for a vote.
- Both processes must select the upgraded dialect before entering the legacy
  loop. No V3 magic/opcode probe, extra legacy words or one-sided startup
  collective. The first CPU slice does not reserve a live opcode, choose the
  bootstrap channel, implement this header, or change worker compatibility.

## Minimal test-first implementation gates

1. Golden singleton v1/v2 decode and prefill transcripts: exact LE words,
   full-prompt versus scheduled-chunk counts, scalar/bulk boundaries; interleave
   decode -> final prefill -> decode and prove normalize/consume ordering.
2. Standalone E0/replacement/shutdown compatibility goldens; V1 multi-slot plans,
   speculative opcodes, unknown command values and command-token collisions
   reject. No accidental decode batching or lifecycle synthesis from work items.
3. Staged malformed headers: zero/too-large N/full/chunk, start+count overflow,
   byte-size overflow/conversion, staging capacity smaller than profile maximum,
   exact-limit success and limit+1 rejection before payload storage is reachable.
4. Bad/truncated/extra payloads, duplicate/permuted/out-of-range IDs, token bounds,
   noncanonical spans and borrowed encode/decode round trips with exact order.
5. Live binding: stale generation, wrong session/step/revision/prompt/position,
   unallocated/retiring/cancelled request, insufficient capacity and unsupported
   storage/profile. A higher generation supplied in a plan never authorizes reuse.
6. Two ranks with different KV blocks agree; legacy differing SSM slots reject.
   Inject each binding failure on rank0 and rank1 separately: both reject with
   zero compute effects and identical next protocol phase.
7. Reuse/cancel after planning invalidates binding. Before agreement no mutation;
   after agreement no unilateral skip; interrupted execution poisons the model
   session. Duplicate completion and step/generation overflow fail closed.
8. Omit decode or final-prefill result consumption: interpreter refuses the next
   arena-writing operation and completion. Complete all obligations: exactly one
   successful position/step advancement, with no duplicated normalization.

Start with codec tests failing against missing implementation, then binding and
interpreter tests. CPU Cargo gates and independent review precede any runtime
adapter proposal. Existing images, legacy routes, memory reserves and rollback
remain unchanged throughout this slice.

## Authorized codec-only implementation record

The implemented scope is test groups1–4 only. `traits/ep_execution_codec.rs`
contains explicit limits, borrowed scalar/bulk calls, exact LE serialization,
separate compatibility-command encoding and immutable plan transcripts.
`traits/ep_execution_parser.rs` validates at most5 exact four-byte scalar calls
before exposing private validated bulk sizes. Its payload parser borrows LE-byte
views without alignment-dependent casts or a full-prompt copy. New modules are
exported through `traits.rs`; nothing invokes them from a serving path.

E0 encoding/decoding is restricted to the canonical V2 sender and sentinel0;
this is a supported-profile restriction, not a claim that the old V1 worker
rejects E0. Shutdown encoding uses slot0, while decoding preserves the existing
worker's ignored preamble. Decode/prompt/batch token values reject the entire
reserved E0..FFFFFFFF control region in addition to vocabulary bounds. Prefill
plans without their explicit local NormalizeSsm obligation reject rather than
quietly changing the worker's normalization semantics. No local obligation is
serialized as a command, and decode rows are never regrouped.

TDD: the initial8 test groups failed compilation against the missing API; after
implementation all8 passed. Three further edge groups cover borrowed round trips,
bulk token/slot rejection and checked overflow/boundary cases. Final focused
result:11/11 PASS; complete no-default-features model unit result:732/732 PASS
in3.87s using the supplied local CPU test libraries. Rustfmt and diff checks
pass. Sources are frozen for independent review; no commit, native/GPU run or
node operation was performed by the implementing agent. All new Rust files
remain below500 lines.
