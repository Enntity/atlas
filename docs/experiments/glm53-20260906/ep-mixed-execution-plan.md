# EP mixed execution: typed intent before packed execution

Status: stage A's pure execution-plan contract and bounded legacy codec are
implemented, CPU-tested, and **unused by serving routes** (updated 2026-09-07).
`2463786b` adds the validated sequential plan; `79c01f03` adds legacy transcript
encoding and staged parsing. Rank-state binding, the two-rank acceptance
simulator/rendezvous, scheduler/worker adapters and framed transport are **not
implemented**. The [codec and remaining binding plan](ep-execution-codec-plan.md)
records the narrower implemented scope and next gates.

The codec's 11 focused tests and the full 732 model CPU tests pass. Full-suite
receipt: `/tmp/atlas-glm53-phase5-20260907.gu145h/execution-codec-model-tests.log`.
Neither commit changes transmitted bytes or adds generation protection to v1/v2.
No GPU experiment or performance claim follows from these CPU-only foundations.
Follow-on to
[the infrastructure roadmap](vllm-infrastructure-roadmap.md), not permission to
remove existing EP/MLA exclusions or enable generic mixed-scheduling flags.

## Existing boundaries and why they matter

- `spark-server/src/scheduler/phase_continue_prefills.rs` excludes EP from
  both batched-prefill and batched-mixed dispatch. Its `run_standard.rs`
  also requires `!model.is_ep()` for the single-prefill mixed branch.
  `phase_start_prefills.rs` excludes EP from mixed chunk-zero deferral.
- `spark-model/src/model/trait_impl/decode_b.rs::mixed_forward_dispatch`
  additionally rejects the fused implementation for any communicator or MLA.
  Its sequential fallback returns a decode-logits pointer and then runs
  prefill; that is not a proof that the pointer survives buffer reuse.
- `traits/model.rs::mixed_forward_batch` already records two necessary
  invariants: retire decode work before another stream reuses its arena, and
  give pending decode/prefill logits disjoint lifetimes/rows. The ordinary
  `mixed_forward` fallback does not establish those guarantees.
- `trait_impl/prefill_b/batch.rs` and `batch_kernel.rs` call `zero_all` for
  multi-rank prefill. This clears logits and scratch, including decode
  metadata. Parking logits or an indexed-KDA slot table inside that arena
  across a prefill call is unsafe without explicit lifetime handling.
- `model/impl_a2.rs` currently implements scalar prefill and decode plus the
  v2 independent-decode command `0xFFFFFFE0`: sentinel sequence ID, opcode,
  N, ordered sequence IDs, and tokens. Prefill `0xFFFFFFF0` carries chunk
  length/start/full length and the entire prompt. It has no mixed work list.
  `ep_worker_decode_batch` reconstructs references in transmitted order.
- Legacy allocation `0xFFFFFFF1` replaces the slot occupant. There is no
  transmitted lifetime generation to reject a stale command for a reused
  slot. Do not treat a request slot, a KV block, an SSM slot, and a token-row
  ordinal as interchangeable identities.
- Worker prefill derives `is_last` from the transmitted bounds and performs
  SSM normalization after each chunk. Both are collective/state ordering
  requirements, not optional scheduler bookkeeping. Prefix-cache agreement
  in `prefill_b/prefix_lookup.rs` can change the effective compute span.

Paths above are relative to `crates/`. Keep these exclusions until an explicit
implementation capability covers the complete path.

## Pinned upstream reference

Use official vLLM revision `6865e67f0be02d53694517f6f71d7fb96492792d`, as in
the roadmap; do not use the older local June checkout as this model's oracle.
No upstream source copying is proposed.

vLLM's `SchedulerOutput` carries new/cached request updates, per-request and
total scheduled-token counts, explicit speculative tokens, finished requests,
and cache-maintenance information. This supports separating scheduling intent
from worker-local execution state.
[Pinned output types](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/core/sched/output.py#L205).

Its scheduler maintains token/input budgets and caps each request's scheduled
work before allocation/dispatch; adopt the explicit budget accounting, not its
whole scheduler or hardware assumptions.
[Pinned scheduling loop](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/core/sched/scheduler.py#L509).

GLM KDA distinguishes prefill, independent decode, and speculative work, using
query boundaries and state indices rather than interpreting every token row as
an independent sequence. Atlas's newly indexed N2/N3/N4 decode kernel still
cannot process a multi-token prefill as independent rows.
[Pinned KDA orchestration](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/kda.py#L401).

## Smallest useful implementation slice

The pure, immutable **scheduled intent → validated substep plan** and its legacy
transcript lowering are implemented. They represent existing independent decode
and single-request prefill operations in order, without executing them. This is
a scheduling contract, not enabled mixed scheduling or fused model execution,
and does not promise weight-sharing or throughput improvement.

Suggested new modules, each bounded below the Rust file-size cap:

- `spark-model/src/traits/execution_plan.rs`: implemented shared semantic types,
  checked construction and sequential result-consumption obligations. Exported
  through `traits.rs`; no GPU/communication dependency in planning logic.
- `spark-model/src/model/ep_execution_plan.rs`: rank-local binding and
  lowering into existing compute paths; reuse `ssm_indexed_decode` and
  `glm_cache_plan` instead of duplicating pointer/geometry rules.
- `spark-model/src/traits/ep_execution_codec.rs` and `ep_execution_parser.rs`:
  implemented bounded legacy-command lowering and parsing. No transport calls,
  rank binding or framed-protocol codec is implemented.
- `spark-server/src/scheduler/execution_plan.rs`: adapter over current active
  and prefilling requests plus existing token-budget policy, not a second
  request queue. Existing continuation/finalization helpers remain owners of
  sampling, cancellation, streaming, and request-state advancement.
- Later, a small matched sender/receiver adapter called by `impl_a2.rs` and
  the scheduler. Do not grow the existing large protocol file with another
  ad hoc parser or invoke head-side broadcasting methods from a worker.

### Types and identities

| Proposed type | Meaning and invariant |
| --- | --- |
| `RequestKey { wire_slot: u32, generation: u64 }` | Stable cross-rank logical identity. Generation increments on replacement and never wraps silently. Not an SSM pool index. |
| `StepId { session: u64, sequence: u64 }` | Orders plans within a model/protocol session; no replay after reload, retirement, or duplicate completion. |
| `WorkItem::DecodeOne` | Exactly one token at the request's expected computed position; never speculative time. |
| `WorkItem::PrefillChunk` | Prompt identity/revision, checked start/count, total length, expected computed position. Derive final-chunk status; zero work is a distinct no-op, not a fake decode. |
| `TokenSpan { offset, count }` | Checked range into a bounded token payload, not a pointer; no overlap/gaps in the canonical packed payload. |
| `ScheduledIntent` | Ordered unique request keys, work items, explicit token budget and selected execution capability. No device pointers or rank-local block IDs. |
| `ValidatedStepPlan` | Immutable, capability-checked substep order plus exact expected position updates and result-consumption obligations. No live mutation during construction. |
| `RankBoundStep` | Local request/state references, SSM guards, local block tables and slot capacities, and prepared metadata. Resolve independently on each rank. |
| `ResultDisposition` | Consume decode logits before prefill arena reuse, or use explicitly owned storage whose lifetime is proven. A bare `DevicePtr` is insufficient. |

Keep geometry, storage format, model revision and TP/EP partition in a shared
`ExecutionSignature`. Capabilities are explicit enum variants such as
`LegacySerial`, `IndependentDecode`, and later `PackedGlmMixed`; not a collection
of independently inferred booleans. Unknown capabilities are rejected before
submission. MTP/verification is not representable by these initial work items.

### Required invariants

1. Each live request appears once per step. Decode count, prefill request count,
   and total token count are different quantities; all sums/products and
   start+count checks precede payload slicing or allocation. Respect existing
   context/admission limits and actual arena/KV budgets, including dummy blocks.
2. Both ranks agree on logical request order, generations, token payload,
   requested chunk bounds and effective work order. Physical SSM/KV IDs may
   differ; bind them locally and validate guards/pointers. Do not require
   bitwise equality of rank-local weights, buffers, or physical block IDs.
3. Preserve temporal order within each prefill. Maintain GLM raw-tail writes,
   four-token pooled-index completion and latent KV writes for each request.
   Token count is not pooled-entry count. No cross-request pooling or index
   maintenance borrowed from an independently decoded row.
4. Prefix lookup is a preparation subphase, not a guessed scheduler fact.
   Retain the existing rank agreement on matched lengths and SSM restoration;
   finalize effective compute spans only afterward. Cold/no-prefix traffic
   is the first hardware scope. Do not remove rank-local prefix-sync collectives.
5. One default compute stream and explicit substep order initially. Decode
   results must be consumed before prefill clearing/writes. If graph decode is
   reused, refresh all metadata again before the next decode; a prefill clear
   invalidates any assumption that the previous slot table is still resident.
6. Preserve the full per-layer collective sequence and sizes on both ranks:
   TP projections, EP routed reduction, shared-expert placement, normalization
   and cache-maintenance collectives. No rank-local fallback or error recovery
   after its peer has begun the next collective. A kernel/transport failure
   after state mutation requires coordinated teardown, not transparent retry.
7. No new SSM slots, persistent arenas, KV overcommit, or memory utilization
   increase. Keep the campaign's ≥4 GiB measured post-load/warmup headroom on
   each node for any eventual full-model test. The pure plan allocates no GPU
   memory and does not imply a lower reserve.

## v1/v2 compatibility and future framing

First ship codec/planner tests and optional diagnostic comparison without
changing transmitted bytes. Golden fixtures must retain v1's singleton command
stream and v2's sequence preamble, decode-batch payload, prefill payload,
allocation, verification and shutdown commands exactly. The initial plan can
lower into those existing substeps; it must not claim generation protection on
the wire while generations remain host-only.

A new framed plan is a **separate negotiated protocol revision**, not extra
words appended to `0xFFFFFFE0`, and not an opcode trial sent to an old worker.
Unknown values can enter legacy token decoding; sending a new command before
peer capability agreement risks the historical mismatched-collective hang.
Prefer an explicit `PlannedV3` dialect while retaining `LegacyV1/LegacyV2`.
Both processes must select/validate the same dialect before entering the worker
command loop. Bootstrap capability exchange must itself be supported by both
versions; do not add a one-sided NCCL handshake to old startup code. Initially
require matching upgraded binaries/session configuration; mismatch fails startup.

Future frame: fixed magic/revision/session/step ID, bounded payload byte count,
request/token counts, capability/geometry signature, then canonical little-endian
work descriptors and token payload. No Rust struct transmute, native `usize`,
raw pointers, unrestricted deserialization, or allocation from an unvalidated
length. Start with a 1 MiB absolute control-frame ceiling and tighter profile
limits; reject trailing/truncated data, unknown enums and reserved-bit changes.
Cache the full prompt only after a separate acknowledged registration/revision
contract; do not stop sending legacy full prompts merely because chunks are small.

Validation rendezvous has a fixed, testable collective schedule. Both ranks
finish bounded frame parsing and local validation, exchange status and canonical
plan/signature agreement, then either both execute or neither executes. A local
validation error participates in the status rendezvous instead of returning
while its peer enters compute. An oversized header is rejected through the
agreed abort path before payload allocation/transfer. A disconnected peer needs
the existing coordinated transport teardown; this is not an atomic rollback
protocol. Record completion only after all planned state updates succeed;
never replay a partially executed plan under the same step ID.

## Tests-first gates and staged rollout

**A — CPU-only contract/codec (partially complete).** Pure contract and legacy
codec are implemented in the commits above; new-frame encoding, live-state
binding and replay/generation authority below remain future work. No scheduler
route changes. Test exact v1/v2
golden bytes and pure lowering order; new-frame encode/decode round trips;
unknown/truncated/oversized fields, checked overflow, invalid token IDs, duplicate
requests, zero-length chunks, bad computed positions, stale generations and
repeated step IDs. Property-test canonicalization and bounded parser allocation.
Unsupported profiles produce `LegacySerial` or an explicit pre-submission
rejection, never a partially accepted mixed plan.

**B — Two-rank CPU interpreter.** Use intentionally different local SSM/KV
indices and permuted logical request slots. Compare logical work order, effective
positions, result-consumption boundaries and every collective kind/count/dtype.
Inject one-rank missing state, FP16 tags, capacity exhaustion, absent kernels,
prefix-hit disagreement, generation reuse/cancellation and malformed payloads.
Require both ranks to reject before model mutation; prove no unmatched next
collective. Include request completion between plan construction and binding.
Freeze reservation/generation changes while the validated plan is in flight.

**C — Existing-kernel sequential mixed scheduling.** Only after A/B pass,
root runs one short prefill (initial slice ≤64 tokens) alongside 1–3 independent
decoders, total resident requests ≤4, nonspeculative text-only TP2/EP2, BF16 KV,
FP32 KDA, context ≤2048. Exclude LoRA, vision, prompt-logprobs, warm-prefix reuse
and swapping initially. Sample/consume decode results before prefill instead
of returning two unprotected pointers. Preserve worker normalization exactly.
No packed GLM kernel dispatch yet. Compare with the existing separate scheduler
steps for quality, consumed token counts, per-request progress and drain/reuse.

**D — Framed rank-agreed submission.** Upgrade both peers, exercise bounded
transport-only round trips first, then run the same sequential executor. Measure
control overhead separately; use an old-image rollback for both ranks. Never
mix upgraded and legacy workers in a live model session.

**E — Packed GPU work, separate design and oracle.** Only now consider a fused
layer plan or ragged prefill kernel. Explicitly partition mHC streams, logits,
attention/index scratch and KDA temporal spans; the current indexed decode
pair is eligible only for its one-token-per-request slice. Derive per-layer
collective signatures before submission and gate full-state numerical equality
against sequential execution. Graphs remain pure-decode only until mixed graph
keys, shape bounds and metadata freshness have their own proof.

Later context expansion tests must include chunk seams 3/4/5 and 15/16/17,
2047/2048/2049 sparse transitions, unequal histories, C4→C3→C2→C1 drains,
cancellation/reset/reuse and warm-prefix restoration. Initial absence of those
features is a typed capability restriction, not a claim that generic mixed
paths already support them.

Report TTFT, per-stream inter-token p50/p95/p99, completed throughput, actual
decode width, prefill tokens per step and per-node memory. Use identical arrival
traces and prompt/output lengths. Sequential fairness or reduced repeated prompt
transfer may help latency; only later packed execution can claim weight-sharing.
No speedup follows merely from introducing this plan.
