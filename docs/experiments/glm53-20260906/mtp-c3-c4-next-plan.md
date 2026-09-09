# Next concurrency increment: four retained MTP owners

2026-09-09 source audit. Proposed implementation, not activated or measured.
The qualified selected path still has two request slots. Nonspeculative C1..C8
support does not provide speculative accepted-prefix or rollback semantics.

Owner-storage prerequisite is now implemented and CPU-qualified: one checked
capacity2..4 derives the private cache/index reserves, fixed-length owner table
and whole hidden-slab bounds. Actual constructor/lease checks cover three/four
owners, fifth-owner refusal, slot3 retirement/reuse, failed-retirement quarantine
and an overlap visible only in the larger slab. Existing capacity/cleanup checks
and both original/shared-M10 all25 Model/worker continuation controls pass.
The constructor test first failed at the actual capacity-four refusal before
implementation. Logs: campaign `c4-owner-capacity-{red,green,controls}.log` and
`c4-owner-pair-controls.log`. No native serving gain is attributed to this slice.
Factory remains literal capacity2. Scheduler, recipe admission, target memory
accounting and large-context limits are not opened by the storage change.

Physical-group mapping is now implemented and CPU-qualified. Model capacity
checks require agreement between the actual private pool, target SSM slot pool
and decode levers. E6 binds its existing preamble and owner records to complete
physical groups[0,1] or[2,3]; packed rows and verdict arrays remain pair-local.
Only the addressed group writes private state, and one exclusive producer is
retained through both commits. The actual four-owner Model/worker test first
failed at the old two-slot wire restriction, then passed alternating groups,
independent0/4 acceptance, exact committed tokens, E1 repair/continuation, full
unselected private/target-byte preservation, and wrong-preamble refusal before
worker writers with globally terminal retention. Existing handoff controls,
including both original/shared-M10 all25 continuation checks, also pass.
Evidence: campaign `c4-model-group-{red,green,controls}.log`; these are host-side
ownership checks with recorded numerical kernels, not native C4 qualification.
The last native-qualified serving source remains290cf248 with two owners.

The inner selected scheduler now derives its owner map from that actual Model
capacity and visits every physical pair, rather than returning after group0/1.
Actual scheduler/Model/worker checks first failed at the old occupancy1..2
restriction, then passed C4 in reversed vector order, both groups' checked
selections/commits/E1, C3 with singleton2, and drain to physical survivor3
including final completion without another E1. Existing selected two-owner
scheduler controls pass. Evidence: `c4-scheduler-group-{red,green,controls}.log`.
These fixtures use local prefill and explicit Model cleanup; they do not prove
serving F0/F1, registration, retirement or native arithmetic. Outer serving
admission, registered worker startup and retirement remain capacity2 pending
the coordinated next integration below.

## Architectural reference

Current upstream vLLM batches request-indexed target queries and independently
tracks accepted counts. Its autoregressive proposer advances all requests at
each dependent draft step. These are useful implementation references, not
proof of the exact revision/workload behind MiaAI's published table:

- [Scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py).
- [GPU runner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py).
- [Autoregressive proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py).
- [MiaAI TP2 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark).

## First bounded milestone: residency and ownership

1. Thread one explicit capacity2..4 through selected admission, the actual MTP
   pool/private-cache constructor, model ownership and selected worker. The
   current `Pool.slots=[Slot;2]`, two full private KV reserves and98,304-byte
   hidden slab are real limits. Derive every reserve from admitted capacity.
2. Keep the existing two-owner compute workspace and one exclusive producer.
   Execute physical groups[0,1] and[2,3] serially. Missing, cold, stopped or
   draftless members use existing single-owner bootstrap/F5/E1 transactions.
   Preserve physical slots on drain; survivors[1,2] initially remain singles.
3. Separate pair-local row ordinal from physical owner identity throughout
   `paired_joint`, `paired_joint_verdict`, `paired_verdict`, model pair compute
   and E6 transport. A two-element verdict must never be indexed with slot3.
   Mark only actual group writers, not every retained pool slot. Preserve the
   global failure latch and all ownership retention after irreversible failure.
4. Keep the26-word E6 packet, but bind its existing owner records to actual
   physical slots and its EP-v2 preamble to the group base. Validate the full
   expected packet and admitted group before worker target writes. Complete
   detach, trim and commit before any next producer reuses shared scratch.
5. Update selected scheduling, retirement, worker shutdown, recipe validation
   and selected preflight together. Do not enable the nonspec C4 branch or
   weaken rank identity, digest pinning, quiescence/release or terminal handling.

Principal paths: `layers/glm5_mtp/{paired,new,paired_joint,paired_joint_verdict,
paired_verdict}.rs`; `model/glm_c2_{pair_verify,pair_transport,sequence_allocation,
sequence_ownership,transport,handoff}.rs`; selected scheduler/worker modules;
`atlas-glm-pair-wire/src/recipe_validate.rs`; serving preflight.

Memory must include each additional private cache/index reserve,49,152-byte
hidden segment, target KV and FP32 state/checkpoint/K5 snapshot ownership.
Existing EP-v2 gives full-width snapshot sizing, but its reserve accounting
still must be applied at the new capacity. No C4 MTP memory receipt exists yet.
Retain context2044/prefill1024 and current no-swap/headroom guards initially.

Concrete next integration boundaries (source audit, not enabled):

- Recipe `max_sequences` already participates in identity: admit2..4 without
  changing the two-rank transport/release arrays. Selected registration must
  compare recipe capacity with the actual Model capacity getter while armed.
- Factory must pass its validated capacity into the actual private-head
  constructor; the compatibility constructor remains two-owner.
- Selected worker retains exactly that many actual physical slots. Scheduler
  validates the whole active slice, then visits both physical groups without
  returning after the first successful pair. Cold/missing/stopped partners use
  the existing singleton path. Active-vector order never defines identity.
- Selected retirement visits actual physical slots in order, preserving
  free/F1/health/Done ordering and terminal retention. Shutdown retains all
  owners through the existing single rank-pair quiescence/release.
- Preflight must keep selected C4 out of the ordinary nonspeculative C4 branch;
  validate capacity before backend initialization, preserving all context,
  sampling, rank, cache and no-swap restrictions.

The private-head payload is currently not an explicit preflight line item.
At context2044 with the current pool4/index128 configuration, the existing
`GlmCachePlan` gives41,984 bytes per private block. Each owner has128 blocks
plus49,152 hidden bytes:5,423,104 bytes per owner, or21,692,416 bytes at C4.
This excludes allocator overhead and shared/loaded weights. Refactor one
private storage plan for constructor and public reserve quote, then add it
exactly once to selected `inference_reserve`. Existing post-load audit and
factory fresh-free-memory clamps carry that reserve into target-KV budgeting;
do not subtract the arena or private allowance twice, or consume the4GiB
headroom. Verify against actual allocation accounting before native admission.
The existing snapshot reserve functions remain authoritative for target state;
for MTP4 snapshot mode their live/verify term is `(6C+1)H+(7C+1)V`, where H/V
are aggregate TP-local H-state/conv bytes. Keep the separate prefix snapshot
and arena terms. A failed post-load memory query currently permits an estimate;
that estimate is not a measured headroom receipt.

Use existing real owner/worker fixtures: group2/3 preserving0/1 bytes, alternating
groups, singleton3, C3 occupancy, survivors1/2, retirement/reuse generations,
wrong group/preamble before target writes, independent0/4 acceptance and global
failure retention. Reuse intra-pair all25 coverage rather than inventing a625-case
four-owner acceptance suite. Then run guarded warm C1/C2/C3/C4 quality and timing.

## Performance milestone after residency

Serialized pairs enable C4 but may stay near C2 aggregate throughput: each pair
still rereads target weights and pays collectives. Do not claim scaling from
admission alone. Follow with one layer-major traversal across three/four owners:
retain independent K5 attention/state, then combine15/20 rows for shared/routed
FFN and reduction. Explicitly enlarge and validate row/worklist staging, saved
tails and producer lifetime; do not silently widen fixed-M10 validators.
Qualify exact arithmetic and bounded allocations before full-model activation.

The standalone arithmetic comparison is now native-qualified at15/20 rows:
exact checked outputs, zero memcheck errors/leaks and about2x faster than
M10 chunks (plus a literal K5 tail for C3), repeated in fresh processes.
See `owner-batch-ffn-native-results.md`. This supports the wider traversal but
does not implement it or establish any additional serving tok/s gain.

Batched draft generation is a separate subsequent optimization. The standalone
first-draft BF16 batch2 projection is exact and faster, but that alone does not
implement request-batched E1 proposal or establish a serving gain.

Detailed audit retained in the campaign's `c3-c4-mtp-next-plan.md`, SHA256
`e79fe34e0e983d02b4e1b476a5a0596e8a83970787879e5447ceb2db96e86246`.

## Required follow-up: fresh large-context qualification

User reminder, 2026-09-09: explicitly revisit large context after the bounded
C3/C4 MTP integration. Historical 10K/32K retrieval runs on older images are
not qualification of the current numerical paths or the new MTP serving path.

Before raising the current 2044-token MTP cap, implement and validate semantic
indexing beyond the 2048-token boundary for target verification and draft
continuation, plus context-dependent cache, metadata and rollback reserves.
Do not bypass existing context guards just to launch a benchmark.

Once those prerequisites hold, test increasing contexts (4K, 8K, 16K, then
32K where safe), starting at C1 and then C2/C3/C4 within an explicitly computed
per-rank budget. Each larger step requires measured headroom at the smaller
step, zero swap, bounded supervision and recoverable shutdown. Keep the user-
permitted reduced total context if larger allocations are unsafe.

For each qualified profile retain fresh-process, warmed uncached prefill rates,
TTFT, decode rates and output counts; coherence/coding spot checks, real tool
calls, and distinct per-request needles at early/middle/late positions with
foreign-needle detection. Include semantic-index threshold crossings and
unequal request lengths/drains. Report speculative and nonspeculative profiles
separately. No 128K/256K capability claim without direct safe qualification.
