# Next concurrency increment: four retained MTP owners

2026-09-09 source audit. Proposed implementation, not activated or measured.
The qualified selected path still has two request slots. Nonspeculative C1..C8
support does not provide speculative accepted-prefix or rollback semantics.

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

Batched draft generation is a separate subsequent optimization. The standalone
first-draft BF16 batch2 projection is exact and faster, but that alone does not
implement request-batched E1 proposal or establish a serving gain.

Detailed audit retained in the campaign's `c3-c4-mtp-next-plan.md`, SHA256
`e79fe34e0e983d02b4e1b476a5a0596e8a83970787879e5447ceb2db96e86246`.
