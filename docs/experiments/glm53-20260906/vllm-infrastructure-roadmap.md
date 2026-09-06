# GLM infrastructure roadmap using vLLM as a reference

User direction, 2026-09-06: build missing execution infrastructure in logical,
tested stages; do not substitute flag sweeps for implementation.

Reference: official vLLM revision
[`6865e67f0be02d53694517f6f71d7fb96492792d`](https://github.com/vllm-project/vllm/commit/6865e67f0be02d53694517f6f71d7fb96492792d),
resolved from GitHub on 2026-09-06. The local `dcp-audit/vllm` checkout is a
June revision and predates this GLM support. Reference architecture is not a
claim that every upstream backend works on SM121 with TP2/EP2. No vLLM source
has been copied into Atlas by this campaign.

Progress: the first milestone's independent-row KDA execution and the second
milestone's existing-layout typed cache contract are implemented and validated
in [phase 4](phase4-infrastructure-results.md). Single-latent ownership and
bounded tails remain unimplemented. The next bounded slice is the
[EP execution-plan contract](ep-mixed-execution-plan.md), not a flag sweep.

## First milestone: state-indexed KDA execution

vLLM submits decode convolution and recurrence with device-side state indices
for the whole batch. At the start of this milestone Atlas batched projections
but launched those stateful updates separately for each independent row.
[Upstream KDA orchestration](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/kda.py),
[upstream recurrence](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py).

Selected implementation step: prototype one indexed convolution and one indexed
recurrence using **existing Atlas arithmetic**, not the temporal MTP verifier.
Keep FP32 recurrent state and current projection formats. Use explicit row
strides, state-slot indices, and checked pool capacity. Reject duplicate live
slots before launching; inactive rows cannot mutate state. The first prototype
lives under `scripts/dev/`, outside production dispatch.

Gate complete outputs, complete recurrent matrices and convolution histories
against the current per-row kernels, not just generated text. Exercise N1–N4,
noncontiguous/permuted slots, C4→C3→C2→C1 drains, slot reset/reuse and fixed-address
graph replay with changing metadata. Hard prototype allocation cap: 64 MiB.
Only after GPU oracle/memory checks pass should the existing KDA multi-sequence
execution path adopt it. No extra batch-width tuning switch is the objective.

## Second milestone: explicit latent/index cache ownership

vLLM's MLA cache specification represents one latent vector, and GLM's raw
pooling tail is bounded per request. Atlas currently stores identical NoPE512
latent K/V separately and retains raw index keys/gates across cached tokens.
[MLA cache specification](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/kv_cache_interface.py),
[GLM cache roles](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/attention.py).

Start with a typed GLM cache plan: latent width512, no RoPE, storage dtype,
token block size, pooling ratio4, pooled-key width128, physical capacity and
ownership. Derive kernel binding, allocation and byte accounting from that
contract. The recently fixed HDIM576/512 mismatch demonstrates why strings and
runtime strides alone are insufficient.

Then implement single-latent ownership and a bounded per-request index tail as
separate changes. Do not merely alias allocations or shrink buffers. Cover
allocation/free, copying, rollback, prefix reuse, chunked prefill seeding,
pool-completion boundaries and request retirement. At four full16K sessions,
byte accounting suggests roughly704 MiB/rank of duplicated latent storage and
352 MiB/rank of raw-tail storage could be avoided. These are live-capacity
calculations, not measured allocation recovery or throughput gains.

The current eager long-C4 work establishes an explicit guarded lane using the
existing ownership model first. It does not claim to implement this redesign.

## Third milestone: explicit batch/worker execution plans

vLLM describes scheduled tokens and request updates in `SchedulerOutput`; its
scheduler budgets tokens per iteration across work types. Atlas already has
batched/mixed-prefill machinery, but its relevant dispatch explicitly excludes
EP while the worker protocol remains missing for that path.
[Scheduler](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/core/sched/scheduler.py),
[execution-plan data](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/core/sched/output.py).

Build on Atlas's existing scheduler/traits rather than replacing them. Specify
ordered request identities, SSM slots, token spans, positions, block tables,
work type, and persistent-state generation in a validated execution plan.
Distinguish independent decode rows from speculative time steps explicitly.
First make metadata/worker round trips and capability rejection testable; then
add a small EP-compatible packed prefill or mixed step. Never enable a generic
mixed flag while the worker/state contract is absent.

For semantic indexing, distinguish token length from pooled-entry length, mask
inactive rows, and maintain within-request update order. Test pooling seams,
2048 sparse transitions, cancellation/reuse and rank agreement before batching
index maintenance or permitting ragged speculative layouts.
[Upstream indexer metadata](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/model_executor/layers/sparse_attn_indexer_kpool.py).

## Later: general grouped MoE contract

The working Atlas grouped route/sort/prequant/shared-after-EP pipeline should
eventually be driven by validated row count and backend capabilities, not
separate C3/C4/verifier branches. Retain explicit quantization semantics and
checkpoint topK8. vLLM's factory is an architectural reference, not a drop-in
kernel: several SM12x expert backends reject EP, and automatic backend selection
has SM121-specific restrictions.
[GLM MoE assembly](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/model.py),
[NVFP4 backend selection](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py).

Preserve the rejected compact-down result: its historical stall/regression is
unexplained, and no worklist hypothesis establishes a fix. Do not revive it as
part of this refactor. Every stage retains bounded memory, standalone numerical
gates, independent-request checks and matched full-model measurements.
