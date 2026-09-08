# GLM paired verification: source seams after request ownership

2026-09-08. Read-only design audit for Partition C, not authorization to bypass
Partitions A/B in `glm_c2_speculation_plan.md`. Atlas source at6165ce1a; the
separate unused MLA M10 export is qualified in f6df67b7. No C2 admission here.

## KDA: batch projections, preserve two temporal histories

The smallest new layer seam is GLM's implementation of
`TransformerLayer::decode_verify_multi`; its current inherited implementation
is unsupported. `Glm5KdaLayer::decode_batched` treats all rows as ONE temporal
sequence. Passing ten rows there is not paired verification.

Factor the stateful boundary in `glm5_kda.rs::forward_inner` around packed
QKV/convolution/recurrence. First reuse the existing K5 convolution and
`kda_recurrent_bf16_verify_snap` twice at row offsets0/5, with each owner's
actual SsmLayerState pointers. Shared stateless projections can cover ten rows;
history and rollback cannot cross that boundary. Indexed C2..4 kernels process
one token per owner and cannot represent five time steps by repeated slot IDs.

Per KDA layer/slot at local heads32/dim128/conv4: canonical H2,097,152 bytes,
convolution196,608 bytes. Actual K5 allocation requires four H intermediates
and the existing five-entry convolution allocation (GLM uses four snapshots).
H intermediates use tiered prefix offsets; convolution uses uniform slot
strides. Bind actual spans; never invent one common `slot*K` stride or allow
an uncovered slot's dummy pointer. Check both owners before any mutation.

Accepted drafts a0..3 select snapshot index a: the seed row is committed too.
a4 retains the canonical final state. `commit_accepted_prefix_dispatch` uses
the secondary stream; both owner commits must complete before another
recurrent consumer. Future fused stateful kernels must match all canonical
states, four snapshots per owner and BF16 outputs against these two calls.

The current `project_hot_verify` exact-M NVFP4 path stops at8; M10 falls into
prefill GEMM. Existing fused QKV, side projections and TP/mHC arms are M5-only.
The generic W4A16 family has a dynamic-M batch16 tier, but selecting it for
GLM M10 still needs exact output and timing qualification against two M5 calls.
Do not assume the wider tier is faster, or retain extra BF16 weights to imitate
another engine at the expense of Spark memory safety.

## MLA: per-row causal metadata is reusable; M10 dispatch is not present

`multi_seq/mla_glm.rs::ms_glm_mla_decode` supports2..5 rows. Its batched cache
write and dense BF16 paged-attention kernel already receive per-row slots,
causal lengths and block tables. For two K5 owners, rows0..4 reference the first
owner's table with lengths base0+1..base0+5; rows5..9 independently reference
the second table and base1. Writing all candidate KV first is safe only with
those per-row causal clamps. No missing physical block may fall back to0.

The actual GLM kernel resolves `paged_decode_attn_512`, not the inherited
576-dimension MLA kernel. Its row dimension indexes separate block-table and
length entries. This permits an initial reuse candidate, not a native proof.
Generic `verify_e` demonstrates flattened metadata but excludes EP and K5;
its Qwen-oriented eligibility cannot qualify GLM.

Dedicated MLA projection/FFN callers still reject or fall back above5 rows.
M10 Q absorption/V extraction now has an unused, standalone-qualified export;
the remaining projection paths, live scratch capacities, GLM mHC and grouped
MoE need explicit ten-row contracts. Sparse index update/selection is a
separate per-owner obligation; dense short-context attention does not justify
silently using another request's sparse metadata.

## Upstream reference, not a drop-in implementation

Pinned vLLM source1a522b69491f1d4f9199769bf44137e0f1de0912:

- [GLM KDA](https://github.com/vllm-project/vllm/blob/1a522b69491f1d4f9199769bf44137e0f1de0912/vllm/models/glm5next/nvidia/kda.py#L401-L486)
  supplies segmented query offsets, per-request state indices and accepted
  counts to convolution and recurrent execution. Its BF16 merged projection
  layout differs from Atlas's memory-tight quantized KDA weights.
- [Autoregressive proposer](https://github.com/vllm-project/vllm/blob/1a522b69491f1d4f9199769bf44137e0f1de0912/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py#L407-L447)
  selects each request's last hidden by last-token indices and retains it for
  subsequent drafting; it does not rely on one shared last-hidden row.

Minimum later integration matrix: owner order[0,1]/[1,0], different nonzero
histories, unequal prefix lengths across block boundaries, all25 acceptance
pairs then continuation, untouched inactive-owner sentinels, cancellation at
transaction boundaries, retirement/reuse, stale/aliased/undersized spans and
fixed-pointer graph replay. Start eager against serialized B before graphs.
No engine throughput estimate follows from this audit or the MLA microtest.

## Follow-up: actual C2 FFN topology and the M10 scheduling cliff

Source audit at `b66f1d34`, independently reviewed on 2026-09-08. These are
dispatch facts, not new GPU timing measurements.

Both `glm5_kda/multi_seq.rs` and
`qwen3_attention/trait_impl/multi_seq/mod.rs` retain per-row scalar MoE for
independent C2. C3 and C4 can use their grouped FFN arms. Thus C2 repeats the
router, routed gate/up/down, routed reduction, replicated shared-expert blend
and per-row mHC consumption. A shared output arena must be consumed before the
next scalar row overwrites it; changing the row loop alone is not batching.

In `moe/forward.rs`, each scalar GLM EP2 row reduces 8192 bytes of routed BF16
output and adds the replicated shared expert **after** reduction, exactly once.
C2 therefore issues two such reductions per MoE layer, compared with one
24576-byte grouped C3 or one 32768-byte grouped C4 reduction. C4's separate
scalar control issues four. These counts exclude attention/projection
collectives. `nccl_backend/comm_impl.rs::all_reduce_async` orders each reduction
with two event records and two stream waits; its configured two-rank fast path
uses send/receive plus a local BF16 add. These are GPU dependencies, not routine
host synchronization. Captured execution uses a different reduction entry.

The current hot path replicates routing and reduces rank-local expert outputs;
it is not the unused `forward_ep_dispatch` token-all-to-all scaffold. Likewise,
a TP2 vLLM recipe does not establish the same expert partition as Atlas EP2.
An earlier C2 batched FFN experiment was neutral on GB10
(`docs/glm53-dual-spark.md`, concurrent decode section). Do not simply re-enable
that older batch-two kernel and predict a gain; any new candidate must identify
which dispatch, weight reuse or collective cost actually changed.

For future paired verification, calling generic grouped prefill with ten rows
has another important condition in `moe/forward_prefill_routed.rs`:

- With unpublished B-tile storage, default NVFP4 exact-tile sizing and eager
  execution, its conservative bound is `ceil(rows * top_k / 64)`.
- K5 has 40 routes and one tile, so exact sizing does not read offsets back.
  M10 has 80 routes and two tiles, enabling a GPU-to-host read of all 289 expert
  offsets: 1156 bytes **per MoE layer**, with the associated stream-draining
  boundary. Setting a truncating load-factor cap is not a valid optimization.
- Published resident B-tile storage already bypasses this readback and uses
  `ceil(rows / 64)`: unique top-k routing visits each expert at most once per
  token. That resident path is not activated in the serving baseline and must
  not be silently assumed by the first paired verifier.
- Existing compact-worklist selection names K5/C3/C4 and excludes M10. A proper
  ten-row path needs explicit GPU-resident work planning and checked workspace
  capacity, not just a wider GEMM. Two K5 FFN reductions move 40960 bytes each;
  one paired reduction could move 81920 bytes, with the same per-owner results
  and the shared expert added only after the routed sum.

The smallest useful later profile covers one KDA and one MLA FFN from routing
through mHC: collective call counts/bytes, dense versus compact tiles, expert
offset readback counts/bytes, and actual eager/graph topology. Diagnostic
synchronization must not be enabled in qualifying throughput measurements.
This refines Partition C; it does not expand the current Gate 2 ownership work.
