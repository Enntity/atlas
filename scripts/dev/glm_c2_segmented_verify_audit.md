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
