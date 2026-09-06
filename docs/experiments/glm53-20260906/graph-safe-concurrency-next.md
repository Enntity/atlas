# GLM graph-safe independent concurrency: proposed next step

Status: design only, 2026-09-06. This document does not describe an implemented
or GPU-validated graph path. The current `ATLAS_GLM_MULTI_SEQ_SPARSE=1` path is
deliberately eager. Scope is independent, non-speculative C2/C3 on the existing
BF16 KV/index geometry. C1 remains eager, including C3 -> C2 -> C1 drain. No
MTP threshold relaxation, FP8 cache change, persistent GPU allocation, or new
concurrency limit is proposed.

## Why graphs are currently unsafe

`crates/spark-model/src/layers/qwen3_attention/decode/glm_index.rs`, method
`glm_index_decode_update_and_select`, embeds the host token position in the
scorer/top-k arguments, derives the scorer width from that position, and takes
a host dense/sparse branch at sequence length 2048. Capturing this chain once
would freeze those values, even if its first replay occurred below 2048.

The multi-sequence caller in
`layers/qwen3_attention/trait_impl/multi_seq/mla_glm_sparse.rs` constructs
row-private metadata and consumes each selector's scratch before advancing to
the next row. Its all-row Q absorption and main KV writes precede all selectors;
V extraction follows all selectors. Preserve that ordering.

`model/trait_impl/decode_a2.rs` currently disables graph lookup for this flag.
`model/trait_impl/decode_a.rs` also keeps the C1 drain eager. These guards must
not be removed until the replacement device-driven chain is validated.

## Smallest exact-dense-preserving design

Add an explicit, default-off graph opt-in under the existing sparse flag. Keep
the eager implementation available as the A/B oracle. First implementation
retains the scalar eight-pool decode scorer, not the new row8 WMMA prefill
scorer; unrelated numerical/kernel changes would obscure graph validation.

Three new CUDA entry points suffice:

1. **Device-length scorer.** Refactor the existing
   `glm_index_logits_bf16` implementation in
   `kernels/gb10/common/glm_indexer.cu` into a shared device implementation,
   preserving its original ABI and arithmetic for existing callers. A new
   single-row wrapper reads `meta_i.seq_len[0]` instead of taking host
   `seq_len_start`. Launch a fixed, capacity-bounded grid. Return before any
   paged-key read for short histories or pools outside `L / 4`; mask unused
   score entries to negative infinity. No future/dummy block-table access.
2. **Device-length top-k.** Similarly preserve the old
   `glm_index_topk_expand` ABI and share its exact radix selection/expansion
   body with a new single-row wrapper. Read device length L and write one
   scratch `u32 dense_len = L <= 2048 ? L : 0`. For short histories, initialize
   selected IDs to -1 and return without reading logits or running radix
   selection. For long histories, preserve 512 selected four-token pools and
   the existing unfinished tail at offsets 2048..2050. All output entries
   must be initialized on every execution, including slot reuse.
3. **Device-guarded sparse attention.** Preserve the existing
   `glm_sparse_mla_prefill_bf16` entry point and factor its body so a new
   wrapper reads device L and returns uniformly for L <= 2048. For longer
   histories use the same 2051-wide sparse attention body, arithmetic, and
   launch geometry as the current single-head decode path.

Capture the existing BF16 dense attention call too, passing the scratch
`dense_len` pointer instead of the original sequence-length pointer. The
existing `paged_decode_attn` in `kernels/gb10/common/paged_decode_attn.cu`
already returns before work for `seq_len == 0`. Thus one captured chain can
execute either old dense attention or old sparse attention without modifying
the common dense kernel or computing both attentions. The inactive node does
not write output. Guards preceding barriers must be uniform over the entire
CTA; scorer warp exits must retain the existing synchronization contract.

The two index query/weight GEMVs can run unconditionally in this first graph
version. Their shapes and addresses are fixed. This adds work to short rows,
but avoids introducing another gated GEMV implementation; measure the cost
before pursuing conditional projections. Index tail write/finalization always
runs, including short histories, so crossing 2048 preserves semantic history.

### Rust integration points

- Add the new launch wrappers beside existing functions in
  `layers/ops/glm_indexer.rs`, with pure checked-geometry tests. Keep the old
  wrappers and standalone harness ABIs intact.
- Add narrowly scoped kernel handles/probes in
  `layers/qwen3_attention/types.rs` and `init.rs`; register the new symbols in
  the GLM build path. Avoid changing non-GLM dispatch.
- Add a separate device-driven selector helper beside `decode/glm_index.rs`.
  It returns fixed scratch pointers, not a host `Option` dependent on length.
- In `trait_impl/multi_seq/mla_glm_sparse.rs`, retain all existing scratch
  liveness ordering and row-private pointer offsets, selecting the new chain
  only under the graph opt-in. Validate fixed-capacity scratch and metadata.
- In `model/trait_impl/decode_a2.rs`, permit graph lookup only for the new
  validated device-driven mode. Preserve profiling/LoRA opt-outs, exact EP
  widths, non-speculative checks, and existing graph capture/replay handling.
  Per-step host range checks belong before replay, not solely in layer code
  that runs only during capture. Keep `decode_a.rs` C1 eager guard unchanged.

## Fixed bounds and scratch ownership

Use the process-fixed block-table row capacity, not current history and not
the checkpoint's million-token nominal model limit:

```
capacity_tokens = checked(meta.max_blocks_per_seq * kv_block_size)
logits_stride   = capacity_tokens / 4
score grid     = (ceil(logits_stride / 8), 1, 1), block 256
top-k grid     = (1, 1, 1), block 256
attention grid = (local_query_heads, 1, 1), block 256
```

Require positive block size divisible by four; checked arithmetic; device L
within the fixed capacity; and the server's existing per-request length cap.
`model/impl_b1.rs::upload_batch_metadata_fixed` already uploads current token
positions, L = position + 1, physical write slots and complete fixed-stride
block tables before graph replay. No new host-to-device transfer is needed.
`model/impl_a1.rs` derives block-table capacity as `max_seq_len / block_size + 1`.
At a 16K cap and block size 16, that means 1025 blocks, 4100 FP32 scores,
16,400 score bytes, and 513 scorer CTAs per row. Inactive CTAs must exit
without key-cache reads. Do not size the grid to the much larger KV pool's
aggregate capacity.

Rows remain sequential and reuse `expert_down_out` for scores and `qkv_output`
for selected IDs. Reserve the extra four-byte dense length immediately after
2051 i32 IDs: require at least 8208 bytes in `qkv_output`, also retaining the
existing all-row KV assembly size requirement. That scalar lives only from
top-k through this row's attention. The next row can overwrite it in stream
order. No new persistent allocation is necessary. Validate every borrowed
arena against the fixed maximum, not the length observed during capture.

## Graph keys and distributed ordering

The existing ordered SSM-slot vector in
`model/trait_impl/decode_graph_key.rs` already encodes all per-row recurrent
state addresses and exact C2/C3 width. Fixed scratch/metadata addresses and
device-read lengths leave no new per-step host value to put in the key. Slot
reuse can retain the graph because KV tables and metadata are refreshed.

Do not key only by batch size; `[0, 2]` differs from `[0, 1]`, and `[2, 0]`
differs from `[0, 2]`. No borrowing C3 graphs for EP C2 drains: the existing
EP path already disallows graph borrowing. If future work adds history buckets,
bucket bounds must join the key and agree across ranks; avoid buckets in this
first patch. Both ranks must use identical configuration and preserve all
existing collective submission counts/order. Device guards may skip local
attention work, never rank-dependent collectives.

## Can fixed-top-k sparse attention replace dense below 2048?

It can represent exactly the same **token set**: fill IDs 0..L-1 and mask the
remaining positions through width 2051. Even the old top-k expansion has enough
capacity for all complete pools plus the tail when L <= 2048, although its
atomic output order is not stable and a direct causal fill is simpler.

It is **not a bit-exact replacement for existing dense attention**. The dense
kernel partitions history between eight warps, batches four positions, uses
vectorized contiguous dot products, `__expf`, and an inter-warp reduction.
Sparse decode uses eight-token tiles, strided dot-product lanes, `expf`, and a
different online-softmax update order. BF16 outputs can differ even with
identical causal token support. A unified sparse path would need a separate
numerical/quality acceptance decision and could be slower for short histories
because of its fixed-width masked work. Preserve runtime-guarded dense in this
graph-only step.

## Validation gates

Before model testing, extend a standalone small-allocation CUDA harness:

- Eager old chain versus captured new chain, replaying the same graph while
  device L changes through 1, 3, 4, 15, 16, 2047, 2048, 2049, 2050, 2051,
  2052, and the bounded maximum. Exercise all four pool phases and page edges.
- Mixed C3 rows below/at/above 2048 and unequal long histories; shuffled
  physical pages; slot/table changes with fixed pointer addresses. Poison
  masked score regions and unused output entries to expose stale reads.
- Exact dense outputs against the existing dense kernel; long score arithmetic
  against the scalar scorer; correct top-512 pool sets and complete tail.
  Account for pre-existing atomic ordering/tie freedom when comparing IDs;
  do not incorrectly demand identical index arrays in tied cases.
- Verify exactly one attention path writes each output, inactive branches do
  not read invalid pages, and tail/pool cache updates match the eager chain.
- Pure host tests for capacity arithmetic, insufficient arenas, zero/overflow
  bounds, row-private offsets and slot-vector separation. Confirm no launch
  geometry depends on current device length.

Then root-controlled hardware validation with unchanged 16K caps and memory
guards: eager-versus-graph A/B; unequal 3x-long independent needles; mixed
threshold crossings; C3 -> C2 `[0, 2]` -> C1 drain; new occupants in freed
slots; short-context correctness and latency regression checks. Disable phase
profilers for throughput measurements. Keep the opt-in off if any correctness,
communication, memory-headroom or capture-stability gate fails.

## Expected value and risk

Graphs remove repeated host submission overhead, not index scoring, radix
top-k or sparse attention work. Long C3 still performs eleven layers times
three sequential selectors; fixed maximum scorer grids add empty CTAs for
short histories, and unconditional query projections add short-row work.
No percentage gain is claimed without a same-image A/B. Existing non-GLM
graph receipts in `decode_graph_key.rs` show only roughly 3% end-to-end, so
large GLM gains should not be assumed from launch counts alone.

This is a bounded several-file kernel/API/dispatch change, followed by a
mandatory capture/replay correctness campaign, not a safe one-line flag
change. Main risks are stale captured lengths, uninitialized dense-length or
index scratch, capacity-based out-of-bounds reads, early returns around
barriers, and rank-divergent capture/collective order. Keeping C1 eager and
retaining the dense arithmetic limits the first implementation's scope.
