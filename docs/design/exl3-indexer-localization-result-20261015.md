# EXL3 reference indexer non-determinism localization — result (2026-10-15)

Result of the [registered diagnostic](exl3-indexer-localization-registration.md).
This is reference-only, not qualification; the gate-v2 campaign stays
UNVERIFIABLE. Evidence root: `$EV/indexer-localization-20261015/`. Dumps are
retained on the Sparks at `/home/colum/idx-20261015/attempts/<a>/dump`
(13.9 GB per rank per attempt), with SHA256 lists archived locally.

## Attempts

idx01, idx02 and idx03 all completed, both ranks exiting 0 each time, on the
loop routed path (`EXL3_FUSED_MOE=0`; no `routed-weighted-f32` stage appears,
confirming the fused kernel did not run). The dumps reproduce the
determinism experiment's pattern in the final logits: the 2,051-token
control row is byte-identical across all three, and each 2,052–2,054-token
row has 3 byte classes.

## Finding: the indexer top-k emits the right set in a run-dependent order

Localization covers 3 pairs × 4 rows × 2 ranks, all self-checks passing
(`localization/localize-<pair>-rank<r>.json`, `hsl_localize.py`, chunk-aware):

- **Control row** (2,051 tokens): identical in all 6 pair × rank comparisons,
  across 3,927 tensors each.
- **Every differing (row, rank)** first differs in the **final prefill
  chunk** (`computed_before = 2048`, the only chunk whose queries see more
  than 2,051 keys), at MLA layer 3 or 7:
  - **14 of 18**: the first differing tensor is the indexer op's top-k
    index buffer (`attn-indexer_indexer_op`). It is rank-local, and every
    indexer input (hidden states, QKV-A, Q-B, indexer Wq-B, Wk/weights) is
    byte-identical in that chunk and in all 16 earlier chunks.
  - **4 of 18**: the first differing tensor is the reduced `o_proj` output on
    the other rank. Its own indices and local attention match; the
    difference arrives through the TP reduce. Every differing row has at
    least one rank whose first difference is the index buffer.
- **Same set, different order.** Comparing the index buffers token by token
  (`localization/topk-order-*.json`, `idx_topk_order.py`): of 216 valid
  token rows, 38 differ in bytes, and **none** of the 38 selects a different
  key set. Only the order differs, with up to 2,044 of 2,044 positions
  permuted. Every differing row has 2,044–2,046 valid indices, the regime
  where a pool has been dropped (513+ complete pools of 4, select 512, plus
  the tail). No row with trivial selection ever differs.
- Downstream, sparse MLA attention accumulates over the indices in buffer
  order, so a permuted order changes the FP32 result. That difference then
  propagates through the reduce and the residual stream.

## Kernel-level confirmation (prior oracle, synthetic inputs)

`$EV/arithmetic-audit/reference-topk-repeat32.json`, which ran
`reference_topk_oracle.py` on the installed `vllm._C_stable_libtorch`
(`top_k_per_row_prefill`, 513 pools, select 512), repeated 32 times within
one process with identical inputs:

| scores | distinct output orders | distinct selected sets |
|---|---|---|
| ascending, distinct | 3 | 1 |
| descending, distinct | 7 | 1 |
| modulo-7 ties | 4 | 2 |
| all ties | 13 | 4 |

With distinct scores the kernel returns a fixed set in a non-deterministic
order, exactly what the model runs show. With exact ties, even the
membership varies. The prefill path calls it at
`sparse_attn_indexer_kpool.py:559` (`torch.ops._C.top_k_per_row_prefill`),
before pool expansion and the tail append.

## Summary of the reference's non-determinism (all three experiments)

Both sources are the same failure mode: a correct result accumulated in a
run-dependent order.
1. **Routed MoE (all lengths):** the fused EXL3 kernel atomically adds the
   top-8 expert contributions into FP32
   ([HSL result](exl3-hidden-state-localization-result-20261015.md),
   [determinism result](exl3-fused-moe-determinism-result-20261015.md)).
2. **Sparse attention (above 2,051 tokens):** the indexer's top-k returns
   its selected pools in a run-dependent order, and sparse attention
   accumulates in that order.

Removing (1) with `EXL3_FUSED_MOE=0` makes every row of up to 2,051 tokens
deterministic. A fixed order for (2), for example sorting the selected pool
IDs before expansion, is the evident candidate for the remaining rows. It is
untested here.
