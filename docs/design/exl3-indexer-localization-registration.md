# EXL3 reference indexer non-determinism localization — registration

Registered 2026-10-15. This is a reference-only **diagnostic** that follows
the [fused-MoE determinism result](exl3-fused-moe-determinism-result-20261015.md).
With `EXL3_FUSED_MOE=0`, every row below 2,052 tokens was byte-identical
across processes, but rows at or beyond 2,052 tokens were not. That
threshold is exactly where the indexer's pool top-k
(`index_topk / index_kpool` = 512 pools of 4, with the incomplete tail always
kept) must first drop a pool. This run localizes the second source. It is
not a gate amendment, and no sample from it is a qualification draw.

## Attempt list (frozen)

Fixed order, fresh processes. Every attempt is reported. No attempt is
added, replaced or retried unchanged, and a failed attempt is retained.

    idx01, idx02, idx03

## Profile (frozen)

- The gate-v2 campaign reference create commands with the deterministic
  routed path, `EXL3_FUSED_MOE=0`, so the fused-MoE noise cannot mask the
  second source.
- Observer: the L1-tooling **context-v5** package
  (`q2-l1-tooling-20261013/ref-overlay`). It is the campaign's v4 observer
  plus two environment-gated additions: the MLA attention trace
  (`reference_attn_trace.py`, which records the output of every `self_attn`
  submodule that runs, including the indexer's top-k indices) and
  multi-layer KDA dumps (unused here). The chunk and FFN traces are
  byte-identical to v4. The package is deployed read-only to
  `/home/colum/idx-20261015/observer` and verified by hash.
- Dumps on the 11 MLA layers only (3, 7, …, 43), for the attention trace and
  the FFN trace. No mHC dumps. Environment:
  `ATLAS_EXL3_FFN_DUMP_DIR=/tmp/ffn-capture`, `ATLAS_EXL3_FFN_DUMP_LAYERS`
  and `ATLAS_EXL3_ATTN_DUMP_LAYERS` set to the MLA layers.
- Collection: only the campaign's `index-boundary-2051` long case (file
  `driver/cases-long-2051.json`, copied verbatim from the campaign case set),
  in the campaign long-collector procedure. Its four teacher-forced rows have
  2,051 tokens (an **in-run control**: expected identical, since selection
  is trivial), then 2,052, 2,053 and 2,054 tokens (expected to differ). Each
  row is a fresh 17-chunk prefill at chunk 128.
- Caches: an isolated copy of the probe's frozen snapshot per attempt, which
  must hash equal before start; fixed mount sources frozen before `idx01`;
  at least 20 GiB free per rank (expected dumps are about 13.5 GiB per rank
  per attempt). Dumps stay on the Sparks with SHA256 lists archived.
- Runner: `tests/exl3_validation/ref_experiment_runner.py idx`. Provenance
  comes from the gate-v2 finalizer variant, with the campaign draft plus the
  environment delta and the v5 hook hashes.

## Analysis (frozen before collection)

`tests/exl3_validation/hsl_localize.py`, chunk-aware: tensors are keyed by
(prefill chunk `computed_before`, layer, stage) and ordered chunk-major. In
each MLA layer the attention stages run in dataflow order:
1. input hidden states;
2. QKV-A, Q-B and the indexer's Wq-B and Wk/weights projections;
3. the indexer op's top-k indices;
4. sparse MLA attention;
5. o_proj input and output;
6. the attention output.

Then the layer's MoE stages follow. For each pair and each (row, rank) the
analyzer reports the first differing tensor. The analyzer was
regression-checked to reproduce the HSL result (`routed-weighted-f32` at L3
for all 16 requests of hsl01/hsl02).

Interpretation, registered now:
- If the indexer **top-k indices** are the first difference, with every
  indexer input projection identical in that chunk and in all earlier
  chunks, the variation originates inside the indexer op: pool scoring or
  top-k selection. Splitting those two needs a separately registered capture
  of the op's internal scores.
- If the first difference is the sparse-attention output with identical
  top-k indices, the sparse attention kernel varies.
- If the first difference is an input projection, the source lies upstream.
  Report it as found.
- The 2,051-token control row must be identical across all three attempts.
  If it is not, report that the dump profile itself introduced variation.

## Execution record

- idx01, idx02, idx03 **complete** (2026-10-15 03:08–03:5xZ; both ranks exit
  0 each time; 13.9 GB of dumps per rank per attempt).
- Result: the first difference is the indexer top-k index buffer (or, on the
  other rank, the reduced `o_proj`) in the final prefill chunk. The selected
  set is always identical; only the order is permuted. The 2,051-token
  control row is identical throughout. See the
  [result](exl3-indexer-localization-result-20261015.md). An analyzer
  stage-label correction (the MLA wrapper output follows the reduced
  `o_proj`) was applied before this record; the first pass is kept as
  `localization/*-v1-stagelabel.json`.
