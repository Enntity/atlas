# EXL3 reference fused-MoE determinism — result (2026-10-15)

Result of the [registered one-variable test](exl3-fused-moe-determinism-registration.md)
(`EXL3_FUSED_MOE=1→0`). This is reference-only, not qualification; the
gate-v2 campaign stays UNVERIFIABLE. Evidence root:
`$EV/fused-moe-determinism-20261015/` (`determinism-rows.json`,
`secondary-descriptive.json`, `attempts/det0{1,2,3}/`).

All three attempts completed, both ranks exiting 0 each time, with caches
from the frozen snapshot. A first launch error (wrong working directory) ran
nothing and consumed no attempt (`runner-det01-launch-error-nothing-ran.log`).

## Primary criterion: FAIL, as registered

25 of 32 final-logit rows are byte-identical across det01, det02 and det03.
7 differ, each with 3 distinct byte classes and up to 1.66 maximum absolute
logit difference. As registered, the primary criterion fails: the loop
profile is **not** fully deterministic.

The 7 rows are exactly the long-suite rows with **2,052 or more** prefix
tokens:

| rows | prefix tokens | across det01–03 |
|---|---|---|
| all 16 short16 rows | 5–39 | identical |
| `index-boundary-2047` r0–r3, `-2048` r0–r3 | 2,047–2,051 | identical |
| `index-boundary-2051` r0 | 2,051 | identical |
| `index-boundary-2051` r1–r3 | 2,052–2,054 | 3 classes each |
| `index-boundary-2053` r0–r3 | 2,053–2,056 | 3 classes each |

## Two separate findings

**1. The fused-MoE attribution is strengthened, not refuted.** With the
fused kernel replaced, every row below 2,052 tokens becomes byte-identical
across processes. That includes all 16 short16 rows, where the hidden-state
localization found the first difference in the fused kernel's atomically
accumulated FP32 output in 96 of 96 comparisons. For fused draws, those same
rows were never all identical: the 15-draw campaign has 4–17 byte classes per
short16 row, and every long row is byte-unique. The one changed variable
removes the variation the localization attributed to it.

**2. A second, independent non-deterministic source starts at exactly
2,052 tokens.** The checkpoint's sparse-attention indexer selects
`index_topk / index_kpool` = 2048 / 4 = 512 key pools of 4 tokens, and always
keeps the incomplete tail pool (`index_kpool_always_select_tail: true`). Up
to 2,051 tokens there are at most 512 complete pools plus a tail, so every
key is selected and the selection is trivial. At 2,052 there are 513
complete pools and one must be dropped. Non-determinism appears at exactly
that point. The attribution to the indexer's pool top-k selection regime
rests on this exact threshold match. The specific kernel (score computation
or top-k selection) is **not localized**; that needs a separately
registered capture.

## Secondary (descriptive, no verdict)

- Descriptive H3 against the 15 fused-path campaign references: each det draw
  passes 52 of 56 items and fails the same 4 short16 items (`arithmetic` r0,
  r2 and `longer` r0, r1) at 1.19–1.93× the limit. The three draws are
  identical there.
- Per-row symmetric KL from det01 to its nearest campaign reference, against
  the campaign's own spread: median 0.28×, maximum 26.9×.

So the loop profile is not a neutral pin. Its separate gate/up/down GEMMs and
FP32 activation move short rows measurably outside the fused ensemble's
envelope.

## Consequences for qualification

- A fully deterministic reference at this profile needs both sources
  removed: the fused-MoE accumulation order, and the indexer selection
  beyond 2,051 tokens. The gate's long `index-boundary` cases sit
  precisely in the second regime.
- Atlas's own routed-MoE combine is already deterministic. Each expert writes
  an FP32 slot, and a fixed top-k-slot-order sum follows
  (`kernels/gb10/common/exl3_moe_precision.cu:132–176`). Atlas candidate
  repeats were byte-identical in the v2 campaign.
