# EXL3 reference hidden-state localization — result (2026-10-15)

Result of the [registered diagnostic](exl3-hidden-state-localization-registration.md).
This is reference-only evidence, not qualification; the gate-v2 campaign
stays UNVERIFIABLE. Evidence root:
`$EV/hidden-state-localization-20261015/` (`$EV=/home/colum/Workspace/atlas-exl3-validation-20261010`).
Dumps for `hsl02`/`hsl03` are retained on the Sparks at
`/home/colum/hsl-20261015/attempts/<a>/dump`, with SHA256 lists archived
locally. `hsl01`'s dumps are also copied locally.

## Attempts

| Attempt | Status | Notes |
|---|---|---|
| hsl01 | complete | both ranks exit 0; 49,847 dump files, 5.1 GiB per rank |
| hsl02 | complete | both ranks exit 0 |
| hsl03 | complete | both ranks exit 0 |

All three ran from copies of the probe's frozen cache snapshot. Descriptive
short16 H3 against the 15 campaign references passes all 28 items for each
draw (worst 0.10 of the limit, `h3-descriptive-short16.json`). The dumps did
not push draws outside the campaign envelope.

## Finding: the reference's run-to-run divergence starts in the EXL3 fused-MoE accumulation

Localization (`localization/localize-<pair>-rank<r>.json`, by
`tests/exl3_validation/hsl_localize.py`) covers 3 pairs × 16 requests × 2
ranks = 96 comparisons. Every self-check passed: the joins hold and the key
sets are equal.

1. **The first differing tensor is the same in all 96 cases:**
   `routed-weighted-f32`. That is the rank-local FP32 output of
   `apply_exl3_fused_moe`, the routed experts' weighted sum before the bf16
   cast and the tensor-parallel reduce. It is at layer 3, the first MoE
   layer, in 91 cases, and at layer 4 or 5 in 5. Everything earlier in
   dataflow order is byte-identical: all of layers 0–2, every mHC and
   attention boundary, and that layer's MoE input, router logits, selected
   expert IDs and routing weights, and shared-expert outputs.
2. **It recurs independently in nearly every MoE layer.** The fused FP32
   output differs in 36–42 of the 42 MoE layers per request, while those
   layers' inputs are still byte-identical (`localization/absorb-*.json`,
   `tests/exl3_validation/hsl_absorb.py`). At the first such layer, up to
   11% of elements differ, by at most 4.8×10⁻⁷ against values up to
   about 9: last-bit FP32 differences.
3. **The noise is absorbed until one rounding flip escapes.** The bf16 cast
   and the rank sum usually round both runs to the same bytes. The
   post-reduce MoE output first differs at a row-dependent layer between 6
   and 36, and never in 6 of 96 cases. The next layer's FFN input then
   differs in 80 of 86 cases one layer later (2–4 layers for the rest),
   after which the hidden states diverge. In 10 of 96 cases the FFN input
   never differs. No stage other than the post-reduce output and its mHC
   consumers differs before the first FFN-input difference. This explains
   the probe's observation that router gate inputs diverge first at a
   row-dependent layer (8–33) with routing flips following: routing ties
   are downstream consequences.

## Mechanism (source, consistent with the data)

The two routed paths that ran, read from the dumps'
`effective-routed-path` receipts (504 thin `exl3_moe` calls and 168
grouped fat calls), both accumulate each token's top-8 expert contributions
into one zero-initialized FP32 buffer with atomic adds:
- thin experts: `exl3_moe_kernel.cuh:249` calls `had_hf_r_128_d_inner`,
  "o += float(out_had(i)), atomic" (`hadamard_inner.cuh:415–473`);
- grouped fat experts: `exl3_fat_moe.cu:456`, `atomicAdd` on `float4`.

Expert work is handed out through atomic ticket scheduling
(`exl3_moe_kernel.cuh:264`). The order in which the eight FP32 terms reach
a given output element can therefore vary between runs, and FP32 addition is
not associative. That fits every observation: the tensor (the atomically
accumulated buffer), the magnitude (last bits), identical inputs, and
occurrence in every layer on both ranks. **Causality is not yet
demonstrated by an intervention.**

## Consequences

- The reference engine is intrinsically non-deterministic at this
  configuration. The campaign's per-row byte classes and its occasional
  excursions (v2ref14, v2ref15) are explained by this kind of drift
  surviving rounding at a row-dependent depth. No discrete runtime choice
  (autotune plan, NCCL algorithm, routing tie) is needed to explain them.
- A reference "pin" would need deterministic routed-expert accumulation.
  The installed plugin has a fixed-order alternative: `EXL3_FUSED_MOE=0`
  selects `apply_exl3_python_loop`, which sums experts in loop order with
  `index_add_`, one write per token per expert call. It also changes the
  arithmetic, so its numerics must be measured and it would define a new
  reference profile. It is not a neutral pin.
- Registered next step (separately): a one-variable `EXL3_FUSED_MOE=0`
  determinism experiment. Run fresh repeats and require byte identity of
  `routed-output` and final logits across processes, then measure its
  distance from the fused-path ensemble. Only after that could a
  reference-profile amendment be proposed, as a newly registered
  qualification campaign that retains the UNVERIFIABLE result.
- Atlas implication: the candidate is being compared with a reference that
  has this noise floor. The gate's reference-only calibration already
  measures that floor, but v2's H3/H4 rules treat rare escapes as modes;
  any amendment must account for this mechanism.

## Artifacts

- `localization/localize-<pair>-rank<r>.json`: first-difference results.
  `*-v1-orderbug.json` is the first pass, before `routed-weighted-f32` was
  placed ahead of `routed-output`, the order in which the hooks write them.
  It is kept for the record; the conclusion is unchanged.
- `localization/absorb-<pair>-rank<r>.json`: per-request magnitudes and
  absorption.
- `h3-descriptive-short16.json`: descriptive envelope check.
- `attempts/<a>/`: receipts, container logs, cache manifests, captures and
  dump hash lists.
