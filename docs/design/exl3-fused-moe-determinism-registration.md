# EXL3 reference fused-MoE determinism — registration

Registered 2026-10-15. This is a reference-only, one-variable **diagnostic**
that follows the
[hidden-state localization result](exl3-hidden-state-localization-result-20261015.md).
That result found the reference's first run-to-run difference in the
atomically accumulated FP32 output of `apply_exl3_fused_moe`, in all 96
comparisons. This experiment tests that attribution by intervention.

It is not a gate amendment and changes no H3/H4 constant, component count,
fixed campaign N or reference identity. No sample from it is ever a
qualification draw. The fixed v2 campaign stays UNVERIFIABLE.

## Attempt list (frozen)

Fixed order, fresh processes. Every attempt is reported. No attempt is
added, replaced or retried unchanged, and a failed attempt is retained.

    det01, det02, det03

## Profile (frozen)

- Exactly the gate-v2 campaign reference create commands, including the
  campaign observer (context-v4) at `/opt/atlas-logits-patch`, with **one
  variable changed: `EXL3_FUSED_MOE=1` → `0`.** The installed plugin then
  selects `apply_exl3_python_loop`
  (`atlas-reference-active-exl3_mul1.py:1023`). That path runs unique
  experts in ascending ID order through `LinearEXL3.forward` (the code path
  of the shared experts and the attention projections, which were
  byte-identical across runs in the localization) and accumulates with one
  `index_add_` per expert, which writes each token row at most once. It also
  changes the routed arithmetic: separate gate, up and down GEMMs replace
  the fused kernel. So it defines a different reference profile, not a
  neutral pin.
- No dump hooks. Collection is the campaign's full short16-then-long order:
  32 final-logit rows.
- Caches: each attempt gets an isolated copy of the probe's frozen per-rank
  snapshot, which must hash equal before start. It is dropped after its
  post-shutdown manifest. Fixed mount sources are frozen in `mount-freeze/`
  before `det01`.
- Runner: `tests/exl3_validation/ref_experiment_runner.py det`, the lifecycle
  of the executed HSL runner, parametrized. Provenance comes from the gate-v2
  finalizer variant `driver/finalize-reference.py` (byte-identical to the
  HSL and probe copies), with the campaign draft plus the one variable.
- Evidence root: `fused-moe-determinism-20261015/`.

## Criteria and interpretation (frozen before collection)

**Primary (determinism):** all 32 final-logit rows (FP32 bytes) are
identical across det01, det02 and det03.
- **Pass:** the loop path is run-to-run deterministic at this profile. The
  fused path's atomic accumulation is then demonstrated as the source of
  the reference's run-to-run variation. It was the only change, and fused
  draws were never byte-identical on long rows across the 15-draw campaign
  or the probe. A deterministic reference variant then exists.
- **Fail** (any differing row): another non-deterministic component
  remains, or the loop path is itself order-dependent. Report the differing
  rows. Localization needs a separately registered dump run. Do not label
  the fused-path attribution refuted: it stands on its own evidence.

**Secondary (descriptive, no threshold, never a gate verdict):**
- descriptive H3 of each det draw against the 15 fused-path campaign
  references;
- per-row symmetric KL from each det draw to its nearest campaign
  reference, compared with the campaign's own spread. This measures how far
  the deterministic variant sits from the fused ensemble, which any future
  reference-profile amendment would have to account for.

Any adoption of a deterministic reference profile needs a separately
registered amendment and a new qualification campaign that retains the
existing UNVERIFIABLE result.

## Execution record

- det01, det02, det03 **complete** (2026-10-15 01:55–02:5xZ; both ranks exit
  0 each time). An earlier launch error ran nothing and consumed no attempt.
- **Primary: FAIL as registered.** 25 of 32 rows are identical; the 7
  differing rows are exactly those with 2,052 or more tokens. The
  fused-MoE attribution is strengthened: all variation below 2,052 tokens
  disappears. A second source starts at the indexer's pool-selection
  threshold. See the [result](exl3-fused-moe-determinism-result-20261015.md).
