# Gate v2 evaluator verification — 2026-10-12

This report's original A2.4 verification is historical. The 2026-10-14
post-collection erratum and qualification outcome are appended below; the
pre-collection findings and their receipts remain unchanged.

## Result

**The v2 evaluator and requested implementation checks are complete; the operator has authorised qualification collection after these checks.** The evaluator is separate from the frozen v1 path and has production-path fixtures for the Amendment 2 semantics, campaign plan/order and receipt contract, and held-out isolation. The archived reference-only H5 replay is bit-exact for all six held-outs, and the amended H4 recheck reproduces the single registered mode row with no LOO failures. The operator authorised collection in the follow-up request to finish replay/reporting and launch Luna for qualification. Evaluator review/freeze and host/image preflight must finish before the first qualification sample. No qualification verdict is made here.

## Implementation

- Added `tests/exl3_validation/statistical_gate_v2.py`. It reuses v1 artifact loading, row metrics, discrete checks and rank scoring without editing v1. H3 computes every gated row and recomputes the envelope from exactly the other 14 references for each leave-one-out check. H4 implements canonical FP32 byte classes, the two-class zero-spread exception, positive-distance gap scanning with first-tie selection, exactly-two-component single-linkage, singleton/zero-denominator handling, measured-radius membership with no epsilon, LOO membership, mode-frequency intervals and byte/ULP evidence.
- The campaign evaluator requires exactly 15 references, 3 held-outs, 2 byte-identical candidate runs, five individually named semantic perturbations and one MDE run. It validates the frozen seeded permutation, recorded engine builds, attempt order from non-overlapping UTC cache receipts, and the pre/post cache mount manifests. It only reads artifacts and receipts; it cannot collect, retry, top up or substitute attempts. Missing evidence returns `INCOMPLETE`, failed reference LOO returns `UNVERIFIABLE`, and only complete numerical/control evaluation can return `PASS` or `FAIL`.
- Added production-path fixtures in `tests/exl3_validation/test_statistical_gate_v2.py`. The test suite exercises actual evaluator functions for H3/H4, including edge cases and all campaign state logic. The v1 evaluator and historical behavior remain unchanged.
- Added CLI and campaign input documentation to `tests/exl3_validation/README.md`.

## Verification performed

1. `PYTHONPATH=tests/exl3_validation python3 -m unittest test_statistical_gate_v2 test_statistical_gate` — **PASS**, 23 tests (13 v2 fixtures and 10 unchanged v1 tests).
2. `python3 -m py_compile tests/exl3_validation/statistical_gate_v2.py tests/exl3_validation/test_statistical_gate_v2.py` — **PASS**.
3. Plan CLI smoke test with a deterministic fixture plan — **PASS**, reports 20 primary attempts, six fixed post attempts, and `collection_executed: false`. Output: `/tmp/atlas-gate-v2-plan-report.json`.
4. Archived H5 comparison — **PASS, exact equality** for `T_candidate`, all `T_references`, and `p_value` on each of `ref13`, `ref13rep1` through `ref13rep5`, evaluated against the archived nine-reference set through v2's H5 implementation. It uses the unchanged v1 `rank_scores` primitive. Receipt: `docs/design/evidence/gate-v2-verification-20261012/atlas-gate-v2-t-check.json`, SHA256 `8cf562579614b0a92c391d4aac433371459a86a235c5b6ecec0a8976811e8120`.
5. Amendment 2 H4 recheck on all 32 rows of the 15 archived reference-only tuning draws — **PASS**. Exactly one row demonstrates modes: `short16/longer/r0`, sizes 12/3, separation 25.614482, `rho=0.0013953445`; exact CP95 intervals are `[0.519109, 0.956688]` and `[0.043312, 0.480891]`. The evaluator ran all 15 leave-one-out fits for that demonstrated row: **0 failures**. No other row demonstrated modes. This reuses tuning data only for implementation rechecking; it is not qualification. Receipt: `docs/design/evidence/gate-v2-verification-20261012/atlas-gate-v2-h4-recheck.json`, SHA256 `ff96d9fc1806ba6b7d545835ded1ce2bfabecd76c6e9342c3418219c0d86c004`.

Evaluator SHA256 at verification: `8ae3122e2aa942fe4c50b78c158653bcb7eea620dadebf8bd3b484ffa5af176d`. Fixture SHA256: `b9e0f055ee09b63cd631c84f6995e72f6da3a1c160bbb0b6615259ee4d553965`.

The archived H4 comparison matches the prior calibration's full-set detection, separation, component sizes and radius. This pass does not repeat the archived exhaustive subset scan. It does not run a complete campaign because no v2 qualification artifacts exist.

## A2.4 coverage

| Requirement | Result |
|---|---|
| H3 leave-one-out exclusion | Fixture verifies each target is excluded and the fit has exactly 14 references; production report carries per-item envelope diagnostics. |
| H4 leave-one-out exclusion | Fixture exercises all 15 exclusions; archived demonstrated-row recheck is 0/15 failures. |
| Exactly two byte classes; positive cross-distance and zero within-distance | Production H4 special case covered by fixture; reports `rho=0`, separation `+Infinity`. |
| Zero cross-distance between distinct byte classes | Fixture uses byte-distinct, additive-shifted logits with equal distributions; no mode is declared. |
| Zero-radius ULP control failure | Fixture applies a one-f32-ULP mutation through `evaluate_h4`; it fails byte-identity membership. |
| Singleton component and tied largest gap | H4 production helper fixtures verify singleton spread zero, `+Infinity`/undefined separation semantics, and first tied gap selection. |
| Fixed attempt order/counts and no top-ups | Plan fixtures reject missing/extra names and a permutation inconsistent with the recorded seed; receipt fixtures reject observed order violations. |
| Blocked/incomplete states | Fixtures check `INCOMPLETE`, `UNVERIFIABLE`, and failed complete campaigns cannot pass. |
| Held-out isolation | The 3 held-outs are evaluated separately against exactly 15 references; fixtures assert held-outs are not inserted into the fit. |
| Unchanged T vs archived reference-only calibration | All six archived evaluations match exactly for candidate/reference T and p. |
| Amendment 2 H4 calibration recheck | All 32 archived rows and the 15 LOO fits for the sole demonstrated row were rerun; no detection/radius/separation difference was found at N=15. |
| Cache receipt and attempt plan contract | CLI schema requires explicit pre/post receipts for five cache categories and verifies observed serial order; no collection is performed. |
| Stable provenance implementation | The additional stable-field validator passes over the 15 archived reference tuning attempts; per-attempt preregistration digests remain individually loader-verified rather than required to match each other. |

## Evidence identity and limits

The calibration inputs are under `/home/colum/Workspace/atlas-exl3-validation-20261010` and were read only. Replay receipts were initially written under `/tmp` and are now archived under `docs/design/evidence/gate-v2-verification-20261012/`. The corrected H5 receipt hash above binds the final replay; the earlier report quoted a previous receipt hash. This was CPU-only: no GPU was used, no new collection was run, no candidate results were read for calibration, and no remote mutation, publish, commit or PR action occurred. Archived tuning draws are not eligible v2 qualification samples. Collection was subsequently authorized and completed; see the erratum below.

## Post-collection evaluator erratum — 2026-10-14

The original frozen evaluator (`statistical_gate_v2.py` SHA256
`426701ccc0f717ea782116354188b24f421862a676afbdf76f14f6c84f7465dc`)
incorrectly called the v1 `validate_runs` candidate-pair validator on the
three independent held-out reference attempts. That validator enforces
byte-identical logits across its second argument; the v2 registration
requires byte identity only for `v2cand01` and `v2cand02`, and explicitly
says distinct reference draws must not be required to match. The first
campaign evaluation therefore returned `INCOMPLETE` before statistical
evaluation. Its report and evaluator snapshot remain preserved in the
campaign evidence root.

The correction adds an optional `require_identical_candidates` parameter to
the v1 validator, defaulting to `true`, and disables only that check when v2
validates held-outs. Shared identity, provenance, and finite-logit checks
remain active; candidate byte identity remains enforced. A new end-to-end
campaign regression supplies distinct held-out logits and still routes the
candidate pair through its byte-identity check. The focused v1/v2 gate tests
pass 24/24. No threshold, metric, or collection artifact changed.

The corrected evaluator (SHA256
`96423dbb11555f064423f90e3f8782bfb634d641ed0b2df23b4f8fcb37a535fd`)
evaluated the complete 26-attempt campaign and returned **UNVERIFIABLE**:
H3 and H4 reference leave-one-out checks failed. H3 has one failed exclusion:
index 13 (`v2ref14`), item `0/longer/r3/sym_kl` (nearest `0.0508702476`, spread
`0.0017906041`, allowed limit `0.0053718123`). H4 has one failed membership
exclusion: index 14 (`v2ref15`), `short16/code` row 1 (distances `0.0002733329` and
`0.0001366067`, limit `0.0000238237`). Under Amendment 2, an unverifiable
reference set blocks qualification; the evaluator correctly stopped before
candidate, held-out, perturbation, and MDE gate results. The candidate is
therefore **not qualified**; this is not a candidate FAIL verdict.

The campaign contains 26 completed attempts and 832 request receipts,
including 16 short and 16 long MDE requests. Full report, inputs, and archive
hashes are indexed at
[`evidence/gate-v2-qualification-20261014/summary.json`](evidence/gate-v2-qualification-20261014/summary.json)
and under `/home/colum/Workspace/atlas-exl3-validation-20261010/gate-v2-qualification-20261013-luna1/`.
