# EXL3 statistical gate v2 qualification — 2026-10-14

## Verdict

**UNVERIFIABLE; the candidate is not qualified.** The fixed campaign completed,
but the 15-reference ensemble failed both H3 and H4 leave-one-out
verifiability. Under the pre-registered Amendment 2 semantics, this blocks
qualification before candidate and control scoring. This is not a candidate
FAIL verdict.

The [pre-registration](exl3-statistical-gate-v2.md) and its thresholds remain
unchanged. The full
[verification report](exl3-statistical-gate-v2-verification-20261012.md)
records the evaluator correction and original frozen-evaluator result.

## Campaign result

The fixed order contains 26 completed attempts and 832 request receipts:
15 references, 3 designated held-outs, 2 candidate repeats, 5 semantic
perturbations, and MDE. The MDE attempt has 16 short and 16 long responses;
its archived logits have 32 sidecars and passed parity loading for 8 cases
(32 rows). Both Spark ranks were shut down cleanly after collection, their
GPUs were idle, and no campaign containers remained.

The corrected evaluator found:

- H3: one failing leave-one-out exclusion, index 13 (`v2ref14`), item
  `0/longer/r3/sym_kl`: nearest `0.0508702476`, spread `0.0017906041`, limit
  `0.0053718123`.
- H4: one failing membership exclusion, index 14 (`v2ref15`), `short16/code` row 1:
  distances `0.0002733329` and `0.0001366067`, limit `0.0000238237`.

The evaluator short-circuited as specified after reference verifiability
failed. Candidate, held-out, perturbation, and MDE gate results were not
scored. No samples were repeated, replaced, or topped up.

## Evaluator correction

The frozen evaluator initially returned INCOMPLETE because it applied v1's
candidate-repeat byte-identity requirement to the three distinct held-out
references. The corrected implementation disables only that check for
held-outs while keeping identity, provenance, and finite-logit checks active.
Candidate pair byte identity remains required and passes for both registered
candidate attempts. The correction changes neither metrics nor thresholds.

The frozen evaluator SHA256 is
`426701ccc0f717ea782116354188b24f421862a676afbdf76f14f6c84f7465dc`; the
corrected evaluator SHA256 is
`96423dbb11555f064423f90e3f8782bfb634d641ed0b2df23b4f8fcb37a535fd`.
Focused v1/v2 gate tests pass 24/24, including an end-to-end campaign test
with distinct held-out logits.

## Evidence

The evidence index binds the frozen plan and campaign manifest, both
evaluator versions, tests, corrected report, and exact H3/H4 failure records:
[summary.json](evidence/gate-v2-qualification-20261014/summary.json).
The full report and raw campaign evidence remain under
`/home/colum/Workspace/atlas-exl3-validation-20261010/gate-v2-qualification-20261013-luna1/`.
