# Decision log

Short, direct entries: code/commit reference, major/minor, the choices, the
decision, and why. Runtime evidence and assumptions live in the append-only
experiment log (`q2-diagnostic.md`) and session checkpoints (`continue.md`);
this file references them, it does not duplicate them. Newest first.
Superseded entries are kept as one-line stubs marked **SUPERSEDED**, so the
audit trail survives; their full text is in git history.

## 2026-10-12 — C1 re-scoped: close from native captures (major)

- Operator decision 2026-10-12, after the independent review
  (`CODE_REVIEW_2026-10-12.md`, finding A1).
- Choices: (1) close C1 from the existing native captures; (2) rebuild the
  binding on trellis bytes and replay both engines through one EXL3
  backend; (3) fold C1 into the C2 in-situ intervention. **Decision: (1).**
- Why: the BF16 common-space replay premise was invalid. The reference
  `apply` runs the exllamav3 trellis path with an FP16 input for
  `rows <= 144` (every C1 row), and the candidate runs EXL3 trellis kernels.
  Neither multiplies the bound BF16 weight, so the self-anchor could never
  pass. The native captures already answer C1 locally: zero same-input
  projection difference on 8/8 identical-input rows; the natural norm delta
  vanishes on rank0 and stays confined to one token on rank1. The open
  question is downstream causality, which only C2 measures.
- **FLAG (carried into every C1 citation):** the corrected-input answer
  (v24 → reference o_proj output) is *inferred* from 8 identical-input rows
  of 2 prompts, not measured. The candidate o_proj input precision is not
  attested by any receipt. The answers are local to layer-0 o_proj.
- Effect: the C1 replay harness, the engine-native WIP and the Rust capture
  hooks are archived (`651ff083`; tag `archive/c1-bf16-o-proj-replay-20261012`;
  `tests/exl3_validation/archive/c1-local-o-proj-replay/README.md`). The
  evidence erratum is at `$EV/q2-c1-execution-20261012/ERRATUM-20261012.md`.
  C2 is the next causality experiment.

## 2026-10-12 — C1 "engine-native replay" remedy withdrawn (major)

- That remedy was approved by an orchestrating agent (q2-diagnostic C1
  entry, `3956b63c`), not by the operator. Its reference half replays the
  `rows > 144` `F.linear` branch that C1 rows never take. It is withdrawn,
  superseded by the C1 re-scope above.

## 2026-10-12 — gate v2 Amendment 1 (major, pending sign-off)

- Source: independent review finding A3. Applied pre-collection to
  `docs/design/exl3-statistical-gate-v2.md` (`fd7b530b`), together with the
  six P2 review fixes.
- Content:
  - Randomised, interleaved run order.
  - A reference-only composition check: VOID with a capped top-up (N ≤ 25)
    instead of FAIL.
  - Disclosures that H3 is v1's rule unchanged and that H4 is stricter than
    the floor.
  - The v1 diagnosis restated as "underpowered at N=9, drift not excluded".
  - Added residual risks and calibration-script errata.
- No threshold is loosened. **Operator sign-off on the amended document is
  required before any v2 collection.** A probe of the ref13 discrete
  mechanism (runtime-choice pinning) is recommended before collection.

## 2026-10-12 — Q2 gate: full v2 pre-registration authorized (major)

- Operator decision 2026-10-12: author a full v2 pre-registration of the
  statistical gate. **Supersedes** the 2026-10-11 "Q2 gate stays frozen"
  entry and the "no gate change" clauses of the ref13-track and
  incorporate-review entries below.
- Evidence (reference-side only): ref13 investigation. The reference fails
  its own must-pass held-out control in 3/5 fresh draws. The catastrophic
  floor (KL 0.005) sits below the reference's own cross-mode distance at
  short16 `longer`/r0 (≈0.056). The rank condition is marginal for held-out
  draws at N=9. Diagnosis per Amendment 1: v1 was underpowered at N=9 under
  a bimodal row, with session drift not excluded.
- Choices: minimal v2 addendum / full v2 / keep frozen / defer. **Full v2**:
  - N>=15 references and H=3 held-outs;
  - an explicit leave-one-out rank exemption;
  - v1's envelope rule stated for every row;
  - a mode-membership rule for bimodal rows;
  - T kept.

  Explicitly surrendered: direct comparability with the registered v1 FAIL
  run, which is kept as the historical record.
- Discipline: v2 is authored from reference-only data (no candidate number
  sets any constant) and committed before any v2 collection. The existing
  *reference* runs are v2 tuning data, not qualification samples. The
  redesign repairs the control, not the verdict.

## 2026-10-12 — long-suite/deep-layer scope extension granted (major)

- Operator decision 2026-10-12: grant the explicit scope extension for the
  2048-token long-suite slice and layer captures beyond the current dumps.
- Choices: grant now, or defer until C1/C2 results. **Granted.** C1/C2 are
  layer-0-only on the existing 16 fixtures and consume none of the extended
  scope. The extension is spent by the G1/G5 fallbacks and by the separately
  unlocalized `index-boundary-2048` long-suite failure, which needs it under
  either C1/C2 outcome.
- Effect: the out-of-scope notes in `docs/plans/q2-norm-causality-plan.md`
  for that slice and for G1 deep-layer dumps are lifted as of this date.
  Fresh-attempt and evidence-preservation constraints are unchanged.

## 2026-10-12 — ref13 parallel track; delegated execution (minor)

- Operator decision 2026-10-12: run the ref13 held-out investigation in
  parallel with the causality chain, and delegate implementation/execution
  to subagents with independent subagent review.
- Why: the must-pass control blocks qualification regardless of candidate
  progress. (This entry's original "no gate change under any outcome" is
  superseded by the v2 decision.) Later delegation preference: Pi agents on
  openrouter inkling-small for implementation; Opus for review.

## 2026-10-12 — incorporate independent Q2 review (minor)

- Operator request: update existing docs with the review findings, then
  revisit the vLLM reports for missed useful findings.
- Revision:
  - Retain fixture-local norm attribution and the failed Q2 verdict.
  - Record the two ablation validation defects (since repaired:
    `9ff39728`/`c71bb557`).
  - Correct the flag-name-based KV asymmetry claim.
  - Specify the C2 intervention/metrics before execution.
  - Prioritize the observed longer L0→L1 boundary, and require
    config/provider binding before the affected replays.
- Authority: `docs/plans/q2-norm-causality-plan.md` owns sequencing;
  `todo.md` owns open checks; `q2-diagnostic.md` owns evidence and
  superseded claims.
- Limits: documentation/review only. (The "no gate redesign" limit is
  superseded by the v2 decision.)

## 2026-10-12 — validation-campaign logging convention (minor)

- Choices: (a) global AGENTS.md `decisions.md` + `assumptions.md`, (b) project
  practice only (`q2-diagnostic.md`/`continue.md` append-only logs, `docs/adr/`),
  (c) merge: `decisions.md` references the project logs; assumptions stay in
  experiment entries; no `assumptions.md`.
- Decision: **(c)**. Operator choice 2026-10-12.
- Why: the project logs are the registered evidence surface for the Q2
  campaign; a second parallel assumption log would split the record without
  adding an authoritative source (SSOT).

## 2026-10-12 — C1 local o_proj binding: load-time BF16 common space (minor) — **SUPERSEDED**

Superseded by "C1 re-scoped" above. The BF16 common-space binding was not
what either engine multiplies. The code is archived; the original text is in
git history (tag `archive/c1-bf16-o-proj-replay-20261012`).

## 2026-10-12 — provenance before new GPU attempts (major)

- Choices: (a) commit the uncommitted binding-capture Rust edits + replay
  tooling first, (b) run replay-only work uncommitted and commit at build time.
- Decision: **(a)**. Commits `1def024c` (Rust binding capture; content-identical
  to clean diagnostic commit `0dc3881e` — `git diff 0dc3881e 1def024c -- crates/`
  empty), `62780568` (Python replay tooling). Operator choice 2026-10-12.
- Why: every new attempt/build needs clean source attribution. The burned
  image `src-c12cc52e2ac2` cites `0dc3881e`, and content equivalence keeps that
  provenance resolvable.

## 2026-10-12 — GPU session scope (major)

- Choices: (a) ablation only, (b) ablation + o_proj causality and downstream
  intervention, (c) push through fresh build + registered Q2 qualification.
- Decision: **(b)**, with the ref13 held-out stability investigation run in
  the same session. Operator choice 2026-10-12. (The o_proj part was
  resolved by the C1 re-scope.)
- Why: the ablation alone cannot establish Q2 relevance. The qualification
  rerun (c) stays out of scope until a justified fix exists.

## 2026-10-12 — ablation axis decomposition (minor)

- Choices: (a) reproduce the reference reduction tree exactly in the candidate
  grid, (b) one variant per named arithmetic axis, (c) emulate both arithmetic
  chains on CPU.
- Decision: **(b)**: `v1-reduction-order`, `v2-rsqrt-sqrt-div`,
  `v3-sigmoid-exp`, `v4-sigmoid-reciprocal-mul`, plus the `v0-copy` anchor
  and the `v-all` composition. Evidence:
  `q2-norm-ablation-20261012/ptx-inspection/`.
- Why: (a) is impossible, because the reference fma-chain+butterfly tree does
  not map onto the candidate's one-row-per-block grid; (c) cannot reproduce
  `rsqrt.approx`/`ex2.approx` approximation bits on CPU. Limit: an axis that
  did not reproduce the reference is "inert on the 16 fixtures", not proven
  a non-source.

## Review revision — neighboring configuration coverage

- Decision: shared-code corrections require a source-derived neighboring
  configuration matrix, as specified in the Q2 causality plan. Cover reachable
  supported branches and rejection contracts; record implemented but
  unqualified configurations separately. GLM's numerical policy is not a
  default for every model using the same primitive.
- Reason: stage-local GLM parity cannot establish correctness of GDN/Mamba2
  norm variants, other weight/cache formats or different dispatch thresholds.
- Status: matrix instantiated (`q2-neighbor-matrix-20261012/`, 49 rows);
  the repeat Rust/vLLM/source review on actual candidate artifacts is open.

## 2026-10-11 — Q2 gate stays frozen (major) — **SUPERSEDED**

Superseded by "Q2 gate: full v2 pre-registration authorized" (2026-10-12).
The v1 rule and its FAIL run remain the historical record. No rerun
selection and no discarding of reference rows: these still hold under v2.
