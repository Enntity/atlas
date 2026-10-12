# Decision log

Short, direct entries: code/commit reference, major/minor, the choices, the
decision, and why. Runtime evidence and assumptions live in the append-only
experiment log (`q2-diagnostic.md`) and session checkpoints (`continue.md`);
this file references them, it does not duplicate them.

## 2026-10-12 — long-suite/deep-layer scope extension granted (major)

- Operator decision 2026-10-12: grant the explicit scope extension for the
  2048-token long-suite slice and layer captures beyond the current dumps.
- Choices: grant now, or defer until C1/C2 results. **Granted** — C1/C2 are
  layer-0-only on the existing 16 fixtures and consume none of the extended
  scope; the extension is spent by the G1/G5 fallbacks and the separately
  unlocalized `index-boundary-2048` long-suite failure, which needs it under
  either C1/C2 outcome.
- Effect: the out-of-scope notes in `docs/plans/q2-norm-causality-plan.md`
  for that slice and for G1 deep-layer dumps are lifted as of this date.
  Frozen gate, fresh-attempt, and evidence-preservation constraints unchanged.

## 2026-10-12 — ref13 parallel track; delegated execution (minor)

- Operator decision 2026-10-12: run the ref13 held-out investigation in
  parallel with the causality chain (as the plan prescribes), and delegate
  code implementation/execution work to mimo-v2.6-flash subagents
  (`xiaomi-token-plan-sgp/mimo-v2.6-flash`) with independent subagent review.
- Why: the must-pass control blocks qualification regardless of candidate
  progress; parallelism costs no gate integrity. Diagnosis outcomes are
  reference-chaos-consistent, environment mismatch, or unresolved; no gate
  change under any outcome.

## 2026-10-12 — incorporate independent Q2 review (minor)

- Operator request: update existing docs with the review findings, then
  revisit the vLLM reports for missed useful findings.
- Revision: retain fixture-local norm attribution and the failed Q2 verdict;
  record the two open ablation validation defects, correct the flag-name-based
  KV asymmetry claim, and specify the C2 intervention/metrics before execution.
  Prioritize the observed longer L0→L1 boundary and require config/provider
  binding before the affected replays.
- Authority: `docs/plans/q2-norm-causality-plan.md` owns revised sequencing;
  `todo.md` owns open implementation checks; the appended independent-review
  and vLLM-reconciliation entries in `q2-diagnostic.md` own evidence and
  superseded claims. Recover/reuse archived config and component receipts
  before treating those areas as missing work.
- Limits: documentation/review only; no new GPU authorization, gate redesign,
  production repair or qualification claim.

## 2026-10-12 — validation-campaign logging convention (minor)

- Choices: (a) global AGENTS.md `decisions.md` + `assumptions.md`, (b) project
  practice only (`q2-diagnostic.md`/`continue.md` append-only logs, `docs/adr/`),
  (c) merge: `decisions.md` references the project logs; assumptions stay in
  experiment entries; no `assumptions.md`.
- Decision: **(c)**. Operator choice 2026-10-12.
- Why: the project logs are the registered evidence surface for the Q2
  campaign; a second parallel assumption log would split the record without
  adding an authoritative source (SSOT).

## 2026-10-12 — provenance before new GPU attempts (major)

- Choices: (a) commit the uncommitted binding-capture Rust edits + replay
  tooling first, (b) run replay-only work uncommitted and commit at build time.
- Decision: **(a)**. Commits `1def024c` (Rust binding capture; content-identical
  to clean diagnostic commit `0dc3881e` — `git diff 0dc3881e 1def024c -- crates/`
  empty), `62780568` (Python replay tooling). Operator choice 2026-10-12.
- Why: every new attempt/build needs clean source attribution; the burned
  image `src-c12cc52e2ac2` cites `0dc3881e`, and content equivalence keeps that
  provenance resolvable.

## 2026-10-12 — GPU session scope (major)

- Choices: (a) ablation only (next-repair #1), (b) ablation + o_proj causality
  and downstream intervention (next-repair #1–3), (c) push through fresh build
  + frozen registered Q2 qualification (next-repair #4).
- Decision: **(b)**, with the ref13 held-out stability investigation
  (required experiment 4) run in the same session. Operator choice 2026-10-12.
- Why: the ablation alone cannot establish Q2 relevance; the causality slice
  decides whether the ≤1-ULP gated-norm difference matters downstream. The
  qualification rerun (c) stays out of scope until a justified fix exists.
  The 2048-token long-suite slice still needs an explicit scope extension
  before it may run.

## 2026-10-12 — ablation axis decomposition (minor)

- Choices: (a) reproduce the reference reduction tree exactly in the candidate
  grid, (b) one variant per named arithmetic axis, test order sensitivity in
  the reference's direction, (c) emulate both arithmetic chains on CPU.
- Decision: **(b)** — `v1-reduction-order`, `v2-rsqrt-sqrt-div`,
  `v3-sigmoid-exp`, `v4-sigmoid-reciprocal-mul`, plus `v0-copy` anchor and
  `v-all` composition. Evidence: `q2-norm-ablation-20261012/ptx-inspection/`.
- Why: (a) is impossible — the reference fma-chain+butterfly tree does not
  map onto the candidate's one-row-per-block grid (stated in the inspection
  limits); (c) cannot reproduce `rsqrt.approx`/`ex2.approx` hardware
  approximation bits on CPU. Compiled-PTX inspection verified three
  non-sources (div-by-128 exactness, eps-add commutativity, product
  commutativity) before the matrix was fixed.

## Review revision — neighboring configuration coverage

- Decision: shared-code corrections require a source-derived neighboring
  configuration matrix, as specified in the Q2 causality plan. Cover reachable
  supported branches and rejection contracts; record implemented but
  unqualified configurations separately. GLM's numerical policy is not a
  default for every model using the same primitive.
- Reason: stage-local GLM parity cannot establish correctness of GDN/Mamba2
  norm variants, other weight/cache formats or different dispatch thresholds.
- Verification: the expanded plan and reusable review prompt are documentation
  only. Runtime coverage remains unchecked in `todo.md`. Correctness and
  performance need separate evidence; global optimality is not established by
  reference parity or by winning a limited comparison.

## 2026-10-11 — Q2 gate stays frozen (major, pre-existing)

- Reference: `q2-diagnostic.md` "Objective and current verdict".
- Decision: repair must pass the unchanged registered Q2 gate with its
  controls; no gate redesign, no rerun selection, no discarding the failing
  `ref13` row.
- Why: pre-registered thresholds and controls are the qualification contract;
  redesign would invalidate comparability with the registered evidence.

## 2026-10-12 — C1 local o_proj binding: load-time BF16 common space + resident sub-record (minor, Q2 C1)

- Reference: `docs/plans/q2-norm-causality-plan.md` §"Experiment C1";
  `q2-c1-prep-20261012/binding-status.md` (`atlas-local-o-proj-binding-v1`);
  supervisor decision A+ of the C1 capture stage (2026-10-12).
- Decision: the candidate capture is load-time first — with
  `ATLAS_LOCAL_O_PROJ_BINDING_DIR` + `ATLAS_LOCAL_O_PROJ_BINDING_LAYERS=0`
  (unset = zero behavioral difference) the layer-0 o_proj shard is written
  once as the BF16 common comparison space (`reference_reconstruct` →
  RNE-BF16, mirroring vLLM's load-time dequant) together with the verbatim
  resident quantized bytes and recipe identity in the additive `resident`
  sub-record; the forward hot path only stamps the per-invocation
  `atlas-local-o-proj-binding-v1` JSON plus GEMM receipt (no tensor copies,
  no synchronizing work). The reference overlay mirrors this symmetrically
  behind `ATLAS_LOCAL_O_PROJ_BINDING=1`.
- Why: every candidate o_proj loader path is quantized (no BF16 device copy
  exists), the frozen replay harness requires BF16 weight bytes, and
  reconstruction must not run in the forward path; the BF16 bytes are the
  comparison space, NOT a claim about the native GEMM's exact bits — that
  claim is tested fail-closed by the C1 self-anchor, with the algorithm
  receipt as the diagnostic if it fails.
- Code: `weight_loader/glm5/o_proj_binding.rs`,
  `layers/glm5_kda/{upstream,upstream_tests,o_proj_receipt}.rs`,
  `spark-runtime weights/exl3/ffn_io.rs`,
  `tests/exl3_validation/local_o_proj_capture.py`.
