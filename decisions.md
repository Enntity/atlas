# Decision log

Short, direct entries: code/commit reference, major/minor, the choices, the
decision, and why. Runtime evidence and assumptions live in the append-only
experiment log (`q2-diagnostic.md`) and session checkpoints (`continue.md`);
this file references them, it does not duplicate them. Newest first.
Superseded entries are kept as one-line stubs marked **SUPERSEDED**, so the
audit trail survives; their full text is in git history.

## 2026-10-14 — L1 escalation + v24-prerequisite tooling batch (major)

- Choices: (a) wait for GPU windows and implement per-item then, (b)
  pre-stage every CPU-only piece now so windows execute runs instead of
  growing code. Operator choice 2026-10-14: **(b)** for the L1 tooling
  (todo.md "Now" #4) and the v24 prerequisites (todo.md "Next").
- Decision: implement D1–D5 per
  [docs/plans/l1-escalation-tooling-spec.md](docs/plans/l1-escalation-tooling-spec.md)
  — candidate L0→L1 capture + mHC boundary binding (`upstream.rs`,
  `hc.rs::mhc_residual_precision` SSOT), reference overlay extension
  (`$EV/q2-l1-tooling-20261013/`), `l1_bisect.py` driver, v24 engagement
  controls, and the `nvcc-production` ablation compile lane. Batch commit
  of 2026-10-14 (this entry lives inside it; hash is self-referential).
- Why: the Sparks are the scarce resource; a free window must run tests,
  not host implementation work. All GPU obligations stay recorded as owed
  (spec per-deliverable sections; no GPU evidence claimed).
- Test recipe note: Rust tests DO run on this host via the C1-era link
  stubs (`LD_LIBRARY_PATH=/tmp/c1-cuda-stubs RUSTFLAGS="-L
  /tmp/c1-cuda-stubs"` + `ATLAS_SKIP_BUILD=1`), contradicting the
  2026-10-13 entry's "cannot link on this host" — 1654 spark-model tests
  pass. The stubs are link-only; no CUDA numerics are exercised.

## 2026-10-14 — ablation variant compile flags: recorded build manifest is the SSOT (minor)

- Choices: (a) hand-supplied flag lists, (b) re-derive from KERNEL.toml/
  MODEL.toml in Python, (c) replay the flags recorded by the build.
- Decision: **(c)** — `variant_compile.py` reads
  `atlas-exl3-build-manifest-v1` `compiler_flags` (`module_flags.kda` else
  `flags`), tied to the attested embedded PTX (sha) + its `.target` arch +
  the local nvcc compiler identity, with argv parity to
  `atlas-kernels/build_target.rs` (`--ptx -arch=… -O3 [win32 -std=c++17]
  <flags> --Werror all-warnings <src> -o <out>`).
- Why: (a) is drift-by-hand; (b) duplicates build.rs merge semantics in a
  second language. (c) replays exactly what built the binary being
  attributed. Both lanes (`nvrtc` historical, `nvcc-production`) are
  required-explicit and recorded per run so a v24 claim can never be read
  across lanes. The `--Werror` strict default is mirrored deliberately; a
  future nonempty warning-suppress list would surface as a compile failure
  here, not silent drift.

## 2026-10-14 — L1 bisection boundary decisions (minor)

- Local o_proj is **not bisectable cross-engine** (assumption B13): the
  reference `o_proj` is a `RowParallelLinear` (reduce inside `forward`),
  so its hook output `kda-o-proj` is post-reduce. Decision: the spine uses
  the reduced-o_proj step, `kda-o-proj-input0` stays a diagnostic pair,
  and a pre-reduce reference capture is an owed overlay extension. Why:
  capturing it requires wrapping the reference's internal reduce — deferred
  rather than guessed; C1 already closed the local projection question.
- L1 capture layers are the explicit {0,1} boundary on both engines (why:
  narrowest scope of the granted escalation; G1's layers 4–44 widen it in
  their own reviewed change). The driver fails on incomplete bisection
  AFTER writing the report (why: a partial pair table is evidence; the
  analyze_localization refusal pattern already does this).

## 2026-10-14 — independent review of the tooling batch: BLOCK fixed (major)

- Reviewer verdict BLOCK on three D3 majors: layout mapping not reused
  (real reference core/gated shapes would error every mapped pair); pair 5
  silently dropped with a wrong spec citation; report lost on failure.
  Minors: disabled capture validated the binding; binding schema drifted
  (`local_rows`); harness hash omitted `variant_compile.py`; tautological
  win32 assertion; D2 recorded unexposed facts as `absent`/null.
- Decision: all fixed in this batch — `kda_layout` mapping + tests,
  spec/assumption amendment (B13) + citation fix, report-before-fail,
  binding built only when armed (pinned by test), spec-conformant `rows`
  schema, harness hash complete, real win32 argv test, `unknown`+reason
  records. Re-tested: 62 in-repo Python, 19 overlay, 1654 spark-model.
- Why: GPU windows must not carry a driver that cannot parse the captures
  it was built for. Accepted notes (no change): strict-flag fork risk
  (recorded in code + above), `--layer` constrained to {0,1} (done).

## 2026-10-14 — repo-health drift stays out of the tooling batch (minor)

- Operator choice 2026-10-14: fmt (~207 files) / clippy (~70 findings
  across spark-runtime+spark-model, incl. the C2 batch's `factory.rs:180`)
  / typos (56 findings) drift is verified "pre-existing, none in changed
  files" and stays for its own cleanup PR (todo.md housekeeping).
- Why: bundling ~200 files of mechanical cleanup into tooling would bury
  the reviewable change; the batch ships with the measured no-new-findings
  bar instead of a green gate.

## 2026-10-13 — M1-parallel CPU batch landed; first pass ready for review/testing (minor)

- Landed the CPU-only items todo.md authorises alongside M1 GPU work:
  Q3 native-opener tests (`44da2486`), Q3 per-turn thinking-toggle cases
  (`918369ab`), norm-substitution module split (`05dcaca8`), dense-prefill
  flag test triage (`a2d36244`), exl3_checkpoint engagement test
  (`2fc58662`), P0 trace-driver command review
  (`p0-trace-prep-20261011/REVIEW-20261013.md`), KV plan-doc triage
  (`2a3b995b` — decided commit, not delete; cross-referenced from todo.md).
- Scope discipline: no frozen-profile surface touched, no GPU window
  consumed. Rust tests compile clean but cannot link on this host (no CUDA
  libs) — execution is owed on a Spark / CUDA CI. The exl3_checkpoint test
  was deliberately built mock-GPU so it runs on CPU CI, exceeding the
  "GPU-side test" ask.
- P0 review verdict: driver/interfaces PASS; two decisions owed before
  `--execute` (image pin choice; three-run schedule with decode-heavy
  capped ≤600 output tokens for the ~5-min nsys limit). `--execute` stays
  unscheduled per the README contract and the M1 GPU-window rule.
- Discovered, recorded in todo.md housekeeping, NOT fixed here: repo-wide
  fmt drift (~180 files under both local rustfmts) and three pre-existing
  current-stable clippy errors. Fixing them is its own PR (one logical
  change per commit).

## 2026-10-13 — KV column-codec experiment scoped and planned (minor)

- Operator request: plan (plan only, no implementation) for ClickHouse-like
  column codecs applied to KV cache compression, staged implementation and
  test plan required.
- Scope chosen by operator: all tiers incl. GPU-resident (contraindicated vs
  TQ+ kernel work — staged last with kill criteria); all serialized state
  incl. SSM; Gorilla-TSDB core + T64 + per-block adaptive selection; success
  gate is end-to-end tok/s under spill pressure with bit-exact roundtrip.
- Plan lives at `docs/plans/atlas-kv-column-codecs-plan.md` (DMAIC, Stages
  0–3 with entry/exit/kill criteria). Codecs compose after the existing lossy
  quantizers; host tiers (`kv_spill`, high-speed-swap, prefix-cache) are the
  primary target. Stage 3 (GPU-resident) is killable without failing the
  experiment.
- Why: host-tier KV is uncompressed today; ClickHouse-style prep+compressor
  pipelines are the cheapest falsifiable hypothesis (Stage 0 is offline).

## 2026-10-12 — v2 qualification collection authorised (major)

- Operator instruction: finish the replay and verification reports, then launch
  a Luna subagent to perform the required qualification collection runs.
- Final replay completed: all six archived H5 comparisons exact; parent reran
  the 23 v1/v2 CPU tests. Durable receipts and corrected final H5 receipt hash
  are in `docs/design/exl3-statistical-gate-v2-verification-20261012.md`.
- This authorises the amended fixed-N campaign and necessary build/preflight
  work. Do not ask again for collection sign-off. Review/freeze the evaluator,
  inspect both hosts and ownership, pin a fresh candidate image and prove
  neutrality/provider identity before qualification requests. If a concrete
  prerequisite fails, report it and preserve the attempt; do not relax rules.
- Luna owns both ranks for this campaign. No publication or numerical PASS is
  implied by this authorisation; the full evaluator and controls decide.

## 2026-10-12 — gate v2 Amendment 2 and pinned vLLM source review (major)

- Operator request: make the review corrections after launching a Luna
  subagent to review vLLM source, compare its report with the plan, and use
  vLLM as the source of truth for reference behavior. This authorises the
  documentation revision, not a new qualification run or numerical verdict.
- Implemented recommendation: retain fixed N=15 and H=3, remove adaptive
  composition top-ups and their unproved final-N rank bound. Preserve all
  attempts; unverifiable reference composition blocks qualification. Explicit
  H4 zero-distance/zero-radius semantics and evaluator checks are now required.
- Source review reconciled in `docs/design/exl3-vllm-reference-contract.md`:
  pinned installed plugin/observer paths override generic checkout assumptions;
  SM120 short prefill reads packed FP8 cache; capture precedes sampling;
  seed/fresh process is not a forward-determinism guarantee. Mode cause remains
  unproven. Statistical qualification policy is Atlas's responsibility.
- Amendment 2 in `docs/design/exl3-statistical-gate-v2.md` supersedes A1.2.
  Randomisation remains. PASS is scoped to fixtures/profile, not distributional
  identity, task quality or release. Final rule sign-off and evaluator
  verification remain prerequisites for collection; no data collected here.
- Lesson: inspect the exact installed vLLM path and existing reconciliation
  before using a generic source branch or archived audit narrative as authority.

## 2026-10-12 — C2 batch: full execution tooling, in-engine request binding (major)

- Operator decisions 2026-10-12 (batch kickoff questionnaire).
- Scope: (a) Rust knob only; (b) knob + fixture tooling; (c) full C2
  execution tooling. **Decision: (c)** — knob, fixture assembly, A/B run
  driver and the plan-fixed scoring, all CPU-side. Contract:
  `docs/plans/c2-tooling-spec.md`.
- Request binding: (a) invocation manifest + out-of-band receipt;
  (b) thread request ID into `ForwardContext`; (c) one request per
  process. **Decision: (b).** Strictest in-engine enforcement
  (`request_id` mismatch errors the run); accepted the production-path
  edit and its owed GPU neutrality proof. Why not (a): the plan demands
  binding to exact request IDs, and an out-of-band receipt cannot reject
  in-engine. Why not (c): 8 requests × 2 ranks × A/B × repeats of
  container restarts is operationally worse than one added field.
- Tensor-hash semantics: (a) payload integrity + recorded pre-substitution
  hash; (b) strict A/B pre-hash binding; (c) payload integrity only.
  **Decision: (a).** Reject now on missing/mismatched fixture; record the
  candidate's pre-substitution `gated` sha256 so A/B alignment and byte
  determinism are checked post-hoc. Why not (b): a run-B abort on any
  benign upstream change destroys the treatment run for no gain — the
  alignment evidence is identical, only the enforcement point moves.
- Effect: implementation delegated per root AGENTS.md (medium/large task),
  independent review subagent on the diff.

## 2026-10-12 — Host inspection command in `scripts/`, not a Justfile (minor)

- Contraindication raised: root AGENTS.md routes remote-host manual ops to
  the project's Justfile; this repo has no Justfile and its pattern is
  `scripts/*.sh` (`scripts/start-glm53-exl3-tp2.sh`).
- Operator decision 2026-10-12: **`scripts/inspect-sparks.sh`.** Repo
  consistency wins for this repo; the Justfile rule is explicitly
  overridden here. Revisit if Atlas adopts `just`. Logged as A13 in
  `docs/plans/c2-tooling-spec.md` §Constraints and assumptions.

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

**Partially superseded by Amendment 2 above:** the composition VOID/top-up
rule is withdrawn. This entry records the earlier proposal, not current policy.

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

## 2026-10-12 — assumption log folded back: logging-convention entry stands (minor)

- Contraindication surfaced by the independent review (m4): this batch
  created root `assumptions.md` per global AGENTS.md, contradicting the
  standing "validation-campaign logging convention" entry below (option
  (c): no `assumptions.md`, SSOT rationale).
- Operator decision 2026-10-12: **honor the standing entry.** The file
  existed only within this batch and is deleted; its A1–A13 content is
  folded into `docs/plans/c2-tooling-spec.md` §"Constraints and
  assumptions" with IDs stable, so tooling citations keep resolving.
- Why: the entry's SSOT rationale holds; the global rule is overridden
  for this project by that operator choice. Lesson: read `decisions.md`
  in full before creating new project-level artifacts.

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
