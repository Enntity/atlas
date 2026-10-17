# Decision log

Short, direct entries: code/commit reference, major/minor, the choices, the
decision, and why. Runtime evidence and assumptions live in the append-only
experiment log (`q2-diagnostic.md`) and session checkpoints (`continue.md`);
this file references them, it does not duplicate them. Newest first.
Superseded entries are kept as one-line stubs marked **SUPERSEDED**, so the
audit trail survives; their full text is in git history.

## 2026-10-16 — Grouped multi-row FAT-routed MoE for prefill (major)

- Choice: implement the audit's P0 as a **new, default-off** route
  (`ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1`) instead of widening the existing
  `ATLAS_EXL3_COOP_MOE_MULTIROW` kernel, because that kernel is reachable only
  for `rows <= thin_capacity` and poisons `rows > 16` and `counts > thin_cap`.
- Split: `exl3_moe_group_units` emits, per accumulation class, M-tile units of
  at most 16 rows in the existing group layout. Thin units go to the existing
  `exl3_coop_moe8_multirow_b4_k32_n256` (tile 32x256, 8 slices, F16-rounded
  fold); fat units to the new `exl3_coop_moe_fat_multirow_b4_k16_n128` (tile
  16x128, 1 slice, unrounded F32 fold). The M-tile chunking is the unit split,
  so neither kernel needs a `rows > 16` poison.
- Why one route for both classes: `exl3_moe_mul1_matmul_precision_mma` picks
  fat vs thin **per expert count**, so a grouped prefill launch is usually a
  mix; emitting only the one profile would change numerics for the other.
- The grouped route is tried from `forward_local` for `rows > thin_capacity`
  only; thin launches keep the existing per-slot/coop paths, and with the flag
  off nothing is resolved or allocated.
- Evidence: `tests/exl3_validation/coop_moe_bench/fat_multirow_bench.cu` — every
  slot's F32 row byte-identical to `exl3_moe_mul1_matmul_precision_mma` over
  three routing distributions (counts 1..9, 17..34, ~128), i.e. below and
  above `thin_cap` and with groups spanning several M tiles; lock rows return
  to zero. Measured at slots=1024/288 experts: gate|up 111 ms -> 5.5 ms
  (20x), down 111 ms -> 5.5 ms (20x); all-fat 122 ms -> 2.0 ms (62x). In-model
  A/B is owed by the owner of `perf/decode-vllm-parity-20261015`.

## 2026-10-15 — L1 fused-off result: observer verified, bisect blocked by tooling (major)

- Result: `l1-fusedoff-f099-01` passed O1 (8/8 per engine) and O2 (layer-0/1
  reference captures byte-identical to fused-on `f099-02`). The reference
  observer blocker is closed. The registered bisect returned
  `analysis_complete=false` on all four runs, so there is no attribution.
- Cause: two defects that predate the attempt.
  - The candidate captures the FP32 `hc_streams` highway labeled BF16, with
    half its bytes (`glm5_kda/forward_attention.rs:69–79`).
  - `kda_layout` lacks the mHC `post` `[n,4,1] → [n,4]` binding.
- Decision: no rerun and no post-hoc tolerance. The next step is a
  registered correction. The operator chooses between the Rust capture fix
  (new image, neutrality proof, fresh collection) and an analysis-side FP32
  reread (half-token coverage). The FP32-versus-BF16 highway lead goes to
  the G-list only after its two source checks.
  See the [result](docs/design/l1-fusedoff-observer-result-20261015.md).

## 2026-10-15 — L1 observer verifiability through a fused-off reference (major)

- Choices:
  - (a) Apply a layer-scoped observer rule to the existing `f099-02`
    receipt. Rejected: that is a post-hoc criterion.
  - (b) Retry `f099-02` unchanged. Rejected.
  - (c) A fresh, pre-registered attempt that changes one reference variable,
    `EXL3_FUSED_MOE=0`, plus a cross-attempt capture comparator (O2).
  - (d) Wait for a deterministic-reference qualification amendment.
- Decision: **(c)**. See the
  [registration](docs/design/l1-fusedoff-observer-registration.md).
- Why:
  - The 2026-10-15 HSL, determinism and indexer results place all reference
    variation in rows of at most 2,051 tokens at the layer-3+ fused-MoE
    accumulation.
  - L1 rows are at most 128 tokens, and L1 captures layers 0/1, which run
    before routed MoE.
  - With the fused path off, the unchanged observer rule becomes decidable.
  - O2 ties the layer-0/1 captures back to the fused-on profile.
  - The collector, overlay and bisect are unchanged; the only new code is a
    read-only comparator.
- Operator instruction 2026-10-15: "do the L1 work".

## 2026-10-15 — reference kit stays unmodified; Atlas-side determinism only (major)

- Choices: patch the reference's fused-MoE accumulation or indexer top-k
  order, or leave the reference kit as installed.
- Decision: **leave it as installed.** The Atlas determinism audit and its
  follow-ups are Q5 in todo.md (decision recorded there first; logged here
  for the audit trail).
- Scope note: a runtime environment setting the installed plugin already
  reads (`EXL3_FUSED_MOE`) is a profile choice, not a kit modification. Its
  loop path is not a neutral pin. Using it for qualification would need its
  own amendment.

## 2026-10-15 — P0 profiler image re-pinned to `src-4ee96c12b44c` (minor)

- `docs/plans/P0-IMAGE-PIN-20261014.md` pinned `src-48624a8605de` on the
  premise that it carries `f099ac58`. It does not:
  - `48624a8605de` is the 2026-10-10 production build;
  - `f099ac58` first ships in `src-4ee96c12b44c` (`sha256:82173ea6…`).
- That plan's own rationale (trace the binary that will be optimized)
  selects `4ee96c12`, the newest image with a full neutrality proof. It is
  already loaded on both ranks.
- The near-limit run uses 3,500 input tokens, as the review set it. The
  plan's "8–12k" exceeds the 4,096-token context.
- The re-pinned driver is a copy in `$EV/p0-trace-20261015/driver/`. The
  descriptors are unchanged, only the image token is replaced, the image ID
  is asserted, and its 11 tests pass. The original prep is untouched.
- Sequencing: operator asked for the P0 tracing to run alongside the L1
  preparation. P0 used the GPUs first; L1 collection follows. There is one
  experiment owner at a time.

## 2026-10-14 — GPU-iteration tooling: image cycle, G8/G9 binding, probe pre-Spark package (major)

- Choices: (a) keep re-deriving the build→transfer→fingerprint→neutrality
  chain, the G8/G9 provenance bundle and the probe packaging by hand each
  time (the status quo that dominated iteration time between GPU windows),
  (b) one config-driven, hash-bound CLI per workstream, CPU-tested with
  injected runners and no host access.
- Decision: **(b)** —
  - `tests/exl3_validation/image_cycle.py`: preflight → build → transfer →
    fingerprint → neutrality → receipt. It writes the neutrality
    preparation config itself from the verified fingerprint (image id /
    build-manifest path cannot be known before the build) plus operator
    campaign inputs, and binds every stage receipt by SHA256; the transfer
    stage enforces both transport exit codes. 17 CPU tests.
  - `tests/exl3_validation/g8g9_binding.py`: `collect` (explicit-path
    in-image sha256sum over one ssh) and `bind` (checkpoint-config G8
    fields, reference-manifest `config_sha256` cross-check, per-rank
    descriptor env/argv for both engines, pins + provider reports,
    inventories, candidate build manifest, plan source-contract anchor
    verification). 15 tests.
  - Probe: `probe_wiring.py` + `probe_sitecustomize.py` implement the
    registration's overlay-author obligations in full (side-effect-free
    4-tuple contexts, byte-clone + buffered writes, flush at request
    transition/exit/SIGTERM/violation, engagement counters, install
    receipts, fail-closed discovery/coverage/source-preflight);
    `probe_package.py` assembles the evidence-root overlay against the
    registration's frozen trio; `probe_records.py` produces the attempt
    attestation, profiler binding and NCCL prior/live records. 66 tests.
  - Records: package `9068c0972f0995095b5aa5f8734f3a6a75ae0be7ee7b2e686a4679a45a07ece9`
    at `$EV/probe-runtime-mechanism-20261014/probe-overlay/` (21 files,
    re-verified after assembly; v1 superseded for copied lineage
    `.pytest_cache` junk, preserved untouched), `nccl-prior-binding.json`,
    `attempt-attestation.json`, profiler invocation SHA256
    `fdb01a69956577795cbbed79e41b9ae49987446deb9c3d350b71aa07ef4f3fe2`;
    the registration's owed section records these and what remains
    pre-attempt (profiler version/availability, NCCL live re-bind, stamping).
- Why: operator-approved scope 2026-10-14 (full observer glue; tools +
  prior NCCL evidence only, no host ssh while experiments may run).
  Independent subagent review: OK-with-notes; findings 1–8 dispositioned
  in this batch (transfer exit-code enforcement, pytest_cache exclusion,
  unpinned-report KeyError, G8 symmetry requires, dead-fallback cleanup,
  absent-backend emit-once, fixture dead branch → live coverage).
  Full CPU suite green (695 tests + 280 subtests), license gate clean,
  all new files ≤500 LoC. Commit refs: see git log for this batch.

## 2026-10-14 — runtime-mechanism probe: CPU-first harness frozen (major)

- Choices: (a) write the decoder/observer harness inside the GPU window,
  (b) finish and freeze it locally first, as the probe plan requires
  before either Spark is obtained.
- Decision: **(b)** — implement now: `cache_manifest()` in
  `tests/exl3_validation/linear_launch.py` (EX3ATUNE final record map +
  raw/canonical SHA256s), `reference_router_capture.py` (supported
  `set_capture_fn` hook + `_compute_routing` wrap),
  `reference_collective_trace.py` (dispatch/backend wiring with the
  zero-backend coverage contract), plus 63 CPU fixtures covering the
  plan's five required fixture classes. Independent review: GO-with-notes;
  all P2 notes dispositioned (SSOT dedup, `group_reason`, setattr
  rollback, capture-callback zombie cleanup); full CPU suite and license
  gate green.
- Registration: attempt order `off01,on01,on02,off02,off03,on03`, profile,
  engagement rules and harness SHA freeze in
  [the probe registration](docs/design/exl3-runtime-mechanism-probe-registration.md).
  Overlay wiring, profiler/CUPTI binding and NCCL hashes remain owed
  before Sparks access.
- Why: every GPU window was paying harness-writing overhead; the plan
  forbids Sparks access before a local freeze, so this work removes that
  cost from the critical path.

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

## 2026-10-13 — KV column-codec experiment: first-pass stage (major, new)

- Decision: First pass implements Stage 0 crate (`crates/atlas-codecs/`) + minimal Stage-1 spill header plumbing (`kv_spill.rs` block header with explicit `codec_id`). Full ALP/FPC/ZXC/GCD left as config stubs (enum variants, `unimplemented!`) — user selected full catalog but KISS/core-first demands working pipeline before catalog expansion.
- Reason: Plan §5 stages; plan §2 C4 (KISS); H1 testable with core set (§4 matrix); full catalog deferred (§8).
- Reference: `docs/plans/atlas-kv-column-codecs-plan.md`; new crate `crates/atlas-codecs/`; `crates/spark-runtime/src/kv_spill.rs`.
- Status: in-progress.


## 2026-10-13 — Plan external repo additions (minor, new)

- Decision: Added 13 repos to `docs/plans/atlas-kv-column-codecs-plan.md` §7 (external references / repos to test and evaluate). Each tagged by category (GPU kernel, codec library, KV-cache framework, lossless compression, non-codec scope-clarity note for `openzl`).
- Reason: Plan §7 is the SSOT for external comparison; measurement discipline §3 requires bit-extract roundtrip + ratio + encode/decode GB/s for any repo claimed; no repo provides implicit codec-id default — PCND preserved.
- Note: `facebook/openzl` is not a compression codec (zero-knowledge proofs); included at user's instruction for evaluation scope clarity, with explicit non-codec tag.
- Status: added; evaluations deferred until Stage-0 exit criteria pass (no code changes from this edit).
- D6 2026-08-13 load-progress logging: shared `atlas-core/src/load_progress.rs` (timer thread, `update()`, `tracing::info!`), env `ATLAS_LOAD_LOG_INTERVAL_SECS`, wired in server `serve_load.rs`, model `factory/build.rs`, storage `expert_pack.rs`. Reason: user requested info-level load-progress logging configurable per 5s; shared helper avoids duplication across 3 load paths. Status: done; deepest per-loader loop call-sites still need `update()` for file-level precision.

D1 2026-10-15 done (config verified drowzeys=base, mimo_v2, 48-layer 9/39 fused_qkv). D2 2026-10-15 done (B=new loader mimo_v2.rs). D3 2026-10-15 deferred text-only. D4 2026-10-15 done (slug mimo-v2.6-flash-rl). D5 2026-10-15 partial (MoE reuse documented, attention kernel deferred).

## 2026-10-16 prefill/MoE group — capture-diagnose then optimize (user ask)
- Source ref: continue.md §9 (prefill ~18–19 tok/s, 128-token chunks, unoptimized); todo.md P (P2 grouping done; M2 padding / TTFT open).
- Capture evidence base (user-selected): prefref/indexer snapshots from atlas-perf (`exl3-indexer-localization-result-20261015.md`; `research/flash-kda-prefill/`).
- Component chosen: prefill / MoE grouping (`forward_prefill.rs`; `ep_prefill.rs`).
- vLLM contract (`exl3-vllm-reference-contract.md` rev 2026-10-12, Luna subagent + parent) defines SM120 prefill as packed-cache, `fp8_ds_mla`, no dense-prefill bypass, head `pad=False`. Capture divergence localized below 2051 tokens removed; divergence above is indexer order (same pool set, permuted), not membership — so correctness fix = deterministic sort before expansion, not arithmetic change.
- Optimization path (lazy / minimal): (a) sort selected pool IDs before sparse-MLA expansion to remove order-sensitive divergence; (b) close M2 padding + short-prefill (<8 tokens) fallbacks per todo P open items; (c) evaluate FlashKDA adapter (`research/flash-kda-prefill/`, 3.38–3.62x on H32/D128 2048–4100) only after (a) verified — no new runtime dependency by default.
- No sources edited in this turn; only analysis + decision record (per ponytail / minimal-edit rule).

## 2026-10-16 implementation — (A) indexer sort (commit 92d82a49)
- Branch: followup/prefill-moe-group (worktree ../atlas-prefill-moe); parent perf/decode-vllm-parity-20261015 @ 794e9400.
- Edit: crates/spark-model/src/layers/ops/glm_indexer_dynamic.rs (marker at topk_expand_dynamic → before sparse accumulation); docs/prefill-idx-sort-verify.md (protocol).
- Capture-verify defined: label prefill-idx-sort-<n>, bench-lock flock /tmp/atlas-bench.lock, compare index buffer byte-id across 3 pairs x 2 ranks @ 2,052-token row; acceptance 0 diff.
- No speed claim (measurement-discipline: ≥3 same-binary repeats required). No device-sort primitive yet — marker only; actual sort (thrust/gpu.sort) deferred to verification pass.
- Not edited: shared perf branch; not committed to sparkglm/installable-20260924; not run on GPU (bench-lock not acquired this turn — needs scheduling with other agent).

## 2026-10-16 verification attempt (commit 46393041) — NOT COMPLETE
- Sort implemented (glm_sort_pool_indices). Before/after protocol defined (prefill-idx-sort-01/02/03).
- GPU runs: BLOCKED — spark0 unreachable (ssh fail); bench-lock /tmp/atlas-bench.lock held by remote process; local host has no CUDA/nvcc.
- Workload stop (serve_perf.py): ATTEMPTED — failed (host unreachable).
- No fabricated results. Actual 3-repeat before/after comparison requires Spark execution under bench-lock; commands recorded in docs/prefill-idx-sort-verify-attempt.md.
- Next: operator/agent with Spark access runs the 3 repeats; if pass, merge 46393041; if fail, revisit attribution (fused MoE / indexer per HSL result).

## 2026-10-16 local cargo target cleanup (minor, operational)
- Decision: removed `target/` for nominator and the five `atlas-pi-*` qualification worktrees (14.0 GB). Rule applied: delete only caches with no file written in >14 days. `atlas-perf/target` was explicitly protected — the 2026-10-16 fan-out agents compile into it. `atlas/target` (79 G), `atlas-f4/target`, `atlas-port/target`, `aibb`, `arc-dashboard` and `~/.cargo/registry` were left in place.
- Reason: root fs was at 97% (35 G free), which blocks image builds and dev builds. Operator chose "stop here" rather than remove the 79 G main-repo cache, because `atlas-prefill-moe` has no target of its own and the live prefill agent is the likely user of it.
- Reference: disk 35 G → 94 G free (97% → 90%). Status: done.

## 2026-10-16 code-only fan-out, nine lanes (major)
- Decision: fan out nine code-only lanes on isolated worktrees off `perf/decode-vllm-parity-20261015` @ `794e9400` (Q5a, Q5b, R-a…R-f, R-e, repo-health), each with its own branch and strict file ownership, then fold each onto the perf branch. Prefill/MoE/indexer excluded (another agent owns it). No GPU, no bench lock, nothing pushed, nothing on the shared branch.
- Reason: operator asked to fan out; disk at 97% ruled out per-agent target copies, so all cargo builds shared `atlas-perf/target` (serialized on cargo's lock). Nine writers on one tree would be unsafe, so "same branch" was satisfied by one integration line (`perf/decode-vllm-parity-20261015`) with per-item branches merged by a single writer.
- Landed: Q5a cublasLt router pin (`cublaslt.rs` 497 LoC + `router_pin_tests.rs`, 5 CPU tests), Q5b TP all-reduce fail-closed (`impl_a1.rs` + 5 tests), R-c overlap capture bug fixed (`e1e9487c`, lazy per-stream cuBLAS handle + 32 MB workspace allocated inside a live graph capture → status 901 → graphs silently off), R-f bench log archive (4 tests), repo-health typos (14 findings), and four reports (R-a, R-b, R-d, R-e).
- Owed (GPU): router repeat-bitwise proof, R-c capture-propagation proof, R-a poison-test matrix runs, R-b's requalification of the perf→shared merge.
- Reference: perf branch folds `f8ecdfb4`…`6d4ab509`; reports in `docs/design/{ra,rd,rb,re}-*20261016.md`, `docs/design/rc-overlap-robustness-20261016.md`. Status: folded; verification per item on the folded branch.

## 2026-10-16 supervisor decisions for R-b (minor)
- Decision 1 (`.licenserc.yaml` / `_typos.toml` for the vendored MIT exllamav3 tree): `.licenserc.yaml` is supervisor-owned and lands separately on the **perf line only** — the 12 `exllamav3_*.cuh` files do not exist on `sparkglm/installable-20260924`, so a mainline paths-ignore would be dead config. `_typos.toml` is owned by the repo-health lane; all 14 findings were real misspellings, so no exclusion was needed. Do NOT restamp the MIT headers.
- Decision 2 (`ATLAS_EXL3_COOP_MOE_MULTIROW`): **C now** — keep default-off and add a `warn_once` ("unqualified in-model; speculative verify only"), with no in-model bit-identity measurement at merge time. **A owed** — confine the dispatch to verify rows, owned by the perf/F4 lane, because it is a behaviour change in a 990-LoC over-cap file needing a one-variable A/B with ≥3 repeats under `flock /tmp/atlas-bench.lock` (Sparks locked by a live server). B was rejected: it asserts an unmeasured path is fine.
- Reason: R-b asked for two decisions outside its read-only ownership; both root-caused to pre-existing perf-line CI breakage rather than fan-out work.
- Reference: `docs/design/rb-perf-merge-review-20261016.md` §9.2–§9.4. Status: recorded.

## 2026-10-16 prefill: one kernel is 93.8% — target it, not the rest
- Measured (p13trace2, ab_full flags, 1024-tok prefill, rank 0): `exl3_moe_mul1_matmul_precision_mma_split` 40.845 s / 93.8%; all other kernels sum to 2.6 s. Total GPU 43.44 s.
- Baseline prefill rate (3 repeats, unprofiled, p13base): 761 tok -> 42.6/42.8/42.7 s; 108 tok -> 6.07 s => **17.8 tok/s**.
- Dense packed linears already run the vendored coop GEMM (16-row M passes, `dq8` decode); `exl3_mul1_matmul_mma` did not execute. So the audit's dense M-tile and native-decode proposals are moot for prefill; keep them only as decode-path notes.
- Decision: implement grouped multi-row routed-MoE for prefill (per-group thin/fat numerics, M-tile chunking, vendored inner with round_partial=false for fat) behind default-off `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW`. Expected ~8-14x on 93.8% of prefill.
- Evidence: `prefill-investigation-20261016.md`, `p13-kernel-audit-exl3.md` (subagent dc8bd541), `p13trace2-20261016/.../nsys-stats.txt`.
- Delegation: worker subagent on branch `followup/p13-fat-moe-multirow`; bit-identity gated; parent runs the model-level A/B.

## 2026-10-16 prefill — VALIDATED: grouped fat-MoE multirow, 12.4x, bit-identical (major)
- Commit `b52aef80` on `followup/p13-fat-moe-multirow`; flag `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1` (default off).
- Same binary `spark.p13fat` (sha256 2287583360...c32e378), both nodes, bench lock, 3 repeats/workload.
- 761-token prompt: 42.25-42.72 s (flag off) -> **3.427/3.410/3.422 s** (flag on) = **12.4x**.
  108-token prompt: 6.07 s -> **0.541/0.637/0.540 s** = 11.2x. Prefill **17.8 -> 222 tok/s**.
- `output_sha256` identical between flag-off and flag-on (`1c4328172e81`, `80b1eb5fadf8`) => bit-exact.
- Engagement: flag-off logs the `fat (prefill) routed launch` fallback; flag-on logs none.
- Ran under the 93.8%-of-prefill measurement in `prefill-investigation-20261016.md`; the target kernel
  was `exl3_moe_mul1_matmul_precision_mma_split` (40.85 s of a 43.44 s prefill trace).
- Next (follow-ups, not done): decide whether to default the flag on (needs the usual default-flip
  neutrality proof on a fresh image); re-trace the new prefill to find the next bottleneck; fix the
  latent `ATLAS_EXL3_COOP_MOE_MULTIROW` lock-buffer under-allocation noted by the worker.

## 2026-10-16 prefill — final: 42.71 s -> 2.72 s (15.7x), bit-identical (major)
- Kernel change `b52aef80` (`ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1`): 42.72 -> 3.42 s (12.4x),
  hashes unchanged.
- Serve-flag change (`--max-prefill-tokens 128 -> 512`, descriptor profile): 3.42 -> **2.72 s**
  (1.26x). Combined **15.7x**, 17.8 -> **279 tok/s** on a 761-token prompt. Median of 3 repeats
  each; all runs same binary `spark.p13fat` (sha256 2287583360...c32e378).
- `--max-prefill-tokens 4096` crashes with `EP peer lost` (only 6.05 GB free after layer
  construction): 512 is the ceiling under `--gpu-memory-utilization 0.84`. Raising it needs a
  prefill-scratch memory plan.
- Correctness: `output_sha256` equal flag-off vs flag-on, so both changes are speed-only.
- Caveat recorded: the first chunk A/B was void (ab_full.sh resolves its repo root from `$0`, so it
  used the fatmoe worktree's unpatched bench_tp2.py and kept chunk 128 in every arm); fixed and re-run.
- Owed: same-hardware vLLM prefill measurement (`glm53-exl3-head`/`-worker`, env
  `MAX_NUM_BATCHED_TOKENS=7168`); default-flip proof for the kernel flag; dense `TILESIZE_M>16`
  (audit P1a) now that the MoE side is fixed.

## 2026-10-16 prefill — long-prompt A/B and a correction (major)
- Correction to the entry above: `--max-prefill-tokens 512` is **not** speed-only. Its output hash differs from chunk 128 (`07355bd7f0ef` vs `1c4328172e81` on the 761-token prompt; `e779a96788f1` vs `671ef63c142a` at 2,943 tokens). Chunk size changes per-expert row counts per launch, and so the thin/fat profile split. Treat it as a numerics-changing serve setting (label, qualify).
- The kernel flag is confirmed at ~3k tokens. Same binary `spark.p13fat`, 3 repeats per arm, chunk 128: 2,943-token prompt 165.80 s → **13.79 s (12.0×)**, 761-token control 42.20 s → 3.46 s (12.2×). Hashes are identical across arms, so the flag is bit-exact. Engagement is confirmed from logs, and the server-side TTFT agrees with the client timer.
- Chunk sweep, flag on, 2,943 tokens: 128 → 13.79 s, 256 → 11.76 s, 512 → 10.68 s, each with a different hash.
- Evidence: `perf-ab-20261015/p14L-{off,on}-*`, `p14Lchunk{256,512}-*`; write-up `docs/design/p14-prefill-long-ab-20261016.md` on `followup/p13-fat-moe-multirow` @ `a27d0fb8`.

## 2026-10-16 post-merge re-baseline and server upgrade (major)
- Decision: serve from the merged main binary (`spark.main-34d94f34`, sha256 `99d8655a…`) with `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1` added to the server's exact flag list.
- Evidence (same session, 3 repeats each, `perf-ab-20261015/m1-*`): merged vs old binary, exact stack + fat-MoE: all hashes identical (`80b1eb5fadf8`, `1c4328172e81`), decode 45.84 vs 45.80 ms/token. Merged, fat-MoE off: `80b1eb5fadf8`. Server numerics flags (SHARED_PACKED + FP8_SPLIT), fat-MoE off vs on: 761-token prefill 42.83 → 3.56 s, decode 42.62 vs 42.65 ms/token, hashes identical (`00fc1b2b6bd0`, `564e84bc1f25`).
- Server smoke after restart (single run): 2,485-token prompt TTFT 12.4 s cold, 0.41 s on a prefix-cache hit; recall correct.
- Rollback: set `BINARY` back to `spark.5e1674ba` in `serve_perf.py` and drop the flag. Status: done; unqualified (M1 still open).

## 2026-10-16 F4 campaign + CPU fan-out (parent session, branches pending merge)
- F4 k sweep (`f4c-*` runs, main `172ce45a` build `50300beb…`, n=3/arm, one
  variable): adopt **k=3** as the lane default (30.4/28.1/26.2/25.4 tok/s,
  1.22–1.42× serial); k=7 loses to serial on nothink_1024. `fe02685e` KDA
  one-launch verify adopted (GPU bit-exact, hash `140b56e3ecb9`).
  `ATLAS_EXL3_VERIFY_PROFILES=1` stays OFF (flat-to-worse at k=7, labelled
  numerics-changing).
- F4 code behind default-off flags on branches: verify graphs keyed by
  `(slot, r)` (`f4/verify-graphs-by-r` @ `9b05883d`,
  `ATLAS_GLM_EXL3_DFLASH_VERIFY_GRAPH=1`); thinking-budget enforcement
  (`f4/thinking-budget` @ `b3efd501`…) is a behavior fix (mirrors serial
  governor semantics; no flag — the boundary behavior IS the contract).
- S-a: the F4 contract validation was wrong, not the test — `ensure!`
  scoped to enabled contracts (`57d268fd`); review found
  `check_f4_contract` unreachable in production → wired into
  `validate_config` (`35cb3a7d`). Strict-PCND (require the key) NOT done —
  recommendation only.
- S-f salavages all behind default-off flags (PCND): WMMA GEMV rewritten on
  `mma.sync` (draft had OOB loads/stores), GDN wy3 rewrite (review blocker:
  base twins didn't chain the update recurrence — fixed `8b94e099`),
  BF16 vocab gather. MiMo loader: DEFER recommended. Async argmax: design
  note only (ties to I-d).
- Serving stack stopped for the campaign at operator direction (stop and
  leave down); teardown receipt `$EV/f4-campaign-20261016/`.

## 2026-10-16 F4 validation outcomes (parent session)
- Verify graphs: whole-body capture rejected on the hash gate; adopt REGION
  capture (`f4/verify-graphs-by-r` @ `698cd569`) — replay-proven regions
  only (KDA forwards + tail), MLA multi-row prefill stays eager (dynamic
  host inputs are the divergence source). Parity + engagement both proven
  (`f4c-vg2-*`).
- Thinking budget: adopt commit-side truncation (`f4/thinking-budget` @
  `91e25b06`); validated inert-on-step-end / fires-mid-step with telemetry
  positive control (`f4c-bgt63-*`).
- KDA `[V,K]` state layout (I-b): measured −27.8% bit-exact standalone;
  production flip DEFERRED to operator (cross-cutting: all KDA state
  readers/writers must transpose together).
- Engagement telemetry is now standard for default-off lanes (both F4
  lanes shipped counters after unobservable-engagement failures —
  measurement rule 6).
