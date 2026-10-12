# GLM-5.3 EXL3 — remaining high performance inference work

## Next batch (staged 2026-10-12)

Ordered execution; sequencing SSOT:
[docs/plans/q2-norm-causality-plan.md](docs/plans/q2-norm-causality-plan.md).
The ref13 held-out diagnostic COMPLETED (verdict: reference-internal bimodal
chaos; both Sparks idle again with post-stop receipts) and the
neighboring-types matrix is instantiated (`q2-neighbor-matrix-20261012/`,
49 rows). C1 preparation is in flight locally.

1. **C1** — local `o_proj` replay before TP (original vs reference-exact vs
   v24-corrected norm output; each engine vs its own captured projection
   first). **Sparks free since ref13 finished 2026-10-12; C1-prep running.**
2. **C2** — layer-0 norm-output intervention on both ranks (o_proj → routing →
   MoE → final logits against the failing rows).
3. **L1 escalation** — for `longer`, trace complete mHC residual streams and
   L1 KDA between matching L0 outputs and differing L1 FFN inputs, before
   broad late-layer bisection.
4. **G2–G7 fallbacks** — G8/G9 runtime + binary binding precedes each
   affected replay.
5. **Neighboring-types matrix + repeat Rust/vLLM/source review** — HARD
   PREREQUISITE before any shared-code fix (matrix instantiated:
   `q2-neighbor-matrix-20261012/`; the repeat Rust/vLLM/source review on
   actual candidate artifacts remains open).
6. **Production v24 gated-norm fix** — only behind causality evidence, the
   plan's step-5 qualification obligations, and a fresh frozen Q2 run.

**Operator decision required (from the ref13 investigation):** the
pre-registered must-pass held-out control fails for the reference engine
itself (3/5 pure reference draws: 2 catastrophic-row at `longer`/r0, 1
rank-only) — the rule is structurally unpassable as written whenever a draw
lands in the failing mode. Options: (a) pursue reference-faithful numerics
and keep the gate as registered; (b) author a NEW pre-registration (any
redesign is an operator act, never unilateral). Evidence:
`q2-ref13-investigation-20261012/`.

**Current Q2 review after `614a2e24`:** norm correction is proven only on
the 16 captured fixtures; downstream causality and qualification remain open.
Use the revised [causality plan](docs/plans/q2-norm-causality-plan.md) and the
active follow-ups at the end of this file. The dated checkpoint below is
historical: later binding/replay commits and stopped-host receipts supersede
its documentation-only HEAD and unverified-host statements. This offline
review did not inspect live hosts.

Reviewed 2026-10-11 against HEAD `f0ba28fd` (documentation-only changes after
tested source `ee5eaf0f`). **Q2 statistical gate: candidate FAILS** (9 fresh
references; held-out reference also fails one row) — see [continue.md](continue.md).
Merged seed/stop/API code has a clean rebuilt-image neutrality pass on all 32
frozen cases (`cand-merged-e333-01`); its focused Q3 run exposed the
documented first-thinking-token grammar bug. Image `ee5eaf0f` now has **full
short16+long byte neutrality** (`cand-grammar-ee5e-02` vs `cand-fp8decode02`,
both archives byte-identical) and a **15/15 focused Q3 pass on GPU** proving
the `19551fc6` JSON-opener fix and the `8334ab33` completion-identity fix;
exact terminal-EOS-ID verification is recorded untested (harness skip).
Host state 2026-10-12: `gate-ref-0/1` are the **IN-FLIGHT ref13 diagnostic
reference engines — do not stop them**; coordinate with the ref13 track
(`q2-ref13-investigation-20261012/`) before any other GPU work. The incomplete
`cand-grammar-ee5e-01` attempt is preserved
and unreused. Q2 investigation report (read-only, local):
`q2-investigation-20261011/report.md` under the durable root — candidate
excess dominates; MLA multi-key causal link still unproven. Reviewed localization
finds layer-0 gated-normalization differences and layer-3 routing membership
changes; original same-input GEMM and dominant MoE-amplification claims were
invalid. See the corrected report and `analysis/review-validation.json` under
`q2-diagnostic-20261011/`.
Q4 and F2/F3 harnesses remain GPU-unvalidated. R2 records remain unmerged on
`pi/r2records-20261011` (benchmark boundary edits invalidate required records).
Original roadmap review was against `11671adf`. Native TP=2 / EP=1 boot and generation
are achieved. Full numerical qualification, a repeated performance baseline,
and the expanded serving features below remain open. This is the authoritative
work checklist; [continue.md](continue.md) holds the launch profile and evidence.

Order: **Q0/Q1 → Q2–Q4 → measured optimization → feature qualification → R1/R2**.
P0 diagnostic profiling can proceed while parity is investigated, with the
candidate labeled unqualified. P1–P6 order is provisional until P0 identifies
costs. Each item should be a focused change with its own acceptance evidence;
this roadmap spans multiple PRs. Check an item only with a run/artifact ID tied
to the source and profile that passed.

## Completed — evidence pointers

One line per completed item, pointer only; receipts and full detail live in
git history and under the durable evidence root
`/home/colum/Workspace/atlas-exl3-validation-20261010` (row index in
[continue.md](continue.md) §Evidence index). Do not delete these paths.

- Native EXL3 mul1 loading, packed kernels, TP=2 / EP=1 KDA/MLA/MoE/mHC
  integration and launcher; five sequential nonstreaming chat smoke receipts
  (`atlas-launch-smoke-20261010/smoke.json`).
- Shared routed MoE workspace and ordered teardown; rank-local GPU guard and
  A→B→A reuse proof on both hosts — does not qualify overlapping streams
  (`arithmetic-audit/native-aux-scratch-gpu-20261010/`).
- Opt-in FFN stage profiling; coherent cuBLAS 13.1 classic/Lt profiles; merged
  KDA auxiliary with recorded in-situ M1/M5/M128 comparisons
  (`native-aux-insitu-20261010/`).
- KDA reference-policy kernels and dispatch (`72d64e4c`, `069e4351`),
  both-rank GPU stage/state oracle and layer 0–2 in-situ byte comparisons
  (`arithmetic-audit/kda-reference-stage-oracle-20261010/`,
  `kda-fla-candidate-20261010/`); Q1 stays open until its full-model Q2
  dependency passes.
- 2049-token FP8 chunk-tail crash reproduced, reader selection repaired
  (`fb4f4b41`), clean-built candidate completed that failing request on the
  real TP pair (`statistical-gate-20261011/resume-diag01/`).
- Full 23-shard rehash on both hosts, final-image scratch replay/guards and
  both-rank serving-process cuBLAS-map attestation
  (`checkpoint-rehash-20261011/comparison.json`,
  `fp8decode-candidate-20261011/scratch/`).
- Final-image launch smoke (2026-10-11 05:28 UTC), 5/5 requests
  (`fp8decode-candidate-20261011/final-image-smoke/smoke.json`).
- `cand-fp8decode02` full collection complete, short16 16/16 + long 16/16
  (`statistical-gate-20261011/cand-fp8decode02/`).
- Correct OpenAI chat stream usage order (`4b49a9c4`) with CPU red/green
  encoder evidence; rebuild and live validation remain part of Q3.
- **Q0 COMPLETE** — reproducible coherent-13.1 recipe in-tree (`a6f64c46`,
  `59b0e113`); clean no-cache image `src-8b6b47f14d5b-rust1.97.1` /
  `sha256:e744b81d…` built from `fb4f4b41` and loaded on both nodes; final-image
  smoke 5/5; all 23 shards rehashed on both hosts
  (`fp8decode-candidate-20261011/`, `checkpoint-rehash-20261011/comparison.json`).
- Ablation summary validity repaired — `build_summary` / `enforce_summary_gate`
  in `tests/exl3_validation/gated_norm_ablation.py` retain all engagement
  controls, set `interpretation_blocked=true` on any absent/failing control and
  fail the command on a blocked summary (commit `9ff39728`). Attribution: the
  summary/exit logic is in `gated_norm_ablation.py`, **not**
  `gated_norm_variants.py` (variant derivation only).
- Ablation fixture validation repaired — `validate_fixture` in
  `tests/exl3_validation/gated_norm_ablation.py` enforces schema/rank/
  invocation, recorded + actual file hashes, cross-engine operand equality and
  effective epsilon bits, with failure-asserting negative tests (commit
  `c71bb557`).
- D4 — installed head padding resolved: plugin explicitly uses `pad=false`;
  prior head-M1 component receipts confirm actual rows=1
  (`linear-head-m1-production13-rank{0,1}-results/`); current worker dispatch
  and failing-input parity remain G2 work.
- G8 checkpoint evidence — exact config bytes recovered from
  `vllm-reference.json`; SHA256 matches all 24 registered suite manifests; layer
  map, index_topk=2048, NoPE, clamp=10 and router dtype are available.
- Gated-norm ablation attribution + minimal fix (2026-10-12): `v24-minimal`
  (rsqrt -> sqrt.approx.ftz+div.full; divide -> reciprocal multiply) reproduces
  the reference byte-exactly on all 16 fixtures
  (`q2-norm-ablation-20261012/analysis-20261012.md`); the production kernel
  carry-over is staged as Next batch item 6.

## Deferred hardening (harness repair review)

Minor, non-blocking findings from the independent repair review
(`q2-norm-ablation-20261012/harness-repair-20261012/review-summary.md`):

1. No test drives `main()` wiring of `validate_fixture` /
   `enforce_summary_gate`; both calls are assured by inspection only.
2. `red-defect1.txt` failure line numbers are not byte-identical to the final
   tests (red ran against a slightly earlier draft; constructs match
   semantically).
3. Finding-3 attribution gap: helper-module hashing is present in
   `gated_norm_ablation.py` but undeclared alongside findings 1–2 in the repair
   report.
4. Untested validator branches: manifest key-set, file-identity/dtype,
   shape-record and `_f32_bits` rejection paths.

## Q — qualification and reproducibility

- [ ] **Q1 — Implement and validate the EXL3 reference-policy KDA core.** Follow
  [T2–T8](docs/design/exl3-kda-reference-core.md): M5 prefill, exact PTX operation
  probes, M128 and residual closure, F32 recurrent decode/state, then explicit
  EXL3-only policy/dispatch. Reject incompatible flags/geometries; preserve
  existing dispatch when unset. Pin reference Triton cache/PTX choices; add
  fixtures for ≥3 chunks, chunk tails, convolution history and repeated decode.
  Restore the external oracle scripts/fixtures and record missing-fixture skips
  as untested. **Done when:** both ranks' stage/output/state comparisons pass
  byte gates at M5/M128/decode and multi-chunk boundaries, negative controls fail,
  and the rebuilt image passes Q2. The CPU M128 residual counts are evidence,
  not permission for a CUDA mismatch. Diagnose later divergences if they appear.
  *Status 2026-10-11:* kernels + dispatch done (`72d64e4c`, `069e4351`); every
  stage, carried state (3 chunks + tail, 4 decodes) and the composed pipeline are
  0-mismatch vs the GPU oracle on both ranks (`tests/exl3_validation/kda_fla/run.sh`);
  historical in-situ layer 0–2 byte equality is scoped to the earlier M5 request.
  Reviewed M10–13/M36–39 captures have matching KDA core/g1/g2 but one-ULP
  gated-normalization differences; bind constants/weights and isolate that stage.
  Open: failed Q2 dependency, a performance pass (correctness-first kernels,
  ~12 launches per KDA prefill), and qualification across long-prefix boundaries.
  Full short16+long collection now exists; its registered gate fails.

- [ ] **Q2 — Close full-model numerical and repeatability gates.** Rerun frozen
  short16, long-prefix/chunk-boundary, persistent decode and raw-logit comparisons
  on the exact candidate image. Apply the pre-registered statistical gate below;
  retain frozen abs/rel/KL/CE/top-K reports as diagnostics and keep the required
  negative controls. Retry the indexer scale-boundary oracle with the exact
  frozen reference source overlay; its latest attempt failed before GPU execution.
  Investigate reference repeat variation at routed combine independently and
  check candidate repeatability. **Done when:** all required candidate gates pass
  on both ranks, with raw tensors/provenance; reference variability is explained
  or explicitly retained as an unresolved constraint, never used to waive failure.
  *Status 2026-10-11 05:30 UTC:* the user chose a statistical gate; it is
  pre-registered (`docs/design/exl3-statistical-gate.md`, `3cf66612`, reviewed
  by Fable 5.1; candidate-profile addendum `87e4b08b`). Historical M5
  localization found a layer-3 MLA core-output divergence: the reference silently uses an
  `fp8_ds_mla` latent KV cache. `--kv-cache-dtype fp8_g128` reproduces its
  single-key rows; the candidate profile now uses it. Layers 0–2 and
  layer-3 projections matched in that earlier M5 capture, not all gate rows. The resumed run reproduced the candidate's
  2049-token crash: its one-token tail entered MLA decode without an enabled
  FP8 reader. `fb4f4b41` selects and initializes that existing reader for
  serial TP2/EP1. The repaired final image completed that request.
  **Candidate run 1 (`cand-fp8decode02`) collection complete** at 05:11 UTC:
  short16 16/16, long 16/16, manifests present. Evidence archived to
  `statistical-gate-20261011/cand-fp8decode02/`.
  **Batch 2 completed 08:47 UTC — gate FAILS:** candidate T 7.20 vs refs
  ≤5.53 (p=0.1) with 3 catastrophic short16 `longer` rows; cand03 byte-identical
  to cand02; candkda0 (MDE) T 7.68; held-out ref13 fails one row (reference
  bimodality at short16 `longer` row 0). Reports in
  `statistical-gate-20261011/final/`. The earlier MLA-port versus gate-redesign
  choice is not supported by a localized dominant cause; isolate the newly
  reviewed KDA/routing differences first. Keep the registered gate unchanged.
  Read-only investigation (`q2-investigation-20261011/report.md` under the
  durable root) recomputed T=7.20 from raw artifacts: candidate excess
  dominates (29× the worst reference on short16 `arithmetic`, 3.6× `longer`,
  2.1–2.2× on 2047/2048) and the FAIL is robust to the ref13 row-0 anomaly;
  MLA multi-key ulp claims verify from raw bytes, causal link unproven.
  Diagnostic-plan review: add `longer` prefixes 36–39, one long-suite case,
  a layer escalation path, and a build-identity prerequisite before GPU runs.
  *User authorization 2026-10-11 18:30 UTC (`extended-plan.json`):*
  extended diagnostic authorized
  (`+longer`, escalation path, build prerequisite — platform = neutrality-proven
  `ee5eaf0f` image) plus offline stage-2 propagation analysis; the
  reference-faithful-MLA vs gate-re-design decision remains open.
  *Extended diagnostic executed (session 3b):* attempts `cand-q2diag-ee5eaf0f-02`
  + `ref-q2diag-ctxv4diag-03`, dumps archived under `q2-diagnostic-20261011/`.
  **Reviewed localization** (`q2-diagnostic-20261011/analysis/localization-report.md`):
  KDA core/g1/g2 match all 16 rank/prefix pairs. Gated normalization differs
  by max 1 BF16 ULP: 1 element per arithmetic prefix on rank 1; 6 per longer
  prefix on rank 0. Longer sampled L0 outputs match, but internals do not.
  Arithmetic FFN inputs already differ, so dense gate_up is not same-input;
  reduced o_proj includes a rank with differing direct input. Correct i32
  routing shows expert-membership changes at L3. Correct post-TP MoE pairing
  gives p99 64–71 ULP; the reported 1507 used incompatible boundaries.
  **Next:** fix analyzer dtypes/signed-ULP/mappings with controls, bind/replay
  gated-normalization constants and dynamic inputs, then isolate local o_proj
  and dense projections with identical inputs. Measure effects on routing and
  full-model logits before assigning a dominant cause. Extend L1 KDA/later
  bisection for residual divergence; the 2048 long-suite case was excluded
  from this diagnostic and remains unlocalized. Original analyzer/results
  remain preserved; `analysis/review_localization.py` reproduces corrections.
  Semantic must-fail knobs exist (`ATLAS_EXL3_GATE_PERTURBATION`, merged), but
  GPU controls are still owed. Rebuild/register only when changing runtime;
  current HEAD after `ee5eaf0f` changes documentation only.
  Both-rank failure logs and raw evidence are preserved in
  `statistical-gate-20261011/resume-diag01/`. The earlier ref01/ref02 enabled
  intermediate tensor captures and are diagnostic only. A collector-only
  reference profile is pinned before fresh collection; the gate also
  checks environment/build/hook identity within each engine.
  The completed batch already ran `statistical_gate.py`. Rerun registered
  candidate/control collection after a justified arithmetic change; do not
  recollect unchanged references simply to seek a passing verdict.

- [ ] **Q3 — Qualify usable API behavior.** Extend beyond the launch smoke to
  OpenAI chat/completions and Anthropic, streaming/nonstreaming, reasoning on/off
  and budgets, stop/EOS and token accounting, structured output, tools (single,
  parallel and dependent result follow-up), multi-turn and sampled requests.
  Check `poolside_v1`, template-seeded thinking, special-token stripping and
  finish reasons against this checkpoint; do not copy the reference's `glm47` /
  `glm45` parser settings without evidence. Exercise client cancellation,
  timeout and the next request after each. Reuse
  [behavior_protocol.py](tests/exl3_validation/behavior_protocol.py) and add the
  missing surfaces. **Done when:** behavior receipts and assertions pass without
  leaked state or protocol drift; fix Atlas failures
  even when the reference also fails its own behavior suite.
  *Status 2026-10-11 16:47 UTC:* first-token/grammar/n>1/MTP/Responses seed,
  blocking/multi-token stops, token-ID stops, min_tokens, stop inclusion,
  Anthropic stop mapping and the live harness are merged. `cand-merged-e333-01`
  validates 11/15 focused live cases on the exact rebuilt image, including
  seeded chat/Responses repeats, stop surfaces, streaming chat and Anthropic.
  Its four preserved failures exposed two harness issues (timing metadata in
  completion identity and a terminal EOS ID not exposed by design) and the
  first-token grammar bug. `8334ab33` compares stable usage counters;
  `ee5eaf0f` skips only unobservable terminal-ID verification, retaining wrong-ID
  and counter-drift controls. `19551fc6` pauses first-token grammar while
  reasoning, matching later decode and preserving template-forced thinking.
  CPU regression reproduces consumed JSON opener before the fix and passes
  afterward. *Session 3 (2026-10-11 18:02 UTC):* fresh attempt
  `cand-grammar-ee5e-02` completed **full short16+long byte neutrality** and
  the focused live slice **15/15 on GPU** — `seed-json-schema`
  parses/schema-valid (JSON-opener fix proven under template-forced thinking),
  both `stream_matches_blocking` cases pass, `stream:last_token_is_eos`
  recorded untested. Evidence: `merged-image-20261011-session2/grammar-r2/`
  under the durable root in continue.md. **Q3 remains open:** the full behavioral surface, tools,
  reasoning budgets, multi-turn, cancellation/timeout recovery and explicit
  skipped assertions still need qualification. Repeated-seed identity alone
  does not validate later-token RNG independence.

- [ ] **Q4 — Protect task quality beyond reference parity.** Freeze a held-out
  corpus with objective checks for arithmetic, executable code, exact tool
  arguments, structured output, multilingual text and long-context retrieval.
  Include sustained generation to expose repetition or premature termination.
  Reuse applicable [suite checks](tests/single_gpu_suite.py), extending them
  beyond keyword/nonempty-output assertions. Establish results with the qualified
  serial profile; evaluate each arithmetic-changing, cache or speculative profile
  on the same cases with fixed sampling settings and enough trials to report
  uncertainty. Set acceptance thresholds before tuning and retain per-case outputs.
  **Done when:** each promoted profile passes the agreed task-quality thresholds
  and Q2's frozen numerical gates; a matching reference error cannot count as
  a correct task answer. Keep tuning cases separate from this regression corpus.
  *Status 2026-10-11:* corpus (81 held-out + 10 tuning), checkers, runner with
  Wilson intervals/`--compare` and provisional thresholds are in
  `tests/exl3_validation/q4/` with CPU tests (`test_q4.py`). No live run yet:
  the serial-profile measurement, final thresholds and per-profile results
  remain open.

## P — measure and optimize the actual native EXL3 path

- [ ] **P0 — Establish the baseline and performance contract.** Collect Nsight
  Systems traces on both ranks for C1 decode and prefill at short and near-limit
  contexts. Attribute packed projections, routed/shared MoE, KDA, MLA/indexer,
  mHC, vocabulary head, collectives, host gaps and synchronization. Use existing
  `--profile` only for diagnostics; it synchronizes stages. Record GPU/RAM peak,
  workspace/KV bytes and load-time headroom, not just kernel time. For dominant
  kernels, collect occupancy, useful work, local-memory traffic and DRAM bytes;
  compare achieved bandwidth with a measured per-rank device-memory read ceiling.
  Account for exposed communication and host gaps without double-counting overlap.
  Use this cost breakdown to estimate recoverable step time and choose work;
  theoretical rooflines and profiler replay timings do not establish a speedup.
  Benchmark cold/warm TTFT, prefill tokens/s, decode TPOT, end-to-end latency,
  p50/p95/p99, and later per-request plus aggregate throughput at qualified
  concurrency. Include prose, code, tools and deliberate reasoning modes; avoid
  counting-only headlines. Full fingerprints must bind image/source/PTX, providers,
  checkpoint/tokenizer, both hosts, all flags/env, ISL/actual OSL, seed/temp,
  cache state, topology/fabric, clocks/thermal/compaction state, harness and date.
  **Done when:** ≥3 matched repeats, raw run records and independent stream/server
  timing cross-checks establish a baseline; explicit latency/throughput targets
  and permitted quality/memory tradeoffs are recorded for each intended workload.
  Repeat after Q2 if initial profiling used the unqualified arithmetic path.
  *Status 2026-10-11:* harness integrated in `72e4e1a3`, with 22 CPU tests.
  It requires explicit initial-state metadata, refuses output overwrite and
  rejects malformed/truncated streams. The saved six-request diagnostic has
  populated `repeat_index`/`cold_start` fields; the earlier null-field claim
  names absent fields (`repeat`/`cold`) and is unsupported by the raw records.
  No saved query establishes how that claim was produced. Unqualified
  diagnostic on the production image: decode ~0.45 s/token, 2573-token TTFT
  ~139 s (`p0-unqualified-baseline-20261010/`). Decode speed is the dominant
  serving problem; trace it before any other P item.
  A paired named-session Nsight driver is prepared in durable
  `p0-trace-prep-20261011/` (11 CPU tests). It uses `nsys launch`, starts
  collection after health, preserves reports before statistics processing and
  removes the logits observer in its explicitly recorded attribution profile.
  **No live profiler execution yet.** Review the generated remote commands,
  choose explicit workload/output paths, archive the active gate attempt and
  stop its containers before using the same port/GPU pair. Timing from the
  instrumented request is attribution only; uninstrumented repeats remain owed.

- [ ] **P1 — Packed decoder and decode occupancy.** Inspect runtime JIT SASS
  and Nsight Compute counters, building on the completed offline LDL/STL audit.
  Specialize actual bitrates/tile profiles to eliminate runtime accumulator
  indexing/local traffic; reuse packed-word windows and exact half values.
  Evaluate ordered split-K and N128/N256 tile variants for measured shapes.
  Preserve slice fold order, partial precision, padding and thin/fat policies
  in `exl3_numerics.rs` and `moe/policy.rs`; avoid nondeterministic atomic sums.
  **Done when:** packed differential oracles and Q2 pass, runtime instruction /
  occupancy evidence confirms the intended change, and matched end-to-end
  decode timings improve without unacceptable memory or prefill cost.

- [ ] **P2 — Real multi-row GEMM and expert-grouped prefill.** Current packed
  MMA repeats work per row; routed execution grids over `rows * topk`. Pack
  multiple useful rows into MMA tiles and build GPU expert/tile worklists with
  an inverse mapping to original route slots. Preserve route-count-dependent
  thin/fat selection (boundary 32), stable ties and weighted combine order.
  Qualify M2 padding, tails, uneven expert occupancy, row-bucket boundaries and
  capacities. Compare packed, shared-memory decode and transient BF16 GEMM
  only where measured, budgeting transient storage explicitly. Tune prefill
  chunk sizes after capacity/Q1 policy support is established.
  **Done when:** byte/component and model gates pass, TTFT/prefill throughput
  improve across declared prompt lengths, and mixed workloads keep their latency
  target. This is a prerequisite for efficient F2 concurrency and F4 verification.

- [ ] **P3 — mHC, indexer and routing hot spots.** Parallelize mHC projection,
  norm, Sinkhorn and post/contract work while preserving reduction/contraction
  order. Profile/widen indexer gate GEMV; consider warp top-k with lowest-ID ties.
  Optimize KDA recurrence and sparse MLA only when traces identify their cost;
  qualify any alternate recurrence against Q1 state/output fixtures. Check
  indexer pool/score allocations for token-versus-pool over-allocation.
  **Done when:** boundary/component tests and Q2 pass, critical-path trace time
  and repeated end-to-end metrics improve. Prior-engine estimates are hypotheses.

- [ ] **P4 — Reduce repeated transforms, launches and F32 intermediates.**
  Evaluate a combined native QKV projection (currently three calls), reuse of
  gate/up input transforms, Hadamard prologue/epilogue fusion, and ordered
  weighted-down/combine fusion. Cache repeated kernel lookups. Evaluate EXL3
  PDL only with matching producer/consumer dependencies; the current allowlist
  contains no EXL3 kernels. **Done when:** byte gates preserve logical offsets,
  accumulation order and casts, and traces show fewer launches/bytes with an
  end-to-end win. Revalidate F1 graph replay after each relevant change.

- [ ] **P5 — Head transport and shared-expert storage.** First evaluate BF16
  head gather/reorder to avoid widening BF16 projection results through full
  F32 vocabulary buffers. Preserve active-vocabulary trim, rank-major layout,
  sampled/logprob consumers and the explicit F32 capture path. Separately
  evaluate packed shared experts versus reconstructed BF16 and safe overlap
  with routed work; this changes arithmetic/resource ownership and requires
  independent qualification. **Done when:** head/layout/collective and full-model
  gates pass with lower bytes/memory and better latency. Packed shared experts
  additionally need quality proof and scratch/stream/graph lifetime tests.

- [ ] **P6 — Two-node communication and host tails.** Trace actual collective
  dispatch before tuning: small decode latency and large prefill bandwidth are
  separate workloads. Measure all-reduce/all-gather counts, payloads, rank skew,
  synchronization and NCCL versus eligible existing RDMA paths. Evaluate overlap
  and a second rail only after verifying cable/HCA topology and matched workloads.
  Log CPU scheduling, managed allocations, compaction, idle recovery and thermal
  behavior during a sustained soak; alter host settings only for an attributed
  bottleneck and retain rollback values. **Done when:** collective correctness,
  repeated p50/p99 and end-to-end results improve with no hangs under restart or
  peer failure. Lossy collective compression is a separate quality-gated profile.

## F — independently qualify the serving features

- [ ] **F1 — CUDA decode graphs.** Audit cuBLAS setup/lazy allocation, shared
  scratch lifetime, TP collectives, per-request state and capture-unsafe copies
  or synchronization. Qualify C1 eager versus graph capture/replay first, then
  each F2 batch shape and F4 verify path. Keep debug captures out of timed runs.
  **Done when:** evidence shows actual replay, identical qualified outputs/state,
  repeated requests and cancellation are safe, graph memory is budgeted and
  uninstrumented latency improves. Only then change launcher graph flags.

- [ ] **F2 — Concurrent and continuous batching.** The launcher admits one
  sequence; existing multi-sequence code is not EXL3 proof. Validate C2, then C4
  and higher only within measured headroom: KDA owner/state isolation, MLA/indexer
  positions, mHC row layout, MoE scratch/stream leases, padded M2, head batching,
  chunked-prefill/decode mixing, eviction/preemption, cancellation and admission.
  Extend reference-policy indexed/multi-sequence KDA dispatch explicitly (the
  first Q1 design excludes it). **Done when:** interleaved outputs match isolated
  requests under the declared policy, soak shows no corruption/deadlock/leak,
  and aggregate throughput improves while per-request p95/p99 meets its target.

- [ ] **F3 — Context, cache capacity and prefix reuse.** Establish actual BF16
  KV/indexer/KDA memory limits; qualify progressively beyond 4096 before
  advertising the checkpoint's much larger configured context. Include chunk,
  sparse-top-k and pool boundaries, long generation, needle/code/tool-string
  fidelity, OOM admission and recovery. Qualify FP8 KV separately against BF16;
  do not import NVFP4 MLA projection or KV results. For prefix caching prove
  synchronized restoration of KDA recurrent/convolution state and MLA/indexer
  state, cache-hit/miss equivalence, ownership, eviction and memory bounds.
  **Done when:** numerical/behavior and long-context retrieval gates pass for
  every published context/dtype/cache profile, with measured TTFT and capacity.
  Offload and cross-node cache sharing are later capacity experiments, not an
  assumed doubling of usable KV memory or an automatic latency improvement.

- [ ] **F4 — Implement DFlash/speculative decoding support.** Native EXL3
  explicitly rejects speculation in
  [serve loading](crates/spark-server/src/main_modules/serve_phases/weights.rs),
  [factory assembly](crates/spark-model/src/factory/build.rs) and
  [preflight](crates/spark-model/src/preflight.rs); KDA paired verification also
  excludes EXL3. Implement the native loading/verification/state paths and retain
  these guards until their replacements are qualified. Establish a compatible
  drafter, tokenizer/vocabulary and target hidden-layer capture contract; budget its
  weights/KV. Qualify EXL3 multi-row verification and rollback of KDA history,
  MLA/indexer caches and rejection/cancellation state. Native MTP needs a separate
  compatible loader/weights path; `default_num_drafts=0` is the current target
  policy. Profile hidden-state capture copies. Replace or extend the existing
  `dflash_width.rs` cost table with EXL3-specific measurements after P2 wins;
  include history, owners and realized width/acceptance distributions.
  **Done when:** serial-equivalence/sampling checks and Q3 pass, logs prove
  speculation engages for intended reasoning modes, and accepted output tokens
  per wall second improve over matched serial runs including draft/verify cost.

## R — readiness and closure

- [ ] **R1 — Publish a qualified configuration and regression record.** Run the
  repository's required CPU checks and relevant real-model regression slices on
  a fresh image at the final source commit. Cover shared runtime/build/comm and
  server changes across the support matrix. Complete sustained two-node service
  tests for memory growth, overload, idle recovery, health, timeout, restart and
  peer failure. Update the support matrix, MODEL.toml behavior/optional-kernel
  notes, launcher and build/run documentation to the actually qualified profiles.
  Remove inherited NVFP4/EP=2 wording from the EXL3 target where misleading.
  **Done when:** correctness, API behavior, resource bounds and agreed latency /
  throughput targets have matching raw records and a reproducible image digest;
  unsupported profiles and remaining limitations are explicit. Use the release
  workflow when publication is requested.

- [ ] **R2 — Make performance regressions detectable.** After P0 and numerical
  qualification, add EXL3 TP2 workload records and baselines to the existing
  `.benchmarks/` / `spark benchmark` gate machinery; derive model benchmark
  summaries from those records. Cover short interactive decode, long prefill,
  qualified concurrent service and speculation separately, including memory and
  Q4 quality results. Set tolerances from repeat variability and the agreed
  service targets. GPU runs produce measurements; the existing
  [CI check](.github/workflows/ci.yml) validates committed records and coverage.
  **Done when:** missing/stale records and deliberately regressed fixtures fail
  the check, records cover the final measured tree under the
  [merge-queue protocol](docs/gate-queue-protocol.md), and a documented command
  reproduces each workload on both Sparks. Never lower a baseline to pass a
  regression or treat a CPU record check as a fresh GPU measurement.

For every optimization: one-variable A/B, ≥3 repeats for headline results,
independent timing confirmation, same quality gates, and a rollbackable change.
No current smoke rate, offline SASS audit or prior-engine benchmark proves an
Atlas speedup. The goal is met only when the chosen serving profiles satisfy
both their correctness and measured performance contract.

## Q2 code-path gap follow-ups (2026-10-12)

Short summaries only; sequencing authority and fallback-suspect ordering in
[docs/plans/q2-norm-causality-plan.md](docs/plans/q2-norm-causality-plan.md)
(folded from the audit gap list); evidence and rationale in
`/home/colum/Workspace/atlas-exl3-validation-20261010/q2-codepath-audit-20261012/comparison-report.md` §5,
as corrected by the appended vLLM reconciliation in `q2-diagnostic.md`.

- [ ] Before a shared-code fix, instantiate the causality plan's neighboring
  types/configurations matrix from current callers, enums, loaders and kernel
  dispatch. Record supported, implemented-but-unqualified and rejected paths;
  cover changed branches, admission checks, shape boundaries and coupled axes.
  Include reachable norm/gate, weight/cache dtype, routing, attention/state,
  topology and serving-mode neighbors. Attach regression evidence per row;
  mark missing hardware or reference coverage untested, not passed.
- [ ] Repeat the plan's Rust/vLLM/source/log review on the actual candidate
  artifacts. Reuse its prompt and record source/runtime binding, harness
  validity, earliest divergence and complete state before causal conclusions.
- [ ] Evaluate performance under the plan's explicit workload/objective and
  measurement protocol after correctness is qualified. Compare named eligible
  alternatives; report best observed scope and unresolved evidence rather than
  declare global optimality from source parity or a single timing result.

- [ ] C1/C2 — execute the revised layer-0 projection/intervention plan;
  explicit layer scope and full-vocabulary ensemble metrics are now specified.
- [ ] L1 escalation — for `longer`, trace complete mHC residual streams and
  L1 KDA between matching L0 attention/FFN outputs and differing L1 FFN inputs
  on both ranks; do this before broad late-layer bisection.
- [ ] G2 — bind final RMSNorm and reuse/rebind the existing head-M1 replay
  (`linear-head-m1-production13-rank{0,1}-results/`) to current failing inputs,
  resident workers and embedded PTX; older NVRTC component proof is scoped.
- [ ] G3 — router binding + same-input router-logit comparison + tie-rule
  evidence for the failing rows. Reuse coherent131 M1/M5/M10/M128 receipts;
  checkpoint specifies float32, but actual GEMM/cast/sigmoid dtype needs binding.
- [ ] G4 — reuse/rebind prior thin/fat expert replays and 32/33 occupancy
  controls. Capture FP16 route-weight casts, local IDs, token permutation,
  per-expert counts and routed FP32 sums. Archived L3 longer counts stay below
  cap32 on both engines/ranks; other layers/long contexts need their own counts.
  Thin half2/post-SiLU arithmetic and atomics-order probes remain relevant.
- [ ] G1 — layers 4–44 hidden-state bisection captures on both engines
  (operator scope extension granted 2026-10-12; see decisions.md).
- [ ] G5 — effective cache/attention-path equivalence: reference auto resolves
  to fp8_ds_mla, which candidate fp8_g128 was selected to match. Bind QDQ
  placement and actual read/write paths on short prompts too: the bound SM120
  backend uses cached MQA even for fresh short prefill. Long-suite tracing
  operator scope extension was granted 2026-10-12 (see decisions.md).
- [ ] G6/G7 — mHC ≤16-row path-branch capture at a representative layer;
  bind deferred `(x,residual,post,comb)`, local rows and provider/splits.
  MLA/indexer replay must compare actual selected keys: installed reference
  truncates one pool only in ranked selection beyond the short-prefill cutoff.
  Evaluate actual continuation prefix/chunk lengths, not suite case names.
- [ ] G8 runtime evidence — bind effective config overrides and selected
  providers; do not reuse the archived JSON's historical serving environment.
- [ ] G9 — compiled-binary ↔ source binding (EXL3 extension, `_C.*`,
  FlashInfer, cuBLAS algorithms).
- [ ] Minor: confirm preserved gate request payloads carry no masks/penalties
  (both audits note processor absence is unattested from "greedy" alone).
- [ ] Deferred review nits (harness review 2026-10-12, non-blocking): failure
  receipt on mid-row crashes; ULP one-liner dedup vs `metrics`; shared compile
  options constant; `name` shadowing; v1 loop `i < dim` comment. Reference:
  `q2-norm-ablation-20261012/harness-review-20261012.md` findings 4-8.
