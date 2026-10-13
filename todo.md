# GLM-5.3 EXL3 on two Sparks — roadmap and work checklist

This is the authoritative work list. [continue.md](continue.md) holds restart
context, current state, hosts, profile and the evidence index; nothing there
is duplicated here. Q2 diagnostic sequencing detail lives in
[docs/plans/q2-norm-causality-plan.md](docs/plans/q2-norm-causality-plan.md);
decisions in [decisions.md](decisions.md); experiment results in
[q2-diagnostic.md](q2-diagnostic.md). Evidence paths are relative to the
durable root `/home/colum/Workspace/atlas-exl3-validation-20261010`.

## Objective

Serve `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3` on the native Atlas
EXL3 path across both Sparks (TP=2 / EP=1) as a **qualified,
high-performance** service. The objective is met when every published
profile has:
- **numerical parity** with the reference engine under the registered
  statistical gate (Q2);
- **qualified API behavior** (Q3) and **task quality** (Q4);
- a **measured performance contract** (P0 baseline plus targets, with the
  optimizations that meet them);
- **independently qualified serving features** (F1–F4) where offered;
- a **reproducible release record** with regression detection (R1/R2).

Rules that apply throughout:
- Check an item only with a run/artifact ID tied to the source and profile
  that passed.
- Every optimization is a one-variable A/B with ≥3 repeats, independent
  timing confirmation, the same quality gates and a rollbackable change.
- No smoke rate, offline audit or prior-engine benchmark proves an Atlas
  speedup.
- The Sparks are a single shared resource: one experiment owner across
  both ranks.

## Roadmap

| Milestone | Scope | Exit criterion | State |
|---|---|---|---|
| **M1 — Numerical qualification** (critical path) | Q2, closing Q1 | Fresh image passes gate v2 with all controls on both ranks | **Active** |
| M2 — Behavior and quality | Q3, Q4 | Behavior receipts pass; Q4 thresholds met on the qualified profile | Q3 partial; Q4 harness only |
| M3 — Performance | P0 → P1–P6 | Baseline + targets recorded; optimizations meet them without regressing M1/M2 | P0 prepared, not run |
| M4 — Serving features | F1–F4 | Each feature qualified independently | Not started |
| M5 — Release | R1, R2 | Qualified image + regression gate | Not started |

**Dependencies and parallelism**
- M1 gates M2's Q4, M3's optimizations, M4 and M5.
- CPU-only work may proceed alongside M1 GPU work, provided it does not
  touch the frozen comparison profile:
  - Q3 surfaces and harnesses;
  - the C2 knob implementation;
  - P0 command review.
- P0 profiling may run on the unqualified image as a labeled diagnostic, but
  only in a GPU window M1 does not need. Repeat it after M1.

### M1 phases

1. **Gate readiness:** collection authorised; evaluator checks passed (A2.4).
   Luna completes review/freeze and operational preflight. Probe the
   ref13 mode mechanism. Run the owed GPU controls.
2. **Causality:** C2, then the L1 escalation, then the G-fallbacks (G8/G9
   binding first). This finds the dominant final-logit cause.
3. **Fix:** the repeat Rust/vLLM/source review, neighbor coverage, and the
   production fix with its obligations (v24 or whatever causality points to).
4. **Qualify:**
   - fresh no-cache image, fingerprints and neutrality proofs;
   - v2 collection in randomised order (fixed 15 references + 3 held-outs + 2
     candidates + perturbations + MDE);
   - verdict.

## Now — active batch (M1 phases 1–2)

Luna `collect_gate_v2` owns the authorised qualification preflight/campaign.
Inspect both hosts before GPU work (continue.md "Before any GPU work").

1. **Gate v2 collection authorised; final review/freeze and preflight.** Review
   [docs/design/exl3-statistical-gate-v2.md](docs/design/exl3-statistical-gate-v2.md)
   (P2 fixes + Amendments 1–2). A2.4 implementation checks passed; operator
   authorised collection after replay/reporting. Freeze evaluator/plan and
   pass host/image preflight before samples. Fixed N=15, no top-ups;
   reference unverifiability blocks qualification. Use the
   [pinned vLLM source contract](docs/design/exl3-vllm-reference-contract.md)
   for engine semantics, not superseded audit prose.
2. **ref13 mode-mechanism probe** (recommended before v2 collection).
   - [ ] The 15 reference draws fall into 6 byte classes, with a recurring
     mode at `longer`/r0 in 3/15. Look for a discrete runtime choice:
     autotune/plan selection, NCCL algorithm, routing tie.
   - [ ] Keep probes in separate diagnostic attempts. If a runtime policy
     can be pinned, propose an amendment before changing qualification
     (H4 may become unnecessary); do not mix profiles. Evidence:
     `q2-ref13-investigation-20261012/`.
3. **C2 — layer-0 norm-output intervention** (plan §Experiment C2). Substitute
   reference-exact gated-norm output (v24 ≡ reference-exact 32/32) at layer
   0 on both ranks, then trace o_proj → reduced attention → FFN inputs → L3
   routing → MoE → final logits on the failing rows.
   - [x] CPU: off-by-default substitution knob bound to request ID, layer,
     rank, invocation and tensor hash. Captures via `Exl3FfnRouter`; do
     not reuse the archived C1 code. **Done:** `f8e9859f` (knob +
     request-ID threading + factory refusal for non-KDA loads),
     `3f9e5ee4` (fixture assembly, A/B driver, plan-fixed scoring),
     `95204789` (host inspection); contract + A1–A13 in
     [docs/plans/c2-tooling-spec.md](docs/plans/c2-tooling-spec.md).
     Independent review: OK with notes; M1/m1/m2/m3/n2 dispositioned.
   - [ ] Fresh image build and neutrality proof: with the knob off, the
     final-logit archive is byte-identical to the frozen baseline.
   - [ ] **First measurement (discharges the C1 flag):** the treatment's
     local o_proj output is byte-identical to the reference's captured
     local output. If not, record it and re-open C1.
   - [ ] Baseline and treatment each repeated for byte-determinism.
   - [ ] Score per the plan against every ref04–ref12 member. Report mixed
     results as mixed. (Scorer ready — `tests/exl3_validation/c2_score.py`;
     the collection itself is GPU-phase.)
4. **L1 escalation** (if C2 leaves residual failure): for `longer`, capture
   the complete mHC residual streams and the L1 attention input, then
   bisect L1 core → gated norm → o_proj → FFN mHC input.
5. **G-fallbacks**, in the plan's order (G2, G3, G4, G1, G5, G6/G7), each
   preceded by its G8/G9 binding. See "M1 diagnostic backlog" below.

## Next — M1 phases 3–4

- [ ] **Repeat the Rust/vLLM/source/log review** on the actual candidate
  artifacts (plan §Repeatable review, reusable prompt). It records
  source/runtime binding, harness validity, earliest divergence and
  complete state.
- [ ] **Neighbor coverage** for the chosen fix. The matrix is instantiated
  (`q2-neighbor-matrix-20261012/`, 49 rows, none passed); attach per-row
  regression evidence. v24 reaches `kernels/gb10/common/kda.cu:565` at
  runtime, is build-reachable by all 30 targets, and fused twins must be
  classified.
- [ ] **Production gated-norm fix (v24)**, only behind all of:
  - [ ] C2 causality evidence;
  - [ ] an ablation GPU rerun under the repaired harness (never yet run on
    a GPU) with `v24-minimal`'s own engagement controls (it had none);
  - [ ] v24 proven under production compile flags (the ablation used NVRTC
    `--fmad=false`; production flags are unverified);
  - [ ] optionally, the untested variant pairs, to support "minimal";
  - [ ] the neighbor coverage above.
- [ ] **Owed GPU controls** for qualification:
  - [ ] the `ATLAS_EXL3_GATE_PERTURBATION` must-fail runs;
  - [ ] the indexer scale-boundary oracle retry with the exact frozen
    reference source overlay;
  - [ ] confirmation that gate request payloads carry no masks or
    penalties.
- [ ] **Qualification run** under gate v2 on a fresh image (randomised,
  interleaved order; composition VOID + top-up rule). Do not recollect
  unchanged references to seek a pass.

## Work items

### Q — qualification and reproducibility

- [ ] **Q1 — EXL3 reference-policy KDA core.**
  ([T2–T8](docs/design/exl3-kda-reference-core.md))
  - **Done when:** both ranks' stage/output/state comparisons pass byte
    gates at M5/M128/decode and multi-chunk boundaries, negative controls
    fail, and the rebuilt image passes Q2.
  - *State:* kernels + dispatch done (`72d64e4c`, `069e4351`). Every stage,
    the carried state (3 chunks + tail, 4 decodes) and the composed
    pipeline are 0-mismatch against the GPU oracle on both ranks
    (`tests/exl3_validation/kda_fla/run.sh`).
  - Open: the Q2 dependency (M1); a performance pass (~12 launches per KDA
    prefill); long-prefix boundary qualification.

- [ ] **Q2 — Full-model numerical and repeatability gates.** Short16,
  long-prefix/chunk-boundary, persistent decode and raw-logit comparisons
  on the exact candidate image, under the registered statistical gate
  (v2 once signed off), with abs/rel/KL/CE/top-K as diagnostics and all
  required controls.
  - **Done when:** all required candidate gates and controls pass on both
    ranks with raw tensors/provenance. Reference variability is explained
    or retained as a constraint, never used to waive failure.
  - *State:* v1 FAIL (historical; candidate T 7.20 vs refs ≤5.53, 3
    catastrophic `longer` rows). Gate v2 awaits sign-off. Work is
    sequenced in M1 above; current facts are in continue.md.
  - Any new collection needs a fresh image plus a neutrality proof, because
    source changed after the tested `ee5eaf0f` image.

- [ ] **Q3 — Usable API behavior.** OpenAI chat/completions and Anthropic,
  streaming and nonstreaming, reasoning on/off and budgets, stop/EOS and
  token accounting, structured output, tools (single, parallel, dependent
  follow-up), multi-turn and sampled requests. Check `poolside_v1`,
  template-seeded thinking, special-token stripping and finish reasons
  against this checkpoint; do not copy the reference's `glm47`/`glm45`
  parser settings without evidence. Exercise cancellation, timeout and the
  next request after each. Reuse
  [behavior_protocol.py](tests/exl3_validation/behavior_protocol.py).
  - **Done when:** behavior receipts pass without leaked state or protocol
    drift. Fix Atlas failures even when the reference fails too.
  - *State:* seed/stop/grammar/API fixes are merged. Image `ee5eaf0f` has
    full short16+long byte neutrality and a 15/15 focused live slice
    (`merged-image-20261011-session2/grammar-r2/`); terminal-EOS-ID
    verification was skipped as untested.
  - Open: tools, reasoning budgets, multi-turn, cancellation/timeout
    recovery, and the full behavioral surface. Repeated-seed identity does
    not prove later-token RNG independence.
  - [ ] Run the CUDA-gated grammar tests on a Spark
    (`first_thinking_token_preserves_content_grammar`,
    `actual_cold_first_tool_opener_tracks_selected_promotion`), and add a
    test through the real prefill callers and `resolve_with_tool_boundary`.
    Known behavior change: with thinking on, tools and a JSON grammar, the
    first token may now be the native tool opener.

- [ ] **Q4 — Task quality beyond reference parity.** A frozen held-out
  corpus with objective checks (arithmetic, executable code, exact tool
  arguments, structured output, multilingual, long-context retrieval,
  sustained generation). Thresholds are set before tuning and per-case
  outputs retained; tuning cases are kept separate.
  - **Done when:** each promoted profile passes the agreed thresholds and
    Q2. A matching reference error does not count as correct.
  - *State:* corpus (81 held-out + 10 tuning), checkers, a runner with
    Wilson intervals/`--compare` and provisional thresholds are in
    `tests/exl3_validation/q4/` with CPU tests. There has been no live run.

### P — measure and optimize the native EXL3 path

- [ ] **P0 — Baseline and performance contract.** Nsight Systems traces on
  both ranks for single-sequence decode and prefill at short and near-limit
  contexts. Attribute:
  - packed projections, routed/shared MoE, KDA, MLA/indexer, mHC and the
    vocabulary head;
  - collectives, host gaps and synchronization;
  - GPU/RAM peaks and workspace/KV bytes;
  - for dominant kernels, occupancy, local-memory and DRAM bytes against a
    measured read ceiling.

  Benchmark cold/warm TTFT, prefill tokens/s, decode TPOT, end-to-end
  latency and p50/p95/p99 across prose, code, tools and reasoning modes,
  with full fingerprints.
  - **Done when:** ≥3 matched repeats with independent stream/server timing
    establish a baseline, and latency/throughput targets and permitted
    tradeoffs are recorded per workload. Repeat after M1.
  - *State:* harness `72e4e1a3` (22 CPU tests); paired Nsight driver
    prepared in `p0-trace-prep-20261011/` (11 CPU tests); **no live
    profiler run yet**.
  - Unqualified diagnostic: decode ~0.45 s/token and 2573-token TTFT ~139 s
    (`p0-unqualified-baseline-20261010/`). **Decode speed is the dominant
    serving problem: trace it before any other P item.**
- [ ] **P1 — Packed decoder and decode occupancy.**
  - Inspect runtime JIT SASS and NCU counters (building on the offline
    LDL/STL audit).
  - Specialize bitrates/tile profiles to remove runtime accumulator
    indexing and local traffic.
  - Evaluate ordered split-K and N128/N256 tiles.
  - Preserve fold order, partial precision, padding and thin/fat policies;
    no nondeterministic atomics.
  - **Done when:** packed oracles and Q2 pass, instruction/occupancy
    evidence confirms the change, and matched decode timings improve.
- [ ] **P2 — Real multi-row GEMM and expert-grouped prefill.**
  - Pack useful rows into MMA tiles, with GPU expert/tile worklists and an
    inverse route mapping.
  - Preserve thin/fat selection (boundary 32), stable ties and combine
    order.
  - Qualify M2 padding, tails, uneven occupancy and capacities.
  - Prerequisite for efficient F2 and F4.
  - **Done when:** byte/model gates pass and TTFT/prefill throughput
    improve across declared prompt lengths.
- [ ] **P3 — mHC, indexer and routing hot spots.**
  - Parallelize mHC projection, norm, Sinkhorn and post/contract,
    preserving order.
  - Widen the indexer gate GEMV; consider warp top-k with lowest-ID ties.
  - Optimize KDA recurrence and sparse MLA only where traces show cost.
  - **Done when:** component tests and Q2 pass, and critical-path and
    end-to-end metrics improve.
- [ ] **P4 — Repeated transforms, launches and F32 intermediates.**
  - Combined QKV projection; gate/up transform reuse; Hadamard
    prologue/epilogue fusion; ordered weighted-down/combine fusion.
  - Cached kernel lookups; EXL3 PDL only with matching dependencies.
  - **Done when:** byte gates hold and traces show fewer launches/bytes
    with an end-to-end win. Revalidate F1 after each change.
- [ ] **P5 — Head transport and shared-expert storage.**
  - BF16 head gather/reorder to avoid widening through full F32 vocabulary
    buffers.
  - Separately, packed shared experts and overlap with routed work, which
    needs its own quality proof.
  - **Done when:** head/collective/full-model gates pass with lower bytes
    and better latency.
- [ ] **P6 — Two-node communication and host tails.**
  - Trace collective dispatch first; decode latency and prefill bandwidth
    are separate workloads.
  - Measure counts, payloads, rank skew and NCCL versus RDMA paths.
  - Evaluate a second rail only after verifying the topology.
  - Soak-log scheduling, allocations, compaction and thermal behavior.
  - **Done when:** collective correctness and repeated p50/p99 improve with
    no hangs under restart or peer failure.

### F — independently qualified serving features

- [ ] **F1 — CUDA decode graphs.**
  - Audit cuBLAS lazy setup, scratch lifetime, TP collectives, per-request
    state and capture-unsafe copies.
  - Qualify single-sequence eager versus graph replay first, then each F2
    shape and the F4 verify path.
  - **Done when:** actual replay, identical outputs/state, safe
    cancellation, budgeted graph memory and an uninstrumented latency win
    are all shown. Only then change the launcher graph flags.
- [ ] **F2 — Concurrent and continuous batching.** The launcher admits one
  sequence today.
  - Validate concurrency 2, then 4+, within measured headroom.
  - Cover KDA owner/state isolation, MLA/indexer positions, mHC row
    layout, MoE scratch leases, padded M2, head batching, prefill/decode
    mixing, eviction, cancellation and admission.
  - Extend reference-policy multi-sequence KDA dispatch.
  - **Done when:** interleaved outputs match isolated ones, the soak is
    clean, and throughput improves within the p95/p99 targets.
- [ ] **F3 — Context, cache capacity and prefix reuse.**
  - Establish the BF16 KV/indexer/KDA memory limits, then qualify
    progressively beyond 4096 tokens.
  - Qualify FP8 KV separately.
  - For prefix caching, prove synchronized restoration of KDA
    recurrent/conv state and MLA/indexer state.
  - **Done when:** numerical/behavior and retrieval gates pass per
    published context/dtype/cache profile, with measured TTFT and capacity.
- [ ] **F4 — DFlash/speculative decoding.** EXL3 currently rejects
  speculation in
  [serve loading](crates/spark-server/src/main_modules/serve_phases/weights.rs),
  [factory assembly](crates/spark-model/src/factory/build.rs) and
  [preflight](crates/spark-model/src/preflight.rs).
  - Implement native loading, verification and state rollback (KDA
    history, MLA/indexer caches), keeping those guards until the
    replacements are qualified.
  - Establish a compatible drafter and capture contract.
  - Native MTP needs its own loader; `default_num_drafts=0` stands.
  - **Done when:** serial-equivalence and Q3 pass, speculation engages,
    and accepted tokens per wall second improve including draft/verify cost.

### R — readiness and closure

- [ ] **R1 — Qualified configuration and regression record.**
  - Required CPU checks and real-model slices on a fresh image at the
    final commit.
  - Two-node soak: memory, overload, idle, health, timeout, restart and
    peer failure.
  - Update the support matrix, MODEL.toml, launcher and docs to the
    qualified profiles; remove misleading NVFP4/EP=2 wording.
  - **Done when:** correctness, behavior, resource and performance records
    match a reproducible image digest. Publish via the release workflow
    when requested.
- [ ] **R2 — Performance regression detection.**
  - Add EXL3 TP2 workload records and baselines to `.benchmarks/` /
    `spark benchmark`, covering decode, long prefill, concurrent service,
    speculation, memory and Q4.
  - The R2 records branch `pi/r2records-20261011` is unmerged, and its
    benchmark boundary edits invalidate required records.
  - **Done when:** stale or regressed records fail the
    [CI check](.github/workflows/ci.yml) under the
    [merge-queue protocol](docs/gate-queue-protocol.md). Never lower a
    baseline to pass.

## M1 diagnostic backlog (G-list)

Ordering and detail are in the plan (§Fallback-suspect ordering); evidence is
in `q2-codepath-audit-20261012/comparison-report.md` §5, as corrected in
`q2-diagnostic.md`. Bind G8/G9 before each affected replay.

- [ ] **G2** — final RMSNorm + 1-row lm_head. Rebind
  `linear-head-m1-production13-rank{0,1}-results/` to the failing inputs,
  resident workers and embedded PTX.
- [ ] **G3** — router binding, same-input router logits and tie rule.
  Reuse the coherent131 M1/M5/M10/M128 receipts; bind the actual
  GEMM/cast/sigmoid dtype.
- [ ] **G4** — thin/fat expert replays and 32/33 occupancy. Capture FP16
  route-weight casts, local IDs, permutation, counts and FP32 sums. L3
  `longer` is under cap32; other layers need their own counts.
- [ ] **G1** — layers 4–44 hidden-state bisection on both engines (scope
  granted 2026-10-12).
- [ ] **G5** — effective cache/attention-path equivalence (`fp8_g128` vs
  `fp8_ds_mla`). Bind QDQ placement and read/write paths, including short
  prompts (SM120 cached MQA); extend to the 2048 long suite (scope
  granted).
- [ ] **G6/G7** — mHC ≤16-row branch capture; MLA/indexer actual selected
  keys beyond the short-prefill cutoff. Classify actual prefix/chunk
  lengths, not case names.
- [ ] **G8** — bind effective runtime config overrides and selected
  providers (the checkpoint config is already recovered).
- [ ] **G9** — compiled-binary ↔ source binding (EXL3 extension, `_C.*`,
  FlashInfer, cuBLAS algorithms).

## Housekeeping

- [ ] Split `crates/spark-runtime/src/weights/exl3/norm_substitution.rs`
  (471/500 LoC) before it grows (review nit n1).
- [ ] The `exl3_checkpoint()` term of the substitution-engagement
  predicate (`crates/spark-model/src/factory.rs`) is covered by inspection
  only; give it a GPU-side test (m3 follow-up).
- [x] Confirm which image the stopped `glm53-exl3-atlas-*` containers
  reference (`docker inspect` on both hosts). **Answered 2026-10-12:**
  both `glm53-exl3-atlas-0` (spark0) and `glm53-exl3-atlas-1` (spark1)
  reference `atlas-exl3-validation:src-48624a8605de-rust1.97.1`
  (`sha256:82b23cd19e59…`, the recorded production image); all containers
  exited, both GPUs idle. Receipt:
  `$EV/host-inspection-20261012T2204Z/` (driven by
  [scripts/inspect-sparks.sh](scripts/inspect-sparks.sh)).
- [ ] Triage the local failure in
  `spark-model factory::glm_dense_cache::tests::dense_prefill_bf16_explicit_flag`.
  It is unrelated to this work: the test expects `"2"` to be rejected, but
  `3be435f7` made `=2` valid.
- [ ] Decide the fate of the untracked `docs/plans/atlas-kv-{coord,cache-server}-plan.md`
  (an unrelated KV-coordinator design, 2026-10-10).

## Completed — evidence pointers

Pointers only; detail is in git history and the evidence root (index in
continue.md). Do not delete these paths.

**Platform and Q0**
- Native EXL3 mul1 loading, packed kernels, TP=2/EP=1 KDA/MLA/MoE/mHC
  integration and launcher; 5/5 launch smoke
  (`atlas-launch-smoke-20261010/smoke.json`).
- Shared routed-MoE workspace with A→B→A reuse proof
  (`arithmetic-audit/native-aux-scratch-gpu-20261010/`). Opt-in FFN
  profiling; coherent cuBLAS 13.1 classic/Lt; merged KDA auxiliary
  (`native-aux-insitu-20261010/`).
- **Q0:** reproducible coherent-13.1 recipe (`a6f64c46`, `59b0e113`), clean
  image `src-8b6b47f14d5b-rust1.97.1` from `fb4f4b41` on both nodes,
  final-image smoke 5/5, all 23 shards rehashed on both hosts
  (`fp8decode-candidate-20261011/`, `checkpoint-rehash-20261011/`).
- 2049-token FP8 chunk-tail crash repaired (`fb4f4b41`;
  `statistical-gate-20261011/resume-diag01/`).

**Q1/Q2 investigation**
- KDA reference-policy kernels and dispatch with the both-rank stage/state
  oracle (`72d64e4c`, `069e4351`;
  `arithmetic-audit/kda-reference-stage-oracle-20261010/`).
- `cand-fp8decode02` full collection, all 16 short16 + all 16 long cases;
  the v1 gate verdict (`statistical-gate-20261011/final/`).
- Reviewed localization and the repaired analyzer
  (`q2-diagnostic-20261011/analysis/`).
- Runtime-bound gated-norm replay (`q2-norm-replay-20261011/`).
- Gated-norm ablation, fixture-local: `v24-minimal` matches 7/7 differing
  elements in 2 groups
  (`q2-norm-ablation-20261012/` + `ERRATUM-20261012.md`).
- Ablation harness repaired (`9ff39728`, `c71bb557`) and hardened (merge
  `9d14796b`): `main()` wiring tests, validator negative tests, crash
  receipt, per-variant `controls_engaged`.
- Code-path audit G1–G9 (`q2-codepath-audit-20261012/`). D4 head padding
  resolved (`pad=false`, rows=1). G8 checkpoint config recovered (SHA
  matches all 24 manifests).
- ref13 held-out investigation: reference-internal bimodality
  (`q2-ref13-investigation-20261012/`).
- Neighbor matrix, 49 rows (`q2-neighbor-matrix-20261012/`).
- **C1 closed** from native captures, flagged; C2 discharges the flag
  (decisions.md; `q2-c1-execution-20261012/ERRATUM-20261012.md`). The BF16
  replay harness is archived (`651ff083`, tag
  `archive/c1-bf16-o-proj-replay-20261012`).
- Gate v2 pre-registration + P2 fixes + Amendment 1 (`f8061ad8`,
  `fd7b530b`), then Amendment 2/source contract after Luna review:
  fixed N, no top-ups, explicit H4 edge cases; evaluator checks completed,
  collection authorised, final review/freeze and operational preflight next.
- Atomic `Exl3FfnRouter` diagnostic records (`651ff083`).

**Q3**
- OpenAI stream usage order (`4b49a9c4`).
- First-token seed, stop, grammar and completion-identity fixes
  (`7e04679f`, `f7791189`, `19551fc6`, `8334ab33`, `ee5eaf0f`); 15/15
  focused live slice (`merged-image-20261011-session2/grammar-r2/`).
