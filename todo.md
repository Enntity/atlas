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
| **M1 — Numerical qualification** (critical path) | Q2, closing Q1 | Fresh image passes gate v2 with all controls on both ranks | **Blocked at reference verifiability; diagnostics remain** |
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

1. **Gate v2 campaign:** completed 2026-10-14. Corrected evaluator verdict is
   **UNVERIFIABLE**: the reference ensemble failed one H3 and one H4
   leave-one-out check. The evaluator correction changed no metric or
   threshold; see the [qualification report](docs/design/exl3-statistical-gate-v2-qualification-20261014.md).
2. **Causality:** C2, then the L1 escalation, then the G-fallbacks (G8/G9
   binding first). This finds the dominant final-logit cause.
3. **Fix:** the repeat Rust/vLLM/source review, neighbor coverage, and the
   production fix with its obligations (v24 or whatever causality points to).
4. **Qualify:**
   - fresh no-cache image, fingerprints and neutrality proofs;
   - v2 collection in randomised order (fixed 15 references + 3 held-outs + 2
     candidates + perturbations + MDE);
   - verdict.

## Now — reference-verifiability follow-up and M1 causality

The worker transport repair, fresh no-cache image and full 32-request
short16/long neutrality proof are complete. The earlier C2 treatment stopped
because the TP worker lacked the request-ID binding; its evidence is preserved.
Fresh reference-exact C2 completed as `c2-refexact-f099-01`: controls pass,
with mixed arithmetic scores and unchanged `longer` rows. Separate v24 repeats
also completed with identical scores and passing controls. Source-bound
`l1-refexact-f099-01` stopped on reference-worker shutdown after collecting
both candidate windows and eight reference-control rows. Candidate observer
neutrality passes; controlled stop SIGKILL/exit137 was rejected. Lifecycle
correction `1204a210` passed independent review and full CPU validation
(612 tests, 278 subtests). Fresh `l1-refexact-f099-02` completed all 32
requests with every container exiting zero. Candidate observer matches 8/8;
reference observer matches 1/8, with text/thresholds unchanged. Attribution
is withheld. Next establish reference observer verifiability before L1
bisection; [L1 record](docs/design/l1-execution-20261014.md). The completed v2 campaign still
blocks qualification at reference verifiability. Inspect both hosts before
new GPU work (continue.md "Before any GPU work").

1. **Resolve reference-verifiability failures before further qualification.**
   - [x] Complete the fixed 26-attempt collection (832 receipts), archive all
     artifacts and teardown both hosts. Campaign:
     `gate-v2-qualification-20261013-luna1/`.
   - [x] Correct the independent-held-out byte-identity bug with regression
     coverage; 24 focused tests pass. Preserve the frozen INCOMPLETE report
     and corrected UNVERIFIABLE report; thresholds/metrics unchanged.
   - [x] Review and commit the evaluator correction and qualification records.
     **Done:** `757e7da4`; independent review, 24 focused tests and 319
     EXL3 unittest cases pass; archived evaluator/report hashes reverified.
   - [x] Diagnose H3 index 13 / `v2ref14` (`short16/longer/r3`) and
     H4 index 14 / `v2ref15` (`short16/code/r1`) from existing reference-only evidence. Separate these
     failures from candidate C2 causality and historical ref13 mode discovery.
     Reference-only replay reproduces both failures; no recorded profile/cache
     difference explains them. [Diagnosis](docs/design/exl3-reference-verifiability-diagnosis-20261014.md).
   - [x] Decide whether reference runtime pinning or a protocol amendment is
     justified by that diagnosis. Preserve this campaign; no top-ups,
     replacement draws or unchanged retries to seek PASS. Any new campaign
     must be separately registered before collection. **Decision 2026-10-14:**
     investigate runtime pinning in a separate reference-only diagnostic;
     no concrete qualification amendment is justified without causal evidence.
   - [x] Register and execute the reference-only runtime-mechanism diagnostic
     with routing/collective telemetry and engagement controls.
     **Done 2026-10-15:** [result](docs/design/exl3-runtime-mechanism-probe-result-20261015.md),
     [registration](docs/design/exl3-runtime-mechanism-probe-registration.md)
     (pre-collection corrections 1–6, on01 record). Five of six attempts
     complete; on01 failed on a probe wiring defect and is retained. Executed
     EXL3 launch plans were unobservable because nsys does not trace forked
     workers. No draw recurred either singleton or left the campaign H3
     envelope (worst 0.155 of the limit). Router gate inputs diverge first,
     at a row-dependent layer 8–33, and expert flips follow; the collective
     dispatch is identical across draws. Mechanism unresolved.
   - [x] Next mechanism step: per-layer hidden-state capture.
     **Done 2026-10-15:** [result](docs/design/exl3-hidden-state-localization-result-20261015.md).
     In 96/96 comparisons the first differing tensor is the EXL3 fused-MoE
     FP32 output (atomic accumulation of top-8 experts), from layer 3, with
     byte-identical inputs and routing. Rounding absorbs the last-bit noise
     until a row-dependent layer.
   - [x] One-variable `EXL3_FUSED_MOE=0` determinism test. **Done
     2026-10-15:** [result](docs/design/exl3-fused-moe-determinism-result-20261015.md).
     25/32 rows identical (primary FAIL as registered). Every row below 2052
     tokens is deterministic; a second source starts exactly at the indexer
     pool-selection threshold (2052). The loop profile is not a neutral pin.
   - [x] Localize the indexer non-determinism beyond 2051 tokens. **Done
     2026-10-15:** [result](docs/design/exl3-indexer-localization-result-20261015.md).
     `top_k_per_row_prefill` returns the correct pool set in a run-dependent
     order, and sparse MLA attention accumulates in that order.
   - [x] Reference-determinism fix scope. **Decision 2026-10-15:** the
     reference kit is not modified; Atlas-side determinism only. The Atlas
     audit and its follow-ups are tracked as Q5 below.
   See the [qualification report](docs/design/exl3-statistical-gate-v2-qualification-20261014.md).
2. **ref13 mode-mechanism probe** (separate diagnostic, not qualification).
   - [x] The historical tuning set of 15 reference draws falls into 6 byte
     classes, with a recurring
     mode at `longer`/r0 in 3/15. Look for a discrete runtime choice:
     autotune/plan selection, NCCL algorithm, routing tie. **Answered
     2026-10-15:** it is not a discrete choice. Short16 variation (including
     `longer`/r0) is the fused-MoE FP32 atomic accumulation order. It first
     appears at layer 3 with byte-identical inputs (HSL, 96/96), and
     `EXL3_FUSED_MOE=0` removes all variation below 2,052 tokens. Routing
     flips follow; collective dispatch is identical.
   - [ ] Keep probes in separate diagnostic attempts. A pinnable runtime
     policy now exists (`EXL3_FUSED_MOE=0`, deterministic below 2,052 tokens),
     but it is **not neutral**: it moves 4 short16 items to 1.19–1.93× the H3
     limit. Adopting it, or any indexer-order pin, for qualification needs a
     registered amendment and a new campaign; no such decision has been made.
     Do not mix profiles. Evidence: `q2-ref13-investigation-20261012/`.
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
   - [x] The collected `2fbb787c` image has knob-off short16/long byte-neutrality
     against the C1 baseline (`gate-v2-qualification-20261013-luna1/neutrality-proof.json`).
   - [x] Bind collected source `2fbb787c` and diagnostic profile for the first
     C2 attempt. Baseline repeats pass eight-row observer neutrality and all
     rank-local norm archive checks; treatment stops on its first request
     because rank 1 has no request ID. [Receipt](docs/design/c2-execution-20261014.md).
   - [x] Repair worker request-ID delivery, with production-path CPU coverage.
     **Done:** `f099ac58`; bounded F8 metadata preserves absent/empty/UTF-8
     IDs and clears stale worker bindings, without changing the F0 prefill
     payload. Full model suite at the commit: 1660 passed, 17 ignored;
     focused worker and scheduler sender regressions pass.
   - [x] Bind a fresh no-cache image after the worker repair, including
     renewed short16/long neutrality. Build and both-rank load completed:
     `src-4ee96c12b44c-rust1.97.1`, image ID `sha256:82173ea631a8…`;
     binaries/PTX and runtime dependency fingerprints verified in
     `q2-c2-worker-binding-20261014/`. Full32 neutrality passed with both
     wait receipts successful and independent frozen-verifier review. Later runtime or
     capture changes require a fresh build/neutrality proof; the collected
     image does not qualify newer L1/v24 tooling or the C2 treatment.
   - [x] **First measurement (discharges the C1 flag):** the treatment's
     local o_proj output is byte-identical to the reference's captured
     local output. `c2-refexact-f099-01` passes all 32 rank/row/repeat
     comparisons. If a later variant fails, record it and re-open C1.
   - [x] Both variants: baseline and treatment each repeated for
     byte-determinism. Reference-exact and v24-corrected controls pass.
   - [x] Score per the plan against every ref04–ref12 member. Report mixed
     results as mixed. (Scorer ready — `tests/exl3_validation/c2_score.py`;
     the collection itself is GPU-phase.) Reference-exact is complete against
     all nine members: arithmetic two closer/two farther; `longer` four
     byte-unchanged, three still catastrophic in KL. v24 scores match exactly.
     [Repaired C2 report](docs/design/c2-repaired-worker-execution-20261014.md).
4. **L1 escalation** (if C2 leaves residual failure): for `longer`, capture
   the complete mHC residual streams and the L1 attention input, then
   bisect L1 core → gated norm → o_proj → FFN mHC input.
   - [x] CPU: the L1 tooling batch (spec:
     [docs/plans/l1-escalation-tooling-spec.md](docs/plans/l1-escalation-tooling-spec.md)).
     Candidate upstream capture widened to the explicit L0→L1 boundary with
     the mHC boundary binding (rows/precision-path/eps/request binding);
     reference overlay extended in `$EV/q2-l1-tooling-20261013/` (layer-set
     contract + DeepGEMM/TileLang dispatch and split-geometry binding);
     `tests/exl3_validation/l1_bisect.py` earliest-divergence driver.
     **Reviewed 2026-10-14 (independent review: BLOCK on 3 driver majors);
     all findings fixed and CPU-verified.** Local o_proj is
     not bisectable cross-engine (spec B13); the reduced-o_proj step stands.
     A further source review corrected L1's fused phase pairing: attention
     uses `2r`, FFN uses `2r+1`, and materialized highway pairs with reference
     residual output. The CPU driver now binds actual collection envelopes,
     request prefixes, sources and both phase policies, and refuses later
     attribution across missing earlier evidence. Independent review: GO;
     Reference GPU IDs are bound to hash-pinned raw head/response witnesses;
     the reference's prompt-index/UUID suffix is preserved, while candidate
     header matching stays exact. 61 focused L1/localization tests pass;
     two actual archived heads also pass mapping/integrity checks. Runtime/image unchanged.
   - [x] GPU: source-bound L1 capture collection on both engines. Fresh
     `l1-refexact-f099-02` completed all four eight-request windows using
     the repaired image and v5 overlay; every container exited zero.
     Candidate observer matches 8/8; reference matches only 1/8.
   - [x] Establish reference observer verifiability. **Done 2026-10-15**
     ([result](docs/design/l1-fusedoff-observer-result-20261015.md)), from
     `l1-fusedoff-f099-01`:
     - O1 passed (reference 8/8, candidate 8/8).
     - O2 passed: the layer-0/1 reference captures are byte-identical to the
       fused-on `f099-02` captures.
   - [ ] Guarded L1 bisection. The registered bisect ran and gave **no
     attribution** (`analysis_complete=false` on all four runs) because of
     two tooling defects that predate the attempt:
     - the candidate `hc_streams` capture is labeled BF16 but holds the
       FP32 highway (`glm5_kda/forward_attention.rs:69–79`);
     - `kda_layout` has no `[n,4,1] → [n,4]` binding for mHC `post`.

     Next is a registered correction: fix the layout binding, then either
     the Rust capture fix (new image) or an analysis-side FP32 reread.
     Descriptively, every pairable layer-1 stage differs.
     The `f099-02` receipt still withholds attribution
     ([L1 record](docs/design/l1-execution-20261014.md)). Do not relax a
     guard, retry unchanged or add a tolerance.
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
    a GPU) with `v24-minimal`'s own engagement controls — **controls
    implemented** (`v24-minimal` in `CONTROL_RUNS`; interpretation blocked
    without engaged controls); the GPU rerun is still owed;
  - [ ] v24 proven under production compile flags — **the `nvcc-production`
    compile lane is implemented** (`tests/exl3_validation/variant_compile.py`:
    build-manifest flags + attested-PTX/arch/compiler identity, argv parity
    with `build_target.rs`, both lanes recorded per run); the GPU rerun
    under both lanes is owed;
  - [ ] optionally, the untested variant pairs, to support "minimal";
  - [ ] the neighbor coverage above.
- [ ] **Control qualification remains open:**
  - [x] Collect all five `ATLAS_EXL3_GATE_PERTURBATION` runs and the MDE run
    in the completed campaign.
  - [ ] Establish required control outcomes under a verifiable reference
    ensemble. Current campaign scoring stopped before these checks; collection
    alone is not a control PASS.
  - [ ] the indexer scale-boundary oracle retry with the exact frozen
    reference source overlay;
  - [ ] confirmation that gate request payloads carry no masks or
    penalties.
- [ ] **Achieve numerical qualification** after the reference-verifiability
  follow-up and any required candidate fix. The first v2 campaign is complete
  but UNVERIFIABLE. Retain fixed N and no top-ups unless a new rule is explicitly
  registered before collection; do not recollect unchanged references to seek PASS.

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
  (v2 with Amendments 1–2), with abs/rel/KL/CE/top-K as diagnostics and all
  required controls.
  - **Done when:** all required candidate gates and controls pass on both
    ranks with raw tensors/provenance. Reference variability is explained
    or retained as a constraint, never used to waive failure.
  - *State:* v1 FAIL (historical; candidate T 7.20 vs refs ≤5.53, 3
    catastrophic `longer` rows). The complete v2 campaign is UNVERIFIABLE:
    one H3 and one H4 reference LOO failure. Candidate repeats are byte-identical,
    but candidate/held-out/control numerical scoring was not reached. Work is
    sequenced in M1 above; current facts are in continue.md.
  - No-cache build/neutrality evidence exists for collected source `2fbb787c`.
    New runtime/capture changes require renewed image/profile binding and
    neutrality proof; that evidence does not cover the current tip automatically.

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
  - Open: the live behavioral-surface run. The tools, reasoning-budget,
    multi-turn and cancellation/timeout harness code exists
    (`behavior_{tools,robustness,surfaces}.py`); the per-turn
    thinking-toggle gap cases are added (`918369ab`, **first pass
    implementation ready for review and testing**). Repeated-seed identity
    does not prove later-token RNG independence.
  - [ ] Run the CUDA-gated grammar tests on a Spark
    (`first_thinking_token_preserves_content_grammar`,
    `actual_cold_first_tool_opener_tracks_selected_promotion`).
    **First pass implementation ready for review and testing** (`44da2486`):
    unit coverage for `resolve_with_tool_boundary` plus a CPU-runnable
    first-token-builder test with a live JSON GrammarState, covering the
    known behavior change (thinking on + tools + grammar may make the
    first token the native tool opener). The Spark run of the CUDA-gated
    grammar tests still owes a GPU window.

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

- [ ] **Q5 — Atlas run-to-run determinism hardening.** A 2026-10-15
  read-only audit of the GLM-5.3 EXL3 path found neither reference hazard
  in Atlas:
  - the routed-MoE combine is a fixed slot-order FP32 sum with no atomics
    (`exl3_moe_precision.cu:159-176`);
  - the indexer pool top-k emits ascending pool order with a lowest-index
    tie-break (`glm_indexer.cu:358-450`), so sparse MLA accumulates in a
    defined order;
  - the TP=2 sum is a single commutative `__hadd`;
  - argmax ties break the same way everywhere.

  Two follow-ups remain:
  - [ ] **Pin the cuBLASLt router GEMM.** `crates/spark-runtime/src/cublaslt.rs:216-338`
    calls `cublasLtMatmulAlgoGetHeuristic` on every call, takes the top
    result, and sets no reduction-scheme mask or pinned algorithm. Its output
    (M ∈ {1..5, 128} × N=288 × K=4096; `glm5_exl3_ffn/moe/forward.rs:79`) feeds
    expert top-k, so a 1-ulp change flips routing. Restrict the preference to
    a deterministic reduction scheme (no atomic split-K), cache the chosen
    algorithm per shape, log its ID/split-K/reduction scheme at first use,
    and add a repeat-bitwise test. The shared mHC TF32 Lt callers
    (`hc_prefill.rs:81`, `glm_hc_prewarm.rs:209`) share the global
    workspace; confirm they use the same stream.
  - [ ] **Fail closed on the TP all-reduce fallback.** `model/impl_a1.rs:616-620`
    only warns when the `bf16_add_inplace` lookup fails, then routes TP sums
    to `ncclAllReduce` (`comm_impl.rs:23,50`) with an unpinned NCCL tuner.
    Make startup assert which path is live (hard error for GLM-5.3 EXL3).
  - Note, not a hazard: the classic-cuBLAS shared/dense FFN profile is
    pinned (`gpu/gemm.rs:20-26`). Batch-composition variance (thin/fat
    MoE threshold, EXL3 row buckets, mHC split policy) is deterministic for
    a fixed batch and matters only under concurrency.

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
    prepared in `p0-trace-prep-20261011/` (11 CPU tests, re-verified);
    **no live profiler run yet**. CPU command review complete 2026-10-13
    (`p0-trace-prep-20261011/REVIEW-20261013.md`, first pass ready for
    testing): interfaces/descriptors PASS.
    **2026-10-15:** both owed decisions are taken and the schedule ran
    (3/3 runs complete, both ranks).
    [Attribution record](docs/design/p0-attribution-20261015.md), observed
    instrumented and single-shot:
    - Both phases are GPU-bound: kernels are busy for 95–100% of each phase.
    - Prefill is about 97% packed EXL3 GEMMs (routed MoE 55%, projections 42%).
    - Decode is projections 45%, NCCL SendRecv 25% and routed MoE 21%.
    - Still owed for P0: NCU counters on the dominant kernels, ≥ 3 matched
      uninstrumented repeats with independent timing, and the targets.
    - Image pinned to `src-4ee96c12b44c` (`sha256:82173ea6…`), the newest
      image with a full neutrality proof. The production `src-48624a8605de`
      is the 2026-10-10 build.
    - Three-run schedule as reviewed: 128/64, 3500/64 and 128/600 tokens.
    - The re-pinned driver copy and its runs are in `p0-trace-20261015/`; the
      original prep is unchanged.
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

- [x] Split `crates/spark-runtime/src/weights/exl3/norm_substitution.rs`
  (471/500 LoC) before it grows (review nit n1). **Done:** `05dcaca8` —
  manifest parse/validation/staging moved to
  `norm_substitution/manifest.rs`; main file now 304/500 LoC, no
  behaviour change. First pass implementation ready for review and testing.
- [x] The `exl3_checkpoint()` term of the substitution-engagement
  predicate (`crates/spark-model/src/factory.rs`) is covered by inspection
  only; give it a GPU-side test (m3 follow-up). **Done:** `2fc58662` — a
  real `Exl3CheckpointRouter` over a minimal on-disk EXL3 fixture, attached
  via `WeightStore::from_exl3` with no companions; deliberately mock-GPU so
  it runs on CPU CI, not only on a Spark. First pass implementation ready
  for review and testing.
- [x] Confirm which image the stopped `glm53-exl3-atlas-*` containers
  reference (`docker inspect` on both hosts). **Answered 2026-10-12:**
  both `glm53-exl3-atlas-0` (spark0) and `glm53-exl3-atlas-1` (spark1)
  reference `atlas-exl3-validation:src-48624a8605de-rust1.97.1`
  (`sha256:82b23cd19e59…`, the recorded production image); all containers
  exited, both GPUs idle. Receipt:
  `$EV/host-inspection-20261012T2204Z/` (driven by
  [scripts/inspect-sparks.sh](scripts/inspect-sparks.sh)).
- [x] Triage the local failure in
  `spark-model factory::glm_dense_cache::tests::dense_prefill_bf16_explicit_flag`.
  It is unrelated to this work: the test expects `"2"` to be rejected, but
  `3be435f7` made `=2` valid. **Done:** `a2d36244` — 1/2 asserted valid, 3
  moved to the invalid set.
- [x] Decide the fate of the untracked `docs/plans/atlas-kv-{coord,cache-server}-plan.md`
  (an unrelated KV-coordinator design, 2026-10-10). **Decided:** commit
  them together with the column-codec plan and `assumptions.md` (all
  cross-referenced from this file) — `2a3b995b`.
- [ ] Repo-health (new 2026-10-13, unrelated to M1): `cargo fmt --check`
  under BOTH the pinned 1.93.1 and current stable rustfmt flags ~180
  committed files, and current-stable clippy errors in
  `cublaslt/grouped.rs`, `weights/appended_skip_tests.rs` and
  `weights/exl3/dense_concat_tests.rs`. CI uses moving `stable`, so this
  drift will surface there; needs its own cleanup PR, deliberately not
  bundled with this batch. Operator decision 2026-10-14: drift also stays
  out of the L1/v24 tooling batch (verified there by "no new findings in
  changed files"); `typos` carries the same pre-existing drift class (56
  findings outside changed files).
- [x] GPU-iteration tooling (new 2026-10-14; CPU-only, full suite green):
  one-command image lifecycle
  (`tests/exl3_validation/image_cycle.py`), G8/G9 provenance receipt
  (`g8g9_binding.py`), and the probe pre-Spark package/records
  (`probe_wiring.py`, `probe_sitecustomize.py`, `probe_package.py`,
  `probe_records.py`). Records: `$EV/probe-runtime-mechanism-20261014/`;
  decisions/assumptions entries 2026-10-14. Follow-up lives in the probe
  registration's owed section (profiler bind, NCCL live, per-attempt stamp).

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
  fixed N, no top-ups, explicit H4 edge cases. Collection completed (26 attempts,
  832 receipts); corrected evaluator returns UNVERIFIABLE at reference H3/H4 LOO.
  See the qualification report; Q2 remains open.
- Atomic `Exl3FfnRouter` diagnostic records (`651ff083`).

**Q3**
- OpenAI stream usage order (`4b49a9c4`).
- First-token seed, stop, grammar and completion-identity fixes
  (`7e04679f`, `f7791189`, `19551fc6`, `8334ab33`, `ee5eaf0f`); 15/15
  focused live slice (`merged-image-20261011-session2/grammar-r2/`).

## KV column-codec experiment — follow-ups (2026-10-13)

Experiment plan: [docs/plans/atlas-kv-column-codecs-plan.md](docs/plans/atlas-kv-column-codecs-plan.md).
Assumptions: [assumptions.md](assumptions.md). Nothing below starts before
Stage 0 of that plan passes.

- Full ClickHouse catalog sweep (ALP, FPC, ZXC on Grace ARM, GCD) if the
  Stage-0 ratio hypothesis holds — plan §8.
- Adaptive codec *prediction* if smallest-of-N encode cost ever matters.
- ZXC asymmetric codec for compress-once/decompress-many reload paths.
- Cross-replica compressed KV transfer format — link
  [docs/plans/atlas-kv-coord-plan.md](docs/plans/atlas-kv-coord-plan.md).

- DoubleDelta bit-exact verification (preliminary single-shot, not serious claim).
- Full ClickHouse catalog (ALP, FPC, ZXC, GCD) — enum variants exist, implementations stub; implement when H1 supported.
- Stage-1 integration test: live spill → reload under --high-speed-swap with codec=DeltaZstd; bit-exact with dequant.
- Measurement: n≥3 repeat on captured real-block corpora; fingerprint sidecars; block-header codec histogram.
- Stage-3 GPU-resident (contraindicated) — only if Stages 1–2 pass and roofline arithmetic predicts win.
