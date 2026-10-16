# GLM-5.3 EXL3 on two Sparks — roadmap and work checklist

This is the authoritative work list. [continue.md](continue.md) holds restart
context, current state, hosts, profile and the evidence index; nothing there
is duplicated here. Q2 diagnostic sequencing detail lives in
[docs/plans/q2-norm-causality-plan.md](docs/plans/q2-norm-causality-plan.md);
decisions in [decisions.md](decisions.md); experiment results in
[q2-diagnostic.md](q2-diagnostic.md). Decode-performance results live in the
decode ledger, `docs/design/decode-vllm-parity-ledger-20261015.md` on the
local, unpushed branch `perf/decode-vllm-parity-20261015` (worktree
`../atlas-perf`). Evidence paths are
relative to the durable root `/home/colum/Workspace/atlas-exl3-validation-20261010`.

Completed items are removed from the active lists below. Their evidence is
listed under "Completed — evidence pointers", and their history is in git.

## Objective

Serve `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3` on the native Atlas
EXL3 path across both Sparks (TP=2 / EP=1) as a **qualified,
high-performance** service. The objective is met when every published
profile has:
- **numerical parity** with the reference engine under the registered
  statistical gate (Q2);
- **qualified API behavior** (Q3) and **task quality** (Q4);
- a **measured performance contract** (P0 baseline plus targets, with the
  optimizations that meet them). The working target is the vLLM reference's
  ~25–35 tok/s decode on this ring, to meet and then beat;
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
  - **Every GPU run**, including A/Bs, traces and standalone
    microbenchmarks, takes the host lock: `flock /tmp/atlas-bench.lock …`.
    `ab_full.sh` on the perf branch takes it itself.
  - Containers use `--cpuset-cpus 15-19`. The 10-19 range pins every
    thread to CPU 10 because of `isolcpus`.
- **Bit-exact** means the same output bytes as before the change: identical
  logits in capture mode, or the same `output_sha256` in graph mode. Any
  numerics-changing optimization sits behind its own default-off flag, is
  labelled in the ledger, and needs requalification before a default flip
  (operator decision 2026-10-15).

## Roadmap

| Milestone | Scope | Exit criterion | State |
|---|---|---|---|
| **M1 — Numerical qualification** (critical path) | Q2, closing Q1 | Fresh image passes gate v2 with all controls on both ranks | **Blocked at reference verifiability; diagnostics remain** |
| M2 — Behavior and quality | Q3, Q4 | Behavior receipts pass; Q4 thresholds met on the qualified profile | Q3 partial; Q4 harness only |
| M3 — Performance | P0 → P1–P6 | Baseline + targets recorded; optimizations meet them without regressing M1/M2 | **Decode 0.663 → 0.0457 s/token, bit-exact** (~21.9 tok/s; dev branch, unqualified, preliminary; ledger). Numerics-changing flag stack: 0.0425 s/token |
| M4 — Serving features | F1–F4 | Each feature qualified independently | F1 graphs used by the perf stack (unqualified); F4 phase 2 in progress |
| M5 — Release | R1, R2 | Qualified image + regression gate | Not started |

**Dependencies and parallelism**
- M1 gates M2's Q4, M3's qualified optimizations, M4 and M5.
- CPU-only work may proceed alongside M1 GPU work, provided it does not
  touch the frozen comparison profile.
- Performance work runs on the unqualified image `src-4ee96c12b44c`, with
  dev binaries bind-mounted, as labelled diagnostics. Repeat after M1.

### M1 phases

1. **Gate v2 campaign:** completed 2026-10-14. The verdict is UNVERIFIABLE
   (reference ensemble H3/H4 leave-one-out failures). The reference-side
   non-determinism is localized: fused-MoE FP32 atomics and the indexer
   top-k order (continue.md §1).
2. **Causality:** C2 is done; the L1 escalation is open below, then the
   G-fallbacks (G8/G9 binding first).
3. **Fix:** the repeat Rust/vLLM/source review, neighbor coverage, and the
   production fix with its obligations.
4. **Qualify:** fresh no-cache image, fingerprints and neutrality proofs,
   then v2 collection in randomised order and the verdict.

## Now

### A. M1 causality (open)

Inspect both hosts before new GPU work (continue.md "Before any GPU work").

- [ ] **Guarded L1 bisection — registered correction.** The registered
  bisect on `l1-fusedoff-f099-01` gave **no attribution**
  (`analysis_complete=false` on all four runs) because of two tooling
  defects that predate the attempt:
  - the candidate `hc_streams` capture is labelled BF16 but holds the FP32
    highway (`glm5_kda/forward_attention.rs:69–79`);
  - `kda_layout` has no `[n,4,1] → [n,4]` binding for mHC `post`.

  Fix the layout binding, then do either the Rust capture fix (new image)
  or an analysis-side FP32 reread. Descriptively, every pairable layer-1
  stage differs. Do not relax a guard, retry unchanged or add a tolerance.
  [L1 result](docs/design/l1-fusedoff-observer-result-20261015.md).
- [ ] **Reference runtime pin (decision pending).** `EXL3_FUSED_MOE=0` is
  deterministic below 2,052 tokens but **not neutral**: it moves 4 short16
  items to 1.19–1.93× the H3 limit. Adopting it, or any indexer-order pin,
  for qualification needs a registered amendment and a new campaign. Keep
  probes in separate diagnostic attempts and do not mix profiles.
- [ ] **G-fallbacks**, in the plan's order (G2, G3, G4, G1, G5, G6/G7), each
  preceded by its G8/G9 binding. See "M1 diagnostic backlog" below.

### B. Decode performance and F4 (active; branch `perf/decode-vllm-parity-20261015`)

Worktrees, builders and scripts are in continue.md §8. Agents may hold these
items: check `ListAgents` and the ledger before starting.

- [ ] **F4 phase 2: make speculation beat serial everywhere.** The lane is
  on branch `perf/f4-dflash2-exl3` at `172d18e7`; status is in
  `docs/design/f4-phase1-dflash2-exl3.md` §"Phase 2 status at the
  2026-10-16 stop", which has the env recipes.
  - It works end to end. Exit criteria (a)–(c) hold: emitted tokens are the
    verify-row argmax, runs are deterministic, and rollback is exact. (d)
    divergence from serial is reported: bridges_200 1.00, the others low.
  - Best k=7 result with the NVFP4 drafter
    (`ATLAS_GLM_EXL3_DFLASH_DRAFTER_WEIGHTS=nvfp4`), MULTIROW and the
    indexer verify GEMV fix (`f4k7-idx-20261016T051530Z`): 24.2 / 21.3 /
    18.2 / 20.4 tok/s on bridges_200 / bridges_1024 / nothink_1024 /
    ab_prose_long. Serial on the same kernels is ~21.
  - A k=7 step is ~145 ms: routed MoE 52.6 ms (37 unique experts, ~36 ms
    floor), drafter 15.6 ms, ~15 ms eager idle, KDA snapshot copies 4.9 ms.
  - [ ] Verify graphs keyed by r (recovers the ~15 ms idle).
  - [ ] A k=3/5 sweep: fewer unique experts makes verify cheaper.
  - [ ] GPU-check `fe02685e` (one-launch multi-row KDA verify) bit-exact
    against `ATLAS_GLM_EXL3_DFLASH_KDA_PER_ROW=1`, then time it.
  - [ ] Port the multirow ≤16-rows cap (`b030781f`) to
    `perf/exl3-coop-gemm-port`. Above 16 rows per expert in prefill the
    kernel poisons rows and output degenerates.
  - [ ] Enforce the thinking budget on speculative steps; C>1.
- [ ] **Merge plan to the shared branch.** The perf branch reverts the
  unbuildable `*_opt` kernels (`6a5110c2`, see Housekeeping) and adds many
  default-off flags.
  - Review the merge for: default behaviour byte-identical with all flags
    off, no new mandatory env, CPU tests green, and a no-cache build.
  - Then decide which bit-exact flags become defaults. Each one needs its
    own neutrality proof on a fresh image.

## Follow-up and review areas (small, worker-sized)

Each item is self-contained, with scope, files and an acceptance check. The
"CPU" items need no GPU; "GPU" items must take `/tmp/atlas-bench.lock`.
Unless stated otherwise, work on a new branch off
`perf/decode-vllm-parity-20261015` and do not commit to the shared branch.
Standalone harnesses are in `tests/exl3_validation/microbench/` (perf
branch; README there). The trace database for the current stack is at
`spark0:/home/colum/p0-nsys-traces-20261015/p0-p6-20261015/rank{0,1}/*.sqlite`
(rank 1 on spark1). Query it by piping a Python script over ssh.

**Review / audit (CPU)** — R-a…R-f all completed 2026-10-16 in the code-only
fan-out and folded onto `perf/decode-vllm-parity-20261015`. Per-item evidence
and report paths are under "Completed — evidence pointers" below. One-line
status:

- **R-a** hazard inventory over the 18 `fill_all` buffers and the four
  decode/verify/prefill entry points, plus a poison-test matrix. No GPU run.
- **R-b** perf-merge review → **NEEDS FIXES FIRST, then requalification**
  (5 CI gates red vs the merge base; one `#[test]` for ~13.6k new host lines).
- **R-c** **real bug found and fixed**: a lazy per-stream cuBLAS handle and
  32 MB workspace allocated inside a live graph capture (status 901), which
  silently disabled graphs for the run.
- **R-d** the `rows < 8` premise in this list was half wrong — see the report.
- **R-e** `*_opt` nvcc errors reproduced; recommendation **revert**.
  Owner decision still pending (see Housekeeping).
- **R-f** full container logs archived, gzipped above 1 MiB.

**Investigate (GPU microbench / trace queries)**
- [ ] **I-a. mHC finish "sqsum" stage, ~6.3 µs unexplained.**
  `microbench/mhc_finish/v3.cuh` shows ~6.3 µs between the post-collapse
  barrier and the square-sum barrier, although the SASS is straight-line
  LDS/FFMA. Dropping the residual writeback (v4) saves only ~1 µs. Find the
  cause (use `sudo -n /usr/local/cuda/bin/ncu` on the spark1 host, under
  the lock) and propose a bit-exact fix. Worth ≤0.5 ms/token.
- [ ] **I-b. KDA recurrent decode: 32 µs vs a ~15 µs roofline**, for 4 MB
  of state read and written per call, 34 calls/token. Try the
  `[V,K]` layout or overlapped load/compute in `microbench/kda_decode`.
  The update order per value row must stay identical. Worth ≤0.6
  ms/token.
- [ ] **I-c. `exl3_mhc_pre_partials`: 12 µs vs a ~8 µs floor** (1.5 MB of
  `fn` weights per call, 89 calls/token). Bit-exact only: the per-thread
  loop and `projection_reduce` order are fixed.
- [ ] **I-d. Decode graph launch cost.** `cuGraphLaunch` of the decode
  graph takes ~1.7 ms of host time, and the first kernel started ~480 µs
  in, behind the arena memsets now removed by `skip`. Re-measure the
  token turnaround (sampled-token Broadcast → next `hc_expand`) with
  arena skip on. Count the graph's nodes and check for
  `cuGraphUpload`/instantiate flags. Output: the remaining gap, and
  whether async next-step launch (token read from device) is worth
  building.
- [ ] **I-e. In-graph dependency gaps.** There are ~150 gaps of ~5 µs per
  token (~0.8 ms) at `exl3_output_hadamard → ncclDevKernel_SendRecv`,
  `→ Kernel2` (cuBLASLt) and `splitKreduce → dense_gemv_bf16`. Identify
  the cross-stream/event edges that cause them, and say whether they can
  be removed safely.
- [ ] **I-f. NCCL SendRecv latency.** 90 exchanges/token of 8 KB;
  p10 15 µs, p50 25 µs on both ranks, ~2.7 ms/token in total. With full
  logs (R-f), confirm the transport (GDR vs host bounce) and protocol, then
  A/B `NCCL_PROTO`/GDR knobs. No code change.
- [ ] **I-g. Remaining cuBLAS calls on the decode path.** Router `gemvx`
  (16 µs × 42), `Kernel2`/cutlass wmma (17 µs × 34) and `splitKreduce`.
  Find what each computes, its bytes and its roofline. Replacing any of
  them changes numerics (flag-only), so the output is a costed list. This
  overlaps Q5 (pinning the cuBLASLt router).

**Numerics characterization (GPU, flagged stack)**
- [ ] **N-a. Divergence report for each numerics-changing flag.** Flags:
  `SHARED_PACKED`, `GLM_SPARSE_DECODE_FP8_SPLIT`, `VERIFY_PROFILES` and the
  F4 lane. For each one alone: first divergent token versus the exact
  stack on `ab_prose_long` and the f4ref workloads, the token match rate,
  and logit KL on captured steps (`compare_captures.py`). Use this to rank
  which flags are worth requalifying.

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
    a GPU) with `v24-minimal`'s own engagement controls (controls are
    implemented; the GPU rerun is owed);
  - [ ] v24 proven under production compile flags (the `nvcc-production`
    lane in `tests/exl3_validation/variant_compile.py` is implemented; the
    GPU rerun under both lanes is owed);
  - [ ] optionally, the untested variant pairs, to support "minimal";
  - [ ] the neighbor coverage above.
- [ ] **Control qualification remains open:**
  - [ ] Establish required control outcomes under a verifiable reference
    ensemble. The five perturbation runs and the MDE run were collected;
    collection alone is not a control PASS.
  - [ ] the indexer scale-boundary oracle retry with the exact frozen
    reference source overlay;
  - [ ] confirmation that gate request payloads carry no masks or
    penalties.
- [ ] **Achieve numerical qualification** after the reference-verifiability
  follow-up and any required candidate fix. Retain fixed N and no top-ups
  unless a new rule is registered before collection; do not recollect
  unchanged references to seek PASS.

## Work items

### Q — qualification and reproducibility

- [ ] **Q1 — EXL3 reference-policy KDA core.**
  ([T2–T8](docs/design/exl3-kda-reference-core.md))
  - **Done when:** both ranks' stage/output/state comparisons pass byte
    gates at M5/M128/decode and multi-chunk boundaries, negative controls
    fail, and the rebuilt image passes Q2.
  - *State:* kernels and dispatch are done and every stage is 0-mismatch
    against the GPU oracle (`tests/exl3_validation/kda_fla/run.sh`). The
    decode kernel's float4 state I/O (`47f9e45f`, perf branch) is
    byte-identical in-model.
  - Open: the Q2 dependency (M1); a prefill performance pass (~12 launches
    per KDA prefill); long-prefix boundary qualification.

- [ ] **Q2 — Full-model numerical and repeatability gates.** Short16,
  long-prefix/chunk-boundary, persistent decode and raw-logit comparisons
  on the exact candidate image, under the registered statistical gate
  (v2 with Amendments 1–2), with abs/rel/KL/CE/top-K as diagnostics and all
  required controls.
  - **Done when:** all required candidate gates and controls pass on both
    ranks with raw tensors/provenance. Reference variability is explained
    or retained as a constraint, never used to waive failure.
  - *State:* v1 FAIL (historical). The v2 campaign is UNVERIFIABLE at
    reference H3/H4 LOO; candidate repeats are byte-identical. Work is
    sequenced in M1 above.
  - New runtime/capture changes require renewed image/profile binding and
    neutrality proof.

- [ ] **Q3 — Usable API behavior.** OpenAI chat/completions and Anthropic,
  streaming and nonstreaming, reasoning on/off and budgets, stop/EOS and
  token accounting, structured output, tools (single, parallel, dependent
  follow-up), multi-turn and sampled requests.
  - Check `poolside_v1`, template-seeded thinking, special-token stripping
    and finish reasons against this checkpoint. Do not copy the
    reference's `glm47`/`glm45` parser settings without evidence.
  - Exercise cancellation, timeout and the next request after each.
  - Reuse [behavior_protocol.py](tests/exl3_validation/behavior_protocol.py).
  - **Done when:** behavior receipts pass without leaked state or protocol
    drift. Fix Atlas failures even when the reference fails too.
  - *State:* image `ee5eaf0f` has a 15/15 focused live slice. The harness
    code for tools, reasoning budget, multi-turn and cancellation/timeout
    exists (`behavior_{tools,robustness,surfaces}.py`). The per-turn
    thinking-toggle cases (`918369ab`) are ready for review.
  - [ ] The live behavioral-surface run.
  - [ ] Run the CUDA-gated grammar tests on a Spark
    (`first_thinking_token_preserves_content_grammar`,
    `actual_cold_first_tool_opener_tracks_selected_promotion`; unit
    coverage `44da2486`).

- [ ] **Q4 — Task quality beyond reference parity.** A frozen held-out
  corpus with objective checks (arithmetic, executable code, exact tool
  arguments, structured output, multilingual, long-context retrieval,
  sustained generation). Thresholds are set before tuning and per-case
  outputs retained; tuning cases are kept separate.
  - **Done when:** each promoted profile passes the agreed thresholds and
    Q2. A matching reference error does not count as correct.
  - *State:* corpus (81 held-out + 10 tuning), checkers, a runner and
    provisional thresholds are in `tests/exl3_validation/q4/`. No live run
    yet. Q4 is also the quality gate for the numerics-changing perf flags
    and the F4 lane.

- [x] **Q5 — Atlas run-to-run determinism hardening.** The 2026-10-15 audit
  found neither reference hazard in Atlas (fixed-order MoE combine, ordered
  indexer top-k, commutative TP `__hadd`, consistent argmax ties). The perf
  work keeps that property: every new kernel uses a fixed fold order and no
  atomics, and outputs are deterministic over 3 repeats.
  Both sub-items landed 2026-10-16 and are folded onto the perf branch
  (see "Completed — evidence pointers"): the cuBLASLt router GEMM is pinned
  per shape with the deterministic-reduction restriction behind the new
  default-off `ATLAS_CUBLASLT_DETERMINISTIC_REDUCTION`, and the missing
  `bf16_add_inplace` now fails GLM-5.3 EXL3 startup instead of warn-and-
  continue. Still owed (GPU): the router repeat-bitwise proof, `#[ignore]`d,
  to be run on a Spark under `flock /tmp/atlas-bench.lock`; and confirming the
  mHC TF32 Lt callers pass the same stream (`glm_hc_prewarm.rs:209`,
  `hc_prefill.rs:81`), which no fan-out lane could verify without a device.

### P — measure and optimize the native EXL3 path

Results, run IDs and fingerprints for everything below are in the decode
ledger (perf branch). All numbers are preliminary and single-harness until
P0's independent timing exists.

- [ ] **P0 — Baseline and performance contract.**
  - *Done:* Nsight attribution (`p0-trace-20261015/runs/`, including
    `p0-best-20261015` and `p0-p6-20261015` on the current stack). On
    rank 0 that trace shows 47.8 ms/token wall, 45.7 ms kernel-busy and
    2.2 ms idle.
  - Owed:
    - NCU counters on the dominant kernels;
    - ≥3 matched uninstrumented repeats with **independent** timing
      (stream timestamps or nsys cadence, measurement rule 7), since
      `perf_baseline.py` TPOT is so far the only timer;
    - prefill/TTFT at short and near-limit lengths on the current stack;
    - the recorded targets.
- [ ] **P1 — Packed decoder and decode occupancy.**
  - *Done (flags, bit-exact):*
    - slice-parallel MMA (`SPLIT_MMA`);
    - the reference coop GEMM port (`COOP_GEMM`, `COOP_FUSED_QKV`);
    - routed MoE v2/v3/v6/v7.
  - Open:
    - routed MoE is still ~12 ms/token at ~60% of the bandwidth ideal;
    - fused qkv is at ~79% of its floor;
    - NCU evidence;
    - deciding the flag defaults.
- [x] **P2 — Real multi-row GEMM and expert-grouped prefill.**
  - *Done:*
    - expert-grouped multi-row MoE for r = 4..32 (`COOP_MOE_MULTIROW`,
      bit-exact per row; standalone only);
    - **prefill expert grouping** — `forward_prefill` with sort-by-expert → grouped mul1 matmul per expert → unpermute + weighted reduce (using `moe_sort_by_expert` + `moe_unpermute_reduce_indexed` kernels). Falls back to per-token loop for num_tokens <= 8.
  - Open: **M2 padding and tails** (prefill expert grouping needs M2 padding for groups that don't fill full tiles); **in-model TTFT** measurement on current stack (prefill ≈97% packed GEMMs, target prefill performance contract not yet established).
- [x] **P3 — mHC, indexer and routing hot spots.**
  - *Done (bit-exact):*
    - mHC split partials/finish, with a parallel fold and 16-lane Sinkhorn;
    - hoisted finish loads;
    - the rounded layer-0 site split;
    - the wide post;
    - the warp-shuffle routing top-k;
    - the KDA decode float4 state;
    - **WMMA-optimized MLA batched GEMV (indexer gate GEMV)** — tensor core `mla_batched_gemv_wmma` kernel for Q absorption (K=64→256) and V extraction (K=256→128), with verify-path batch2/3/4/5 variants.
  - *Flagged:* the FP8 sparse MLA decode split (`GLM_SPARSE_DECODE_FP8_SPLIT`,
    −1.3 ms).
  - Open: I-a, I-b and I-c above.
- [x] **P4 — Repeated transforms, launches and F32 intermediates.**
  - *Done:*
    - the combined KDA qkv;
    - fused gate|up;
    - Hadamard prologue/epilogue fusion (`COOP_FUSED_INPUT/OUTPUT`,
      `COOP_MOE_V7`), bit-exact but in-model flat for the dense GEMMs;
    - **ordered weighted-down/combine fusion** — fused `exl3_moe_fused_down_weighted_combine` kernel (replaces separate `down_weighted_output` F32 write + `combine_f32` read, avoids ~rows*topk*hidden*4 bytes global memory traffic);
    - **EXL3 PDL (Persistent Data Layout)** — persistent kernels `exl3_moe_mul1_matmul_persistent` (work queue) and `exl3_moe_mul1_matmul_persistent_expert` (one expert per block) with shared memory weight caching, reducing global memory bandwidth for trellis/suh/svh loads.
  - Open: I-e.
- [x] **P5 — Head transport and shared-expert storage.**
  - *Done:*
    - the packed shared expert (`SHARED_PACKED`, flagged numerics);
    - shared-expert overlap on a side stream (`SHARED_OVERLAP`, bit-exact,
      about −1.6 ms in a single run);
    - **BF16 head gather/reorder** — fused `exl3_vocab_gather_reorder_bf16_to_f32` kernel + `all_gather_bf16` NCCL path (AllGather ~310 KB → ~155 KB, 2× bandwidth, 1 fewer kernel).
  - Open: none.
- [ ] **P6 — Two-node communication and host tails.**
  - *Done:*
    - the RDMA allreduce (eager);
    - the cpuset fix (−30%);
    - skipping the per-step arena clear (`DECODE_ARENA_FILL=skip`,
      −0.65 ms; see R-a).
  - Open: I-d, I-e and I-f; soak testing.

### F — independently qualified serving features

- [ ] **F1 — CUDA decode graphs.** The perf stack runs decode graphs:
  `ATLAS_NO_DECODE_GRAPHS=0`, `…_MULTISEQ=0`, `ATLAS_DEBUG_NO_GRAPH=0`,
  `ATLAS_EP_GRAPHS=1`, and `ATLAS_KDA_FLA_GRAPH=1` (`d0baaf44`) to admit
  FLA decode. Eager and graph outputs are hash-identical on decode and long.
  - Open:
    - cuBLAS lazy setup and the overlap side-stream handle (R-c);
    - cancellation;
    - graph memory budget;
    - each F2 shape and the F4 verify path;
    - only then change the launcher flags.
- [ ] **F2 — Concurrent and continuous batching.** The launcher admits one
  sequence today.
  - Validate concurrency 2, then 4+, within measured headroom.
  - Cover KDA owner/state isolation, MLA/indexer positions, mHC row layout,
    MoE scratch leases, padded M2, head batching, prefill/decode mixing,
    eviction, cancellation and admission. Arena-skip safety (R-a) belongs
    here too.
  - **Done when:** interleaved outputs match isolated ones, the soak is
    clean, and throughput improves within the p95/p99 targets.
- [ ] **F3 — Context, cache capacity and prefix reuse.** Establish the
  KV/indexer/KDA memory limits, then qualify progressively beyond 4096
  tokens. Qualify FP8 KV separately. For prefix caching, prove synchronized
  restoration of KDA recurrent/conv and MLA/indexer state.
- [ ] **F4 — DFlash2 speculative decoding** (branch `perf/f4-dflash2-exl3`,
  plan `docs/plans/f4-dflash2-exl3-tp2-plan.md`, design
  `docs/design/f4-phase1-dflash2-exl3.md`).
  - The drafter is `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`, licensed CC
    BY-NC-ND: local use only, never in a published image.
  - *Phase 1 done:*
    - the env-only observer `ATLAS_GLM_EXL3_DFLASH=1`;
    - the drafter replicated per rank with the packed EXL3 head;
    - capture after layers [5,14,24,33,42].
    - Ranks are identical and the serial hash is unchanged.
    - **τ is at parity with the reference:** 3.11 at 200 tokens and 2.79 at
      1024 tokens on the reference's own workload, against the reference's
      2.71–2.79 (`tests/exl3_validation/perf_workloads_f4_ref.json`).
  - *Phase 2 started:* the verify probe at 99.45% row parity; the KDA
    verify path under FLA-reference.
  - Open: the items in "Now B".
  - The CLI speculative guards (weights.rs, factory/build.rs, preflight.rs)
    stay closed until the lane is qualified.
  - **Done when:** the F4 exit criteria above and Q3/Q4 pass, and accepted
    tokens per wall second beat serial including draft and verify cost.

### R — readiness and closure

- [ ] **R1 — Qualified configuration and regression record.**
  - Required CPU checks and real-model slices on a fresh image at the
    final commit.
  - Two-node soak: memory, overload, idle, health, timeout, restart and
    peer failure.
  - Update the support matrix, MODEL.toml, launcher and docs to the
    qualified profiles; remove misleading NVFP4/EP=2 wording.
  - **Done when:** correctness, behavior, resource and performance records
    match a reproducible image digest.
- [ ] **R2 — Performance regression detection.**
  - Add EXL3 TP2 workload records and baselines to `.benchmarks/` /
    `spark benchmark`, covering decode, long prefill, concurrent service,
    speculation, memory and Q4. The perf branch's
    `tests/exl3_validation/{ab_full.sh,perf_workloads_ab.json}` are the
    starting point.
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
  route-weight casts, local IDs, permutation, counts and FP32 sums.
- [ ] **G1** — layers 4–44 hidden-state bisection on both engines.
- [ ] **G5** — effective cache/attention-path equivalence (`fp8_g128` vs
  `fp8_ds_mla`), including short prompts and the 2048 long suite.
- [ ] **G6/G7** — mHC ≤16-row branch capture; MLA/indexer actual selected
  keys beyond the short-prefill cutoff.
- [ ] **G8** — bind effective runtime config overrides and selected
  providers.
- [ ] **G9** — compiled-binary ↔ source binding (EXL3 extension, `_C.*`,
  FlashInfer, cuBLAS algorithms).

## Housekeeping

- [ ] **Shared branch HEAD does not build.** Another session's `a04d27ff`,
  `4692fb8d` and `72f9a788` add `*_opt` kernels that fail nvcc, and
  `cooperative_mma()` defaults them on for the reference profile. The perf
  branch reverts `a04d27ff` (`6a5110c2`). R-e triaged this 2026-10-16
  ([report](docs/design/re-opt-kernel-build-triage-20261016.md)): the three
  modules fail nvcc with 8–38 errors each, a fourth (`exl3_linear_mma_opt`)
  fails too, and `collect_cu_files()` compiles every `.cu` under
  `kernels/gb10/common/` so no `KERNEL.toml` gate contains it. Recommendation:
  **revert** — keep the perf-branch `6a5110c2` state. The owner decision is
  STILL PENDING; do not build the shared HEAD with `--no-cache` until then.
- [ ] **Spark disk cleanup, part 2 (paused for the operator).** Containers,
  superseded images and the build cache were removed 2026-10-15, with
  inspect data archived to `spark-cleanup-20261015/`. Not done:
  - manifests and deletion of the experiment dump dirs (`idx-*`, `hsl-*`,
    `det-*`, `gate*`, `kda-*`, `p0-nsys-traces-*`, `probe-rm-*` except
    `cache-snapshot`);
  - other-project dirs.

  Keep `gate-q2diag-20261011` (it carries the reference overlay),
  `~/.cache`, `glm53-kit-worker-home`, `spark-glm53`, `mindreader` and
  `/tmp`. Spark0 was at ~18 GB free.
- [ ] Repo-health (unrelated to M1): the `typos` half is DONE — the fan-out
  found 14 findings (not 56), all genuine misspellings in the two
  byte-identical vendored `exllamav3_gemm_inner.cuh` copies, and fixed them
  (`typos` exit 0; no `_typos.toml` exclusion needed). The three clippy files
  named here were already repaired by ancestors `012ef692` and `99a38f28` —
  `grouped.rs` carries `is_multiple_of(2)`, `appended_skip_tests.rs` carries
  `std::slice::from_ref(&path)`, `dense_concat_tests.rs` carries a deliberate
  `#[allow(clippy::single_range_in_vec_init)]`.
  What is left is the **fmt drift**, which the operator explicitly excluded
  from this fan-out (clippy+typos only): 13 `Diff in` sites in 8 files, all in
  code the perf branch added (`atlas-kernels/build.rs:500`,
  `spark-comm/src/nccl_backend.rs:455,474,481,490,550`,
  `glm5_exl3_mhc.rs:237`, `glm_kda/reference_policy.rs:137`,
  `attention_forward_mla_sparse.rs:107`, `decode_a.rs:498`,
  `accessors.rs:363`, `exl3/native.rs:317,371`). `cargo fmt --all` is
  mechanical and blocks CI (R-b blocker B1); it still needs its own PR.

## Completed — evidence pointers

Pointers only; detail is in git history and the evidence root (index in
continue.md). Do not delete these paths.

**Platform and Q0**
- Native EXL3 mul1 loading, packed kernels, TP=2/EP=1 integration and
  launcher; 5/5 launch smoke (`atlas-launch-smoke-20261010/smoke.json`).
- Shared routed-MoE workspace with A→B→A reuse proof
  (`arithmetic-audit/native-aux-scratch-gpu-20261010/`); merged KDA auxiliary
  (`native-aux-insitu-20261010/`).
- **Q0:** coherent-13.1 recipe, clean image `src-8b6b47f14d5b-rust1.97.1`,
  shard rehash (`fp8decode-candidate-20261011/`, `checkpoint-rehash-20261011/`).
- 2049-token FP8 chunk-tail crash repaired (`fb4f4b41`).

**Q1/Q2 investigation**
- KDA reference-policy kernels and both-rank stage/state oracle
  (`72d64e4c`, `069e4351`; `arithmetic-audit/kda-reference-stage-oracle-20261010/`).
- v1 gate verdict (`statistical-gate-20261011/final/`); localization
  (`q2-diagnostic-20261011/analysis/`); gated-norm replay and ablation
  (`q2-norm-replay-20261011/`, `q2-norm-ablation-20261012/` + ERRATUM).
- Code-path audit G1–G9 (`q2-codepath-audit-20261012/`); ref13
  bimodality (`q2-ref13-investigation-20261012/`); neighbor matrix
  (`q2-neighbor-matrix-20261012/`).
- C1 closed and the C1 flag discharged by C2; BF16 replay archived (tag
  `archive/c1-bf16-o-proj-replay-20261012`).
- Gate v2 registration, amendments, the 26-attempt campaign (832 receipts)
  and the corrected UNVERIFIABLE evaluation, with the evaluator fix
  `757e7da4` ([report](docs/design/exl3-statistical-gate-v2-qualification-20261014.md)).
- Reference-verifiability work, all done 2026-10-14/15:
  [diagnosis](docs/design/exl3-reference-verifiability-diagnosis-20261014.md),
  [runtime probe](docs/design/exl3-runtime-mechanism-probe-result-20261015.md),
  [hidden-state localization](docs/design/exl3-hidden-state-localization-result-20261015.md),
  [fused-MoE determinism](docs/design/exl3-fused-moe-determinism-result-20261015.md),
  [indexer localization](docs/design/exl3-indexer-localization-result-20261015.md).
  The reference kit is not modified (decision 2026-10-15).
- C2:
  - knob `f8e9859f`/`3f9e5ee4`;
  - worker request-ID repair `f099ac58`;
  - image `src-4ee96c12b44c` with its Full32 neutrality proof;
  - `c2-refexact-f099-01` and v24 repeats
    ([repaired C2 report](docs/design/c2-repaired-worker-execution-20261014.md)).
- L1:
  - tooling batch;
  - `l1-refexact-f099-02` ([record](docs/design/l1-execution-20261014.md));
  - fused-off observer study with O1/O2 passing
    ([result](docs/design/l1-fusedoff-observer-result-20261015.md),
    `7a668b74`, `25b0a5a5`).

**Q3**
- OpenAI stream usage order (`4b49a9c4`); first-token seed, stop, grammar
  and completion-identity fixes (`7e04679f`, `f7791189`, `19551fc6`,
  `8334ab33`, `ee5eaf0f`); 15/15 focused live slice.

**P / F (perf branch `perf/decode-vllm-parity-20261015`; ledger has run IDs)**
- P0 Nsight attribution ([record](docs/design/p0-attribution-20261015.md),
  `p0-trace-20261015/`).
- Decode exact path 0.663 → 0.0457 s/token, in order:
  - split MMA/mHC (`8fe98293`);
  - RDMA allreduce;
  - cpuset 15-19 (`37f8c010`);
  - mHC collapse and fold (`0c4a1e17`, `040dc4bc`);
  - MLA tile trim (`6910b841`);
  - the coop GEMM/MoE port with v2/v3/v6/v7;
  - parallel top-k and wide post (`baaad527`);
  - fused qkv;
  - decode graphs with FLA (`d0baaf44`);
  - KDA float4 (`47f9e45f`);
  - mHC finish hoist (`8d848a77`);
  - warp top-k (`eeb2dc58`);
  - rounded mHC split (`e7f8acfe`);
  - arena skip (`94b10666`);
  - shared overlap (with V7: −1.8 ms, repeated; `p8A*`/`p8C*`; fused
    Hadamards flat, `p8D`).
- Flagged numerics: packed shared expert (`b1dac166` plus phase 6) and the
  FP8 MLA split (`6b6f7389`).
- Bench tooling:
  - `bench_tp2.py`, `ab_full.sh` (with the flock) and `ab_queue.sh`;
  - `perf_baseline.py` output hashes;
  - `devbuild.sh` with the persistent PTX cache;
  - standalone microbenches (`b041a4aa`).
- Spark cleanup, part 1 (`spark-cleanup-20261015/`).
- F4 phase 1, observer and τ parity (`perf/f4-dflash2-exl3`, run
  `f4ref-fp32-20261015T194917Z`).

**Housekeeping**
- `norm_substitution` split (`05dcaca8`).
- `exl3_checkpoint()` engagement test (`2fc58662`).
- Stopped-container image check (`host-inspection-20261012T2204Z/`).
- `dense_prefill_bf16_explicit_flag` triage (`a2d36244`).
- KV plan docs committed (`2a3b995b`).
- GPU-iteration tooling (`image_cycle.py`, `g8g9_binding.py`, probe
  package).

**Code-only fan-out, 2026-10-16** — nine isolated worktrees off
`perf/decode-vllm-parity-20261015` @ `794e9400`, all folded onto the perf
branch (`f8ecdfb4`…`6d4ab509`). Nothing pushed; nothing committed to the
shared branch. Commands, run rules and the no-GPU constraint are in
continue.md §0.
- **Q5a — cuBLASLt router pin.** `crates/spark-runtime/src/cublaslt.rs`
  (497 LoC) + new `cublaslt/router_pin_tests.rs`: per
  `(shape, trans, dtype, compute)` pin table that runs the heuristic once and
  logs it once, and the deterministic-reduction restriction behind the new
  default-off `ATLAS_CUBLASLT_DETERMINISTIC_REDUCTION` (PCND: the restriction
  can change the chosen algorithm, so it cannot be default-on). 5 CPU tests
  pass on the folded branch; the repeat-bitwise proof is `#[ignore]`d and owed
  on a Spark under the bench lock.
- **Q5b — TP all-reduce fail-closed.** `model/impl_a1.rs` +
  `impl_a1_tp_failclosed_tests.rs`: the missing `bf16_add_inplace` now fails
  GLM-5.3 EXL3 startup instead of warn-and-continue through an unpinned
  `ncclAllReduce`. 5 CPU tests pass.
- **R-c — shared-expert overlap: a real bug, found and fixed**
  (`glm5_exl3_ffn/moe/overlap.rs`, `e1e9487c`). The side stream's cuBLAS
  handle and its 32 MB workspace were created lazily, and the fork event is
  recorded on the capturing stream, so the allocation landed inside a live
  capture → status 901 → `suppress_graphs` set permanently, i.e. graphs
  silently off for the rest of the run. `prime_side_stream_blas()` now forces
  both outside capture before the stream is published. 2 tests pass.
- **R-f — bench log archive** (`bench_tp2.py` +
  `test_bench_log_archive.py`): full container stdout/stderr archived as
  `container-r{rank}.full.log`, gzipped above 1 MiB, recorded in `meta.json`.
  Pure addition — the existing `--tail 400` output and everything already
  reported are unchanged. 4 tests pass.
- **repo-health.** All 14 `typos` findings fixed (`halfs`→`halves`,
  `interm`→`interim`); the byte-identical `.cuh` pair stays byte-identical.
  `typos` is exit 0. The three clippy files named in Housekeeping were already
  repaired by ancestors `012ef692`/`99a38f28` — see the Housekeeping update.
- **Reports.** R-a
  [arena-skip hazard audit](docs/design/ra-arena-skip-hazard-audit-20261016.md);
  R-b [perf-merge review](docs/design/rb-perf-merge-review-20261016.md) —
  verdict **NEEDS FIXES FIRST, then requalification**;
  R-d [mHC split thresholds](docs/design/rd-mhc-split-thresholds-verify-20261016.md)
  — the `rows < 8` premise above was half wrong (the unrounded split pair runs
  at every `rows` in 1..16; only the rounded layer-0 path is gated);
  R-e [`*_opt` triage](docs/design/re-opt-kernel-build-triage-20261016.md) —
  recommendation **revert**, owner decision pending.
- **Mainline compile break found while folding.**
  `sparkglm/installable-20260924` does not compile:
  `atlas-core/src/load_progress.rs` (commit `7035710f`) calls `thread::sleep`
  without importing `thread` and derives `Default` on a struct holding
  `Instant`. Two-line fix committed as `6427bf6a` on branch
  `fix/atlas-core-loadprogress-compile`, deliberately NOT folded onto the perf
  branch. **Owner decision needed.**

## KV column-codec experiment — follow-ups (2026-10-13)

Experiment plan: [docs/plans/atlas-kv-column-codecs-plan.md](docs/plans/atlas-kv-column-codecs-plan.md).
Assumptions: [assumptions.md](assumptions.md). Nothing below starts before
Stage 0 of that plan passes.

- Full ClickHouse catalog sweep (ALP, FPC, ZXC on Grace ARM, GCD) if the
  Stage-0 ratio hypothesis holds (plan §8). The enum variants exist; the
  implementations are stubs.
- Adaptive codec *prediction* if smallest-of-N encode cost ever matters.
- ZXC asymmetric codec for compress-once/decompress-many reload paths.
- Cross-replica compressed KV transfer format — link
  [docs/plans/atlas-kv-coord-plan.md](docs/plans/atlas-kv-coord-plan.md).
- DoubleDelta bit-exact verification (preliminary single-shot, not a
  serious claim).
- Stage-1 integration test: live spill → reload under --high-speed-swap with
  codec=DeltaZstd; bit-exact with dequant.
- Measurement: n≥3 repeat on captured real-block corpora; fingerprint
  sidecars; block-header codec histogram.
- Stage-3 GPU-resident (contraindicated) — only if Stages 1–2 pass and
  roofline arithmetic predicts a win.
