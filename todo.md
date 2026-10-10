# GLM-5.3 EXL3 — remaining high performance inference work

Reviewed against `11671adf` on 2026-10-10. Native TP=2 / EP=1 boot and generation
are achieved. Full numerical qualification, a repeated performance baseline,
and the expanded serving features below remain open. This is the authoritative
work checklist; [continue.md](continue.md) holds the launch profile and evidence.

Order: **Q0/Q1 → Q2–Q4 → measured optimization → feature qualification → R1/R2**.
P0 diagnostic profiling can proceed while parity is investigated, with the
candidate labeled unqualified. P1–P6 order is provisional until P0 identifies
costs. Each item should be a focused change with its own acceptance evidence;
this roadmap spans multiple PRs. Check an item only with a run/artifact ID tied
to the source and profile that passed.

## Already delivered — preserve and use

- [x] Native EXL3 mul1 loading, packed kernels, TP=2 / EP=1 KDA/MLA/MoE/mHC
  integration and launcher; five sequential nonstreaming chat smoke receipts.
- [x] Shared routed MoE workspace and ordered teardown; rank-local GPU guard and
  A→B→A reuse proof on both hosts. This does not qualify overlapping streams.
- [x] Opt-in FFN stage profiling; coherent cuBLAS 13.1 classic/Lt profiles;
  merged KDA auxiliary with recorded in-situ M1/M5/M128 comparisons.
- [x] KDA reference-policy design and CPU oracle groundwork. **No production
  enum, flag, reference CUDA kernels or dispatch yet.**

## Q — qualification and reproducibility

- [ ] **Q0 — Reproduce the working image from committed inputs.** Bring the
  external coherent-13.1 build recipe into the EXL3 Docker/build path. Package
  the pinned pair with hashes/licenses and explicit provider resolution; build
  the server and `exl3_moe_scratch_check` from one source snapshot. Bind source,
  Git commit, compiler flags, binaries, embedded PTX, image digest and checkpoint
  revision/shard hashes on both ranks. Complete the outstanding shard re-hash.
  **Done when:** a clean no-cache build boots both nodes, actual runtime mappings
  show only the intended pair, and smoke plus scratch GPU checks pass with archived
  manifests. Current launcher image is a preserved snapshot, not proof of a
  reproducible `11671adf` build.

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

- [ ] **Q2 — Close full-model numerical and repeatability gates.** Rerun frozen
  short16, long-prefix/chunk-boundary, persistent decode and raw-logit comparisons
  on the exact candidate image. Keep abs/rel/KL/CE/top-K/argmax/text requirements
  and negative controls. Retry the indexer scale-boundary oracle with the exact
  frozen reference source overlay; its latest attempt failed before GPU execution.
  Investigate reference repeat variation at routed combine independently and
  check candidate repeatability. **Done when:** all required candidate gates pass
  on both ranks, with raw tensors/provenance; reference variability is explained
  or explicitly retained as an unresolved constraint, never used to waive failure.

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
