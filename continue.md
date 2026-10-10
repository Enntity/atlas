# EXL3 / GLM-5.3 TP=2 continuation state

## LAUNCH ACHIEVED 2026-10-10 (resumed, launch-first goal)

User redefined the goal: launch first, parity second, performance last. Done:
Atlas TP=2 / EP=1 serving `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3`
on both Sparks via `scripts/start-glm53-exl3-tp2.sh` (new, in repo root
`scripts/`). Image `atlas-exl3-validation:src-48624a8605de-rust1.97.1`
(`sha256:82b23cd19e59…`), containers `glm53-exl3-atlas-{0,1}`, `--restart
unless-stopped`, took over port 8000. `/health` ready on both ranks. Smoke
5/5: greedy short, 612-token coherent long generation, multi-turn recall,
sampled. Production vLLM `glm53-exl3-head`/`-worker` preserved stopped as
fallback (not removed). Restart with the script; validate with
`tests/exl3_validation/launch_smoke.py`. Smoke artifacts:
`atlas-launch-smoke-20261010/smoke.json`.

Smoke throughput ~2.17 tok/s (612 tok / 281.7 s) is **preliminary,
single-shot, no repeats** under fingerprint attested-image 48624a8605de /
no-speculation / eager / max_batch=1 / BF16-KV / GB10-TP2 — not a benchmark;
perf work is deferred behind parity. Follow-ups (parity KDA core, M128
residual, perf areas) are in `todo.md`; reference-policy KDA core work is
running in parallel (design doc `docs/design/exl3-kda-reference-core.md`;
groundwork done — spec + plan T1–T8, CPU oracle
`tests/exl3_validation/test_exl3_kda_reference.py` 4/4 with verified
negative control; next is T2, the M5 prefill CUDA kernel).

Below is the prior paused state, retained for the parity/perf record.

Paused again 2026-10-10 at the user's request after resuming. Steps 1-6 done,
step 7 partial (see Resume sequence). No GPU work is in flight; both hosts
idle; production head/worker still `exited` (not restored). Work is
uncommitted; nothing was published. Do not resume until requested.

## Objective and current verdict

Support native EXL3-specific kernels in Atlas for
`MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3`, including multi-device
execution equivalent to vLLM **TP=2 / EP=1**, and complete actual GPU validation
and comparison on the user's spark0 and spark1.

**Full-model numerical qualification still FAILS** (short16 rerun on the new
image 2026-10-10). The coherent 13.1 BLAS pair and merged KDA auxiliary are
implemented, built, attested and byte-exact in situ at layer 0; the earliest
remaining divergence is the KDA recurrent core (see step 7).

- Repository: `/home/colum/Workspace/RustroverProjects/atlas`.
- Branch: `sparkglm/installable-20260924`.
- HEAD: `3a6de55f78ac0a60aabf27b7136ad0eb8106473a`.
- Durable evidence root: `/home/colum/Workspace/atlas-exl3-validation-20261010`.
- There are many modified/untracked files. Preserve all existing work,
  including the external reviews, `translate-tasks.md` and `follow-up-ideas.md`.
- Architecture/status document: `docs/design/exl3-support.md`.
- Harness documentation: `tests/exl3_validation/README.md`.
- Read `AGENTS.md`, `CONTRIBUTING.md` and the measurement-discipline skill
  before continuing. New Rust files must remain at most 500 lines with SPDX.

## Host and service state

Host inventory/configuration lives in `~/Workspace/homelab-ansible`.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 -J colum@192.168.1.50 colum@10.50.0.10
ssh -o BatchMode=yes -o ConnectTimeout=15 -J colum@192.168.1.50 colum@10.50.0.11
```

- spark0: `gx10-eb0e`; spark1: `gx10-235f`.
- Each is a GB10, actual 48 SMs, approximately 121 GiB unified memory.
- Driver 580.159.03; host CUDA 13.0 / nvcc 13.0.88.
- Fabric: `192.168.100.1` / `.2`.
- Previously used interfaces: spark0 `enp1s0f0np0` / `rocep1s0f0`;
  spark1 `enp1s0f1np1` / `rocep1s0f1`.
- No local GPU or CUDA driver. Local Cargo stable is 1.97.1; the pinned
  1.93.1 installation lacks Cargo, so prior checks used `cargo +stable`.

At pause, the GPU owner reports **no validation GPU work in flight**. All
shared/router/auxiliary validators exited; the latest indexer attempt failed
its source guard before Torch/GPU execution. No new Atlas worker was started.
Reference containers are stopped and preserved.

Final read-only pause inventory confirms `docker ps` is empty and the
`nvidia-smi` compute-process query is empty on **both hosts**. Original serving
containers have `.State.Status=exited` and remain intact. All three agents
finished their pause handoffs; they report no running tool sessions or GPU jobs.

**Original production is also still stopped, not restored:**

- spark0 container `glm53-exl3-head`.
- spark1 container `glm53-exl3-worker`.
- Original image `glm53-flash-mikeroz:dt-carry20`, image ID
  `sha256:6eb42fa20c145cabf9f788d7b850d8d3416706ad1955a003448b48962cb2845b`.
- Restore these preserved containers when validation is finished, and verify
  port 8000 health on both hosts. Do not silently replace them with a candidate.
- Original production uses FP8 KV, DFlash and high reasoning; the isolated
  reference comparison deliberately uses a different BF16/eager profile.

## Checkpoint and frozen validation contract

- Checkpoint revision: `dda99c7e935920fa6e235ce410e6f806e1c02720`.
- All 23 indexed shard SHA-256 values match on both Sparks; metadata stayed
  unchanged. Checkpoint staging includes spark0 `/home/colum/glm53-local`,
  the worker cache on spark1 and `/nfs1/hub`.
- H=4096, 45 layers: 34 KDA and 11 MLA; first routed layer 3; 42 MoE layers.
- 288 experts, sigmoid top-8, normalization 2.5, groups 1/1.
- Global expert intermediate 2048, local 1024; one shared expert.
- Physical vocabulary 154880; tokenizer size 154856 / max token 154855.
- `mhc_post_mult_value` is absent: the explicit named reference policy supplies
  2.0 and rejects conflicting checkpoint values.
- Numerical gates stay frozen: abs 0.15, rel 0.02, KL 0.005, CE 0.15,
  top-K 10, exact argmax and HTTP text. Do not relax/rebaseline them.
- Profile: TP=2 / EP=1, max context 4096, prefill chunk 128, BF16 KV,
  GPU admission C1, eager/no graphs, no prefix cache or speculation.
- Long prefixes: 2047/2048/2051/2053, 16 registered continuation rows,
  up to 2057 tokens. Graphs remain unqualified.

## Completed implementation and CPU evidence

Native EXL3 loading, kernel target selection, packed linear/MoE kernels,
KDA/MLA integration, TP head transport, scheduler chunk-budget fix and
source-bound diagnostic captures are in the working tree. Ordinary model
paths retain their existing dispatch. The shared build supports explicit
per-module flags; EXL3 mHC alone declares `--fmad=true` to match reference
sigmoid contraction. This is a cross-cutting build-script change.

### Shared routed scratch, profiling and teardown

- Model-owned `MoeScratch` in
  `crates/spark-model/src/layers/glm5_exl3_ffn/scratch.rs` supplies 11 distinct
  buffers shared between sequential routed layers on the validated stream.
- Checked authoritative geometry, Arc borrowing and an enqueue lease validate
  rank/world/stream/capacity before work. Production capacity remains **129**;
  128 is the configured prefill budget, not the allocation capacity.
- Allocates 60,468,096 bytes per rank once, versus the old 42-fold
  2,539,660,032 bytes. Static recovery: 2,479,191,936 bytes/rank. These are
  allocation arithmetic, not measured speedups.
- Actual two checkpoint-loaded FFNs are tested using the same addresses;
  rollback, failed free retry and unknown completion are covered.
- KV sizing already observes actual free GPU memory after loading.
- FFN profiling uses `ForwardContext.profile`, actual loader layer identity
  and explicit stage policies. Disabled profiling adds no sync or tensor read.
- Model teardown quiesces classic resources and shared scratch before freeing
  operands. Unknown completion retains resources. Known owner-free failures
  skip the final allocation sweep so retained pointers remain retryable.
- Shared teardown/model capability/typed GEMM changes require cross-cutting
  review; do not describe them as isolated kernel-only changes.

Evidence: `arithmetic-audit/shared-moe-scratch-implementation/` beneath the
evidence root, including root independent reports. Key logs:

- `/tmp/atlas-exl3-shared-scratch-model-final-green.log`: earlier FFN 28/28.
- Agent subsequently reports FFN 31/31; retain the particular log/run binding.
- `/tmp/atlas-exl3-known-free-sweep-green3.log`: actual model teardown 3/3.
- `/tmp/atlas-exl3-known-free-legacy-teardown-green.log`: related teardown 10/10.
- `/tmp/atlas-exl3-known-free-server-check.log`: CUDA server compilation PASS.

The actual Rust GPU example
`crates/spark-model/examples/exl3_moe_scratch_check/` is source-ready, CPU guard
tests 5/5 and CUDA-feature example check passed. It uses real Atlas dispatch,
two actual layers, isolated then shared **A→B→A**, raw output equality and
256-byte allocation guards. No GPU execution yet. It proves rank-local scratch
reuse, not TP-sum equivalence. Build it and the server in the same attested image.

### Latest coherent BLAS backend changes

The current working tree now has explicit profiles in
`crates/spark-runtime/src/gpu/gemm.rs`:

- `CublasLt`: existing dispatch.
- `Glm53Classic`: compute 68, algorithm 99, **math 16**, 32 MiB workspace.
- `Glm53AuxiliaryClassic`: compute 68, algorithm 99, **math 0**, 32 MiB workspace.
- Both classic profiles require provider version **130101**.

Per-stream classic resources track/reconfigure the profile under the enqueue
lock, reuse one workspace and retry configuration failures. Actual classic/Lt
symbol providers must resolve to the same explicit directory and version;
both libraries remain owned until resources are quiesced. Ordinary Lt remains
unchanged. No mixed 13.0/13.1 provider loading or implicit numeric fallback.

- `/tmp/atlas-exl3-classic131-green3.log`: classic resource tests **9/9 PASS**.
- `/tmp/atlas-exl3-classic131-profile-red.log`: prior missing-enum RED.
- `/tmp/atlas-exl3-classic131-server-check.log`: CUDA server bin/tests check PASS.
- **After that check**, extra profile/provider assertions were added to the
  existing GEMM test and formatted; these latest assertions have **not rerun**.

### Merged KDA auxiliary projection — implemented 2026-10-10 (CPU-verified)

Resumed and completed the former RED step. GPU execution is still pending.

- `crates/spark-model/src/layers/glm5_kda/native_aux.rs`: `NativeKdaAux`
  borrows the merged weight, owns padded input `[max(cap,3), K]` and merged
  output `[max(cap,3), 288]` scratch, validates stream/rows/live distinct
  buffers before enqueue, runs one `Glm53AuxiliaryClassic` (math 0) GEMM,
  pads M<3 to operator M=3 with zero rows, and scatters physical rows to
  beta/FA/GA with `copy_d2d_2d_async`. Release keeps failed frees retryable.
  Constructor is `new(gpu, weight, inner, heads, dim, capacity, stream)`.
- `weight_loader/glm5/exl3_kda_aux.rs`: `merge_auxiliary` concatenates B,FA,GA
  (reference `extra_slices` order) into new slot `AuxMerged`, then retires the
  sources; a failed retirement keeps the source owned for rollback.
  `weight(BProj|FAProj|GAProj)` returns interior views (offsets 0, heads*K*2,
  (heads+dim)*K*2). `tensor(BProj|FAProj|GAProj)` now errors by design.
- `assembly.rs` builds the aux after all kernel lookups, only for EXL3, using
  q_proj capacity/stream, after asserting the three views are contiguous.
  `exl3.rs` releases aux scratch (never the merged weight; companions own it).
- `forward_aux.rs`: EXL3 branch precedes every batch5/batchm GEMV fusion;
  FB/GB dispatch unchanged. `multi_seq.rs` (C2/C3, outside the qualified C1
  profile) also routes B/FA/GA through the aux when present.
- RED test corrected: `n/k` -> `columns/inner`; the dim=64 geometry case was
  replaced with heads=0 (dim=64 violates no real invariant).
- M=2 pads to 3 but is **unqualified**; only M1->3, M5, M128 are proven.
- Scratch: 1,131,072 B/layer at capacity 129, 38,456,448 B/rank over 34 KDA
  layers (allocation arithmetic, not a measurement).
- Known pre-existing gap: if `Glm5KdaLayer::new` fails after weights move in,
  those weights are dropped without release (same as before this change).

Evidence (`arithmetic-audit/kda-native-merged-aux-implementation/`, local
logs copied there; originals in `/tmp/atlas-kda-native-aux-*.log`):

- Targeted: native aux 2/2, loader 7/7 incl. new
  `auxiliary_merge_failed_retirement_stays_owned_for_rollback`, and
  `native_exl3` layer test (now asserts prefill M3 + decode M1->3 aux GEMMs,
  columns 257, math-0 profile, stream 77, and 22 freed resources).
- Broad no-default model lib: 1605 pass, 1 ignored, 1 fail = the existing
  `factory::glm_dense_cache::tests::dense_prefill_bf16_explicit_flag` baseline.
- Runtime no-default lib 353 pass / 2 ignored, including the previously unrun
  GEMM profile/provider assertions and both concat tests.
- CUDA-feature `spark-server --bins --tests` and `spark-model --examples
  --tests` checks PASS (`CUDARC_CUDA_VERSION=13000`).
- Env needed locally: `ATLAS_SKIP_BUILD=1 ATLAS_NO_RDMA=1` (+ CUDARC version
  for cuda features); fish shell does not word-split `$F`, use `bash -c`.
- rustfmt `--check` clean on all touched files; SPDX on all; new files <=500
  lines; `git diff --check` clean.
- Clippy: plain runs stop on unchanged spark-runtime baseline lints. With
  `RUSTFLAGS=--cap-lints=warn`, no-default spark-model lib+tests clippy reports
  zero findings in touched files (one test-only `single_range_in_vec_init` was
  fixed); log `atlas-kda-native-aux-clippy-caplints2.log`. Pre-existing
  warnings elsewhere and cuda-only integration-test compile errors under
  no-default are untouched.
- Retiring B/FA/GA right after `concat_rows` is safe on CUDA: `copy_d2d` is
  `cuMemcpyDtoDAsync` + `cuStreamSynchronize` and `free` is `cuMemFree_v2`.

## GPU/reference results and provider decision

Fresh reference `reference-upstream-context-v4` uses the actual v2
`vllm.v1.worker.gpu.model_runner.prepare_inputs` capture point; input CPU
positions are bound before model forward. Short strict 16 rows, persistent
requests, three full reference arithmetic repeats and long runs are preserved.

- Long reference geometry PASS both ranks: 269 chunks / 32820 tokens, 128-row
  nonterminal chunks. Long persistent requests include three actual M1 decode
  steps; position/context/head bindings pass.
- Capital max2 includes actual five-row prefill + one-row decode, final counters
  head 45 / context 329 / FFN 326; the auxiliary M1 input is invocation 325.
- All 31 private tuning-cache hashes unchanged before/after requests.
- Full remote archive integrity verified; large raw reference tensors remain
  on hosts where appropriate. Final metadata archives are extracted locally.
- Reference behavior diagnostics: 11 raw receipts, nine strict failures
  including reasoning usage/tool finish fields; dependent tool followup was
  skipped because no valid initial tool ID existed. This is not an Atlas
  regression verdict. Reference flags include `glm47` tool parsing and `glm45`
  reasoning parsing.
- Matched reference short16 vs earlier native candidate1a still FAILS:
  max-abs examples 0.65625 / 2.955078 / 1.695312 / 2.486328, KL up to
  0.128428, longer CE 0.456066. Exact argmax/text alone are insufficient.
- Persistent candidate comparison: three of four pass; longer text fails.
- Reference repeats also differ numerically. Preserve this diagnostic and
  investigate earliest divergence; **do not excuse candidate failures**.

### Coherent 13.1 pair component proof

Candidate 13.0 shared math16 was exact at M1/M5/M10/M128, but merged auxiliary
M128 was not. Same-input controls demonstrate a provider-version difference:

- **13.1 merged EXTRA math0 exact** at M1 padded to 3, M5 and M128, both hosts.
- 13.1 math16 EXTRA fails small shapes; do not use a universal math policy.
- **13.1 shared math16 exact** gate/up and independent down at M1/M5/M10/M128,
  both hosts (16 positive comparisons).
- **13.1 router Lt exact**, same four actual captured sizes, both hosts.
- Current FB/GB PTX exact at M1/M5/M128 on both hosts. Preserve these kernels.
- Checkpoint-to-resident BF16 companion byte checks also pass, both ranks.

Pair from installed `nvidia-cublas` package **13.1.1.3**:

- Classic SHA: `4e346a7b60a01552bd9bf4bbc6c862e11eb851559c3da3572293345de6aaed75`
  (61,910,216 bytes).
- Lt SHA: `930418840b747413584867cb7167dd5911dcf4184135589e0334527e9abbac9f`
  (629,945,952 bytes).
- Source package RECORD, metadata/license, ELF SONAME/dependencies and original
  source image are archived. Pair staged durably on spark1 as
  `reference-blas131-pair` within the validation staging directory.
- Root package RECORD check PASS; root shared raw/provider/source check PASS
  (32 comparisons including controls, 90 bound files).

Evidence under `arithmetic-audit/`:

- `shared-coherent131-controls/root-independent-shared131-raw-check.json`.
- `shared-coherent131-controls/root-independent-pair-package-record-check.json`.
- `kda-reference-classic-controls/independent-bitwise-review.json`:
  agent independently verified uint16 bytes; an earlier invalid u32 review is
  explicitly preserved and corrected.
- `router-coherent131-controls-attempt2/remote-rank{0,1}/results-m{1,5,10,128}/`:
  raw reports and `outputs.npz`; all eight agent-reported exact controls.
- First router attempt was prerequisite-only failure because the classic DLL
  was not loaded; attempt2 explicitly loads Lt then classic. Guard not relaxed.

**Root independent verification now complete (2026-10-10):**

- Router: `router-coherent131-controls-attempt2/root-independent-router131-raw-check.json`
  PASS, 369 checks, all eight rank×M cases byte-exact (`replayed` vs captured).
  The earlier stop was a dtype mismatch: captured logits `.bin` are F32 while
  `captured.npy` is uint16 BF16 (runner takes `>>16`). The root check verifies
  every F32 logit has zero low 16 bits (BF16-representable) so truncation hides
  nothing, then compares bytes. Also checks prereg/source/binding hashes,
  identity/chunk, Lt symbol/version 130101, loaded-pair maps, no Torch.
  Negative control (one flipped replay bit) fails exactly
  `outputs-sha` + `replayed-byte-exact`: `root-independent-router131-negative-control.json`.
- EXTRA: `kda-reference-classic-controls/root-independent-extra131-raw-check.json`
  PASS, 175 checks: math0 byte-exact at M1(op 3)/M5/M128 both ranks, rebuilt
  from raw qkvbfg trailing 288 columns; math16 recomputed counts 74/283/0 and
  69/285/0 match reports (built-in positive-difference control). Resident
  weight SHAs equal the prior root companion byte check.
- Checker scripts are saved beside each JSON (`root-independent-*.py`).

## Image/build state

Previous diagnostic image exists on both hosts, not serving:

- Source snapshot `af4e1f2c96941ab11682290d30bff71fd7b2362199cf3984589f9d9a745147d5`.
- Tag `atlas-exl3-validation:src-af4e1f2c9694-rust1.97.1`.
- Image ID `sha256:4de4bd173e7134030937e547c75a215ad15aef0074645f470c2a69b8e9df4f37`.
- Binary SHA `83488ec6b8c768cba8dafb993398a9e603e02c3876aac2b6601da001f93d0c10`.
- Root verified 3530 build inputs and ten embedded PTX modules; old eight PTX
  byte-identical to parent. It has chunk-budget/upstream captures, **not** the
  latest classic profiles/shared scratch/merged auxiliary changes.
- Reusable builder on spark1:
  `bd293e358226d4be6ab980b221755cc00f11a0221f7daa067109033d03ffff04`.

**No new source archive/image build was started.** Freeze was held for
component proof and complete final source. Temporary preparation scripts
`/tmp/atlas-prepare-shared-freeze.py` and `/tmp/atlas-extract-shared-ptx.py` are
obsolete **13.0-era** and must be updated before use. New EXL3-specific image
must package the coherent 13.1 pair together, retain source/license/provenance,
and independently attest server and scratch-example binary/PTX containment.
Do not affect default containers or co-load conflicting provider pairs.

## External reviews adopted

Preserve the original untracked reports unchanged:

- `CODE_REVIEW_2026-10-10.md`.
- `GB10_KERNEL_OPTIMIZATION_REVIEW_2026-10-10.md`.
- `GB10_PRIOR_ART_CROSSREF_2026-10-10.md`.

Code review P1 shared scratch and P2 opt-in stage timing are implemented and
CPU-verified, GPU proof pending. BF16 head transport is deferred until parity.

Optimization review resulted in an exact embedded-PTX offline assembly audit:
both generic/routed report 64-byte stack and zero register-spill counters, yet
SASS has local accumulator LDL/STL in repeated MMA regions. Root independently
verified output/source hashes, instruction counts and backward regions.
This is **not runtime JIT code or a measured speedup**.

- `arithmetic-audit/exl3-ptxas-spill-audit/report.json`.
- `arithmetic-audit/exl3-ptxas-spill-audit/root-independent-assembly-check.json`.
- `arithmetic-audit/kernel-optimization-review/disposition*.json` records
  accepted diagnostics, deferred arithmetic changes and unchanged gates.

Indexer oracle now covers BF16-representable inputs below/equal/above
`448 * 2^e` AFTER normalized H128 and BF16 rounding, positive and negative:
48 rows / eight exponents including the scale floor. Equality needs paired
impulses; single impulses can skip it. CPU tests **2/2 PASS**, including direct
test-file invocation. Python sweep latest **96/96 PASS** in
`/tmp/atlas-exl3-final-prepair-python-sweep.log`.

**Latest indexer GPU attempt failed prerequisites**, both hosts, before Torch:
the base image's `kpool_compress.py` SHA differs from the frozen source expected
by the old oracle. The prior source overlay/mount was omitted. Preserve failure
logs in `arithmetic-audit/index-scale-boundary-controls`; locate and mount the
exact frozen installed source closure for retry. Do not change expected hashes
or replace log2/ceil/exp2 arithmetic with frexp without reference evidence.
The prerequisite logs are `prerequisite-failure-rank0.log` and
`prerequisite-failure-rank1.log` in that directory. There is no numeric report
or NPZ for this attempt.

Next optimization priorities after parity: profile actual serving path;
decoder specialization/local accumulator removal, ordered split-K, mHC
parallelization, expert grouping. Preserve exact accumulation/row policies.
Packed shared experts, compressed KV/collectives, head transport, graphs and
speculation require separate qualification. Do not apply unrelated legacy
NVFP4/Qwen router-tier recommendations to the native EXL3 path.
Capture host compaction/fabric/thermal metadata before performance experiments;
no sysctls, NIC settings or roofline numbers were changed during this work.

## Known baseline checks / limitations

- Latest full license-header engine PASS: 3064 checked, zero invalid.
- Scoped formatting and `git diff --check` were green before pause.
- Workspace-wide fmt/typos/clippy have pre-existing failures, archived baseline
  proofs. Do not claim all workspace checks passed or clean unrelated code.
- Earlier broad no-default model tests: 1582 PASS, one existing dense-cache
  parser failure, one ignored. Current unfinished native auxiliary TDD adds
  compile failures described above.
- No-default clippy five old diagnostics: cublaslt metal stubs,
  two unchanged NVFP4 cutlass-stub functions, gguf names. CUDA server checks
  passed; no-default server bin has unrelated missing communication methods.
- Runtime rustdoc FP8 tensorwise `[N,K]` issue and workspace formatter/typos
  baselines are preserved. Relevant artifact directories include
  `local-ci-baseline-blockers`, `fmt-baseline-review`,
  `unchanged-dense-cache-baseline`, `native-shared-classic-backend`.

## Resume sequence

1. DONE 2026-10-10: both hosts idle (`docker ps` empty, no compute apps);
   `glm53-exl3-head`/`glm53-exl3-worker` still `exited`. Recheck before GPU work.
2. DONE: root router (369/369) and EXTRA reference_classic (175/175) raw
   checks PASS. The 13.0 `candidate_lt` EXTRA entries were not re-checked
   (diagnostic only).
3. DONE: `NativeKdaAux`, merged ownership, wiring, teardown and tests.
4. DONE: GEMM assertions, CUDA server/examples checks, scoped fmt and
   regressions (only the known dense-cache baseline fails). **Next: step 5.**
5. DONE (except checkpoint re-hash). Source `48624a8605de…` (3553 inputs, 57
   Rust-only changes) built on spark1 as
   `atlas-exl3-validation:src-48624a8605de-rust1.97.1`, image
   `sha256:82b23cd19e595cae5e96616e726a64f2f4b964416d31268210bf24f3e686c737`,
   identical ID/label on spark0 (`spark0-image-transfer-verification.json`).
   Recipe: `build/prepare-native-aux-classic131.py`; evidence in
   `build/native-aux-classic131/` (build/export logs, `attestation/`).
   Attestation: spark `a2b0e44b…` (75,927,336 B), exl3_moe_scratch_check
   `ff859e42…` (37,594,184 B), both equal builder output + fingerprint; all 10
   parent PTX modules byte-identical and contained in both binaries; target
   closure identical (mHC `--fmad=true` once); LD_LIBRARY_PATH starts with
   `/opt/atlas/cublas-13.1.1.3`; spark, example and libcublas resolve Lt there;
   pair hashes + license present. Root local re-check
   `root-independent-attestation-check.json` 15/15 PASS. The runtime image's
   13.0 Lt (`59cf…`) remains on disk; GPU runs must confirm via /proc/maps that
   only the 13.1 pair is loaded (classic is dlopen'd at runtime). Still owed:
   checkpoint shard re-hash on both hosts (prior: `checkpoint-full-sha/`).
6. DONE 2026-10-10. `exl3_moe_scratch_check` from the attested image ran on
   both ranks (rank-local, layers 3/4, capacity 129, captured L3 inputs M1/M5/
   M128). Exit 0; guards intact, inputs unchanged; shared == isolated bytes and
   A==A', A!=B for every case; executable SHA equals attestation; all 10 PTX
   hashes equal (report uses runtime aliases gemv/gemm); `/proc/self/maps`
   shows **only** the 13.1.1.3 pair. `libcublas.so.13` is not a NEEDED
   library, so its mapping proves the runtime classic dlopen resolved the pair
   directory and passed the 130101/same-directory check, and that the
   `Glm53Classic` (math 16) shared GEMMs ran. Not yet proven on hardware: the
   `Glm53AuxiliaryClassic` math-0 profile or per-stream math0<->math16
   reconfiguration (first exercised in step 7). Shared scratch 60,468,096 B/rank.
   Evidence: `arithmetic-audit/native-aux-scratch-gpu-20261010/` incl.
   `root-independent-scratch-check.json` (PASS). Scope: routed MoE scratch
   reuse only; it does **not** exercise NativeKdaAux, TP or reference parity.
   Checkpoint stat guard (index SHA + 23 shard dev/inode/size/mtime/ctime)
   matched on both hosts: `build/native-aux-classic131/checkpoint-stat-guard.jsonl`.
7. PARTIAL 2026-10-10 (evidence `native-aux-insitu-20261010/`). TP2 server
   from the attested image (create commands `rank{0,1}-create-command.json`,
   template = corrected-candidate, upstream dump layer 0, no LD_LIBRARY_PATH
   override). Both ranks logged classic 130101 from the pair dir;
   `/proc/1/maps` shows only the 13.1 pair (`maps-cublas-rank*.txt`).
   - In-situ aux: layer-0 `normed`, merged beta/FA/GA and packed QKV are
     **byte-exact** vs reference at M5 (capital prefill), M1 (decode, padded
     to 3) and M128 (first 2047-prompt chunk), both ranks
     (`insitu-aux-comparison.json`, `insitu-qkv-probe.json`). This is the
     first hardware exercise of math0 aux + math16 shared on one stream.
   - Short16 (frozen collector/comparator, reference
     `reference-upstream-context-v4/reference-logits/teacher-forced-results`,
     provenance with the 7-key execution map + `native_numerical_policy`
     adjunct): **FAIL**. 16/16 argmax + text exact; max-abs 0.59/2.28/1.13/
     2.34, KL max 0.0588, CE err max 0.313 (longer). Negative controls pass.
     `short16-comparison.json`. Different binary from candidate1a; do not read
     the lower KL as qualification.
   - Earliest divergence (capital M5, layer 0, `layer0-stage-divergence-m5.json`):
     normed, QKV, beta/FA/GA, FB/GB outputs exact; **KDA recurrent core output
     is the first stage that differs** (~55% elements, ~1 BF16 ULP: 3.05e-5 r0,
     1.22e-4 r1); gated norm and reduced o_proj inherit ULP-scale gaps. The
     reference `kda-o-proj` hook is post-all-reduce (matches Atlas
     `attention_reduced_output`, not local).
   - NEXT: add Atlas layer-0 captures for conv output / post-transform gate
     (reference `kda-core-input-g1` [1,5,32,128]) / beta (`kda-core-input-beta`)
     / `kda-core-input-qkv_proj_states` [5,12288] to split preprocessing from
     the chunked recurrence; then compare the recurrence arithmetic against
     the reference FLA chunk_kda kernel. Long/persistent/behavior gates not run.
   Validation containers `atlas-exl3-nativeaux-insitu-{0,1}` exited 0 and are
   kept; hosts idle afterwards.
   - KDA core analysis (`arithmetic-audit/kda-core-divergence/`): reference
     core-input hooks are RAW (pre-conv qkv, raw g1, raw beta); core =
     causal_conv1d_fn (BF16 out) + FLA `chunk_kda_with_fused_gate` (prefill) /
     `fused_recurrent_kda` (decode). Kernel sources extracted read-only from
     image 6eb42fa2 to `reference-kda-kernel-source/`. F64 emulation
     (`emulate_core_f64.py`, `emulation-f64-m5.json`): **Atlas core equals the
     correctly rounded exact result** (2/4 of 20480 mismatches, BF16 conv);
     the reference deviates from exact math in ~55% of elements, i.e. the gap
     is the reference chunk algorithm's precision, not an Atlas bug. Closing
     it means reproducing FLA's lossy chunk arithmetic (reference policy).
     Other shapes vs reference: decode inv325 25-31% (confounded by prefill
     state), M128 chunk ~56%. Raw reference tensors for inv 16/324/325 copied
     to `kda-core-divergence/raw/` (inv324 == first-prefix inv0 bytes).
   - Exact reference-policy model of the KDA core (subagent-written,
     root re-ran M5): `emulate_fla_chunk.py` (M5) and `emulate_fla_ext.py`
     (`fla-emulation-ext.json`, details in `findings.md`). M5 prefill 0/0 and
     decode inv325 0/0 bit-exact both ranks; M128 inv16 residual 1728/585 of
     524288 (mostly 1 ulp; conv-silu/Aqk near-ties, hypothesised f32
     ex2.approx/div.full bit behaviour). Required policy: conv out BF16;
     q/k l2norm BF16 (prefill only; decode keeps f32); f32 safe gate +
     sequential f32 chunk cumsum *RCP_LN2; intra Aqk/A f32; inter-sub-chunk
     dot tf32 with truncated f32 operands; solve_tril f32 -> Ai BF16;
     v*beta, k*beta*exp2(g), u, w, kg, v_new, h (per chunk), qg, tril(Aqk) BF16;
     chunk 64 / sub-chunk 16; f32 recurrent state carried between calls;
     decode = fused_recurrent_kda in f32.
   - DECISION PENDING / NEXT: implement an EXL3-only reference-policy KDA core
     in CUDA (chunked prefill per above + decode recurrence), using the same
     PTX ops as Triton (ex2.approx, div.full, TF32 mma on raw f32 bits) to try
     to close the M128 residual on GPU; validate against these captures, then
     rerun short16. This is new kernel work (kernel/PTX change => full image
     rebuild, not a Rust-only delta). Not started; no code written. An advisor
     consult was about to be requested before committing to this approach.
   - Reference self-repeat already fails the frozen gates (max-abs to 3.46, KL
     0.048, CE 0.21; nondeterminism starts at layer-3 routed combine
     `routed-weighted-f32`); preserved, not used to excuse candidate failures.
8. Retry indexer boundary oracle using the exact frozen reference source mount.
9. Only after strict numerical success, collect any performance comparison
   under the measurement-discipline skill with full fingerprints and repeats.
10. Restore original production head/worker and verify both port-8000 health
    checks. No PR, commit, release or image publication was requested.

Paused agents and responsibilities: `/root/tp_audit` owns all remote GPU
launches/captures/cleanup/restoration; `/root/exl3_cuda` owns backend/provider
resources and image build; `/root/glm_exl3_plan` owns merged auxiliary loading,
scratch driver and oracle boundaries; root owns independent verification,
documentation and source/image approval. Avoid concurrent GPU launches.
