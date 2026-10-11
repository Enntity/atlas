# GLM-5.3 EXL3 on two Sparks — continuation state

Reviewed against `11671adf3247d32291e0a7cbcf3a0920786a138f` on 2026-10-10.
The model boots and generates on TP=2 / EP=1. The next objective is qualified,
high performance inference: correct outputs, interactive latency, prompt
processing, concurrent throughput, and reliable operation on both nodes.

[todo.md](todo.md) is the authoritative remaining-work checklist, with task IDs,
dependencies and completion criteria. This file holds the restart context,
implementation status and evidence needed to execute it. The design documents
specify arithmetic; the dated reviews contain hypotheses, not measured Atlas
speedups. Earlier pause handoffs remain in the Git version of this file at
`11671adf`; their idle-host, uncommitted-work and restoration instructions are
superseded by the successful launch below.

## Current status and scope of proof

| Area | Status at review |
|---|---|
| Native checkpoint loading and two-node generation | Implemented; launch smoke recorded |
| Full-model frozen numerical gates | **FAIL** (short16, reference-policy candidate `b6be48ed`). Text/argmax 16/16. The reference fails the same gate against its own repeats by the same magnitude; Atlas is byte-deterministic (3 fresh-server runs). First Atlas-own divergence: layer-3 attention (first DSA/MLA layer) — layers 0–2 are byte-exact in situ |
| FLA reference-policy KDA | **Implemented** (`72d64e4c`, `069e4351`): module `kda_fla`, `ATLAS_KDA_FLA_REFERENCE=1`. Byte-exact vs GPU stage oracle (all stages, carried state, both ranks) and in situ (layer-0 core, gated norm, o_proj after TP, dense FFN; layers 0–2 inputs/outputs) |
| Shared routed MoE scratch | Implemented; rank-local GPU reuse/guard proof recorded on both hosts |
| FFN stage timing | Implemented, opt-in and synchronizing; useful for attribution, unsuitable for headline timings |
| Merged KDA auxiliary projection | Implemented; layer-0 M1 padded to 3, M5 and M128 byte comparisons recorded on both hosts |
| CUDA graphs, concurrent admission, long context, prefix reuse, FP8 KV | Unqualified for this native EXL3 profile; qualify independently |
| DFlash/MTP/self-speculation | Explicitly rejected by native EXL3 startup guards; implementation and qualification required |
| Reproducible deployable image | Recipe in-tree (`a6f64c46`): no-cache build `atlas-exl3-validation:src-efd023f8f7b9-rust1.97.1` = `sha256:b6be48ed…` from `59b0e113`; runtime maps attest only the pinned pair on both ranks; smoke 5/5. Owed: shard re-hash, `exl3_moe_scratch_check` GPU run |
| Performance baseline | Unqualified diagnostic only (`p0-unqualified-baseline-20261010/`): decode ~0.45 s/token (~2.2 tok/s), 2573-token prompt TTFT ~139 s on the production image. Not a headline; P0 harness in branch `pi/p0-20261010` |

This review checked source and committed records. It did not inspect live hosts,
restart services, run a new GPU qualification, or establish a new performance
result. “Recorded” below refers to the earlier evidence, not a rerun today.

## Launch profile and operational handoff

- Repository: `/home/colum/Workspace/RustroverProjects/atlas`.
- Branch at review: `sparkglm/installable-20260924`; reviewed HEAD: `11671adf`.
- Model: `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3`.
- Checkpoint revision: `dda99c7e935920fa6e235ce410e6f806e1c02720`.
- Last recorded serving containers: `glm53-exl3-atlas-0` on spark0 and
  `glm53-exl3-atlas-1` on spark1, port 8000, restart policy `unless-stopped`.
- Image: `atlas-exl3-validation:src-48624a8605de-rust1.97.1`.
- Image ID: `sha256:82b23cd19e595cae5e96616e726a64f2f4b964416d31268210bf24f3e686c737`.
  `48624a8605de` is a source snapshot identifier, **not the reviewed Git commit**.
  Do not assume the running image contains every change in `11671adf`.
- Launcher: [scripts/start-glm53-exl3-tp2.sh](scripts/start-glm53-exl3-tp2.sh).
  On the corresponding host: `RANK=0 bash scripts/start-glm53-exl3-tp2.sh` or
  `RANK=1 bash scripts/start-glm53-exl3-tp2.sh`. It removes/recreates the named
  Atlas container: inspect service ownership and plan downtime before using it.
- Explicit launch limits: TP=2 / EP=1, BF16 KV, default LM head, context 4096,
  prefill chunk 128, max batch 1, max sequences 1, memory utilization 0.84,
  swap 0, OOM guard 1024 MiB, request timeout 1200 seconds, language-model-only.
- Graph disables are `ATLAS_NO_DECODE_GRAPHS=1`,
  `ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1`, `ATLAS_DEBUG_NO_GRAPH=1`.
  Prefix reuse and speculation are outside the frozen comparison profile.
- Preserved fallback: `glm53-exl3-head` / `glm53-exl3-worker`, image
  `glm53-flash-mikeroz:dt-carry20`. They were stopped for Atlas takeover.
  Restore only as a deliberate rollback; the old instruction to restore them
  automatically after validation no longer describes the serving objective.

[Launch smoke receipts](atlas-launch-smoke-20261010/smoke.json) contain five
successful sequential, nonstreaming OpenAI chat requests, including a 612-token
answer, multi-turn recall and sampling. The
[smoke harness](tests/exl3_validation/launch_smoke.py) checks request success and
nonempty output; coherence was judged from transcripts. It does not qualify
streaming, tools, reasoning controls, Anthropic, concurrency or numerical parity.
The previous smoke-derived rate must not become a decode baseline: its wall
interval includes request/prefill time, and the receipt lacks a complete run
fingerprint. Preserve the raw receipt rather than promote its rate to a target.

### Hosts and evidence locations

Inventory/configuration: `~/Workspace/homelab-ansible`.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 -J colum@192.168.1.50 colum@10.50.0.10
ssh -o BatchMode=yes -o ConnectTimeout=15 -J colum@192.168.1.50 colum@10.50.0.11
```

spark0 is `gx10-eb0e`; spark1 is `gx10-235f`. Recorded hardware: GB10, 48 SMs,
approximately 121 GiB unified memory each, driver 580.159.03, host CUDA 13.0 /
nvcc 13.0.88. Fabric addresses are `192.168.100.1` / `.2`. Launcher interfaces:
spark0 `enp1s0f0np0` / `rocep1s0f0`; spark1 `enp1s0f1np1` / `rocep1s0f1`.
It currently selects one HCA per rank and `NCCL_CROSS_NIC=0`.

Durable evidence root: `/home/colum/Workspace/atlas-exl3-validation-20261010`.
Paths in the evidence table below are relative to it. Some large reference
arrays remain on the hosts; `/tmp` logs alone are not durable evidence. Before
GPU work, inspect both hosts' containers, GPU processes, image IDs, flags and
health. Coordinate one experiment owner across both ranks. The old agents and
“work running in parallel” statements describe a finished session.

## Numerical contract and remaining blocker

The checkpoint has hidden size 4096, 45 layers (34 KDA, 11 MLA), 42 routed MoE
layers beginning at layer 3, 288 experts, sigmoid top-8, normalization 2.5,
groups 1/1, expert intermediate 2048 globally / 1024 locally, and one shared
expert. Physical vocabulary is 154880; tokenizer size is 154856. Absent
`mhc_post_mult_value` is handled by the explicit named reference policy (2.0),
with conflicting values rejected.

The frozen comparison uses TP=2 / EP=1, BF16 KV, eager execution, context 4096,
chunk 128, C1 admission, no prefix caching or speculation. Preserve abs 0.15,
rel 0.02, KL 0.005, CE 0.15, top-K 10, exact argmax and HTTP text requirements.
Long-prefix cases use 2047/2048/2051/2053 tokens and registered continuations.
Do not relax these gates to accommodate either engine's nondeterminism.

Status 2026-10-11 (candidate image `sha256:b6be48ed…`, `ATLAS_KDA_FLA_REFERENCE=1`):

- **KDA closed.** The M128 "residual" was the reference's Inductor-compiled beta
  sigmoid plus f32 conv weights (design doc §3.6a), not GPU transcendental
  behaviour. The `kda_fla` kernels are byte-exact against the GPU stage oracle
  and, in the live two-rank server, layer-0 KDA core, gated norm, o_proj after
  TP and the dense FFN match the reference bytes; layers 0–2 FFN inputs and
  outputs match for the capital M5 request on both ranks.
- **short16 still fails** (abs 0.63/2.66/1.85/2.38, KL ≤ 0.054, CE ≤ 0.085,
  text and argmax 16/16, negative controls detected). Atlas is byte-
  deterministic across three fresh-server runs. The *reference* fails the same
  frozen gate against its own repeats by the same magnitude (abs ≤ 2.75,
  KL ≤ 0.057, CE ≤ 0.21; `reference-three-run-diagnosis.json`), so no
  deterministic engine can pass the gate as written. This is a retained
  constraint, not a waiver: see todo.md Q2 for the decision required.
- **Update:** the layer-3 divergence is the reference's silent `fp8_ds_mla`
  latent KV (log: `FLASHINFER_MLA_SPARSE_SM120`, "top-k MQA path only").
  With `--kv-cache-dtype fp8_g128` Atlas reproduces single-key rows; multi-key
  rows differ at implementation level. The Q2 decision was a pre-registered
  statistical gate (`docs/design/exl3-statistical-gate.md`).
- **First Atlas-own divergence (superseded above):** layer-3 FFN input (16147/20480 elements,
  max 0.021) although layer-2 output is exact, i.e. inside layer 3's attention
  block — the first `deepseek_sparse_attention` (MLA + indexer) layer, or its
  mHC pre/post. A fresh canonical reference repeat matched the original byte
  for byte through layer-3 output for this request, so this one is Atlas's.
- The CPU oracle test (`test_exl3_kda_reference.py`) is historical; the GPU
  oracle supersedes it. `KdaRecurrentPath::Reference` is unrelated to the
  policy.

## What the commit already implements

- Native EXL3 format-aware loading, TP partition/upload, packed linear and MoE
  kernels, KDA/MLA/mHC assembly, vocabulary transport and preflight.
  The pinned arithmetic requires TP=2 / EP=1; this is not an EP=2 or general
  EXL3-model qualification. Sequence parallelism and LoRA are also rejected;
  vision is outside this language-model-only launch scope.
- Model-owned `MoeScratch` shares 11 buffers between sequential routed layers
  under rank/world/stream/capacity checks and an enqueue lease. Launch capacity
  is 129 (chunk 128 plus the model's additional row). Recorded shared scratch
  allocation is 60,468,096 bytes/rank; this is allocation size, not a speedup.
  KV budgeting observes free memory after loading. New streams, graph replay
  or larger batches need new lifetime/capacity proof.
- `ForwardContext.profile` drives native FFN stage timing with rows, routes,
  layer/rank and precision metadata. `--profile` synchronizes around stages;
  use Nsight traces and uninstrumented requests to establish performance.
- `NativeKdaAux` merges B/FA/GA into one classic GEMM, pads M<3 to operator M=3,
  scatters results, and retains failed frees for retry. M2 is still unqualified.
  Math-0 auxiliary and math-16 reconstructed shared-expert profiles use a
  coherent classic/Lt provider pair, version 130101, with per-stream resources.
- Shared routed scratch is distinct from packed shared experts. Shared experts
  currently run reconstructed BF16 weights. Native QKV still invokes separate
  packed components. Routed MoE still processes route slots individually.
- Head projection produces BF16 values, widens to F32 for gather/reorder and
  converts back for sampling. A BF16 transport path is still open, with F32
  capture retained for validation consumers.
- Ordered resource teardown, scheduler prefill budget correction and
  completion-text fixes are included. Shared traits, communication, runtime
  GEMM APIs and per-module build flags changed; regression coverage must include
  existing supported models. Only EXL3 mHC declares `--fmad=true`.

The committed [Dockerfile](docker/gb10/glm-5.3-flash-exl3/exl3/Dockerfile) builds
only the server and does not install/copy the required cuBLAS 13.1.1.3 pair or
set its search path. The successful build used the external
`build/prepare-native-aux-classic131.py` recipe. Task Q0 makes that build
reproducible in-tree, including the scratch example used for GPU verification.

## Evidence index

These records preserve prior results; restore and verify their hashes before
reusing them. A build, CPU test, component oracle and full-model gate prove
different scopes.

| Evidence under the durable root | Recorded result / limitation |
|---|---|
| `build/native-aux-classic131/` | Snapshot `48624a8605de`, build/export logs, attestation; both hosts had the same image ID; root attestation 15/15 |
| `arithmetic-audit/native-aux-scratch-gpu-20261010/` | Real layers 3/4, M1/M5/M128, capacity 129; shared vs isolated A→B→A byte equality, guards intact, both ranks; rank-local proof only |
| `native-aux-insitu-20261010/` | Layer-0 aux/QKV M1/M5/M128 comparisons, provider maps and failed short16; long/persistent/behavior not rerun on that candidate |
| `arithmetic-audit/kda-core-divergence/` | Raw tensors, extracted reference source, emulation scripts, findings and residual reports; inputs for T2–T8 |
| `arithmetic-audit/shared-coherent131-controls/` | Shared math16 exact component controls and package RECORD/source/provider checks |
| `arithmetic-audit/router-coherent131-controls-attempt2/` | Router Lt raw check 369/369, both ranks and M1/M5/M10/M128; flipped-bit negative control |
| `arithmetic-audit/kda-reference-classic-controls/` | EXTRA math0 raw check 175/175; math16 negative comparisons retained |
| `arithmetic-audit/index-scale-boundary-controls/` | Latest GPU retry failed frozen-source guard before Torch; no numerical result; restore exact source overlay |
| `arithmetic-audit/exl3-ptxas-spill-audit/` | Offline embedded-PTX audit: local accumulator LDL/STL despite zero reported register spills; not runtime JIT evidence or a speedup |
| `reference-upstream-context-v4/` | Frozen short/long/persistent reference captures, repeats and behavior diagnostics; behavior failures are not Atlas verdicts |
| `arithmetic-audit/kda-reference-stage-oracle-20261010/` | GPU replay of the pinned reference (Inductor beta, f32 conv weights, varlen, original Triton caches): reproduces every captured KDA output byte; per-stage bytes + carried-state `seq*` fixtures; acceptance oracle for `tests/exl3_validation/kda_fla/run.sh` |
| `arithmetic-audit/kda-reference-ptx-spec-20261010/`, `reference-triton-cache-20261010/` | Exactly-used compiled reference variants (PTX/TTGIR/autotune) and both nodes' original Triton caches |
| `kda-fla-candidate-20261010/` | Reference-policy candidate: build log/manifest, create commands, short16 comparison + 2 determinism repeats, attestation, smoke, in-situ layer captures (L0 upstream, L0–4 FFN) and the matching canonical reference L0–3 capture |
| `p0-unqualified-baseline-20261010/` | 6 streaming requests on the production image; harness stopped early (rate too low for the full matrix) |
| `checkpoint-full-sha/` | Earlier full shard hashes; latest build only has stat/index guard, full re-hash still owed |
| `local-ci-baseline-blockers/`, `fmt-baseline-review/`, `unchanged-dense-cache-baseline/` | Earlier baseline failures; reproduce before calling any current failure pre-existing |

Required pair from `nvidia-cublas` 13.1.1.3:

- Classic SHA256: `4e346a7b60a01552bd9bf4bbc6c862e11eb851559c3da3572293345de6aaed75`.
- Lt SHA256: `930418840b747413584867cb7167dd5911dcf4184135589e0334527e9abbac9f`.
- Preserved staging: `reference-blas131-pair` on spark1; original metadata and
  license archived. The attested image uses `/opt/atlas/cublas-13.1.1.3` first
  in the loader path. Verify actual classic **and** Lt mappings after runtime
  dlopen on both ranks; on-disk presence alone is insufficient.

Earlier CPU checks recorded model 1605 pass / 1 ignored / 1 known dense-cache
failure and runtime 353 pass / 2 ignored, plus CUDA-feature compilation checks.
These are historical records, not a clean current workspace gate. The previous
“native aux still RED,” “no GPU scratch run” and “new image not built” statements
were superseded by later evidence in the table.

## Next execution sequence

1. Read [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md), the
   [measurement discipline](.claude/skills/measurement-discipline/SKILL.md), and
   [validation harness contract](tests/exl3_validation/README.md).
2. Get the user's Q2 decision (todo.md Q2: pin reference nondeterminism,
   approve a statistical gate, or retain the constraint). Do not relax the
   frozen thresholds without it.
3. Find Atlas's first own divergence: layer-3 attention (first DSA/MLA layer)
   for the capital M5 request. Layers 0–2 are byte-exact on both ranks
   (`kda-fla-candidate-20261010/cand-l04-capital` vs `ref-l03-capture`); add
   MLA/indexer/mHC stage captures for layer 3 on both engines.
4. Run the long-prefix gate (2047/2048/2051/2053) on image `b6be48ed` with
   `ATLAS_KDA_FLA_REFERENCE=1`; finish Q0 (scratch-check GPU run, shard re-hash).
5. P0: trace C1 decode (≈0.45 s/token unqualified) with Nsight on both ranks
   before any optimization; merge the `pi/p0-20261010` harness after fixing
   its null `repeat`/`cold` fields.
6. Then Q3/Q4, P1–P6, F1–F4, R1/R2 as in todo.md.

Operational: production runs the original image on port 8000; the candidate
image `atlas-exl3-validation:src-efd023f8f7b9-rust1.97.1` is loaded on both
nodes; candidate create commands are in `kda-fla-candidate-20261010/`.

The dated [code review](CODE_REVIEW_2026-10-10.md),
[kernel review](GB10_KERNEL_OPTIMIZATION_REVIEW_2026-10-10.md) and
[prior-art corrections](GB10_PRIOR_ART_CROSSREF_2026-10-10.md) remain useful
background. Their old P1 scratch and P2 timing findings are implemented;
the head transport finding is open. Local-accumulator specialization, ordered
split-K, mHC parallelization and expert grouping are candidates whose priority
must follow an Atlas trace. Legacy NVFP4 router advice and prior-engine rates
must not be treated as native EXL3 results.
