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
| Full-model frozen numerical gates | **FAIL** on the last candidate; KDA core is the earliest identified divergence |
| FLA reference-policy KDA | Design and CPU acceptance oracle exist; policy enum, environment switch, CUDA kernels and dispatch are **not implemented** |
| Shared routed MoE scratch | Implemented; rank-local GPU reuse/guard proof recorded on both hosts |
| FFN stage timing | Implemented, opt-in and synchronizing; useful for attribution, unsuitable for headline timings |
| Merged KDA auxiliary projection | Implemented; layer-0 M1 padded to 3, M5 and M128 byte comparisons recorded on both hosts |
| CUDA graphs, concurrent admission, long context, prefix reuse, FP8 KV | Unqualified for this native EXL3 profile; qualify independently |
| DFlash/MTP/self-speculation | Explicitly rejected by native EXL3 startup guards; implementation and qualification required |
| Reproducible deployable image | Successful external image preserved; committed Docker recipe still lacks its required cuBLAS 13.1 pair |
| Performance baseline | No qualified repeated baseline; launch smoke is preliminary, single-shot evidence only |

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

The last short16 comparison **fails** despite exact argmax/text in all 16
rows. Layer-0 normalized input, QKV, merged beta/FA/GA and FB/GB outputs matched;
the first identified mismatch is the KDA recurrent core. Captured-case F64
emulation supports an arithmetic-policy explanation, but does not prove that
Atlas is universally “correctly rounded,” that the reference is universally
wrong, or that there are no later Atlas bugs. Reference self-repeat differences
beginning at layer-3 routed combine are a separate investigation.

[Reference-policy design](docs/design/exl3-kda-reference-core.md) specifies
chunk 64 / subchunk 16, precise BF16 rounding points, TF32 operand handling,
and F32 state across calls. Its T1 CPU oracle is present. T2–T8 (CUDA prefill,
PTX probes, M128 closure, decode and dispatch/GPU validation) remain open.
`KdaArithmeticPolicy`, `ATLAS_KDA_FLA_REFERENCE`, `kda_fla_chunk_prefill_bf16`
and `kda_fla_recurrent_decode_bf16` are **proposed names**, absent from production
source at `11671adf`; setting the proposed environment variable does not enable
this policy today. The existing `KdaRecurrentPath::Reference` is a different path.

The recorded CPU emulation matches M5 and its following decode on both ranks;
M128 retains 1728/585 mismatches out of 524288 elements. Those counts document
the residual, not an acceptable CUDA tolerance. The GPU target is byte equality.
Resolve exp/div/rsqrt and dot/reduction lowering from actual reference PTX and
stage captures. Add fixtures for at least three chunks and repeated decode:
inter-chunk carry and pinned autotune choices are not fully established.
The CPU test can skip when its external scripts/fixtures are missing; a skip
is not a passing qualification.

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
2. Start Q0 (reproducibility) and Q1 (KDA closure). Preserve the working service
   and all frozen artifacts. Rebuild from scratch for runtime/kernel changes.
3. P0 may collect an explicitly unqualified diagnostic baseline in parallel with
   correctness work. Use it to choose hot spots; any changed arithmetic must pass
   its component and end-to-end gates before adoption or a qualified speed claim.
4. Complete Q2–Q4 (numerical, API and held-out task quality), then pursue P1–P6
   according to measured cost. In particular,
   optimize both C1 decode and grouped prefill; define latency and throughput
   acceptance targets from the baseline instead of importing another engine's.
5. Qualify F1–F4 as separate serving profiles and finish R1/R2, including
   automated EXL3 performance regression records. Do not turn on graphs,
   FP8 KV, concurrency, caching or speculation merely because Atlas supports them
   on another format. Record actual dispatch engagement and resource limits.

The dated [code review](CODE_REVIEW_2026-10-10.md),
[kernel review](GB10_KERNEL_OPTIMIZATION_REVIEW_2026-10-10.md) and
[prior-art corrections](GB10_PRIOR_ART_CROSSREF_2026-10-10.md) remain useful
background. Their old P1 scratch and P2 timing findings are implemented;
the head transport finding is open. Local-accumulator specialization, ordered
split-K, mHC parallelization and expert grouping are candidates whose priority
must follow an Atlas trace. Legacy NVFP4 router advice and prior-engine rates
must not be treated as native EXL3 results.
