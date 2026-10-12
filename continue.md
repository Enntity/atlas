# GLM-5.3 EXL3 on two Sparks — continuation state

[todo.md](todo.md) is the authoritative remaining-work checklist; this file
holds the restart context, current verdict, and the evidence/profile/host
pointers. Sequencing SSOT:
[docs/plans/q2-norm-causality-plan.md](docs/plans/q2-norm-causality-plan.md);
decisions: [decisions.md](decisions.md); append-only experiment log:
[q2-diagnostic.md](q2-diagnostic.md) (its latest entry, "post-review
corrections", supersedes earlier C1/v24/gate-v2 claims). Latest review:
[CODE_REVIEW_2026-10-12.md](CODE_REVIEW_2026-10-12.md). Superseded narrative
is in git history.

## Current verdict (2026-10-12, post-review)

- **Q2: candidate not qualified.** Under the frozen v1 gate (historical
  record) the candidate `cand-fp8decode02` FAILS on rank (T 7.20 vs refs
  5.04–5.53) plus 3 catastrophic short16 `longer` rows; the ulp control
  fails the same way. Reports: `statistical-gate-20261011/final/`.
- **Held-out `ref13` failure explained:** reference-internal bimodality at
  short16 `longer`/r0.
  - Recurring byte-identical mode in 3/15 draws (CP95 [0.043, 0.481]; 2/14
    if the triggering draw is excluded); 15 draws fall into 6 byte classes.
  - The mechanism (likely a discrete runtime choice) is unidentified, and
    "fingerprint-identical" covers configuration only.
  - Diagnosis: v1 was **underpowered at N = 9 under a bimodal row, with
    session drift not excluded**. Three fresh repeats scored T above the
    archived range. Evidence: `q2-ref13-investigation-20261012/`.
- **Gate v2** (`docs/design/exl3-statistical-gate-v2.md`, `f8061ad8` + P2
  fixes and **Amendment 1**, `fd7b530b`) is the qualification rule going
  forward. It was authored from reference-only data. Amendment 1 adds:
  - randomised interleaved run order;
  - a reference-only composition VOID + capped top-up;
  - disclosures that H3 is v1's rule unchanged and H4 is stricter.

  **Not signed off; no v2 data collected.**
- **Gated-norm attribution is fixture-local.** `v24-minimal` reproduces the
  reference on **7/7 differing elements across 2 independent (rank × prompt)
  groups**. The 16 fixtures are nested prefixes, and two groups never
  differed. v24 introduced no new differences.
  - Limits: v24 was chosen after seeing these fixtures, and other variant
    pairs were never run.
  - The replay used NVRTC `--fmad=false`; production flags are unverified.
  - `v24-minimal` carried no engagement controls of its own, and the
    repaired harness has never run on a GPU.
  - Evidence: `q2-norm-ablation-20261012/`.
- **C1 (layer-0 local o_proj) CLOSED from native captures** (operator
  decision 2026-10-12):
  - No same-input projection difference: byte-identical outputs on 8/8
    identical-input rows.
  - The natural ≤1-ULP norm delta is absorbed on rank0 (6 input elements
    differ, 0 output elements) and confined to one token on rank1 (109
    elements, max |Δ| 6.1e-5).
  - **FLAG:** "corrected input ⇒ reference output" is inferred from 8 rows
    of 2 prompts, and the candidate o_proj input precision is unattested.
  - The BF16 replay harness was built on an invalid premise (both engines
    multiply EXL3 trellis weights) and is archived (`651ff083`, tag
    `archive/c1-bf16-o-proj-replay-20261012`). Evidence + erratum:
    `q2-c1-execution-20261012/ERRATUM-20261012.md`.
- **Dominant final-logit cause: unproven.** Per-case excess concentrates in
  short16 `arithmetic`, then `longer`, then `index-boundary-2048`. Known
  internal differences: the layer-0 gated-norm ≤1 ULP and layer-3 expert
  membership changes. Downstream causality is unmeasured; C2 measures it.
- **Neighboring-types matrix instantiated** (49 rows; none passed). v24 reaches
  exactly `kernels/gb10/common/kda.cu:565` at runtime (GLM-KDA only) but is
  build-reachable by all 30 targets; fused twins must be classified by any
  shared-code fix. Evidence: `q2-neighbor-matrix-20261012/`.
- The KV flag-name difference (candidate `fp8_g128` vs reference `auto` →
  `fp8_ds_mla`) is not evidence of dtype asymmetry; G5 concerns effective
  cache/attention-path equivalence.
- No production numerical fix is qualified. No image was published, and
  the production containers remain stopped.

## Host state

Nothing is in flight. At the last checkpoint (2026-10-12, end of the
overnight drive) both Sparks had 0 running containers. This was not
re-verified since; inspect both hosts before any GPU work.

## Next actions (ordered; details in todo.md "Next batch")

1. **Gate v2 sign-off.** The operator reviews the amended
   `exl3-statistical-gate-v2.md`. Recommended first: a cheap probe of the
   ref13 discrete mechanism (autotune/plan, NCCL algorithm, routing tie
   pinning). v2 collection only after sign-off.
2. **C2.** Layer-0 norm-output substitution (reference-exact; v24 is
   byte-identical to it) on both ranks, traced through o_proj → routing →
   MoE → final logits against the failing rows. Needs a new off-by-default
   knob with a neutrality proof. Captures use `Exl3FfnRouter`.
3. **L1 escalation** for `longer`: complete mHC residual streams + L1 KDA
   between matching L0 outputs and differing L1 FFN inputs.
4. **G-fallbacks** (G2–G7 with G8/G9 binding first), reusing archived receipts.
5. **Repeat Rust/vLLM/source review** on actual candidate artifacts. This
   and the neighbor matrix are a hard prerequisite for any shared-code fix.
6. **Production v24 gated-norm fix**, only behind:
   - C2 causality evidence;
   - a GPU rerun with v24's own controls under the repaired harness;
   - proof under production compile flags;
   - neighbor coverage;
   - a fresh v2 qualification run.

## Preserved production profile (currently stopped)

- Repository: `/home/colum/Workspace/RustroverProjects/atlas`.
- Branch: `sparkglm/installable-20260924` (original roadmap review was
  against `11671adf`; current HEAD moves — use `git log`).
- Model: `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3`.
- Checkpoint revision: `dda99c7e935920fa6e235ce410e6f806e1c02720`.
- Last recorded serving containers: `glm53-exl3-atlas-0` on spark0 and
  `glm53-exl3-atlas-1` on spark1, port 8000, restart policy `unless-stopped`.
- Last recorded production image: `atlas-exl3-validation:src-48624a8605de-rust1.97.1`
  (`sha256:82b23cd19e595cae5e96616e726a64f2f4b964416d31268210bf24f3e686c737`).
  The Q0 clean image `src-8b6b47f14d5b-rust1.97.1` (`sha256:e744b81d…`, from
  `fb4f4b41`) was later loaded on both nodes. **Which image the stopped
  `glm53-exl3-atlas-*` containers reference is unverified:** run
  `docker inspect` on both hosts before any restart. Source snapshot IDs are
  not Git commits.
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

If the numeric jump route times out, use the user-corrected jump hostname
`ai-node` (observed Tailscale address `100.121.137.95`) instead of the old
numeric LAN route.

spark0 is `gx10-eb0e`; spark1 is `gx10-235f`. Recorded hardware: GB10, 48 SMs,
approximately 121 GiB unified memory each, driver 580.159.03, host CUDA 13.0 /
nvcc 13.0.88. Fabric addresses are `192.168.100.1` / `.2`. Launcher interfaces:
spark0 `enp1s0f0np0` / `rocep1s0f0`; spark1 `enp1s0f1np1` / `rocep1s0f1`.
It currently selects one HCA per rank and `NCCL_CROSS_NIC=0`.

Durable evidence root: `/home/colum/Workspace/atlas-exl3-validation-20261010`.
Paths in the evidence table below are relative to it. Some large reference
arrays remain on the hosts; `/tmp` logs alone are not durable evidence. Before
GPU work, inspect both hosts' containers, GPU processes, image IDs, flags and
health. Coordinate one experiment owner across both ranks.

## Numerical contract (essentials)

The checkpoint has hidden size 4096, 45 layers (34 KDA, 11 MLA), 42 routed MoE
layers beginning at layer 3, 288 experts, sigmoid top-8, normalization 2.5,
groups 1/1, expert intermediate 2048 globally / 1024 locally, and one shared
expert. Physical vocabulary is 154880; tokenizer size is 154856. Absent
`mhc_post_mult_value` is handled by the explicit named reference policy (2.0),
with conflicting values rejected.

The frozen comparison profile is TP=2 / EP=1, BF16 KV, eager execution,
context 4096, chunk 128, C1 admission, no prefix caching or speculation. Its
absolute-threshold reports remain diagnostic records; the user-approved
statistical gate and registered effective FP8 KV profile decide numerical
qualification. The v1 rule (`docs/design/exl3-statistical-gate.md`) is the
historical record. `docs/design/exl3-statistical-gate-v2.md` (with
Amendment 1) is the rule going forward once signed off. "C1 admission" here
means one-sequence admission, not experiment C1.
Long-prefix cases retain 2047/2048/2051/2053 tokens and registered
continuations.

Earlier stage-level status bullets (KDA closure records, layer-3 `fp8_ds_mla`
identification, absolute-threshold short16 records, first-divergence notes)
are superseded narrative — see Current verdict above and Evidence index below.

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
| `kda-fla-candidate-20261010/{ref-attn3-capture,mla3b-capture,fp8-mla,qdq-*,ref-mhc3-capture,mla3-capture}` | Layer-3 MLA/mHC captures on both engines: inputs/projections exact, core attention diverges; fp8_g128/QDQ reproduces single-key rows |
| `statistical-gate-20261011/` | Old ref01/ref02 diagnostic captures; preserved `resume-diag01` failure; collector-only command descriptors (`collector-only-profile/`) and full mounted-source audit; frozen gate reports in `final/`, runs in `runs/`, batch2 attempts in `batch2-attempts/` |
| `checkpoint-full-sha/` | Earlier full shard hashes, confirmed by both-host full rehash in `checkpoint-rehash-20261011/` |
| `fp8decode-candidate-20261011/` | Replacement source archive, successful no-cache build log/manifest, focused CPU checks and final-image GPU evidence |
| `local-ci-baseline-blockers/`, `fmt-baseline-review/`, `unchanged-dense-cache-baseline/` | Earlier baseline failures; reproduce before calling any current failure pre-existing |
| `statistical-gate-20261011/final/` | Pre-registered gate verdict: candidate FAIL (rank + 3 catastrophic short16 `longer` rows), ulp control fails the same way, held-out `ref13` fails `longer` row 0; per-case breakdowns and leave-one-out records |
| `q2-investigation-20261011/` | Read-only Q2 investigation: `report.md`, `report-stage2-propagation.md`, `report-mla-implementation-delta.md`; candidate excess dominates, MLA causal link unproven |
| `q2-diagnostic-20261011/` | Extended diagnostic dumps (1.1 GB), repaired analyzer + reviewed localization report (`analysis/`), resume-analyzer receipts |
| `q2-norm-replay-20261011/` | Runtime-bound gated-norm replay: 16 fixtures, reference self-replay 11/11 per rank, candidate PTX self-replay 8/8 per rank, epsilon bits agree |
| `q2-norm-ablation-20261012/` | Arithmetic attribution: `v24-minimal` matches 7/7 differing elements across 2 independent fixture groups (16 nested fixtures); `analysis-20261012.md` + `ERRATUM-20261012.md`; pre-repair harness runs; repair + review in `harness-repair-20261012/` |
| `q2-codepath-audit-20261012/` | Merged vLLM step inventory + gap report G1–G9 (`comparison-report.md` §4–§5) |
| `q2-ref13-investigation-20261012/` | ref13 held-out investigation — verdict: reference-internal bimodality (3/15, 6 byte classes), mechanism unidentified (`report.md`, `analysis/`) |
| `q2-neighbor-matrix-20261012/` | 49-row neighboring-types matrix (`matrix.md`, `reachability.md`); none passed |
| `q2-c1-prep-20261012/` | C1 preparation (binding spec); superseded by the C1 re-scope |
| `q2-c1-execution-20261012/` | C1 build/neutrality (PASS, byte-identical)/native pre-reduce captures; **read `ERRATUM-20261012.md` first** — the replay-anchor conclusions are wrong |
| `q2-c1-binding-2026013/` (sic — missing digit, created 2026-10-12; do not rename) | 32 assembled C1 rows: fixture inputs, native own-outputs (valid); BF16 weights are reconstructions, not what either engine multiplies |
| `q2-gate-v2-20261012/` | v2 calibration (reference-only): `calibrate_v2.py`, `calibration.json`, `review-summary.md` (P2 fixes now applied) |
| `merged-image-20261011-session2/` | Corrected-image neutrality (`grammar-r2/`) and focused Q3 receipts, attempt tarballs, provenance; attempt `cand-grammar-ee5e-01` preserved and unreused |
| `p0-trace-prep-20261011/` | Prepared paired named-session Nsight driver/tests/README (11 CPU tests); no live profiler execution yet |

Required pair from `nvidia-cublas` 13.1.1.3:

- Classic SHA256: `4e346a7b60a01552bd9bf4bbc6c862e11eb851559c3da3572293345de6aaed75`.
- Lt SHA256: `930418840b747413584867cb7167dd5911dcf4184135589e0334527e9abbac9f`.
- Preserved staging: `reference-blas131-pair` on spark1; original metadata and
  license archived. The attested image uses `/opt/atlas/cublas-13.1.1.3` first
  in the loader path. Verify actual classic **and** Lt mappings after runtime
  dlopen on both ranks; on-disk presence alone is insufficient.

## Historical fold (preserved elsewhere)

- Q3 source fixes (`7e04679f` first-token seed threading, `f7791189` Anthropic
  streamed stop_sequence; multi-token stop branch
  `6f25cb8c`/`c4644ca5`/`264667b9`/`e39790e0` on `pi/q3stop-20261011`) are
  included in the tested `ee5eaf0f` image; detail removed from this file on
  2026-10-12 — see git history.
- Dated background reviews remain useful: [code review](CODE_REVIEW_2026-10-10.md),
  [kernel review](GB10_KERNEL_OPTIMIZATION_REVIEW_2026-10-10.md),
  [prior-art corrections](GB10_PRIOR_ART_CROSSREF_2026-10-10.md) — head
  transport finding open; legacy NVFP4 advice and prior-engine rates are not
  native EXL3 results.
