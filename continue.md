# GLM-5.3 EXL3 on two Sparks — continuation state

*Superseded historical narrative (the 2026-10-11 resume/analyzer/reviewed-checkpoint blocks, sessions 2/3/3b, gate-batch and status tables, "Resume findings", "Merged CPU-side work", "Gate batch 2", the historical execution sequence and historical Q3 fix detail) was removed on 2026-10-12; it is preserved in git history and under the durable evidence root below.*

[todo.md](todo.md) is the authoritative remaining-work checklist; this file
holds the restart context, current verdict, and the evidence/profile/host
pointers. Sequencing SSOT:
[docs/plans/q2-norm-causality-plan.md](docs/plans/q2-norm-causality-plan.md);
append-only experiment log: latest entries in [q2-diagnostic.md](q2-diagnostic.md).

## Current verdict (2026-10-12)

- **Q2 statistical gate: candidate FAIL — unchanged** (pre-registered rule,
  nothing loosened). `cand-fp8decode02` fails on rank (T 7.20 vs refs
  5.04–5.53) plus 3 catastrophic short16 `longer` rows; the ulp control fails
  the same way; held-out `ref13` fails short16 `longer` row 0 — now
  investigated: the row is reference-internal bimodal chaos (see below).
  Per-case concentration: short16 `arithmetic`, then
  `longer`, then `index-boundary-2048`; the other long cases sit near the
  references' own spread. Full reports: `statistical-gate-20261011/final/`
  under the durable root.
- **v24 attribution proven on the 16 fixtures — stage-local only.**
  `v24-minimal` reproduces the reference gated-norm output byte-exactly 16/16
  (attested-PTX and `v0-copy` anchors also byte-exact; reduction-order and
  exp-construction variants are output-inert, not sources). Downstream
  causality is unmeasured — do not infer this explains the gate failure.
  Evidence: `q2-norm-ablation-20261012/` (analysis-20261012.md) and
  `q2-codepath-audit-20261012/`.
- **Ablation harness defects repaired** (commits `9ff39728`, `c71bb557`);
  independent review verdict "OK with notes, no blockers"
  (`q2-norm-ablation-20261012/harness-repair-20261012/review-summary.md`).
  The harness is cleared for new attribution work; deferred minor findings are
  listed in todo.md.
- **ref13 held-out failure explained (verdict: reference-internal chaos).**
  The failing `longer`/r0 row recurred byte-identically in 2/5 fresh
  fingerprint-identical reference repeats (3/15 draws overall, exact CP95
  [0.043, 0.481]); held-out-control emulation fails 3/5 for pure reference
  draws (2 catastrophic-row, 1 rank-only). No environment/config anomaly
  (all 15 runs' audits hash `3723c602aab8fa7d`). Consequence: the must-pass
  control can fail for the reference itself — **operator decision required**
  (todo.md); the frozen gate was not changed. Evidence:
  `q2-ref13-investigation-20261012/`.
- **Neighboring-types matrix instantiated (49 rows; none "passed").** The v24
  change reaches exactly one kernel entry (`kernels/gb10/common/kda.cu:565
  kda_sigmoid_gated_rms_norm`, GLM-KDA-only at runtime) but is build-reachable
  by all 30 targets; fused twins embedding the same arithmetic must be
  enumerated affected-or-out-of-scope by any shared-code fix. Evidence:
  `q2-neighbor-matrix-20261012/`.
- No production numerical fix is qualified; no image was published; production
  containers remain stopped.
- The KV flag-name difference (candidate `fp8_g128` vs reference `auto`) is
  **not** evidence of effective dtype asymmetry: `auto` resolves to
  `fp8_ds_mla`, the format `fp8_g128` was selected to match. G5 concerns
  effective cache/attention-path equivalence across chunk boundaries.

## In flight (2026-10-12) — check before touching hosts

- **C1 preparation** (binding-prerequisite check + TDD replay harness) running
  locally; evidence dir
  `/home/colum/Workspace/atlas-exl3-validation-20261010/q2-c1-prep-20261012/`.
- Completed earlier today: ref13 held-out diagnostic (both Sparks idle again,
  all engines stopped with post-stop receipts) and the neighboring-types
  matrix. Evidence dirs — do not touch:
  `q2-ref13-investigation-20261012/`, `q2-neighbor-matrix-20261012/`.
- Operator decisions in `decisions.md`: the 2048-token / deep-layer scope
  extension is GRANTED; ref13 runs in parallel with the causality chain;
  implementation and execution are delegated to subagents with independent
  review. Coordinate one experiment owner across both ranks before any other
  GPU work.

## Next actions (ordered; details in todo.md "Next batch")

1. **C1** — local `o_proj` replay before TP (original vs reference-exact vs
   v24-corrected norm output; each engine vs its own captured projection
   first). Sparks free since ref13 finished 2026-10-12 (C1-prep running); the harness repair that gated
   C1 is done.
2. **C2** — controlled layer-0 norm-output intervention on both ranks, tracing
   o_proj → routing → MoE → final logits against the failing rows.
3. **L1 escalation** — for `longer`, trace complete mHC residual streams and
   L1 KDA between matching L0 outputs and differing L1 FFN inputs, before broad
   late-layer bisection.
4. **G2–G7 fallbacks** — G8/G9 runtime + binary binding precedes each affected
   replay; reuse the recovered checkpoint config, head/router/expert component
   receipts and prior replays noted in todo.md instead of re-collecting.
5. **Neighboring-types matrix + repeat Rust/vLLM/source review** — HARD
   PREREQUISITE before any shared-code fix.
6. **Production v24 gated-norm fix** — only behind causality evidence, the
   plan's step-5 qualification obligations, and a fresh frozen Q2 run.

## Preserved production profile (currently stopped)

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
qualification (`docs/design/exl3-statistical-gate.md` is the current rule).
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
| `q2-norm-ablation-20261012/` | 16-fixture arithmetic attribution, `v24-minimal` byte-exact 16/16 (`analysis-20261012.md`), ablation harness, harness repair + independent review (`harness-repair-20261012/`) |
| `q2-codepath-audit-20261012/` | Merged vLLM step inventory + gap report G1–G9 (`comparison-report.md` §4–§5) |
| `q2-ref13-investigation-20261012/` | ref13 held-out diagnostic — IN FLIGHT, do not touch |
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
