# GLM-5.3 EXL3 on two Sparks — continuation state

Restart context for the GLM-5.3 EXL3 serving effort: where things stand, how
to resume safely, and where the evidence is. The work itself (roadmap,
active batch, backlog) is in **[todo.md](todo.md)**. This file does not
duplicate it.

**Read order on restart:**
1. This file, sections 1–3.
2. [todo.md](todo.md) "Roadmap" and "Now", then the
   [completed v2 qualification report](docs/design/exl3-statistical-gate-v2-qualification-20261014.md).
3. [docs/plans/q2-norm-causality-plan.md](docs/plans/q2-norm-causality-plan.md)
   for the active experiment.
4. The last entries of [q2-diagnostic.md](q2-diagnostic.md) and
   [decisions.md](decisions.md).
5. The `AGENTS.md`/`CONTRIBUTING.md` rules and the measurement-discipline
   skill before any numerical claim.

The latest review and its disposition table are in
[CODE_REVIEW_2026-10-12.md](CODE_REVIEW_2026-10-12.md). Superseded narrative
is in git history.

## 1. Status at a glance (2026-10-15)

| Area | Status | Pointer |
|---|---|---|
| Boot and generation (TP=2/EP=1, native EXL3) | Works; 5/5 launch smoke; production containers **stopped** | §4 |
| Numerical qualification (Q2) | **Not qualified.** v1 gate FAIL (historical); complete v2 campaign is **UNVERIFIABLE**: reference ensemble fails H3 and H4 LOO | §2, [qualification report](docs/design/exl3-statistical-gate-v2-qualification-20261014.md) |
| Dominant cause of divergence | **Unproven.** C2 mixed; L1 collected, attribution withheld by reference observer mismatch | todo "Now" #3 |
| API behavior (Q3) | Partial: focused live slice 15/15 on `ee5eaf0f` | todo Q3 |
| Task quality (Q4) | Harness only; no live run | todo Q4 |
| Performance (P0) | Unqualified diagnostic only (decode ~0.45 s/token); Nsight driver prepared, not run | todo P0 |
| Serving features (F1–F4), release (R1/R2) | Not started | todo |
| Reference runtime mechanism | Probe executed 2026-10-15 (5/6 complete, on01 failed and retained). No recurrence of either singleton, all draws inside the campaign H3 envelope; gate inputs diverge first at layers 8–33, routing follows, collective dispatch identical. Mechanism unresolved; EXL3 plans unobservable (nsys misses forked workers) | [probe result](docs/design/exl3-runtime-mechanism-probe-result-20261015.md) |
| In flight | Nothing. C2 complete with mixed scores; L1 attribution withheld (reference observer 1/8). Both hosts released 2026-10-15 00:26Z | [L1 report](docs/design/l1-execution-20261014.md), todo M1 |

## 2. Established findings (what a restart must not re-derive)

- **v1 gate verdict (historical record).** Candidate `cand-fp8decode02`
  FAILS: rank T 7.20 vs refs 5.04–5.53, plus 3 catastrophic short16
  `longer` rows; the ulp control fails the same way
  (`statistical-gate-20261011/final/`). Candidate excess concentrates in
  short16 `arithmetic` (29× the worst reference), then `longer`, then
  `index-boundary-2048`.
- **The reference is bimodal at `longer`/r0.**
  - A byte-identical recurring mode appears in 3/15 draws (CP95
    [0.043, 0.481]; 2/14 excluding the triggering ref13); 15 draws fall
    into 6 byte classes.
  - The mechanism is unidentified (likely a discrete runtime choice).
    "Fingerprint-identical" covers configuration only.
  - v1 was **underpowered at N=9 under a bimodal row, with session drift
    not excluded**. It was not structurally unpassable.
- **Gate v2** ([registered rule](docs/design/exl3-statistical-gate-v2.md),
  original `f8061ad8`; P2 fixes + Amendment 1 `fd7b530b`; Amendment 2
  follows the operator-requested Luna source review) uses reference-only calibration.
  - It uses fixed N=15 references + H=3 held-outs, the v1 rank statistic T, v1's
    envelope rule on every row, and a mode-membership rule.
  - Randomised interleaving remains. Amendment 2 withdraws Amendment 1's
    composition VOID/top-ups: an unverifiable reference set blocks qualification
    at fixed N. H4 explicitly handles zero-distance/zero-radius cases.
  - H3 constants stay fixed, but measured envelopes can widen with N. H4
    can be stricter. The rank-only bound is not a whole-campaign guarantee.
  - The fixed 26-attempt campaign completed (832 request receipts); corrected
    evaluation is **UNVERIFIABLE**, with one H3 and one H4 reference LOO
    failure. No candidate/control gate verdict was reached. The initial frozen
    evaluator report remains preserved; a post-collection correction fixed its
    heldout byte-identity check without changing thresholds. See the
    [qualification report](docs/design/exl3-statistical-gate-v2-qualification-20261014.md).
  - H3 failure: index 13 (`v2ref14`), short16 `longer` row 3. H4 failure:
    index 14 (`v2ref15`), short16 `code` row 1. Reference-only diagnosis
    reproduced both failures and found no recorded
    profile/cache difference explaining them ([report](docs/design/exl3-reference-verifiability-diagnosis-20261014.md)).
    Runtime pinning probes are justified; no concrete protocol amendment is
    supported yet. Investigate these fresh reference-set failures separately
    from the historical `longer`/r0 mode and C2 candidate
    causality work. Do not top up, replace draws or retry unchanged to seek PASS.
  - Both candidate repeats pass byte identity; this is not numerical
    qualification. All five perturbations and MDE were collected, but their
    gate scoring, the held-out scoring and candidate numerical scoring were
    skipped after reference verifiability failed.
  - Frozen evaluator INCOMPLETE and corrected UNVERIFIABLE reports are both
    preserved. The narrow held-out identity fix passed 24 focused tests;
    the correction/report changes were independently reviewed and archived
    hashes reverified on 2026-10-14 before commit. Freeze a future evaluator
    separately. The original thresholds and metrics are unchanged.
- **Localized internal differences** (reviewed,
  `q2-diagnostic-20261011/analysis/`):
  - KDA core/g1/g2 match.
  - Layer-0 gated norm differs by ≤1 BF16 ULP.
  - L3 expert membership changes.
  - Post-TP MoE p99 is 64–71 ULP.
  - The earlier 1507-ULP and same-input-GEMM claims are retracted.
- **Gated-norm attribution is fixture-local.** `v24-minimal` (rsqrt →
  sqrt.approx.ftz + div.full; divide → reciprocal multiply) reproduces the
  reference on **7/7 differing elements across 2 independent groups** (the
  16 fixtures are nested prefixes).
  - v24 was chosen after seeing the fixtures.
  - The replay used NVRTC `--fmad=false`.
  - The historical replay had no `v24-minimal` engagement controls. Those
    controls and the production-compile lane are now implemented; the GPU
    reruns remain owed (todo "Next").
  - The repaired harness has never run on a GPU.
- **C1 (layer-0 local o_proj) is closed** from native captures by operator
  decision:
  - zero same-input projection difference on 8/8 identical-input rows;
  - the natural norm delta is absorbed on rank0 and confined to one token
    on rank1.

  **C2 first measurement passed:** each registered fixture variant matches
  all 32 rank/row/repeat native local outputs to the reference captures.
  This discharges the corrected-input/reference-output flag for the bound
  prefixes; input dtype itself remains unattested. Both engines multiply EXL3 trellis weights (the reference uses an
  FP16 input for ≤144 rows), so never replay o_proj over a BF16
  reconstruction. The old harness is archived (tag
  `archive/c1-bf16-o-proj-replay-20261012`).
- **Neighbor matrix:** 49 rows, none passed. v24 reaches only
  `kernels/gb10/common/kda.cu:565` at runtime (GLM-KDA) but is
  build-reachable by all 30 targets.
- **KV dtype:** candidate `fp8_g128` matches the reference's `auto` →
  `fp8_ds_mla` format. The flag names differ, but that is not a dtype
  asymmetry. G5 concerns effective path equivalence.
- **Pinned vLLM is authoritative for reference behavior.** The
  [source contract](docs/design/exl3-vllm-reference-contract.md) records the
  Luna review and parent source checks: the bound SM120 short-prefill path
  reads packed cache too; logits are captured after the head/TP gather and
  before sampling. Seed 42/fresh processes do not guarantee deterministic
  forward logits. The recurring mode's cause remains unidentified.
- **Latest collected candidate:** runtime source `2fbb787c`, image
  `atlas-exl3-validation:src-40efa67de6d9-rust1.97.1`, image ID
  `sha256:011423015fc1da354e07fbe5ee1b49751efb4e3a886159f2a23f458744cbcff0`.
  A no-cache build and short16/long byte-neutrality proof against the C1
  baseline preceded collection; both cuBLAS providers were attested. Evidence:
  `gate-v2-qualification-20261013-luna1/neutrality-proof.json`.
  This image includes the C2 knob off, not a qualified v24 numerical fix.
  Later source/tooling changes are outside this image's evidence: bind the
  current source and rebuild/reprove neutrality where required before new GPU work.
- **C2 live attempt stopped** (`c2-refexact-02`): both baseline repeats
  match all eight prior-image logit rows and 16 archived norm captures per
  repeat. The first treatment request fails on rank 1 because its
  `ForwardContext` has no request ID. Rank 0 emits one substitution receipt;
  no treatment request completes. Causality remains unresolved and C1's
  first measurement remains undischarged. Both hosts were released; worker
  transport repair, a fresh image and neutrality proof precede a fresh C2
  attempt. [Report](docs/design/c2-execution-20261014.md).
- No production numerical fix is qualified, and no image was published.
- **Worker binding repair:** `f099ac58` delivers bounded request-ID metadata
  to the addressed distributed worker sequence before prefill. Both peers
  require the same image. The full model suite at that commit passed
  (1660 tests, 17 ignored); focused sender/worker regressions also passed.
  The fresh no-cache image `atlas-exl3-validation:src-4ee96c12b44c-rust1.97.1`
  is loaded on both ranks as
  `sha256:82173ea631a8c78e4fb565f0e23bc6d55f94a43090dbfb27c850f8cc1e6b41d9`.
  Both packaged binaries and their 232 embedded PTX modules were verified;
  toolchain, compiler closure, PTX and runtime dependency fingerprints match
  the prior collected image. Live provider mapping and full logit neutrality
  remain tied to each lifecycle. The completed full-suite collection is
  `neutrality-bounded03/worker-binding-neutral03`: all 32 logit rows and text
  match the frozen baseline, both containers exit 0, and the frozen verifier
  independently passes. Proof SHA256:
  `52b8f2de1fc300d7b2e194c273afaf5817d505adbbe127c5a20f5f3d5e64284f`.
  Fresh C2 preparation revalidated this proof. `c2-refexact-f099-01`
  completed: all 32 native o_proj comparisons match; both arms are byte
  deterministic; arithmetic has two closer and two farther rows, while all
  four `longer` rows are unchanged and three retain catastrophic KL. The
  separate `c2-v24-f099-01` repeats completed with identical scores and all
  controls passing. Source-bound `l1-refexact-f099-01` stopped after both candidate windows
  and all eight reference-control requests completed. The reference worker
  received Docker stop SIGTERM then timeout SIGKILL (exit 137, non-OOM),
  which the strict exit-zero guard rejected. The candidate observer matches
  all eight rows/text. Correction `1204a210` registers bounded non-OOM
  worker timeout-stop proof while retaining exit zero for candidate/leader.
  Full CPU harness: 612 tests and 278 subtests passed. Fresh
  `l1-reference-exact-shutdown-v2/l1-refexact-f099-02` completed all four
  eight-request windows, each with both ranks exiting zero. Candidate observer
  matches 8/8; reference observer matches 1/8, with unchanged text/thresholds.
  Attribution is withheld and no L1 bisect ran. Both hosts are released.
  [L1 result](docs/design/l1-execution-20261014.md).
  [Repaired C2 report](docs/design/c2-repaired-worker-execution-20261014.md).
  No Q2 qualification claim. The earlier administrative abort preserved
  16 matching short rows and seven matching long rows without a full proof.
  Evidence: `q2-c2-worker-binding-20261014/`; [validation record](docs/design/c2-worker-binding-validation-20261014.md).

## 3. Before any GPU work

1. Inspect both hosts: containers (`docker ps -a`), GPU processes, image
   IDs, flags and health. Confirm nothing else owns the GPUs, and
   coordinate one experiment owner across both ranks.
2. Record which image the stopped `glm53-exl3-atlas-*` containers reference
   (§4). The last verification was 2026-10-12; recheck current ownership
   and image identity rather than relying on that historical receipt.
3. Use fresh attempt names and capture roots. Never reuse burned names or
   delete failed attempts. `/tmp` logs are not durable evidence; archive
   to the evidence root (§6).
4. For any new image: no-cache build, source/image/PTX fingerprints on both
   ranks, a neutrality proof against the frozen baseline, and
   cuBLAS classic **and** Lt runtime mappings verified (§6).
5. Append a shutdown/teardown receipt when releasing the hosts.

## 4. Production profile (preserved, currently stopped)

- Repository: `/home/colum/Workspace/RustroverProjects/atlas`.
- Branch: `sparkglm/installable-20260924` (original roadmap review was
  against `11671adf`; current HEAD moves — use `git log`).
- Model: `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3`.
- Checkpoint revision: `dda99c7e935920fa6e235ce410e6f806e1c02720`.
- Last recorded serving containers: `glm53-exl3-atlas-0` on spark0 and
  `glm53-exl3-atlas-1` on spark1, port 8000, restart policy `unless-stopped`.
- Last recorded production image: `atlas-exl3-validation:src-48624a8605de-rust1.97.1`
  (`sha256:82b23cd19e595cae5e96616e726a64f2f4b964416d31268210bf24f3e686c737`).
  **Verified 2026-10-12:** the stopped `glm53-exl3-atlas-0/-1` containers
  reference exactly this image on both hosts (receipt:
  `$EV/host-inspection-20261012T2204Z/`; tool:
  [scripts/inspect-sparks.sh](scripts/inspect-sparks.sh)). The Q0 clean
  image `src-8b6b47f14d5b-rust1.97.1` (`sha256:e744b81d…`, from
  `fb4f4b41`) was later loaded on both nodes. Source snapshot IDs are
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

## 5. Hosts and access

Inventory/configuration: `~/Workspace/homelab-ansible`.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 -J colum@192.168.1.50 colum@10.50.0.10
ssh -o BatchMode=yes -o ConnectTimeout=15 -J colum@192.168.1.50 colum@10.50.0.11
```

If the numeric jump route times out, use the jump hostname `ai-node`
(observed Tailscale address `100.121.137.95`). The `~/.local/bin/spark0` /
`spark1` wrappers ssh via the jump host. The Sparks have no SSH trust
between each other.

- spark0 is `gx10-eb0e`; spark1 is `gx10-235f`.
- Hardware: GB10, 48 SMs, ~121 GiB unified memory each, driver 580.159.03,
  host CUDA 13.0 / nvcc 13.0.88.
- Fabric addresses: `192.168.100.1` / `.2`.
- Launcher interfaces: spark0 `enp1s0f0np0` / `rocep1s0f0`; spark1
  `enp1s0f1np1` / `rocep1s0f1`. The launcher uses one HCA per rank and
  `NCCL_CROSS_NIC=0`.

## 6. Numerical contract and evidence

**Checkpoint:**
- hidden size 4096; 45 layers (34 KDA, 11 MLA);
- 42 routed MoE layers from layer 3; 288 experts, sigmoid top-8,
  normalization 2.5, groups 1/1;
- expert intermediate 2048 globally / 1024 locally; one shared expert;
- physical vocabulary 154880, tokenizer 154856;
- absent `mhc_post_mult_value` takes the named reference policy (2.0);
  conflicting values are rejected.

**Frozen comparison profile:** TP=2/EP=1, reference requested KV `auto`
(effective MLA `fp8_ds_mla`), candidate `fp8_g128`, eager, context 4096,
chunk 128, one-sequence admission, no prefix caching or speculation. These
are qualification settings, distinct from the preserved serving launch in §4. The
long-prefix cases keep 2047/2048/2051/2053 tokens and registered
continuations. Absolute-threshold reports are diagnostics. The registered
statistical gate decides qualification: v1
(`docs/design/exl3-statistical-gate.md`) is the historical record, and v2
(with Amendments 1–2) governed the completed campaign. Its corrected result
is UNVERIFIABLE; qualification remains blocked. Prefix caching stays off and
chunk size stays 128 for this registered profile.

### Evidence index

Durable evidence root: `/home/colum/Workspace/atlas-exl3-validation-20261010`
(paths below are relative to it). Some large reference arrays remain on the
hosts. Restore and verify hashes before reusing a record. A build, CPU test,
component oracle and full-model gate each prove a different scope.

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
| `gate-v2-qualification-20261013-luna1/` | Completed fixed 26-attempt campaign; 832 request receipts, neutrality proof, original INCOMPLETE and corrected UNVERIFIABLE evaluations, raw artifacts/cache receipts and final teardown. In-repo [report and evidence index](docs/design/exl3-statistical-gate-v2-qualification-20261014.md) bind the evaluator correction and H3/H4 failures. |
| `q2-gate-v2-20261012/` | v2 calibration (reference-only): `calibrate_v2.py`, `calibration.json`, `review-summary.md` (P2 fixes now applied) |
| `merged-image-20261011-session2/` | Corrected-image neutrality (`grammar-r2/`) and focused Q3 receipts, attempt tarballs, provenance; attempt `cand-grammar-ee5e-01` preserved and unreused |
| `p0-trace-prep-20261011/` | Prepared paired named-session Nsight driver/tests/README (11 CPU tests); no live profiler execution yet |
| `probe-runtime-mechanism-20261014/` | Reference runtime-mechanism probe: package/records, cache and mount freezes, nsys tooling checks (forked workers untraced), engagement smoke, six attempts (on01 failed), byte classes, descriptive H3, router diffs. [Result](docs/design/exl3-runtime-mechanism-probe-result-20261015.md). Per-attempt cache copies + telemetry also retained on the Sparks at `/home/colum/probe-rm-20261014/` |

Required pair from `nvidia-cublas` 13.1.1.3:

- Classic SHA256: `4e346a7b60a01552bd9bf4bbc6c862e11eb851559c3da3572293345de6aaed75`.
- Lt SHA256: `930418840b747413584867cb7167dd5911dcf4184135589e0334527e9abbac9f`.
- Preserved staging: `reference-blas131-pair` on spark1; original metadata and
  license archived. The attested image uses `/opt/atlas/cublas-13.1.1.3` first
  in the loader path. Verify actual classic **and** Lt mappings after runtime
  dlopen on both ranks; on-disk presence alone is insufficient.

## 7. Background references

- Q3 source fixes (`7e04679f` first-token seed threading, `f7791189`
  Anthropic streamed stop_sequence, multi-token stop branch on
  `pi/q3stop-20261011`) are in the tested `ee5eaf0f` image.
- Dated reviews:
  - [code review 2026-10-10](CODE_REVIEW_2026-10-10.md);
  - [code review 2026-10-12](CODE_REVIEW_2026-10-12.md);
  - [kernel review](GB10_KERNEL_OPTIMIZATION_REVIEW_2026-10-10.md);
  - [prior-art corrections](GB10_PRIOR_ART_CROSSREF_2026-10-10.md). The
    head transport finding is still open. Legacy NVFP4 advice and
    prior-engine rates are not native EXL3 results.
