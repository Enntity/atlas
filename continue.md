# GLM-5.3 EXL3 on two Sparks — continuation state

**Stopped at user request 2026-10-11 16:47 UTC — durable restart checkpoint.**

**Host state:** no running containers on either Spark. Production containers
remain stopped. Validation `gate-cand-0/1` now contain the corrected `ee5eaf0f`
image and are stopped. The older `e33340d9` engines are preserved as
`gate-cand-0/1-preserved-before-ee5eaf0f`; the KDA controls remain
`gate-cand-0/1-preserved-before-6f516e5a`. No image was published.

Completed this session:

- `e33340d9`: fixed the real CUDA production build failure by scoping the
  test-only `StopDecodeFn` re-export with `cfg(test)`. Production/test compilation
  and all 32 stop-matcher tests pass.
- Clean no-cache `e33340d9` image on both ranks:
  `atlas-exl3-validation:src-176780fd07e0-rust1.97.1` /
  `sha256:ed8a38adbc2813d9524cca3814be811a19aa0ca8958a3f6552a905b5e49643c0`.
  `cand-merged-e333-01` completed **short16 16/16 + long 16/16**, perturbations
  unset. Both complete NPZ archives, every F32 tensor byte and generated texts
  match `cand-fp8decode02`; profile/checkpoint/collector/provenance checks pass.
  A valid one-ULP mutation fails the comparator. **Merged-image arithmetic
  neutrality passed; this is not Q2 qualification.**
- Initial focused Q3 live run on that image passed **11/15**. The original raw
  receipts and all four failures are preserved. `8334ab33` fixes completion
  identity checks that compared variable timing metadata instead of stable
  usage counters. `19551fc6` fixes a real server bug: first-token grammar
  masking/advancement ran during template-forced thinking, consuming `{"`
  in reasoning and omitting it from the final JSON answer. All admission routes
  now pause grammar during initial thinking, matching later decode; template
  thinking resolution is preserved. The actual production-sampler regression
  fails before the fix and passes afterward on both fixture ranks/owners,
  including a first-token `</think>` boundary.
- `ee5eaf0f` corrects the EOS harness and stale token-ID comments: the exact
  terminal EOS ID is **untested** when it is not exposed by the stream. Atlas
  counts terminal EOS in completion usage without streaming its Token event.
  Complete-ID wrong-EOS and usage-counter-drift negative controls still fail.
- CPU verification: sampler fixture suite 5/5; full server **2568 pass / same
  3 known failures / 12 ignored** in the stub-linked color-capable environment;
  validation Python **274 pass + 193 subtests**. License headers, changed-file
  Rust formatting/typos and current CI source-cap/allowlist checks pass.
  Workspace fmt/clippy/typos fail in unchanged files (plus an existing user-owned
  untracked plan); see `required-check-limitations.json`. These are CPU/stub
  checks, not GPU serving proof.

Current corrected image (already built cleanly and loaded on both hosts):

- Source commit: `ee5eaf0f19370f0dc302565d46b834a88630afc8`.
- Source archive SHA256:
  `455b0da9f7cd7bab369593f212a13c43fa0ad7bc72bdeed5fbb075eb93f75a15`.
- Tag: `atlas-exl3-validation:src-455b0da9f7cd-rust1.97.1`.
- Image: `sha256:7270d7ca16f7896da09e78e4a68cbac56f0aac1749ea9f8c6c671370aa3b94a6`.
- Server SHA256: `bde219ffbd2922b6cac535448861c8452fa93c388afaf7185cad983c8527005a`.
- Rust 1.97.1; all 232 PTX hashes, compiler flags and toolchain records match
  the prior rebuilt image. Both live-rank provenance finalization and cuBLAS
  runtime-map attestation pass (CUDA runtime 13.0.48 / NCCL 2.32.3).

**Attempt `cand-grammar-ee5e-01` is intentionally incomplete; do not reuse it.**
Short16 completed **16/16**, with the full NPZ archive and generated text
byte-identical to `cand-fp8decode02`. Long completed **6/16 rows**, each raw F32
row matching the baseline. At the user's stop request the collector was
suspended to prevent new admission, the in-flight request drained, both rank
containers stopped, then the collector resumed and saved that sixth response.
Its next request got connection refused because the engines were deliberately
stopped; `long.log` records this, not a new Atlas regression. The collector
removed `capture-enabled` in its `finally` block. **No completed long manifest
or NPZ exists; full corrected-image neutrality and live grammar validation
remain pending.** The queued corrected Q3 rerun was cancelled before launch.

Durable evidence: `merged-image-20261011-session2/` under the root below.
Original full-neutrality and Q3 evidence lives there; corrected-image source,
CPU records and the complete stopped attempt are under `grammar-fix/`.
`grammar-fix/driver/` contains both-rank attempt tarballs, exact create commands,
image/build manifests, runtime maps, provenance, short neutrality, partial-long
reports, build log and `stop-checkpoint.json`. Live copies:
`/tmp/atlas-resume-20261011-session2/grammar-image/`; remote attempts:
`~/gate/attempts/cand-grammar-ee5e-01`. Build checkout/log on spark1:
`~/atlas-build/resume-6f516e5a` (detached `ee5eaf0f`) and
`~/atlas-build/resume-ee5eaf0f-build.log`.

Next session, in order:

1. Inspect both hosts and image IDs. Reuse the exact corrected image above;
   documentation-only commits after `ee5eaf0f` do not change it. Do not restart
   the preserved KDA-control containers as the candidate.
2. Create **fresh** attempt `cand-grammar-ee5e-02` and capture root, e.g.
   `/tmp/atlas-ee5eaf0f-neutrality-r2`. Restore/copy `grammar-fix/driver/` into
   a new local working directory. Update `gate_run.sh`'s capture root and remote
   staging path, the profile JSON host capture mounts, and its preserved-container
   rename suffix (the current `preserved-before-ee5eaf0f` names already exist).
   Stage the matching descriptors remotely; keep every env/serve flag unchanged,
   including perturbations **unset**. The driver rejects existing capture roots;
   `run_gate_attempt.sh` rejects reused names. Never delete prior evidence.
3. Run the preservation wrapper with the fresh `GATE_CAPTURE_ROOT`, collect full
   short16 + long and use `check_neutrality.py` against `cand-fp8decode02`.
   Require both complete archives and texts to match before Q3. The stopped
   partial collection cannot substitute for this full run.
4. Copy/update `run_q3.sh` to the fresh attempt/output and source-bound remote
   harness. Rerun the same focused 15-case slice. It must prove the JSON opener
   fix on GPU and resolve the completion/EOS harness failures. Record skipped
   exact-EOS-ID verification as untested. Broader Q3/Q4 remain open.
5. Return to the failed registered Q2 gate without changing its rule. The next
   arithmetic diagnostic is `q2-diagnostic-plan.json`: prefixes 10–13,
   FFN/routes layers 0–3, upstream layer 0, MLA layer 3. The frozen reference
   collector package does not invoke `reference_attn_trace.patch`; create/hash
   a separate diagnostic overlay before MLA capture. MLA is a hypothesis,
   not an established numerical root cause. No gate redesign is authorized.

Prior restart checkpoint: tag `checkpoint/fp8decode-20261011` on branch
`sparkglm/installable-20260924`. Older entries below are historical wherever
the resume above supersedes their image, source, container, or completion state.

**Q2 statistical gate result 2026-10-11 ~09:00 UTC — candidate FAILS (pre-registered rule, nothing loosened).**
Reports: `statistical-gate-20261011/final/` under the durable root; runs in `runs/`,
both-rank attempt trees in `batch2-attempts/`. References ref04–ref12 (9),
held-out ref13; harness = repo `statistical_gate.py` at `27bf8352`+ (unchanged since `9879d73b`).

| Run | T (refs 5.04–5.53) | catastrophic rows | verdict |
|---|---|---|---|
| candidate `cand-fp8decode02` ≡ `cand03` (logits byte-identical) | 7.20, p=0.1 (minimum) | 3 (short16 `longer` rows 0–2, sym_kl) | FAIL: rank + rows; ulp control fails the same way |
| `candkda0` (`ATLAS_KDA_FLA_REFERENCE=0`, MDE) | 7.68 | 1 + 1 argmax/text | detected (expected) |
| held-out `ref13` (must pass) | 5.30, p=0.7 | 1 (short16 `longer` row 0: 0.056 vs limit 0.005) | FAILS one row |

- Leave-one-out is verifiable (0 failures for all 9 references); negative controls pass.
- Both suites put the candidate beyond every reference (short16 7.42 vs ≤5.52;
  long 6.98 vs ≤5.72), so the candidate is systematically distinguishable, not
  failing on one noisy row. It is only modestly closer than the known-wrong KDA
  policy (7.20 vs 7.68). Per-case localization (`final/per_case_breakdown.py`,
  `final/per-case-cand02.json`): the excess concentrates in short16
  `arithmetic` (candidate nearest-reference symKL 0.074 vs reference-to-reference
  0.0018) and `longer` (0.045 vs 0.0004), then `index-boundary-2048` (0.075 vs
  0.017); the other long cases sit near the references' own spread. Cause is
  **not localized** — the multi-key sparse MLA FP8 difference is a candidate
  hypothesis only. Next diagnostic: per-layer captures for short16 `arithmetic`.
- The held-out reference failing one row means the reference is itself
  occasionally bimodal at short16 `longer` row 0, where the 9 references agree
  to < 0.005/3. Per the pre-registration this is a failed must-pass control; the
  rule is not changed. **Decision for the user:** pursue reference-faithful
  multi-key MLA FP8 numerics, or revisit the gate design (new pre-registration).
- Interim 3-reference reports remain in `interim/` (diagnostic only).

**Status 2026-10-11 06:25 UTC:**

- The 05:29 batch (`gate_batch_fresh.sh`) **died**: both ref03 attempts failed
  reference provenance finalization, then `|| exit 1` ended the driver. Cause:
  the reference image maps wheel runtimes by soname only (`libnccl.so.2`,
  `libcudart.so.13`), so no version could be derived from the path. The
  finalizer had never completed for `ref` before (ref01/ref02 predate it).
  The old driver's retry cleanup also `rm -rf`'d attempt 1's evidence; attempt
  2's both-rank logs/inspect/captures are archived in
  `statistical-gate-20261011/failed-ref03/`. `ref03` is burned.
- Finalizer fixed (live copy `/tmp/atlas-resume-20261011/provenance/`, durable
  copy in `collector-only-profile/provenance-tools/`, original kept as
  `finalize_provenance.pre-ref-version-fix.py`): when the path has no version,
  NCCL uses the mapped binary's own `NCCL version X+cuda` banner (cross-checked
  with the owning wheel), cudart uses the owning `*.dist-info/RECORD` entry
  whose sha256 must match the mapped bytes. Reference attests NCCL 2.30.7 /
  CUDA runtime 13.0.96. It also accepts `candkda0` (candidate descriptors plus
  exactly `ATLAS_KDA_FLA_REFERENCE=0`), which the old chain would have rejected.
  Dry runs `ref03-finalizer-dryrun{,2}` passed on both live reference ranks:
  `statistical-gate-20261011/finalizer-dryrun-ref03/`.
- **Batch 2 completed 08:47 UTC** (`collector-only-profile/gate_batch_fresh2.sh`):
  refs **ref04–ref12** (9), held-out **ref13**, cand03, candkda0. It never
  deletes evidence; a failed attempt is retried as `NAME-r2`. Logs:
  `statistical-gate-20261011/batch2-driver.log` and `batch2-logs/NAME.log`;
  completed runs append to `batch2-logs/completed.txt`.
- After batch: `statistical_gate.py` with `--reference` (ref04–ref12),
  `--candidate` (cand-fp8decode02, cand03), `--held-out-reference` ref13,
  `--ulp-control`.
- GPU image remains pinned to `fb4f4b41` (image `sha256:e744b81d…`);
  `4b49a9c4`, `7e04679f`, `f7791189` need a rebuilt image for Q3.

**Host state 2026-10-11 ~09:30 UTC:** no containers running on either Spark.
`gate-cand-0/1` (the **candkda0** control engine, `ATLAS_KDA_FLA_REFERENCE=0`)
were stopped after archiving; do not restart them as "the candidate". Production
containers remain stopped. `gate_batch_fresh.sh` is superseded by
`gate_batch_fresh2.sh` (the old one deleted evidence on retry).

**Merged CPU-side work (2026-10-11, all GPU-unvalidated):** gate perturbation
knobs (`ATLAS_EXL3_GATE_PERTURBATION`, incl. a `round_accum` flag on
`exl3_moe_combine_f32`), first-token/grammar/n>1/MTP/Responses seed fixes,
blocking stop strings, `stop_token_ids`, `min_tokens`,
`include_stop_str_in_output`, Anthropic stop mapping, Q3 live suite
(`q3_behavior_suite.py`), Q4 corpus/runner (`tests/exl3_validation/q4/`), F2/F3
harnesses. Local checks on the merged tree: spark-server 2567 pass / 3 fail
(same 3 as before the merges, stub-linked harness in
`~/.cache/claude-q3-review/`), spark-model 1635 / 1 (known dense-cache test),
validation Python 273 pass. R2 records are on unmerged `pi/r2records-20261011`
(editing benchmark-gate boundary files invalidates all required gate records).

**First GPU step for the next session:** build a clean image from the merged
HEAD, then collect short16 + long with `ATLAS_EXL3_GATE_PERTURBATION` unset and
require `logits.npz` byte-identity with `cand-fp8decode02`. That single check
covers the kernel `round_accum=0` neutrality claim and the scheduler/seed/stop
changes before any Q3/Q4 live run; a byte difference is a finding.

[todo.md](todo.md) is the authoritative remaining-work checklist, with task IDs,
dependencies and completion criteria. This file holds the restart context,
implementation status and evidence needed to execute it. The design documents
specify arithmetic; the dated reviews contain hypotheses, not measured Atlas
speedups.

## Gate batch 2 (completed 08:47 UTC)

Read-only checks:

```bash
cat /home/colum/Workspace/atlas-exl3-validation-20261010/statistical-gate-20261011/batch2-driver.log
cat /home/colum/Workspace/atlas-exl3-validation-20261010/statistical-gate-20261011/batch2-logs/completed.txt
spark0 'docker ps; ls ~/gate ~/gate/attempts'
```

Durable local evidence root:
`/home/colum/Workspace/atlas-exl3-validation-20261010`.
Fresh create commands and drivers are under
`statistical-gate-20261011/collector-only-profile/`; finalized provenance tools
are in `/tmp/atlas-resume-20261011/provenance/` (restore from
`provenance-tools/` if missing).

## Completed since last checkpoint

| Item | Status |
|---|---|
| `cand-fp8decode02` collection | Complete: short16 16/16, long 16/16 |
| Evidence archive (both hosts) | Archived to `statistical-gate-20261011/cand-fp8decode02/` |
| Q0 final-image smoke | Passed: 5/5 requests, receipt archived |
| Fresh batch (9 refs + held-out + cand03 + candkda0) | Running since 05:29 UTC |

Read-only resume checks:

```bash
tail -n 30 /tmp/atlas-resume-20261011/cand-fp8decode02.log
spark0 'docker ps; ls ~/gate/cand-fp8decode02/long; tail -n 10 ~/gate/cand-fp8decode02/long.log'
spark0 'docker logs --tail 20 gate-cand-0'
spark1 'docker logs --tail 20 gate-cand-1'
```

Durable local evidence root:
`/home/colum/Workspace/atlas-exl3-validation-20261010`.
Fresh create commands and drivers are under
`statistical-gate-20261011/collector-only-profile/`; finalized provenance tools
are archived in its `provenance-tools/` subdirectory. The driver currently
invokes `/tmp/atlas-resume-20261011/provenance/finalize_provenance.py`; restore
that directory from the archived tools before a future run if it is missing.
The current run already passed this preflight. Its effective environment,
commands, process maps and library hashes were verified on both ranks.
Actual mappings are CUDA runtime 13.0.48 and NCCL 2.32.3, despite the image's
older NCCL package environment label.

## Commits and verification from this resume

| Commit | Change |
|---|---|
| `92735661` | Preserve failed gate attempts and both ranks' evidence |
| `72e4e1a3` | Integrate strict streaming performance diagnostics |
| `c2774c49` | Reject mixed gate environments |
| `9879d73b` / `7e0bc8d3` | Register fresh collector-only references and pin observers |
| `fb4f4b41` | Initialize/select the required FP8 MLA reader for serial TP2/EP1 |
| `d22672d1` | Register the rebuilt candidate before collection |
| `4b49a9c4` | Emit OpenAI streamed usage after the finish chunk; CPU verified only |

Current CPU evidence is in `fp8decode-candidate-20261011/cpu-checks/`: 185
validation Python tests pass; focused Rust filters pass (reader 6, FP8 85,
decode-TC 5). License headers and changed-file typos pass. Full workspace checks
are not green: the pinned Rust 1.93.1 cargo component is missing; installed
stable 1.97.1 finds formatting differences in 65 unchanged paths, existing
clippy findings and full-tree typos. Logs distinguish these failures from the
focused changes; no unrelated cleanup was made.

The separate API fix has five passing production encoder tests in a temporary
CPU harness, with an old-order red regression, exact extracted wire types,
source hashes and server test compile evidence in `q3-usage-order-20261011/`.
Full server unit-test linking requires CUDA/NCCL unavailable on the local host.
The fix has not been included in a rebuilt image or validated live.

## Resume findings (supersede the earlier collection handoff)

- With the registered image and both sparse-prefill flags, short16 and the
  2047/2048-token prefixes pass collection. The 2049-token prefix fails at
  layer 3: its one-token chunk tail enters MLA decode, whose FP8 reader was
  gated by an absent experimental flag. Both TP ranks exit. Evidence:
  `statistical-gate-20261011/resume-diag01/diagnosis.json`, both-rank logs,
  container identities and raw captures.
- `fb4f4b41` initializes and selects the existing FP8 reader for ordinary serial
  TP2/EP1 inference. Focused CPU tests pass. A clean image was built and loaded
  on both nodes from source archive SHA256
  `8b6b47f14d5b7cefa354b970f24f95619576255d34b8ced063ee30a0eec48b13`.
  Both production containers are stopped during this validation ownership.
- `cand-fp8decode02` completed short16 and the 2047/2048/2049-token requests;
  the repaired one-token tail returns a token and both ranks continue to the
  next request. Full long-suite collection is still running, with no numerical
  verdict. `cand-fp8decode01` failed preflight before any generation request
  because a library-map check falsely rejected `/dev/zero (deleted)`; its
  evidence is preserved and the check now only considers runtime libraries.
- Ref01/ref02 enabled mid-forward FFN/KDA/mHC and head-input tensor captures;
  they are diagnostic data, not qualification samples. The gate document now
  pins a collector-only reference profile before fresh collection. Start all
  nine references plus held-out afresh. No thresholds or controls were relaxed.
- `92735661` adds the tested `run_gate_attempt.sh` wrapper: reserve a new name,
  preserve both ranks' logs/inspection/raw captures and request artifacts, and
  fail if preservation fails. Use it for every subsequent external gate run.
- Both hosts' 23 checkpoint shards match the archived hashes and each other:
  `checkpoint-rehash-20261011/comparison.json`. Final-image scratch GPU checks
  pass on both ranks, including A→B→A reuse, guards and unchanged inputs;
  reports and raw bytes are in `fp8decode-candidate-20261011/scratch/`.
  Serving-process runtime maps on both ranks attest only the pinned cuBLAS pair.
- P0 harness integration is `72e4e1a3`. The saved raw diagnostic has populated
  `repeat_index` and `cold_start`; the old null-field claim does not describe
  those fields. Initial state is now explicit and failed streams are excluded.
- The current user requests **Luna subagents** and grants exclusive use of both
  Sparks, including sudo as needed. This supersedes the old Opus-only handoff.

## Current status and scope of proof

| Area | Status at review |
|---|---|
| Native checkpoint loading and two-node generation | Implemented; launch smoke recorded |
| Full-model numerical gate | Pre-registered statistical rule remains authoritative. The repaired image passed the reproduced 2049-token crash case in `cand-fp8decode02`; full collection is underway. Old ref01/ref02 are diagnostic only; all 9+1 references must be collected afresh with the pinned collector-only profile. No numerical verdict yet |
| FLA reference-policy KDA | **Implemented** (`72d64e4c`, `069e4351`): module `kda_fla`, `ATLAS_KDA_FLA_REFERENCE=1`. Byte-exact vs GPU stage oracle (all stages, carried state, both ranks) and in situ (layer-0 core, gated norm, o_proj after TP, dense FFN; layers 0–2 inputs/outputs) |
| Shared routed MoE scratch | Implemented; rank-local GPU reuse/guard proof recorded on both hosts |
| FFN stage timing | Implemented, opt-in and synchronizing; useful for attribution, unsuitable for headline timings |
| Merged KDA auxiliary projection | Implemented; layer-0 M1 padded to 3, M5 and M128 byte comparisons recorded on both hosts |
| MLA latent KV | Reference silently uses `fp8_ds_mla`; Atlas `--kv-cache-dtype fp8_g128` is the same format and is the candidate profile (needs `ATLAS_GLM_SPARSE_PREFILL_TC=1` + `ATLAS_GLM_SPARSE_PREFILL_KV_REUSE=1`, now enforced at startup by `c64b8c6e`). Layer-3 single-key rows match to 1 ulp; multi-key rows differ at implementation level |
| API behavior (Q3) | Live suite still owed. `4b49a9c4` fixes chat include_usage order with CPU red/green encoder tests; absent from the current GPU image. Remaining predicted defects: multi-token stops, streamed Anthropic stop_sequence and seeded first-token sampling |
| CUDA graphs, concurrent admission, long context, prefix reuse | Unqualified for this native EXL3 profile; qualify independently |
| DFlash/MTP/self-speculation | Explicitly rejected by native EXL3 startup guards; implementation and qualification required |
| Reproducible deployable image | Clean replacement `src-8b6b47f14d5b-rust1.97.1`, image `sha256:e744b81d…`, source `fb4f4b41`, built and loaded on both nodes. Full 23-shard rehash, final-image scratch and serving-process cuBLAS maps pass on both hosts. Final-image smoke remains owed |
| Performance baseline | No qualified baseline or trace yet. P0 harness integrated in `72e4e1a3`; explicit initial state and strict streaming completion checks. Historical raw diagnostic fields `repeat_index` and `cold_start` are populated; the prior null-field claim was unsupported |

The resumed work inspected both hosts, reproduced the failure, rebuilt the image
and started fresh GPU validation. Older evidence below is historical unless
explicitly dated as resumed; no new performance result has been established.

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

The original frozen comparison used TP=2 / EP=1, BF16 KV, eager execution,
context 4096, chunk 128, C1 admission, no prefix caching or speculation. Its
absolute-threshold reports remain diagnostic records. The user-approved
statistical gate and registered effective FP8 KV profile now decide numerical
qualification; use `docs/design/exl3-statistical-gate.md` for the current rule.
Long-prefix cases retain 2047/2048/2051/2053 tokens and registered continuations.

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
| `kda-fla-candidate-20261010/{ref-attn3-capture,mla3b-capture,fp8-mla,qdq-*,ref-mhc3-capture,mla3-capture}` | Layer-3 MLA/mHC captures on both engines: inputs/projections exact, core attention diverges; fp8_g128/QDQ reproduces single-key rows |
| `statistical-gate-20261011/` | Old ref01/ref02 diagnostic captures; preserved `resume-diag01` failure; collector-only command descriptors and full mounted-source audit for fresh collection |
| `checkpoint-full-sha/` | Earlier full shard hashes, confirmed by both-host full rehash in `checkpoint-rehash-20261011/` |
| `fp8decode-candidate-20261011/` | Replacement source archive, successful no-cache build log/manifest, focused CPU checks and final-image GPU evidence |
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

The old tag `checkpoint/stat-gate-20261011` is the starting checkpoint, not the
current host state. Both production containers are stopped during validation.

1. Read [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md), the
   [measurement discipline](.claude/skills/measurement-discipline/SKILL.md),
   [validation harness contract](tests/exl3_validation/README.md) and the
   [statistical gate](docs/design/exl3-statistical-gate.md).
2. Inspect the existing `cand-fp8decode02` process and artifacts first. Finish
   or diagnose its collection and archive both hosts' preserved attempt trees.
   Scratch, shard rehash and runtime library attestation already pass; do not
   rerun them without a new change or failed evidence check. Run final-image
   launch smoke only after collection finishes and its capture marker is gone.
   Use `run_gate_attempt.sh` for every new gate attempt; never delete evidence
   before archiving it.
3. Any further runtime change requires another clean build and preregistered
   image identity before collecting candidate samples.
4. Collect nine fresh collector-only references and a held-out tenth, two
   candidate runs (byte-identical), and candkda0 (minimum-detectable effect). Run
   `tests/exl3_validation/statistical_gate.py` with the controls; the
   semantic must-fail perturbations still need runtime knobs (not built).
5. Run the Q3 behavior suite live against the candidate; fix the predicted
   API defects (Atlas bug-first). `4b49a9c4` needs a rebuilt image and live
   usage-order verification. The seed omission is confirmed by source audit:
   first-token routes do not pass request seed into `sample_first_token` /
   `sample_token`, while later decode passes it. Add a deterministic unit
   regression and thread the seed through every first-token route. No seed
   fix was implemented; greedy numerical collection is unaffected.
6. P0: trace C1 decode with Nsight on both ranks before optimizing; the harness
   is integrated. A named `nsys launch` session followed by `nsys start` after
   model load is documented, but has not yet been exercised on Atlas.
   Prepared driver/tests/README are archived in `p0-trace-prep-20261011/`
   under the evidence root (11 tests pass). The inner Python heredoc escaping
   was corrected and tested with a mocked remote call; all actual remote
   profiler behavior remains untested. Preserve the gate run before freeing
   its port and GPU memory. Use explicit workload limits and unique outputs;
   the trace profile removes the logits observer and cannot supply headline
   performance figures.
7. Close Q0 only with final-image scratch, smoke and both-rank library maps.
   Then Q4, P1–P6, F1–F4, R1/R2 as in todo.md.

Operational: original production containers on port 8000 are preserved and
stopped. The replacement validation image is loaded on both nodes. Fresh
commands are staged in `/home/colum/gate2-profile/`; their durable copies and
all mounted reference hook sources are archived under `statistical-gate-20261011/`.
Subagents: use Luna where possible (current user instruction).

The dated [code review](CODE_REVIEW_2026-10-10.md),
[kernel review](GB10_KERNEL_OPTIMIZATION_REVIEW_2026-10-10.md) and
[prior-art corrections](GB10_PRIOR_ART_CROSSREF_2026-10-10.md) remain useful
background. Their old P1 scratch and P2 timing findings are implemented;
the head transport finding is open. Local-accumulator specialization, ordered
split-K, mHC parallelization and expert grouping are candidates whose priority
must follow an Atlas trace. Legacy NVFP4 router advice and prior-engine rates
must not be treated as native EXL3 results.

## Q3 code fixes (2026-10-11, not yet in GPU image)

Two Q3 defects fixed in source (commits `7e04679f`, `f7791189`); the GPU
image remains pinned to `fb4f4b41` and needs a rebuild + preregistration
before live Q3 validation:

1. **First-token seed threading** (`7e04679f`): `sample_first_token` and
   `sample_token` hardcoded `seed: None`, ignoring the request seed for the
   first sampled token. Added `seed: Option<u64>` parameter threaded from all
   9 callers. 4 deterministic sampler regression tests added.

2. **Anthropic streamed stop_sequence** (`f7791189`): The streaming translator
   always emitted `stop_sequence: null` + `stop_reason: "end_turn"`, even when
   a client stop matched. Added `matched_stop: Option<String>` to
   `StreamDelta::Finish`, threaded from `apply_stop_string_holdback` (now
   returns the matched string) through `StreamState` into the translator.
   Now emits `stop_reason: "stop_sequence"` + echoes the matched string.

Correction: multi-token stops were handled correctly only on the STREAMING
chat path (`apply_stop_string_holdback` with `stop_string_buffer_len`
holdback). Non-streaming chat/completions/Anthropic only sent single-token
stops to the scheduler and stripped an exact suffix, so generation ran past a
multi-token stop and returned it plus everything after; legacy completions
streaming had no stop-string handling at all. Fixed on branch
`pi/q3stop-20261011` (commits `6f25cb8c`, `c4644ca5`, `264667b9`,
`e39790e0`): shared vLLM matcher (`api/stop_match.rs`, earliest completion
wins, ties by request order), a scheduler-side `StopStringWatch` on blocking
sinks that retires the sequence at the cancellation checkpoint, single-token
stops reported as `stop_sequence` via `stop_token_id`, and Anthropic
non-streaming no longer echoes `stop_sequence` on non-Stop finishes. Not yet
validated live (no GPU image built from it).
