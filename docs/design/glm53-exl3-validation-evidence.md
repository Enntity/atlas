# GLM-5.3-Flash EXL3 — validation evidence index

Summary of the correctness evidence behind the native GLM-5.3-Flash EXL3
serving change. This is a receipt index, not a qualification claim: numerical
parity against the reference engine and the performance contract remain open
(see "Known limits").

All rows were measured on the PR tip `2e8b54db` unless noted, with the binary
built by `build-tip.sh` in the isolated builder container:

- **Binary:** `~/atlas-pr/bin/spark.pr-tip.2e8b54db`
  (`sha256 6b09909f3f279d23cfd5ce03d7595a4611f2c99c3175bc05b554d40d04ab58b0`),
  installed on both ranks from one build.
- **Image:** `atlas-exl3-validation:src-4ee96c12b44c-rust1.97.1`
  (`sha256:82173ea631a8c78e4fb565f0e23bc6d55f94a43090dbfb27c850f8cc1e6b41d9`),
  identical on both ranks.
- **Topology:** TP=2 / EP=1, two Sparks (`spark0` `10.50.0.10`,
  `spark1` `10.50.0.11`), one sequence at a time, greedy decoding.
- **Durable evidence root:** `/home/colum/Workspace/atlas-exl3-validation-20261010`
  (`$EV` below). Raw JSON/text receipts for every row live there.

## Build (PR-1)

| Check | Result | Receipt |
|---|---|---|
| PR tip builds with kernels | all gb10 targets + `glm-5.3-flash/exl3` + qwen3.8 arity pins + release server, `BUILD OK` 2026-10-17T23:02Z | `~/atlas-pr/src/logs/build-tip-20261017.log` |
| Kernel target resolution | `glm-5.3-flash-exl3` symlinks and MODEL.toml removed; `find kernels -xtype l` empty; EXL3 resolves via `[build] base_quant = "nvfp4"` | in-tree |

## Unit tests (PR-2)

| Check | Result | Receipt |
|---|---|---|
| Suites vs upstream `a9464810` baseline | no new failure vs base: the three `spark-server` reds and the one `spark-runtime` red also fail on `a9464810` | `~/atlas-pr/src/logs/overnight-tests-summary.log` |
| Rebase-ported tests present and green | `request_identity`, `idle_command`, `cmd_words`, `prefill_stream`, `stop_match`, `choice_seed`, `to_ir`, `responses`, `decode_step`, `exl3`, `glm5_next`, `native_head`, `glm_exl3_teardown`, `glm_sparse_decode`, `sample_step::seed_tests`, `small_m_gemm` | same |

## Functional correctness (PR-3)

| Check | Result | Receipt |
|---|---|---|
| Boot + launch smoke | 5/5, coherent transcripts | `$EV/pr-validation-20261017/pr3/pr3-tip-2e8b54db-20261018T055230Z/smoke.json` |
| Q3 behavior suite | **27/27** | `$EV/pr-validation-20261017/pr3/pr3-tip-2e8b54db-20261018T055230Z/q3-slice/summary.json` |
| Q4 tuning corpus | **10/10** (`INSUFFICIENT_TRIALS` at n=1; thresholds are not a PR gate) | `$EV/pr-validation-20261017/pr3/pr3-tip-2e8b54db-20261018T055230Z/q4-tuning/summary.json` |
| Greedy regression, exact stack, `ab_prose_long`, n=3 | `output_sha256 347172a63cab` — equal to the rebased reference (`b6292f15`) | `$EV/perf-ab-20261015/pr3-tip-greedy-20261018T062745Z/` |
| Prefix cache (`ATLAS_MLA_PREFIX_SKIP=1`) | cold 83.4 s TTFT → 3.24 s on an identical repeat (784 cached tokens), byte-identical greedy output | `$EV/pr-validation-20261017/pr3/pr3-prefix2-20261018T072717Z/prefix.json` |
| Grouped fat-MoE prefill on/off | identical hashes (`ab_prose_prefill 07355bd7f0ef`, `ab_prose_long 2e4f88b338a3`); prefill TPOT 0.04241 → 0.03595 s | `$EV/perf-ab-20261015/` |
| bfcl-subset (MLPerf ST leg) | **not run to completion** — see Known limits | `$EV/pr-validation-20261017/pr3/pr3-bfcl2-20261018T140944Z/PARTIAL.md` |

## Upstream non-regression (PR-4)

| Check | Result | Receipt |
|---|---|---|
| GLM-5.3-Flash **NVFP4** EP=2, PR tip vs upstream `a9464810` | byte-identical on all four workloads, 3/3 repeats each (`c091dbe6d365`, `07355bd7f0ef`, `671ef63c142a`, `25109c60637e`) | `$EV/nvfp4-nonreg/` |
| qwen4_exp EXL3 (upstream's path) | non-invasive: GLM-specific gates scoped to `glm5_next`; build owed | — |

## CI gates (PR-6)

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| clippy (pinned `rust-toolchain.toml` 1.93.1) | 0 findings on our files; upstream `+stable` (1.97) reds are pre-existing |
| SPDX / license headers | green (segment-aware `check_spdx.py`; vendored `exllamav3_*` MIT headers ignored) |
| kernel-structure | green |
| `typos` | green after allow-listing upstream's own `copys` test identifier |
| 500-line cap | inherited red — tip 16 over-cap files vs base 18; strict subset, not introduced here |

## Known limits

- **Numerics are not qualified against the reference.** The registered Q2
  statistical gate is UNVERIFIABLE on the reference side (H3/H4 leave-one-out
  failures); no candidate numerical claim is made. The greedy hash match above
  is a regression check against the rebased reference build, not parity.
- **Performance is not tuned and no target is claimed.** The 500-line cap gate
  is inherited red; measurable perf work is a separate, deferred track.
- **The MLPerf ST leg (bfcl-subset) was not completed.** The golden
  62/10/10 draw is n=995 and costs roughly 22–25 h of GPU time on the
  validated single-sequence profile; it is informational here, and the row is
  left open rather than faked with parameter overrides.
- **Scope is deliberately bounded:** F4 speculative decoding, F1–F3 serving
  features, M1/Q2 causality, and the release/regression record (R1/R2) are
  out of scope for this change.
- Auxiliary state fingerprints cover KDA/conv, not every auxiliary tensor; the
  live structural probe exercised the `</think>` close marker, while the raw thinking-reentry mask is unit-tested only.
