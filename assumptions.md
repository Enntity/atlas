# Assumption log

Assumptions, constraints, and requirements inferred or provided for the
project. Reference these from source documentation when explaining why
something is done a specific way. Keep entries short; newest section first.

## GPU-iteration tooling (2026-10-14)

References: `decisions.md` 2026-10-14 GPU-iteration entry,
`docs/design/exl3-runtime-mechanism-probe-registration.md` (owed section),
`docs/plans/exl3-reference-runtime-mechanism-plan.md`.

Assumptions (inferred; disconfirmed by live evidence is a finding):

- T1. The registered diagnostic profile prepares at most one unpadded
  serial request per `prepare_inputs` (batch-1, no speculation); the probe
  fails closed on concurrency and treats an empty prepare as a request
  boundary (flush + unmark).
- T2. MoE layer identity comes from the `layers.<n>.mlp` construction
  prefix (the registered FFN overlay precedent). Instances with other
  prefixes stay uninstrumented; per-request engagement counters must catch
  a total miss rather than construction failing on auxiliary instances.
- T3. `ATLAS_PROBE_CONFIG.source_targets` entries are
  `module:attribute` references: the frozen harnesses bind
  `__module__`/`__qualname__`, so bare modules cannot bind.
- T4. nsys availability/version on either Spark is unknown until a host
  check; the registered invocation flags and their SHA256 stand
  regardless of availability.
- T5. The prior loaded-NCCL evidence (spark0/spark1 loaded-sha files and
  the reference runtime-evidence JSON) reflects the reference image the
  probe will run; a live re-bind from running processes is owed
  pre-attempt and is stated as a limitation inside the binding.
- T6. `q2-l1-tooling-20261013/ref-overlay` is the lineage SSOT for the
  probe package (v5 binding recorded in its `package-binding.json`).
- T7. The image-cycle driver authors the neutrality preparation config
  from verified fingerprint outputs plus operator campaign inputs;
  `image_id`, `runtime_source` and the build-manifest path cannot be
  known before the build completes.

## Runtime-mechanism probe harness (2026-10-14)

Reference: `docs/design/exl3-runtime-mechanism-probe-registration.md`
(registered) and `docs/plans/exl3-reference-runtime-mechanism-plan.md`
(design SSOT).

Assumptions (inferred; disconfirmed by live evidence is a finding):

- R1. `context()` is side-effect-free and bound per router/communicator
  instance; the bindings cannot cross-check the layer index against the
  resident object, so a mislabeled closure silently misattributes records
  (overlay-author obligation).
- R2. `capture_callback_calls` counts unmarked calls; engagement must be
  judged from per-request counters, or an inert hook could pass.
- R3. Logical IDs equal physical IDs whenever the recorded EPLB status is
  'disabled'/'absent'; the `apply_exl3_experts` boundary stays unwrapped
  unless a live run records EPLB enabled.
- R4. One-forward-per-process TP execution; the collective frame stack is
  not thread-safe.
- R5. Recording every collective dispatch is acceptable only with
  buffered overlay writers (flush boundary defined by the overlay).

## KV column-codec experiment (2026-10-13)

Reference: `docs/plans/atlas-kv-column-codecs-plan.md`.

Assumptions (inferred — to be confirmed in Stage 0):

- A1. KV streams have exploitable regularity along the token axis
  (neighbouring positions correlated). **Unproven — this is the central
  empirical question of Stage 0.** Do not build integration stages on it
  before Stage 0 numbers exist.
- A2. Adjacent token positions of a fixed `(side, kv_head, head_dim)`
  channel form compressible "columns" regardless of exact paged axis order.
- A3. Post-quant streams (codebook indices, group scales) are highly
  compressible by integer codecs (T64/Delta) due to tiny value ranges.
- A4. Codec decode on Grace CPU cores is cheap relative to spill/reload
  bandwidth; decode in GPU kernels (Stage 3) is cheap relative to HBM
  bandwidth saved (requires inline roofline arithmetic before coding).

Constraints (provided):

- C-a. Deliverable is the plan only; implementation follows the staged
  gates in the plan (§5). Each stage is PR-sized and gated.
- C-b. Codec layer is strictly lossless; bit-exact roundtrip is mandatory.
  All lossiness stays in the existing `KvCacheDtype` quantizers.
- C-c. GPU-resident stage (3) is contraindicated vs the TQ+ kernel line;
  it is staged last, restricted to bit-oriented codecs, and killable.
- C-d. Success gate: end-to-end decode tok/s under spill pressure holds or
  improves; all perf claims follow `.agents/skills/measurement-discipline`.
- C-e. New deps (`zstd`, `lz4_flex`) must pass `deny.toml` license policy
  before Stage 0 lands.

## Project-level (from AGENTS.md — stable)

- Target hardware is NVIDIA GB10 / DGX Spark; `gb10` is the only
  implemented kernel hardware target.
- License is AGPL-3.0-only; every source file carries the SPDX header;
  `deny.toml` gates dependency licenses.
- A real build/test cycle requires a CUDA-capable host;
  `ATLAS_SKIP_BUILD=1` gates are for fmt/clippy on the host only.
- Rust source files ≤500 LoC (CI-enforced file-size cap).
- Model×quant support matrix SSOT: `docs/GB10_DEPLOYMENT_GUIDE.md` §2.

## 2026-10-13 — KV column-codec first pass

- Codec layer is lossless and composes strictly after existing lossy `KvCacheDtype` quantizers (§2 C2); bit-exact roundtrip is mandatory regardless of ratio outcome.
- `zstd` and `lz4_flex` licenses (BSD-family) are compatible with `deny.toml` (C3); verified by allow-list inspection (MIT/BSD/Apache permitted); no FFI to system libs.
- Block header carries explicit `codec_id` (PCND — no implicit default); `codec=NONE` is the safe default so no regression on unconfigured tiers.
- Captured corpus source is `data/corpus/` binary dumps from `paged_impl.rs` blocks (synthetic + captured); live serve integration deferred until Stage-1 exit criteria pass (§5).
- File-size cap ≤500 LoC/file enforced by `.github/workflows/file-size-cap.yml`; crate split into lib/codec/block/bench modules accordingly.

- External repo evaluation (13 repos, §7 of plan) is planned, not executed; each must supply explicit codec-id / pipeline declaration (PCND); no repo's default pipeline will be adopted implicitly.
Assumption added to assumptions.md:
- Prefill divergence above 2051 tokens is indexer-order (not arithmetic) per exl3-indexer-localization-result-20261015.md (216 rows, 38 differ, 0 different set). Fix = sort before expansion, not kernel replace.
- FlashKDA adapter (research/flash-kda-prefill) is candidate only, not default; requires ATLAS_KDA_FLASH_PREFILL=1 + lib path. No dependency added.

## 2026-10-16 campaign assumptions (parent session)
- Operator grants: F4 lane ownership + Spark/GPU control for the campaign
  ("stop and leave down"); builder2 + `~/atlas-dev2` are in use (campaign
  binary backed up as `spark.main-172ce45a`). The port agent's builder3,
  routed-MoE kernel lane and the paused Spark cleanup are still NOT ours.
- All campaign numbers are single-harness (`perf_baseline` TPOT), n=3/arm,
  same session, one variable per A/B → "observed under fingerprint F"
  claims (run dirs hold `fingerprint.json`/`meta.json`); no cross-harness
  validation yet.
- F4 arms carry `COOP_MOE_MULTIROW` in both serial and spec arms (inert at
  1 serial row) so the spec lane is the only variable.
- DFLASH flags are the diagnostic A/B class; every new surface
  (`_VERIFY_GRAPH`, `_KDA_PER_ROW`, `GDN_WY3_REWRITE`,
  `EXL3_BF16_VOCAB_GATHER`, `MLA_GEMV_WMMA`) is default-off pending its
  GPU proof.
