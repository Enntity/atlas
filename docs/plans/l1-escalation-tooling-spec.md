# L1 escalation + v24-prerequisite tooling spec

Parent contract: [q2-norm-causality-plan.md](q2-norm-causality-plan.md)
§"Fallback-suspect ordering" (L1 escalation) and §"Constraints" (v24
production obligations) are the authority for semantics; this file is the
implementation contract only and never relaxes the parent. Purpose: implement every
CPU-only piece now so the next GPU windows execute runs instead of growing
code. **No GPU work happens under this spec.** Assumptions/constraints are
folded here (2026-10-12 "validation-campaign logging convention").

## Deliverables

| # | Artifact | Area | Item |
|---|---|---|---|
| D1 | Upstream capture L1 relaxation + mHC binding metadata | `crates/spark-model` | L1 |
| D2 | Reference-side L1 hook extension | evidence-root overlay | L1 |
| D3 | L1 earliest-divergence bisect driver | `tests/exl3_validation/` | L1 |
| D4 | v24-minimal engagement controls | `tests/exl3_validation/` | v24 |
| D5 | Production-compile lane for ablation variants | `tests/exl3_validation/` | v24 |

## Constraints and assumptions (B1–B12)

Append-only; mark retracted entries instead of deleting them.

### Constraints (provided)

- **B1 — Frozen comparison profile** (c2-tooling-spec A1): untouched. All
  capture knobs stay off by default and zero-I/O when off.
- **B2 — Evidence discipline** (c2-tooling-spec A2): durable root
  `/home/colum/Workspace/atlas-exl3-validation-20261010`; fresh attempt
  names/roots; never reuse or delete; `/tmp` is not evidence.
- **B3 — Metric SSOT** (c2-tooling-spec A4): row scoring from
  `statistical_gate.py::row_metrics`; tensor comparisons use
  `analyze_localization.metrics`/`bf16_key` (the localization SSOT) — no
  re-derived metrics.
- **B4 — Capture semantics are capture-only.** The L1 escalation performs
  no substitution (that is C2's `norm_substitution`, layer-0-only, A12).
  Hooks return original values unchanged ("exact-byte diagnostic hooks").
- **B5 — Fail closed (PCND).** Every env-var contract is explicit; unknown
  toolchain/flag/profile values abort rather than default.
- **B6 — Claim discipline.** A NVRTC or nvcc *replay* never claims the
  production build's qualification; the production v24 obligations in the
  parent plan (own engagement controls, production toolchain/flags,
  neighbor coverage, ablation GPU rerun) stay owed until run on GPU.

### Assumptions (inferred — confirm before relying)

- **B7 — RETRACTED 2026-10-14: standalone `hc_pre` at ordinary L1.**
  The original assumption was that L1 is a KDA layer and failing `longer` rows cross the
  L0→L1 boundary at mHC `hc_pre` (parent plan wording "before attributing
  this to L1 KDA"). The reference hook `hc_pre(x,...)` receives the
  complete multi-stream residual as `x` and returns `(post, comb,
  layer-input)`, so the streams at the boundary and the L1 attention input
  are `mhc-pre-input-x` and `mhc-pre-layer-input`
  (`vllm-reference-src/vllm/models/glm5next/nvidia/model.py:515`). Source review
  now establishes that ordinary L1 consumes deferred L0 post/comb and calls
  `hc_fused_post_pre` twice per request (attention, then FFN), not `hc_pre`.
  The corrected mapping and source hashes are recorded in D3 below.
- **B8 — Reference KDA stages at L1** are produced by
  `reference_kda_trace.patch_kda` once its `layers != {0}` guard and
  sitecustomize's `_UPSTREAM != '0'` contract widen; stage names are
  layer-keyed and identical at L0/L1 (`kda-core-output`, `kda-gated-norm`,
  `kda-o-proj`, `kda-g1`, `kda-g2`, …). (inferred)
- **B9 — Candidate upstream stages are layer-generic.** `UpstreamTrace`'s
  `layer == 0` ensure is scope-limiting, not a safety invariant; the stage
  calls in `glm5_kda/forward_attention.rs` execute at every KDA layer.
  Relaxing to the explicit L0→L1 set preserves all existing layer-0
  semantics. (assumed)
- **B10 — DeepGEMM/TileLang and split geometry are reference-side facts**
  (`vllm/model_executor/kernels/mhc/tilelang_kernels.py`;
  `hc_fused_post_pre` binds `n_splits`/`tile_n`). Atlas has no such
  dispatch; the candidate binding instead records its own mHC prenorm
  precision path (`MhcResidualPrecision::for_fused_rows`,
  `glm5_exl3_mhc.rs` — fused ≤16 rows vs materialized BF16). (inferred)
- **B11 — Production compile SSOT** is the build manifest the campaign
  already records: `atlas-exl3-build-manifest-v1` `compiler_flags`
  (`arch`, `flags`, `module_flags`) with `target_key`
  `gb10/glm-5.3-flash-exl3/exl3`; production nvcc argv parity is
  `crates/atlas-kernels/build_target.rs` (`--ptx -arch=<arch> -O3
  -std=c++17 <flags>`). (inferred from the recorded
  `q2-norm-replay-20261011/new-image-fingerprint/build-manifest.json`.)
- **B12 — Reference overlay lineage** lives in evidence roots
  (`q2-diagnostic-20261011/overlay/ref-overlay/` is the current
  installable copy, hash-manifested). D2 continues that lineage in a fresh
  directory; it does not edit the archived copy. (prior art)
- **B13 — local o_proj is not bisectable cross-engine** (independent
  review 2026-10-13). The reference `o_proj` is a `RowParallelLinear`
  (`vllm/models/glm5next/nvidia/kda.py:279,371`): its TP reduce runs inside
  `forward`, so the hook output `kda-o-proj` is post-reduce and the overlay
  has no pre-reduce capture. Capturing it needs an owed overlay extension
  (wrapping the linear's internal reduce without altering outputs); until
  then the bisection spine uses the reduced-o_proj step and records the
  limitation. C1 closed the local projection question from native captures.
  (inferred from installed source)

## D1 — candidate capture: L1 relaxation + mHC binding

`crates/spark-model/src/layers/glm5_kda/upstream.rs` (+ `forward_attention.rs`).

1. Replace the `layer == 0` ensure with an explicit supported set
   `{0, 1}` ("upstream capture supports the L0→L1 boundary"), fail-closed
   for any other layer. `ATLAS_EXL3_UPSTREAM_DUMP_LAYERS` already selects
   layers explicitly; both L0 and L1 may be armed together.
2. Extend `capture_upstream_hc`'s `attn_mhc_norm`/`ffn_mhc_norm` record to
   a `capture_with_metadata` binding: `schema: atlas-mhc-boundary-binding-v1`,
   `rows` (actual local token count `m_hc`), `hc_mult`, `sinkhorn_iters`,
   `hc_eps_bits`, `rms_eps_bits`, `residual_precision` (the
   `Glm5KdaLayer::mhc_residual_precision` selection — the exact value
   `pre_norm` receives), `sequence_parallel` (false — SP capture is refused
   at the call site), and `request_id` from `ForwardContext` when bound.
   `x`/`residual`/`post`/`comb` are the captured tensors themselves
   (`normed`, `highway`, `post`, `comb` keep their names; `incoming_highway`
   carries the boundary streams). The binding is built only after the
   capture is armed: disabled capture returns before any binding validation
   (the module invariant).
3. Existing layer-0 tests keep passing unchanged; new tests (MockGpuBackend):
   layer-1 capture writes `ffn-L1-…` records with the binding metadata;
   a layer-2 request still fails fast; knob-off remains zero-I/O.

Acceptance D1: `cargo fmt --all -- --check`;
`ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace
--tests`; `bash scripts/check-license-headers.sh`; `typos`; tests green.
GPU neutrality for the capture change is owed at image-build time (same
obligation class as the C2 request-ID field).

## D2 — reference-side hook extension (evidence root)

New `…/q2-l1-tooling-20261013/ref-overlay/` (copy of the q2-diagnostic
overlay lineage; new hash manifest; archived copy untouched).

1. `reference_kda_trace.patch_kda`: widen `layers != {0}` to the explicit
   L0→L1 escalation set (`{0}` or `{0,1}`; anything else still raises).
   Stage machinery is layer-keyed already (B8).
2. `sitecustomize.py`: widen the `_UPSTREAM != '0'` env contract to accept
   `0` and `0,1`, and pass the parsed set to `patch_kda` instead of the
   hardcoded `layers={0}`.
3. `reference_mhc_trace.py`: extend `_write_mhc_policy` bindings with the
   dispatch facts of B10: `mhc_pre_op`/`mhc_fused_post_pre_op`
   module+qualname (the DeepGEMM/TileLang choice) and `n_splits`/`tile_n`
   from the bound arguments where present. A fact the installed source
   does not expose is recorded as `"unknown"` with a reason string —
   never guessed.
4. Tests: extend `test_reference_kda_trace.py` (guard widening accepts
   {0}/{0,1}, rejects others) and `test_reference_mhc_trace.py` (dispatch
   and split-geometry bindings present, unexposed facts recorded as
   `unknown` with a reason; original outputs unchanged). Run the overlay
   test files with pytest CPU-only.

Acceptance D2: overlay tests green; `ref-overlay-hashes.txt` recorded;
"original outputs returned unchanged" preserved and re-tested. Installing
the overlay into the reference container and re-binding hook identity is
owed at GPU time.

## D3 — L1 earliest-divergence bisect driver

`tests/exl3_validation/l1_bisect.py` + `test_l1_bisect.py`.

- CLI: `--candidate <capture-dir> --reference <capture-dir>
  --candidate-artifact <collection-dir> --reference-artifact <collection-dir>
  --binding <frozen-input-binding.json> --rank --layer --rows … --out`;
  `--layer` is constrained to the qualified
  {0, 1} boundary (normally 1); capture dirs are the `ffn-capture/` sidecar
  layout both engines already write. Reuse `analyze_localization`'s sidecar
  scan, `metrics`, `bf16_key` and layout mapping (`kda_layout`) by import
  (B3); do not re-implement.
- Pair order (the parent plan's bisection), per row and rank:
  1. **boundary streams** — candidate `incoming_highway.highway` and
     `attn_mhc_norm.highway` vs reference `mhc-fused-residual-output` at
     attention-phase invocation `2r`; `attn_mhc_norm.post/comb` vs
     `mhc-fused-post/comb` at the same invocation. The incoming Atlas highway
     is already materialized after L0 hc_post; the reference fused inputs
     are deferred `(x,residual,post,comb)`, so `mhc-fused-input-x` and
     `mhc-fused-input-residual` are not its semantic counterparts;
  2. **L1 attention input** — `attn_mhc_norm.normed` vs
     `mhc-fused-layer-input` at attention invocation `2r`;
  3. **L1 core** — `recurrent_output.core` vs `kda-core-output`;
  4. **gated norm** — `gated_norm.gated` vs `kda-gated-norm`;
  5. **local o_proj** — NOT bisectable cross-engine (B13): the reference
     hook has no pre-reduce capture. `kda-o-proj-input0` (the gated input)
     is kept as a diagnostic pair against `gated_norm.gated`; a pre-reduce
     reference capture is an owed overlay extension;
  6. **reduced o_proj** — `attention_reduced_output.output` vs `kda-o-proj`;
  7. **FFN mHC input** — `ffn_mhc_norm.normed` vs `mhc-fused-layer-input`
     at FFN invocation `2r+1`. KDA stages use invocation `r`. For the supported
     L0 diagnostic, attention uses `mhc-pre-*` at `r` and FFN uses
     `mhc-fused-*` at `r` (only one fused call per request at L0).
- Output: per-row ordered stage table (exact/unequal/max_ulp per tensor),
  `earliest_divergent_stage`, explicit `missing` entries for absent
  captures, and the phase-specific reference-policy bindings beside each
  row (both attention/FFN policy records, `mhc-fused-policy` for ordinary L1).
  The report states "diagnostic localization only; no gate claim".
- A missing required stage is recorded (`incomplete_pairs`, per-pair
  `missing`/`error` entries) and fails the command AFTER the report is
  written (mirrors `analyze_localization`'s end-of-run refusal), so a GPU
  window keeps the pair table even when the bisection is partial.
- A missing/errored earlier required pair blocks attribution to a later
  divergent pair; later metrics remain evidence without an earliest-stage
  claim. Missing/mismatched mHC policy or request binding blocks attribution.
- Accept the actual Atlas `tp_rank`/`tp_world` sidecar schema and the reference
  `rank` schema, rejecting conflicting identities. Validate both phase policy
  records against candidate mHC binding (rows, streams, epsilon bits,
  iterations, request ID, pure TP and recorded residual-precision label).
  Unknown dispatch/geometry facts require nonempty reason strings.
- Require collection manifests pinned by SHA256 in
  `atlas-l1-bisect-inputs-v1`; recompute prescribed prefix SHA256 from their
  cases, validate case/step/order/request IDs and the recorded HTTP envelope
  `{"request": ..., "response": ...}` against exact prefix/model/policy.
  Candidate mHC binding carries the submitted `x-request-id`. Reference GPU
  IDs are different from HTTP response IDs: bind each row to a hash-pinned
  `head-position-NNNNNN.json` and matching raw logits sidecar/data, then require
  every compared `chunk_position.request_id` to equal that exact observed GPU ID.
  Cold terminal single-chunk prefill and the bound runner source hash must
  match for every compared reference sidecar. The report hashes the used
  sidecars/payloads and records unmatched files; it reads no checkpoint weights.

Source-bound correction (2026-10-14):
`vllm/models/glm5next/nvidia/model.py` SHA256
`a12142897fc5bdc5cff693d4d63c88a782f9ef7f35879963f45e1682d7faaed8`,
lines 445–471 and 486–497, carries deferred state into attention and then FFN.
The frozen v5 overlay's `reference_mhc_trace.py` SHA256
`aca6cd83c786027f637a4a3afeffaa194cb3748b54a269ae45a91c93eeeb603b`
increments `(layer,rank,kind)` counters once per call, producing `2r/2r+1`.
Its tensor sink `reference_ffn_trace.py` SHA256
`4774dc6932e6185fef4360619fcc9eec27a0dfd1fb5553fd5aa68d4c976b191d`
records runner context; the runner SHA256 is
`f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415`.
The installed model's `hook-source-*` receipt must match the registered model
and FFN sink hashes and layer set `[0,1]`. This does not replace the owner's
full overlay/image/source provenance checks before collection.

The explicit binding JSON has `schema: atlas-l1-bisect-inputs-v1`,
`candidate_manifest_sha256`, `reference_manifest_sha256`, and
`reference_source` with `model_sha256`, `ffn_hook_sha256`, `runner_sha256`.
It also requires `reference_head_provenance`, one entry per flattened
collection request, with `head_position_file`, `head_position_sha256`,
`raw_sidecar_file`, `raw_sidecar_sha256`, and `response_sha256`. Files are
archived absolute local paths; the raw sidecar stem and head's `head_capture` must match
the collector manifest's original capture stem. Verify one exported row,
vocabulary/metadata, raw-file and row hashes against that manifest through
`parity_protocol.read_raw`. The head runner context must match the registered
prefix length/source. These head/raw/response files are included in report hashes.

Reference GPU-ID correction (2026-10-14, after independent archived-data check):
pinned `entrypoints/openai/completion/serving.py:188` appends prompt index 0
to the HTTP response ID, then `v1/engine/input_processor.py:242` appends eight
random hexadecimal characters. Source SHA256s are respectively
`b60e42e7a23e4d03aefc40d51cd51306159d69bbee58c30a18b9ef6d540c2b8f`
and `d468edd32fd32a2d1e1c094695462f525028e097ca8c9a8e4e1c56f32c6a43a0`;
`utils/__init__.py:11` (`random_uuid`) SHA256 is
`c58b3f45deeb98bccca22c945526fd15d483a2f5eb8a3c62cf7df218b081885f`.
Accept only the full escaped response ID followed by `-0-` and exactly eight
lowercase hexadecimal characters, **and** the exact ID observed in the
hash-bound head context. Never use prefix matching or strip a suffix.
Profiles disabling randomization or batching multiple prompts require their
own reviewed mapping contract; this driver fails closed for them.

This difference is observed in archived `v2ref14` `longer/r0`: HTTP ID
`cmpl-a1d5209eb257bc58`, GPU ID `cmpl-a1d5209eb257bc58-0-b70f2acc`.
Its `capture-files/head-position-000012.json` SHA256 is
`b0bb12648ce457258683907820ac95be89a59b4f211e3b3e1695cde4f2ba4a9b`,
and matching `logits-000012.json` SHA256 is
`235a3811fe60af69aa5a03c59e4dd03c860c0aa41a516b361f50d80b25a9e3f4`
under `gate-v2-qualification-20261013-luna1/attempts/v2ref14/` in the evidence
root. The corrected helper also validates that row's actual raw data hashes.
These receipts diagnose ID transport only; they do not qualify a model.

Freeze it against the archived collection inputs before analysis; do not
silently repin modified evidence. After owner-run L1 collection of the frozen
eight arithmetic/longer requests, longer corresponds to rows `4,5,6,7`:

```bash
python3 tests/exl3_validation/l1_bisect.py \
  --candidate <fresh-attempt>/candidate/rank0/ffn-capture \
  --reference <fresh-attempt>/reference/rank0/ffn-capture \
  --candidate-artifact <fresh-attempt>/candidate/collection \
  --reference-artifact <fresh-attempt>/reference/collection \
  --binding <fresh-attempt>/l1-input-binding.json \
  --rank 0 --layer 1 --rows 4,5,6,7 --out <new-rank0-report.json>
```

Repeat with rank-1 capture roots and `--rank 1`; never put both ranks in one
capture root. The L1 experiment remains capture-only; no new candidate image
is needed for this Python driver/spec correction. Installing/rebinding the
v5 reference overlay and proving observer engagement/neutrality remain owed.

Acceptance D3: pytest green on synthetic capture trees (real sidecar
schema, hand-computed ULP checks, missing-stage refusal with the report
preserved, mixed divergence stages reported as mixed, singleton-batch KDA
layout mapping via `kda_layout`).

## D4 — v24-minimal engagement controls

`tests/exl3_validation/gated_norm_ablation.py`.

- `CONTROL_RUNS` gains `'v24-minimal'`: the input/gate/weight perturbation
  engagement controls (`native_gated_norm_replay.perturbations`) run and
  gate interpretation for v24's runs exactly as for `attested`/`v0-copy`/
  `v-all`. `interpretation_blocked` then covers v24 (build_summary already
  aggregates `controls_engaged` over `CONTROL_RUNS`).
- Rationale: the parent plan's named gap is "v24-minimal has none";
  derivation integrity (fail-closed `ExactReplace`, recorded unified diff
  and source sha) is unchanged and remains v24's feature-presence proof.

Acceptance D4: existing tests updated and green; a synthetic summary with
v24 controls not engaged is `interpretation_blocked` and fails
`enforce_summary_gate`.

## D5 — production-compile lane for ablation variants

`tests/exl3_validation/gated_norm_ablation.py` (+ small helper module if
the file nears the readability limit).

- New required CLI: `--variant-compile nvrtc|nvcc-production`, plus
  `--build-manifest <path>` required iff `nvcc-production` (B5).
- `nvrtc`: unchanged historical lane (`native_ptx.NVRTC_OPTIONS`).
- `nvcc-production`: compile each variant `.cu` to PTX with argv parity to
  `atlas-kernels/build_target.rs`: `nvcc --ptx -arch=<compiler_flags.arch>
  -O3 -std=c++17 <effective kda flags> -I <common> <source>`, where the
  flags come from the build manifest's `compiler_flags` (`module_flags.kda`
  when present, else `flags` — B11). Fail closed unless:
  - manifest schema is `atlas-exl3-build-manifest-v1`;
  - `embedded_ptx.kda.sha256` equals the `--ptx-binding` `modules/kda`
    sha256 (manifest ↔ attested binary identity);
  - the local `nvcc` reports the same compiler id as
    `compiler_flags.compiler`;
  - `arch` matches the attested PTX `.target sm_*` prefix.
  Compiled PTX is written under the immutable output dir
  (`out/variant-ptx/<name>.ptx`) and replayed via `RawModule(path=…)`.
  The argv mirrors the production strict default (`--Werror all-warnings`;
  `KERNEL_STRICT_DIAG_SUPPRESS` is empty today) and records itself in full.
- Every run record and the summary carry `variant_compile` (toolchain
  label), effective flags, compiler id, argv, manifest sha256 and per-
  variant PTX sha256, so a v24 claim can never be read across lanes.

Acceptance D5: pytest green with a fake `nvcc` (argv capture + canned PTX)
and a synthetic manifest/binding pair; negative tests for each fail-closed
condition. A real GPU replay under both lanes is owed.

## Source-bound L1 collection helper

`tests/exl3_validation/l1_prepare.py` prepares a fresh immutable plan from an
explicit JSON configuration. It requires a completed analyzed C2 campaign,
the selected candidate policy (`baseline` or `selected-treatment`), variant,
fixture, image/profile/source hashes, and the analyzer's first-control allow
field. It recomputes the completed C2 analysis; only the two scorer
`repeat_pair_scores[*].evaluated_utc` fields may differ. A fixture directory
alone does not authorize collection.

`l1_collect.py` executes four serial windows: candidate control, candidate
observed, reference control, reference observed. Each window starts fresh
containers on both hosts and preserves command receipts, failure evidence,
and teardown/retrieval results. The diagnostic marker stays absent in controls;
observed windows arm it only after both ranks pass image, executable/source,
and actual mapped classic/Lt provider checks. Reference uses the complete
hash-bound v5 overlay, layers `0,1`, and cold `serial128`. The separate
`l1_reference_attest.py` identifies one actual mapped GPU worker by registered
process title, Docker process inventory, ancestry, and cgroup; its receipt
binds the full container ID, PID, command line, maps, libraries, and pins.

Controls must match all eight actual logits rows, generated text, thresholds,
and numerical/source profiles. The earlier full32 logits-only proof does not
discharge these L1 observer controls. A mismatch stops binding/interpretation;
there is no implicit tolerance or selective retry policy.

Shutdown is explicitly configured. Candidate containers and the reference
leader require exit zero. A registered headless reference worker may instead
be labelled `controlled-timeout-stop137`, distinct from `clean-stop0`, only
after exact8 durable raw/head/request/source validation and both-rank
attestation. Its exact owned CID must be running and non-OOM before a
successful explicit timeout stop, stopped and non-OOM afterwards, with wait
137 and bounded container events proving SIGTERM15 then SIGKILL9 after at
least the registered timeout, then stop/die137. Missing events, spontaneous
death, OOM, incomplete collection, or any other nonzero exit fail the attempt.
The original attempt01 remains failed; fresh registration binds its receipts.

The concrete offline configuration and audited inputs are under
`q2-c2-worker-binding-20261014/l1-reference-exact-shutdown-config.json` and
`l1-reference-exact-shutdown-registration.json` in the external evidence root.
After reviewing those inputs, the explicit CLI contract is:

```bash
REP=/home/colum/Workspace/atlas-exl3-validation-20261010/q2-c2-worker-binding-20261014
python3 tests/exl3_validation/l1_prepare.py --config "$REP/l1-reference-exact-shutdown-config.json"
L1_PLAN="$REP/l1-reference-exact-shutdown-v2/plan.json"
sha256sum "$L1_PLAN"
# L1_PLAN_SHA256 must contain the reviewed, prepared plan's exact SHA256.
python3 tests/exl3_validation/l1_collect.py --plan "$L1_PLAN" --plan-sha256 "${L1_PLAN_SHA256:?reviewed plan SHA256 required}"
```

Collection artifacts live at each window's `collection/` and rank-0 head
witnesses at `logits/`. Diagnostic and substitution records remain separated
under `rank0/diagnostic`, `rank1/diagnostic`, and each rank's `records/`.
`binding.json` binds exact observed opaque reference GPU IDs through the
head-position, raw sidecar/payload, and HTTP response hashes; it never infers
IDs by removing suffixes. The collection receipt leaves attribution disabled
until the unchanged bisect driver verifies complete stage and policy bindings:

```bash
L1_ATTEMPT="$REP/l1-reference-exact-shutdown-v2/l1-refexact-f099-02"
python3 tests/exl3_validation/l1_bisect.py \
  --candidate "$L1_ATTEMPT/candidate-observed/rank0/diagnostic" \
  --reference "$L1_ATTEMPT/reference-observed/rank0/diagnostic" \
  --candidate-artifact "$L1_ATTEMPT/candidate-observed/collection" \
  --reference-artifact "$L1_ATTEMPT/reference-observed/collection" \
  --binding "$L1_ATTEMPT/binding.json" --rank 0 --layer 1 --rows 4,5,6,7 \
  --out "$REP/l1-reference-exact-longer-rank0-layer1.json"
```

Repeat with rank-1 diagnostic roots, `--rank 1`, and a fresh output file.
The longer rows use attention invocation `2r` and FFN invocation `2r+1`;
local pre-reduce projection still lacks a cross-engine reference counterpart.
These commands document the registered workflow; collection remains separate
from a qualification verdict.

## Out of scope (record, do not build)

- All GPU execution: the ablation rerun, L1 capture collection, image
  builds and neutrality proofs. They remain owed items.
- G1 (layers 4–44 bisection): D1 deliberately permits only `{0,1}`;
  widening is its own reviewed change.
- The production v24 fix itself and any neighbor-coverage matrix work.
- C2 substitution changes.

## References (external to this repo)

- Evidence root: `/home/colum/Workspace/atlas-exl3-validation-20261010/`
  (`q2-norm-ablation-20261012/`, `q2-norm-replay-20261011/` incl.
  `new-image-fingerprint/build-manifest.json` and `replay-ptx/binding.json`,
  `q2-diagnostic-20261011/overlay/ref-overlay/`,
  `reference-upstream-context-v4/` capture layouts).
- Pinned reference source: `/home/colum/Workspace/vllm-reference-src/`
  (`vllm/models/glm5next/nvidia/model.py` `hc_pre`/`hc_fused_post_pre`;
  `vllm/model_executor/kernels/mhc/tilelang_kernels.py`).
- In-repo prior art: `docs/plans/c2-tooling-spec.md` (A1–A13, knob
  obligations), `crates/atlas-kernels/build_target.rs` (nvcc argv),
  `crates/atlas-kernels/build_module_flags.rs` (flag merge semantics),
  `crates/atlas-kernels/tests/module_compile_flags.rs` (override tests),
  `tests/exl3_validation/{analyze_localization,native_ptx,gated_norm_ablation,
  gated_norm_variants}.py`.
