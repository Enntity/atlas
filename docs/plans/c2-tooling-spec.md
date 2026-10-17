# C2 tooling spec — norm-output substitution knob, fixtures, A/B driver, scoring

Parent contract: [q2-norm-causality-plan.md](q2-norm-causality-plan.md)
§"Experiment C2" is the authority for semantics, scoring rules and controls.
This file is the implementation contract only; it never relaxes the parent.
Decisions: operator decisions of 2026-10-12 (C2 batch). Assumptions and
constraints (A1–A13) live in the section below — folded here per the
2026-10-12 "validation-campaign logging convention" decision (no separate
`assumptions.md`). No GPU work happens under this spec (hosts are the
constraint); everything here is CPU-verifiable.

## Deliverables

| # | Artifact | Area |
|---|---|---|
| D1 | Rust substitution knob + request-ID binding | `crates/` |
| D2 | Fixture assembly | `tests/exl3_validation/` |
| D3 | A/B run driver + scoring | `tests/exl3_validation/` |
| D4 | `scripts/inspect-sparks.sh` host inspection | `scripts/` (not delegated) |

## Constraints and assumptions (A1–A13)

IDs are stable and cited by the tooling. Append-only: mark retracted
entries as such instead of deleting them.

### Constraints (provided)

- **A1 — Frozen comparison profile.** TP=2/EP=1, BF16 KV, eager, context
  4096, chunk 128, one-sequence admission, no prefix caching or
  speculation. CPU-only diagnostic tooling must not touch
  it.
- **A2 — Evidence discipline.** Durable root
  `/home/colum/Workspace/atlas-exl3-validation-20261010`; fresh attempt
  names and capture roots; never reuse or delete failed attempts; `/tmp`
  is not durable evidence.
- **A3 — One experiment owner** across both Sparks at a time; re-inspect
  hosts before any GPU work.
- **A4 — Metric SSOT.** Scoring metrics come from
  `tests/exl3_validation/statistical_gate.py::row_metrics`; no re-derived
  metrics in C2 tooling.
- **A5 — Scoring authority.** The plan's "Diagnostic scoring fixed before
  execution" governs; ref04–ref12 ensemble members and frozen limits are
  supplied inputs, not choices the tooling may make.
- **A6 — v24 provenance.** `v24-corrected` and `reference-exact` fixture
  sets stay separate even when byte-identical (plan §Experiment C2).

### Assumptions (inferred/assumed — confirm before relying)

- **A7 — "The eight captured requests"** are the C2 row set captured on
  both engines (rows 0–7 per
  `q2-c1-prep-20261012/binding-status.md`), i.e. the requests whose
  registered failing rows are `longer` 0–2, `arithmetic` and the ref13
  row-0 fingerprint. Confirm at fixture-assembly time; the plan names the
  rows, not the request list. Includes the D2 invocation semantics:
  fixture invocation numbering equals the engine's layer-0 forward-pass
  order. (assumed)
- **A8 — request_id** is the server `x-request-id` value
  (`main_modules/middleware.rs`; client-supplied or generated UUID v4),
  surfaced as `ctx.id`. (inferred)
- **A9 — Single-sequence path.** Under A1 the layer-0 gated norm engages
  `forward_attention.rs`; the `multi_seq.rs` call site is out of scope and
  errors when the substitution knob is engaged. (assumed)
- **A10 — Fixture layout.** `manifest.json` at a fixed name inside an
  explicitly provided directory is new contract (no prior art); see the
  D1 fixture-manifest section. (assumed)
- **A11 — Graph interlock.** Substitution requires the three graph-disable
  flags (same rule as `Exl3FfnRouter::from_env_named`) so fixture bytes are
  never captured into a CUDA graph. (inferred)
- **A12 — Layer-0 only.** The initial intervention is layer 0 on both
  ranks, all token/head elements; all-layer correction is a separate
  experiment (plan §Experiment C2). (provided)
- **A13 — Host inspection lives in `scripts/`** (operator decision
  2026-10-12), overriding the root AGENTS.md "Justfile" convention for this
  repo, which has no Justfile. Revisit if the repo adopts `just`.
  (provided)

## D1 — Rust substitution knob

Target layout (split further if any file nears the 500 LoC cap):

- `crates/spark-runtime/src/weights/exl3/norm_substitution.rs` (+ tests):
  the diagnostic I/O boundary (SBIO): manifest parse/validate, payload
  staging, lookup, hashing. Sibling of `ffn_io.rs` (`Exl3FfnRouter`).
- `crates/spark-model/src/layers/glm5_kda/`: engagement at the layer-0
  `ops::kda_sigmoid_gated_norm` output (`gated`) in `forward_attention.rs`
  (before `project_hot`). Records go through `UpstreamTrace` /
  `Exl3FfnRouter` (`ffn_io.rs`), the atomic-record path from `651ff083`.
- `crates/spark-model/src/layer.rs`: `ForwardContext` request binding plus
  every construction site, including derived contexts (`glm5_mtp.rs`,
  `deepseek_v4_mtp.rs`, `model/impl_b*.rs`, `model/trait_impl/*`, verify
  paths).
- `crates/spark-server/`: populate the binding at admission from `ctx.id`
  (`x-request-id`, `main_modules/middleware.rs`).

Lifecycle:

1. **Off by default.** `ATLAS_EXL3_NORM_SUBSTITUTE_DIR` unset/empty ⇒ knob
   off for the process: zero file I/O and zero device copies afterwards.
   Set ⇒ parse `$DIR/manifest.json` once per model load (pattern:
   `crates/spark-model/src/exl3_gate_perturbation.rs`), fail fast on any
   validation error.
2. **Records directory (PCND).** When the knob is engaged,
   `ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR` is a second **required** env
   var; unset/empty ⇒ abort at model load. It names an existing directory
   (same rule as `Exl3FfnRouter::new`) that receives the substitution
   records; the D3 driver creates a fresh per-attempt directory and sets
   the variable per run (A2: fresh capture roots; repeats never share a
   records dir, so record-name collisions are impossible by
   construction). The fixture directory stays immutable: no writes of any
   kind into `ATLAS_EXL3_NORM_SUBSTITUTE_DIR`.
3. **Graph interlock.** Require `ATLAS_NO_DECODE_GRAPHS=1`,
   `ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1`, `ATLAS_DEBUG_NO_GRAPH=1` (same rule
   as `Exl3FfnRouter::from_env_named`); fixture bytes must never be baked
   into a CUDA graph.
4. **Engage** per forward invocation at layer 0 on each rank: substitute the
   full `gated` tensor when an entry matches; an engaged process with no
   entry for the current invocation errors the run.

Fixture manifest (`$DIR/manifest.json`):

```json
{
  "schema": "atlas-norm-substitution-fixtures-v1",
  "variant": "reference-exact",
  "source": {"capture_root": "...", "engine": "...", "schedule_sha256": "..."},
  "entries": [
    {"request_id": "...", "layer": 0, "tp_rank": 0, "tp_world": 2,
     "invocation": 3, "stage": "gated_norm", "tensor": "gated",
     "dtype": "bf16-le", "shape": [tokens, p],
     "file": "L0-rank0-000003-gated.bin", "sha256": "..."}
  ]
}
```

- `variant` ∈ {`reference-exact`, `v24-corrected`}; recorded in every
  substitution record.
- Entries unique on (layer, tp_rank, invocation, tensor); `layer` must be 0.

Verification contract (PCND — fail fast, no defaults):

| Condition | Behavior |
|---|---|
| env unset | no-op |
| manifest missing / bad schema / bad sha / size mismatch / duplicate entry / `layer != 0` | abort at model load |
| engaged invocation, no entry | run error |
| `request_id` ≠ `ForwardContext` binding | run error |
| shape/dtype ≠ live tensor | run error |
| entry valid | substitute + record |
| entry never engaged | post-hoc receipt gap (driver reports), not in-engine |

Record (via `Exl3FfnRouter::capture_with_metadata` rooted at
`ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR`, atomic commit): stage
`norm_substitution`, metadata schema
`atlas-norm-substitution-v1`: fixture file + sha256, variant, request_id,
invocation, shape, `payload_stage: "post-substitution"`, and the
**pre-substitution sha256 of the candidate
`gated` bytes** (hash semantics decision: integrity + recorded pre-hash).
Order: kernel → D2H pre-image → sha256 → H2D fixture → existing
`gated_norm` capture (post-substitution) → substitution record. The record
payload is the post-substitution `gated` bytes (what the engine actually
consumed downstream); run A/B alignment is checked post-hoc by comparing
the recorded pre-hashes.

Request-ID binding: every forward pass of a request carries the server
request id (`ctx.id`) in `ForwardContext`, propagated through derived
contexts; the knob requires it when engaged. Neutral when off; the field
addition is the only production-path edit and gets its own GPU neutrality
proof at qualification time.

Multi-sequence: the frozen profile admits one sequence. If the multi-seq
path (`multi_seq.rs:284`) runs while the knob is engaged ⇒ run error (do
not substitute there).

Tests (TDD, CPU, `MockGpuBackend`): knob-off zero-I/O; load negatives (bad
sha, bad shape, duplicate, `layer != 0`, missing manifest); missing-entry
error; request mismatch error; happy path copies fixture bytes and records
the pre-hash; multi-seq fail-fast; request-id propagation.

Acceptance D1: `cargo fmt --all -- --check`;
`ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace
--tests`; `bash scripts/check-license-headers.sh`; `typos`; new and
existing tests pass. The GPU neutrality proof (knob off ⇒ final-logit
archive byte-identical to the frozen baseline) is owed at GPU time — record
it as owed, never claim it.

## D2 — fixture assembly

`tests/exl3_validation/make_norm_fixtures.py` + `test_make_norm_fixtures.py`.

- Input: reference-side layer-0 `gated_norm` captures (upstream reference
  package dump layout, e.g. `$EV/reference-upstream-context-v4/`), a run
  schedule JSON (request order → request_id), `--variant`.
- Output: fixture directory per the manifest schema above; immutable
  (`mkdir(exist_ok=False)` convention from the ablation runner).
- Validation discipline copied from
  `gated_norm_ablation.validate_fixture` (fail-closed identity checks,
  sha256 records, dtype/shape/finiteness); import/reuse it where the seam
  fits — never re-implement its checks.
- request_id ↔ invocation mapping comes from the schedule plus serial
  request execution; any mismatch ⇒ refuse to emit.
- **Invocation semantics (review finding m5):** fixture invocation numbers
  are the engine's layer-0 forward-pass order (`norm_substitution.rs`
  per-process counter). The replay/v24 lane currently expresses exactly
  one layer-0 invocation per request (`order == invocation`); a request
  with multiple layer-0 passes (e.g. chunked prefill) must fail closed at
  assembly, not mis-number. The upstream-dump lane binds multi-invocation
  captures contiguously per request. Assembly must confirm at emit time
  (A7) that the schedule's invocation numbering equals the engine's
  layer-0 pass order.
- Both variants: `reference-exact` from reference captures;
  `v24-corrected` from the ablation replay
  (`q2-norm-ablation-20261012/`). Byte-identity of the two on proven
  fixtures is reported as a fact, never collapsed (separate provenance).

## D3 — A/B driver + scoring

`tests/exl3_validation/c2_ab_driver.py`, `tests/exl3_validation/c2_score.py`
+ tests.

- Driver: run A = knob off; run B = knob on (`--variant`), both ranks,
  identical serve flags and requests; **fresh attempt names** (refuse
  existing paths; never reuse or delete); ≥2 repeats per arm to confirm
  byte-determinism (any variation is reported and causality stays
  unresolved); full fingerprints (image, source snapshot, flags) in
  receipts; every attempt preserved. Per run it creates a fresh records
  directory under the attempt root and sets
  `ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR` (run A leaves the substitution
  env unset entirely); it locates run-B receipts there and reports
  unconsumed manifest entries as a post-hoc gap. It wraps the existing
  collector
  (`collect_logits.py` / the collector-only command descriptors in
  `statistical-gate-20261011/collector-only-profile/`); it does not
  re-implement collection.
- Scorer: exactly the plan's fixed rules — `statistical_gate.row_metrics`
  (SSOT) against **every** member of the frozen ref04–ref12 ensemble; per
  row: A and B mean symmetric KL, nearest-reference symmetric KL,
  continuation CE error where defined, top-10 miss; primary row delta =
  mean symmetric KL(B, refs) − mean symmetric KL(A, refs) (negative =
  closer); raw-logit byte equality reported separately; per-row
  catastrophic checks against the frozen limits; mixed row signs reported
  as mixed; no favorable reference/row/vocab subset selection anywhere;
  the five-token fingerprint is explanatory only; the report never claims
  a Q2 gate pass.

Acceptance D3: pytest green including hand-computed metric checks;
driver refusal tests (existing attempt dir, repeat count, mixed
configurations); scorer mixed-sign reporting test.

## Out of scope (record, do not build)

- GPU execution of C2 (fresh image + neutrality proof + collection happen
  when the Sparks are free, per the plan).
- Gate v2 sign-off (operator decision) and the ref13 mode probe (separate
  batch item).
