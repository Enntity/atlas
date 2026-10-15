# L1 reference observer under `EXL3_FUSED_MOE=0` — result (2026-10-15)

Result of the [registered diagnostic](l1-fusedoff-observer-registration.md).
Diagnostic only; no gate claim. The gate-v2 campaign stays UNVERIFIABLE.
Evidence: `$EV/l1-observer-fusedoff-20261015/`
(`$EV=/home/colum/Workspace/atlas-exl3-validation-20261010`). That includes
the receipt `prep/l1-fusedoff-f099-01/receipt.json`, `analysis/` (registered
outputs with `SHA256SUMS`), `analysis-descriptive/` (post-registration
diagnosis), `autotune/` and the pre/post host inspections.

## Verdict

**Reference observer verifiability for L1 is established.** This is
interpretation row 1 (O1 pass, O2 pass). **The registered bisect produced no
attribution:** all four runs ended with `analysis_complete=false` because of
two tooling defects. Neither defect is new to this attempt.

## Collection

`l1-fusedoff-f099-01` ran 05:19–05:48Z. All four windows completed, and the
collector exited 0, with status `collected-and-request-bound`. The binding
is `binding.json`, SHA256 `e8d62f8b0d4f…`.

Autotune cache:
- Both ranks were verified frozen and backed up before the run.
- The post-state equals the determinism post-state on both ranks
  (`e78106dc…`, `e66f8bd2…`). So control appended exactly the
  determinism-test records and observed added none.
- Both ranks are restored to the frozen state (`62e13d3d…`, `6844e72b…`).
- The host backups stay at `/home/colum/l1-fusedoff-20261015/autotune-backup/`
  as evidence.

## Registered criteria

| Criterion | Result |
|---|---|
| O1 candidate | **pass**, 8/8 rows |
| O1 reference | **pass**, 8/8 rows (was 1/8 in `f099-02` with the fused path on) |
| O2 | **pass** on both ranks. All 2,857 files were receipt-verified on each side. 1,421 files and 686 payloads per rank are byte-identical to `f099-02`. Sidecars are equal apart from the registered per-run fields, and hook-source multisets are equal (9/9, 6/6). |
| Bisect, `longer` rows 4–7, ranks 0 and 1 | `analysis_complete=false`. Incomplete pairs `boundary_streams` and `boundary_post` on every row. All rows `attribution_blocked` → **no attribution** |
| Bisect, `arithmetic` rows 0–3, ranks 0 and 1 (secondary) | same as `longer` |

Descriptive (no verdict):
- Both candidate windows are byte-identical to `f099-02` on 8/8 rows.
- Both reference windows differ from `f099-02` on 8/8 rows, as expected,
  because the loop path changes routed arithmetic.

**What the passes establish:**
- Arming the v5 capture does not perturb the reference.
- Layer-0/1 reference captures reproduce byte-for-byte across processes,
  attempts and the MoE setting.
- The comparison profile is the `f099-02` L1 diagnostic profile: the q2diag
  reference descriptors, the v5 overlay and cold `serial128`. The fused-on
  layer-0/1 captures of that profile are therefore valid reference evidence.

This closes the reference-observer blocker recorded in the
[L1 record](l1-execution-20261014.md). It does not show that the L1
diagnostic profile is identical to the gate-v2 qualification profile. That
relationship stays as the L1 spec records it.

## Why the bisect could not attribute

Both defects predate this attempt. The candidate image and capture code are
the same as in `f099-02`, so a bisect there would have failed the same way.
The refusal path in `analyze_localization.metrics` did its job.

1. **The candidate highway capture is mislabeled.**
   `crates/spark-model/src/layers/glm5_kda/forward_attention.rs:69–79`
   captures `ctx.buffers.hc_streams()` as `Bf16`, shape
   `[m, hc_mult, hidden]`. That buffer is the FP32 mHC highway
   (`crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/hc_ffn.rs:107`).
   The payload is therefore the first half of an FP32 buffer, labeled BF16:
   - read as BF16, it shows 1,096–1,188 "nonfinite" values per row, so the
     `boundary_streams` pairing refuses;
   - read as FP32, it is finite and plausible.

   The `attn_mhc_norm` highway capture has the same signature. Read as u16,
   its words alternate zero and value, which is FP32 storage of BF16-exact
   values. So the `boundary_streams_at_norm` metrics (0.04% exact, p50 15,188
   ULP) are a dtype artifact, not a divergence measurement.
2. **The bisect cannot bind the mHC `post` layout.** The reference `post` is
   `[n, 4, 1]` FP32 and the candidate's is `[n, 4]`. `kda_layout`
   (`tests/exl3_validation/analyze_localization.py:98–113`) has no rule for
   that trailing singleton, so `boundary_post` errors with
   "unbound KDA layout".

A layer-0 bisect on this data would hit defect 1 as well, because its
`boundary_streams` uses the same capture (`l1_bisect.py:28`).

## Descriptive observations (post-registration, not attribution)

- **Every pairable layer-1 stage differs on every row and rank:**
  `boundary_comb`, `attn_input` (75.6–76.6% of elements exact), `l1_core`,
  `gated_norm`, `reduced_oproj`, `ffn_mhc_input` and `oproj_input_identity`.
  The registered "no divergence with complete prefix" rule does not fire,
  because the prefix is incomplete.
- **Reread as FP32, the candidate highway does not match the reference.**
  The incoming-highway payload, rounded to BF16 (RNE assumed), equals the
  reference `mhc-fused-residual-output` on 81–82% of elements, and on no full
  token (`analysis-descriptive/boundary_dtype_diagnosis.json`). This is a
  hypothesis lead, not a measurement:
  - the dtype reread is post hoc;
  - the rounding mode is assumed;
  - it covers only tokens 0–17/19, never the terminal token the logit row
    sits on;
  - it is confounded by storage precision (next bullet).
- **Lead for the G-list, outside this registration.** The candidate's
  incoming highway carries full FP32 mantissas. The reference overlay
  records the residual output as BF16, while the same hook records
  `post`/`comb` as FP32. If the reference stream is BF16 between layers and
  the candidate's is FP32, that is a precision-path difference at every mHC
  boundary. Two checks are owed before it becomes a hypothesis:
  - the dtype of the pinned reference's `mhc_fused_post_pre_op` residual
    (as opposed to a capture-time cast);
  - what rounds the candidate highway between `incoming_highway` and
    `attn_mhc_norm`.

## Next (operator decision; nothing has run)

A registered correction is required before any attribution:
- **(a) `kda_layout` squeeze.** Bind `[n, 4, 1] → [n, 4]` for mHC `post`.
  This is CPU-only and small.
- **(b) Highway dtype, either:**
  - fix the candidate capture label/size in Rust. That needs a new image,
    neutrality proof and a fresh L1 collection, which can reuse this
    verified observer design; **or**
  - register an analysis-side reread of the existing payloads as FP32. That
    gives half-token coverage and never reaches the terminal token.
- **(c) Investigate the precision-path lead** with the two checks above.

The stop rules still hold: no rerun with changed inputs and no tolerance.
