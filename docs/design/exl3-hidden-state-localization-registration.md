# EXL3 reference hidden-state localization — registration

Registered 2026-10-15. This is a reference-only **diagnostic**: the next
mechanism step recommended by the
[runtime-mechanism probe result](exl3-runtime-mechanism-probe-result-20261015.md).
It is not a gate amendment and changes no H3/H4 constant, component count,
fixed campaign N or reference identity. No sample from it is ever a
qualification draw. The fixed v2 campaign stays UNVERIFIABLE.

## Question

Between two fresh reference processes, where in the decoder does the first
byte difference enter a request's computation? The probe established that
router gate inputs diverge first, at a row-dependent layer between 8 and 33,
with expert flips following. Localizing the first differing tensor
separates three things: a rank-local expert computation (EXL3 kernels), the
tensor-parallel reduce, and attention (whose internals are not captured
here).

## Attempt list (frozen)

Fixed order, fresh processes. Every attempt is reported. No attempt is
added, replaced or retried unchanged, and a failed attempt is retained.

    hsl01, hsl02, hsl03

The three draws give three pairs. Rows that are byte-identical within a pair
give no localization (the probe saw this for `code` r0–r2), which is why
there are three draws rather than two.

## Profile (frozen)

- Exactly the gate-v2 campaign reference create commands
  (`gate-v2-qualification-20261013-luna1/driver/profile/ref-rank{0,1}-*`),
  including the **campaign observer** at `/opt/atlas-logits-patch`
  (context-v4, sitecustomize `81771af5…`, unchanged; not the probe overlay).
- Diagnostic delta, the same in every attempt: three environment
  variables, all handled by the campaign observer's existing exact-byte hooks
  (`reference_ffn_trace`, `reference_mhc_trace`):
  `ATLAS_EXL3_FFN_DUMP_DIR=/tmp/ffn-capture` (the campaign's existing mount
  slot, attempt-scoped), and `ATLAS_EXL3_FFN_DUMP_LAYERS` and
  `ATLAS_EXL3_MHC_DUMP_LAYERS` set to `0,…,44` (every decoder layer, no
  MTP). The hooks return original outputs, but they add device-to-host
  copies and synchronous file writes. Dumped draws are therefore compared
  **with each other**. Comparison against the campaign envelope is
  descriptive only: an earlier hooked-versus-unhooked diagnosis found
  hooked draws up to KL 0.048 on `arithmetic`.
- Dumps are enabled the way the proven q2-diagnostic run did it. A
  `capture-enabled` marker is created in each rank's dump directory after
  the API is ready and before collection, because the collector marks only
  rank 0's logits directory.
- Collection: **short16 only**, in the campaign collector order. Short16
  runs first in the campaign order, so execution is identical up to that
  point, and the long suite's dump volume is avoided.
- Caches: each attempt gets an isolated copy of the probe's frozen per-rank
  snapshot (`probe-runtime-mechanism-20261014/cache-freeze/`). The copy must
  hash equal to the snapshot before start. It is dropped after its
  post-shutdown manifest is recorded, since the snapshot reproduces it.
- Fixed mount sources are frozen by type and hash in `mount-freeze/` before
  `hsl01`; every attempt must match.
- Runner `tests/exl3_validation/hsl_runner.py` (the probe runner's
  lifecycle: preflight before the point of no return, guarded teardown and
  archive, order always advances). Provenance comes from the gate-v2
  finalizer variant `driver/finalize-hsl-reference.py` (byte-identical to
  the probe's), with the campaign draft plus the three variables.
- Evidence root: `hidden-state-localization-20261015/`. Dumps above 6 GiB per
  rank stay on the Spark at `/home/colum/hsl-20261015/attempts/<a>/dump`, with
  a full SHA256 list archived locally.

## Analysis (frozen before collection)

`tests/exl3_validation/hsl_localize.py`, for each pair and each
(request, rank):
- Join every tensor to its request through the embedded `chunk_position`
  request ID, never by hook counters.
- Order tensors in decoder dataflow order:
  1. the attention-side mHC call (inputs, then outputs; `layer-input` is
     the attention input);
  2. attention (internals not captured);
  3. the FFN-side mHC call (input `x` is the attention output after
     reduce);
  4. FFN input, router logits, routes, the rank-local shared/routed outputs
     before reduce, then the post-reduce output.
- Report the first tensor whose bytes differ, and whether it is rank-local.
- Self-checks, which fail the analysis rather than report:
  - the FFN `input` equals the FFN-side mHC `layer-input` bytes;
  - both runs carry the same tensor keys;
  - router logits never differ before their FFN input.

  The analyzer was validated on the earlier q2-diagnostic context-v4 dumps
  (self-comparison identical; join check holds on real data).

Interpretation, registered now:
- First difference at a **rank-local** FFN stage on one rank, with an
  identical FFN input: that rank's expert computation varies run to run.
- First difference at `post-tp-output` with identical rank-local stages on
  both ranks: the reduce varies.
- First difference at the FFN-side mHC `input-x` (attention output) with an
  identical attention input: attention, or its reduce, varies. Its internals
  need a separately registered attention trace.
- First difference inside an mHC call with identical inputs: the mHC
  kernels vary.

Descriptive H3 against the 15 campaign references is reported for context
only. Byte classes are not compared with the campaign, because the observer
configuration differs.

## Execution record

- `hsl01` **complete** (2026-10-15 01:05–01:3xZ): both ranks exit 0, short16
  collected, 49,847 dump files and 5.1 GiB per rank, copied locally.
  Analyzer self-comparison on each Spark: 16 requests × 1,469 tensors each,
  join self-check holds on every layer, identical.
- Storage change for `hsl02`/`hsl03` (capture unchanged): local disk cannot
  hold three attempts' dumps, so the runner keeps them on the Sparks
  (`LOCAL_COPY_LIMIT = 0`), with full SHA256 lists archived locally. Pairwise
  analysis runs on each Spark, where all three attempts' dumps for that
  rank sit side by side.
- `hsl02`, `hsl03` **complete** (both ranks exit 0). All six pairwise
  localizations passed their self-checks. **Result:** the
  [result](exl3-hidden-state-localization-result-20261015.md). The first
  differing tensor is the EXL3 fused-MoE FP32 accumulation in all 96 cases.

