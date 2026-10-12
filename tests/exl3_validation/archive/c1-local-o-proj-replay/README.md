# ARCHIVED — C1 local o_proj BF16 common-space replay (do not use)

Archived 2026-10-12 by operator decision (C1 re-scoped to "close from native
captures", see `decisions.md`). Full pre-removal tree: git tag
`archive/c1-bf16-o-proj-replay-20261012` (at `4e00b222`).

## Contents

| File | Was |
|---|---|
| `local_o_proj_capture.py`, `test_local_o_proj_capture.py` | C1 capture-record validator and binding-row assembler (`e5e18056`, `6568110b`) |
| `local_o_proj_replay.py`, `test_local_o_proj_replay.py` | f32-GEMM/RNE replay over bound BF16 weights with self-anchor gate (`5412f043`) |
| `local_o_proj_engine_native.py` | Unintegrated WIP "engine-native" backend (`bbd2d267`) |
| `rust/o_proj_binding.rs`, `rust/o_proj_receipt.rs` | Removed spark-model load-time BF16 weight capture and GEMM receipt (`b85623c0`; the forward/loader hooks and `Exl3FfnRouter::write_binding_with_receipt` were reverted) |

The Python modules raise `ImportError` unless `ATLAS_ALLOW_ARCHIVED_C1=1`
(forensic reading only); `conftest.py` stops pytest collecting them.

## Why the premise is invalid

- The reference `Glm53PackedLinearMethod.apply`
  (`$EV/arithmetic-audit/atlas-reference-active-exl3_mul1.py:3087-3110`)
  uses `F.linear(x, layer.weight)` only for `rows > 144`. Every C1 row
  (10–39 tokens) took the exllamav3 trellis path with an FP16 input cast.
  The overlay's receipt string `"torch linear (cuBLAS)"` was hard-coded,
  not observed.
- The candidate's KDA projections run the EXL3 trellis kernels (FP16-input
  numerics family). The BF16 "common comparison space" weight that the
  binding recorded is a host-side reconstruction from the checkpoint. It was
  never checked against the resident bytes.
- So neither engine multiplies the bound BF16 weight. No f32/RNE replay over
  it can pass the self-anchor, and the WIP reference backend replays the same
  wrong (`rows > 144`) branch. The 6.3% cross-engine weight delta compares
  two BF16 reconstructions of one trellis, not what the engines compute.

## What remains valid

The data under `$EV/q2-c1-execution-20261012/` and `$EV/q2-c1-binding-2026013/`
(sic): fresh inputs equal to the fixtures, native pre-reduce captures, and
the neutrality pass. The C1 answers are derived from those native captures.
See the `ERRATUM-20261012.md` beside the execution evidence.

## If o_proj replay is ever needed again

1. Bind the trellis bytes the kernel reads, checked against the resident
   upload.
2. Attest the input precision and `apply` branch at run time.
3. Replay through one EXL3 kernel for both engines.

Do not revive these files.
