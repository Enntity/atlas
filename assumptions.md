# Assumption log

Assumptions, constraints, and requirements inferred or provided for the
project. Reference these from source documentation when explaining why
something is done a specific way. Keep entries short; newest section first.

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
