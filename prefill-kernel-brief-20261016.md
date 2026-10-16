# Prefill kernel redesign brief — 2026-10-16

Worktree: `/home/colum/Workspace/RustroverProjects/atlas-perf`
Branch: `perf/decode-vllm-parity-20261015` (local, unpushed). New work on `followup/prefill-*`.

## Established (evidence-backed)

1. **Prefill ≈ 0.106 s per prompt token → ~9.5 tok/s marginal**, plus ~0.4 s fixed.
   108 tok → 11.8 s TTFT; 2573 tok → 272.3 s (p0 traces, instrumented).
   An M-tiled prefill should be ~flat in prompt length until compute-bound.
2. **95 % of prefill is two packed-matmul kernels** (p0-short-20261015 rank0, 108 tok, busy 11.25 s):
   | kernel | s | % |
   |---|---|---|
   | `exl3_moe_mul1_matmul_precision_mma` | 6.035 | 53.6 |
   | `exl3_mul1_matmul_mma` | 4.644 | 41.3 |
   | `ncclDevKernel_SendRecv` | 0.346 | 3.1 |
   | all others | <0.05 each | ~2 |
3. **Root cause: no M reuse.** Both kernels launch `grid.y = rows` (MoE: `rows*topk`).
   `kernels/gb10/common/exl3_mma.cuh`:
   > `// All warp lanes participate. One actual row is padded to M16; only lane<4 owns
   > that row's C0/C1 outputs.`
   Per CTA, the full K×N trellis decode (`decoded()` → `tile_state()`) runs, but only 1 of
   16 MMA rows is real. The decode+load repeats per row. This is why cost is linear in tokens
   and why achieved bandwidth is far below the GB10 memory roofline (memory-bound device).
4. **The vendored cooperative GEMM tiles M only to 16**
   (`exllamav3_gemm_inner.cuh:238`): `static_assert(TILESIZE_M == 16, ...)`, and
   `exl3_coop_gemm_body` loops `size_m` in 16-row passes. So even the coop path reads every
   dense weight `ceil(rows/16)` times per chunk.
5. **The grouped multi-row MoE kernel already exists but is unusable for prefill**
   (`kernels/gb10/common/exl3_coop_gemm.cu`, `exl3_coop_moe8_multirow_b4_k32_n256`):
   it groups slots by expert and calls the GEMM with `size_m = group_rows` (real M reuse),
   but (a) it hard-poisons any group with `rows > 16`, and (b) it hard-poisons
   `counts[expert] > thin_cap`, i.e. it implements only the *thin* (FP16-rounded) numerics.
   It is also unreachable for prefill because `moe/coop.rs:467` returns early when
   `rows > thin_capacity` (constant 32), with the runtime warning
   `ATLAS_EXL3_COOP_MOE: fat (prefill) routed launch; using the existing routed EXL3 matmul`.
6. **Fat numerics contract** (`exl3_moe_mma.cu`): slots whose expert has
   `counts[expert] > thin_cap` must accumulate **unrounded F32**
   (`fat_tile_k=16, fat_tile_n=128, fat_slices=1, fat_half_partial=0`); thin slots round the
   partial to FP16 after every k-fold. Any new kernel must preserve exactly this split.
7. Serve profile: `--max-prefill-tokens 128`, `--max-batch-size 1`, `--max-num-seqs 1`,
   `--max-seq-len 4096`. Chunk size does **not** change the number of weight passes
   (passes = ceil(rows/16) per matrix regardless of how rows are chunked), so chunk size is
   not the lever; M-tile size is.

## Requested changes (in priority order)

### P1 — grouped fat MoE with real M tiles (largest single win, 54 % of prefill)
Extend the multirow MoE path to fat launches:
- Reachable when `rows > thin_capacity` (remove/replace the early return at `moe/coop.rs:467`
  for path selection; keep the thin path for `rows <= thin_capacity`).
- Group slots by expert (`exl3_moe_group_slots` already exists) and run the GEMM per group with
  `size_m = group_rows`, **chunking groups whose count exceeds the M tile** instead of poisoning
  (replacing the `rows > 16` poison at the multirow kernel).
- Emit the **fat** numerics for every group whose expert count `> thin_cap`
  (`fat_tile_k=16, fat_tile_n=128, fat_slices=1, fat_half_partial=0`), and the thin numerics
  otherwise, matching `exl3_moe_mul1_matmul_precision_mma` bit-for-bit per row.
- Target: per-chunk weight reads drop from `rows*topk` slot-GEMMs to
  `Σ_experts ceil(count_e / M_tile)` ≈ number of touched experts (~3.5× for a 128-token chunk;
  more as rows/expert grows toward the tile).
- Acceptance: bit-identical output vs the current fat kernel on the same inputs
  (per-row F32), then a measured TTFT improvement ≥3 repeats.

### P2 — larger M tile in the dense coop GEMM (41 % of prefill)
The vendored inner is fixed at `TILESIZE_M == 16`. Options, cheapest first:
- (a) Interleave: keep TILESIZE_M=16 but have one CTA process several 16-row chunks for the same
  column tile **while keeping the B fragments staged**, so the decode is shared across chunks.
- (b) Add a genuine larger-M variant (e.g. TILESIZE_M=64) with matching A staging/shared-memory
  plan; the A staging is `TILESIZE_M * TILESIZE_K` halfs, so check `SMEM_MAX = 90 KiB` and the
  register budget against the fragment layout.
- Acceptance: bit-identical F32 output vs the existing coop kernel for the same profile geometry,
  then measured prefill improvement.

### P3 — reduce the trellis decode itself
`exl3_common.cuh` `tile_state`/`mul1` is the hot instruction sequence. Every kernel calls it
`(k/16)*(n/16)` times per M tile. Consider: precomputed decode LUTs in shared memory, warp-cooperative
decode with shuffles, or reordering so decoded values are reused across the K loop. GB10 is
memory-speed constrained, so prefer changes that raise bytes/instruction.

## Constraints (non-negotiable)
- SPDX `AGPL-3.0-only` header on every source file.
- Rust files ≤ 500 LoC (CI cap). CUDA files: keep the existing structure.
- No implicit defaults in production paths (PCND); fail fast on unsupported geometry and keep the
  existing NaN-poison contract for unsupported shapes.
- Do not change numerics silently. Per-row outputs must match the current kernel exactly, or the
  change must be behind an explicit, default-off flag with its own neutrality proof.
- Bench discipline: `flock /tmp/atlas-bench.lock`, label every run, ≥3 repeats, same binary in the
  same session for any speed claim. See `.agents/skills/measurement-discipline/SKILL.md`.
- Build into a unique path (`~/atlas-dev/bin/spark.<tag>`), never overwrite
  `~/atlas-dev/bin/spark` (other agents and the running baseline use it).

## How to measure
```bash
cd /home/colum/Workspace/RustroverProjects/atlas-perf
E=(--env ATLAS_EXL3_COOP_MOE_V2=1 --env ATLAS_EXL3_COOP_MOE_V3=1 --env ATLAS_EXL3_COOP_MOE_V6=1 \
   --env ATLAS_EXL3_COOP_MOE_V7=1 --env ATLAS_EXL3_SHARED_OVERLAP=1 --env ATLAS_DECODE_ARENA_FILL=skip \
   --env ATLAS_NO_DECODE_GRAPHS=0 --env ATLAS_NO_DECODE_GRAPHS_MULTISEQ=0 --env ATLAS_DEBUG_NO_GRAPH=0 \
   --env ATLAS_EP_GRAPHS=1 --env ATLAS_KDA_FLA_GRAPH=1 \
   --workload ab_prose_prefill --workload ab_prose_long)
bash tests/exl3_validation/ab_full.sh <tag> /home/colum/atlas-dev/bin/spark.<tag> "${E[@]}"
```
`ab_full.sh` adds SPLIT_MMA, SPLIT_MHC, RDMA_ALLREDUCE, COOP_GEMM, COOP_MOE, COOP_FUSED_QKV, PAR_SMALL.
Outputs land in `/home/colum/Workspace/atlas-exl3-validation-20261010/perf-ab-20261015/`.
Report `first_output_token_s` (real prefill+first token) and `ttft_s` from `summary.json` /
`raw.jsonl`, plus `output_sha256` for the numerics claim.

## Build
```bash
# source sync + build in the persistent builder on spark1, never touching the shared binary
rsync -az --delete -e "ssh -o BatchMode=yes -J colum@192.168.1.50" \
    Cargo.toml Cargo.lock rust-toolchain.toml crates kernels vendor jinja-templates tests \
    colum@10.50.0.11:atlas-dev/src/
# then cargo build --release -p spark-server inside atlas-dev-builder with the PTX cache
```
