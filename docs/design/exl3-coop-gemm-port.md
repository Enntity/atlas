# EXL3 cooperative GEMM port (phase 1: packed linear projections)

Status: implemented behind `ATLAS_EXL3_COOP_GEMM=1` (default off). Not yet
GPU-validated; bit-identity against `exl3_mma.cuh` is argued below and must be
confirmed by the identity check before the flag is used for qualification.

## Enabling

```
ATLAS_EXL3_COOP_GEMM=1
```

Applies to every `Exl3LinearWorkspace` built with an explicit numerics profile
(`new_with_numerics`; GLM-5.3 TP2 profiles in
`spark-model/src/weight_loader/glm5/exl3_numerics.rs`). Workspaces without a
profile, and any row bucket whose profile is not covered (below), keep the
existing `exl3_input_hadamard_f16 -> exl3_mul1_matmul_mma[_split] ->
exl3_output_hadamard[_f32]` path; each uncovered case logs one
`ATLAS_EXL3_COOP_GEMM: ...` warning. `ATLAS_EXL3_SPLIT_MMA` only affects the
fallback path.

## Pipeline

Same stream, same scratch, same outer contract:

1. `exl3_coop_input_hadamard_f16h` (Atlas): the arithmetic of
   `exl3_input_hadamard_f16`, storing the F16 result as `__half` (the existing
   kernel stores the same F16 value widened to F32). Written into the existing
   F32 input scratch (half its bytes).
2. `exl3_coop_gemm_b{5,6}_k{tk}_n{tn}` (vendored core): cooperative launch,
   F32 output (`c_fp32 = true`) into the existing F32 output scratch.
3. `exl3_output_hadamard` / `exl3_output_hadamard_f32`: unchanged.

The Hadamards are deliberately not fused: the reference's in-kernel transforms
(`had_hf_r_128_inner` / `had_ff_r_128_inner`) round differently from Atlas's
`hadamard128` + separate scale multiplies, so fusing would break bit-identity.

## Launch geometry (as the reference chooses it for a profile)

For a profile `(tile_k, tile_n, slices)` (the reference's pinned autotune record
`num_sms`, `block_dim`, shape):

| (tile_k, tile_n) | reference shape | block | SH_STAGES/FRAG_STAGES |
|---|---|---|---|
| (32, 128) | 2 | 512 | 4 / 3 |
| (32, 256) | 3 | 512 | 4 / 3 |
| (16, 512) | 4 | 256 | 4 / 3 |

Dynamic shared memory is always `SMEM_MAX` = 92160 B (the reference launches
every shape with it); `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` is set
before `cuLaunchCooperativeKernel`. The reference grid is `slices` CTAs; Atlas
launches only the contiguous sub-range of those slices whose split-K range
intersects the component (`slice_base`, `grid`), because the others return
immediately (fused-QKV components cover 1/3 of the logical matrix). This does
not change any CTA's work.

Rows: the kernel loops over 16-row passes exactly like upstream
`exl3_gemm_kernel` (`TILESIZE_M == 16`), so prefill up to the workspace capacity
(e.g. M = 128) is covered with the same split-K partition per pass. Upstream
separates passes with `grid.sync()` because it reuses one lock row; Atlas gives
each pass its own lock row and needs no grid barrier (the per-pass arithmetic is
unchanged).

Locks: a per-workspace buffer of `ceil(capacity/16) * (n/128)` ints, zeroed at
allocation, freed with the workspace. The reference uses one per-device
`(MAX_TILES_C + 2*MAX_BARRIERS + MOE_SCHED_INTS)` int buffer; per-workspace
locks are used here because a workspace is bound to one stream, so two
workspaces on different streams can never share a lock word. Both rely on the
same self-resetting protocol (the last chunk of each column stores 0).

## What was vendored

MIT, Copyright (c) 2025 Turboderp; the full notice heads every vendored file.
Sources: `atlas-exl3-validation-20261010/arithmetic-audit/installed-extra-source/`
(`source-sha256.json`) and, for the helper headers that tree does not contain,
the installed-head snapshot `mindreader/artifacts/moe-source-audit-v1/head/`
(`SHA256SUMS`).

| Atlas file (in `kernels/gb10/common/` and `kernels/gb10/glm-5.3-flash-exl3/exl3/`, byte-identical) | upstream | SHA256 |
|---|---|---|
| `exllamav3_gemm_inner.cuh` | `quant/exl3_gemm_inner.cuh` | `46aefd5e631ef8bd21b816939d9ed0511decf9ae42241b6ad745a57042b84abf` |
| `exllamav3_dq.cuh` | `quant/exl3_dq.cuh` (verbatim) | `7e3009b0f70635e8a98edf8b929832ca0fe16b6a5ad20f13b77ebfa72723feac` |
| `exllamav3_codebook.cuh` | `quant/codebook.cuh` (verbatim) | `0e3c63b323f8d3cc15c6a8f2e2b3816efafd71e149a99997b30b2f806375138a` |
| `exllamav3_kernel_map.cuh` | `quant/exl3_kernel_map.cuh` (shape table, T-args) | `59c49719b489764410a129a73ec3fddfab9eb96bfeedbf365d020ce0a39b26d4` |
| `exllamav3_ptx.cuh` | `ptx.cuh` (subset, verbatim) | `8ceacb1b321af5877a9a3d651e038fb1f44b543c3011e2f814f448830efe27ee` |
| `exllamav3_util.cuh` | `util.h` / `util.cuh` (macros, unions) | `ba89ac6793bf31cfe123d13baa2e34531a18a24bc3b2ff4add8cf4856b75a1c8` / `1907cea115260db7c3b0de540e375e7733b7ffc9019bec190c1d92abea1d964e` |

Consulted, not vendored: `quant/exl3_gemm.cu` (`2455154041c8573832fbe7ef814efe7bef0cf2cd6be1c3ed5050fb16b1f5f8fc`,
host dispatch and cooperative launch), `quant/exl3_gemm_kernel.cuh`
(`f73b2e578db1343a7ee1700bba9c3ec12819224964660d21dd817b87c0d5f90f`, row-pass
loop), `quant/exl3_kernel_map.cu` (`ad98208c05841bed99030e85502419da481a5e1fe4abd002af91078af9645649`),
`quant/exl3_devctx.cuh` (`effb1827e9b6c61ba95287ccfa5d6a0b6ccef103c0d588d992359be9291834c0`),
`quant/coop_autotune.cu` (`f6096afdc617b038c09b42cd4fab331a929ef5d191843e9c80366affde01987a`).
Not vendored: `hadamard_inner.cuh` (including the MoE atomic scatter), the
mgemm/MoE kernels, the GEMV paths, torch/ATen glue.

Atlas-authored (AGPL): `exl3_coop_gemm.cu` (entry points, row-pass loop, input
transform twin), `crates/spark-runtime/src/weights/exl3/native/coop.rs`, the
cooperative launch in `atlas-core` registry / `GpuBackend::launch_cooperative_typed`.

## Adaptations to the GEMM core (each marked `ATLAS:` in the source)

1. Logical split-K window: the partition `tiles_k * logical_tiles_n * s /
   slices` runs over the logical (fused) width and is clamped to the component's
   column-tile window, then rebased to physical indices. With logical ==
   physical this is upstream's partition. Requires `logical_offset % tile_n ==
   0` (true for every GLM profile; otherwise fallback).
2. `shmem_out_had` path removed (Atlas output Hadamard kept), i.e. the
   `shmem_out_had == false` instantiation upstream uses in `exl3_mgemm_kernel`.
3. sm_86-only FP16-accumulate path removed (compiled out upstream on sm_121).
4. First partial of each column tile is `__fadd_rn(p, 0.0f)` (see numerics).
5. `register` removed (ill-formed C++17), unused locals dropped.

Instantiated: mul1 codebook (cb 2), bits 5 and 6 (the checkpoint's bitrates;
`exl3_numerics.rs`), F32 output, shapes 2/3/4. PTX compiles under
`--fmad=false --Werror all-warnings`; a standalone sm_121a compile reported
(ptxas) 111 / 128 / 183 registers for (32,128) / (32,256) / (16,512), no
spills. The Atlas build (devbuild2) compiles the module for sm_121f.

## Numerics argument: bit-identical to `exl3_mma.cuh` `accumulate_logical`

For each output element, both compute the same sequence of IEEE operations:

- A operand: the same F16 values (step 1 above; `pair_bits` in Atlas rounds an
  F32 that already holds an exact F16 value). Rows other than the real ones are
  zero in Atlas and stale shared memory here; MMA rows are independent, and
  only real rows are stored.
- B operand: Atlas `mul1(tile_state(...))` is byte-sum of `state * 0x83dcd12d`
  plus 0x6400 as an F16 bit pattern, one `__hfma` with 0x1eee/0xc931, widened
  and re-rounded (exact). Upstream `decode_3inst_2<2>` computes the same byte
  sum with `__dp4a(x, 0x01010101, 0x6400)` (max 0x67FC, no overflow) and the
  same single-rounding `__hfma2`. Same weights per lane, derived: for bits 5/6
  `dq_dispatch` is `dq4(lane*8)` + `dq4(lane*8+4)`, so lane `g*4+t` decodes
  stream positions `g*32 + t*8 + j`, j = 0..7, which the m16n8k16 B layout
  places at k = 2t + (j&1) + 8*((j>>1)&1), n = g + 8*(j>>2). Inverting gives
  `stream = (n&7)*32 + ((k>>1)&3)*8 + ((n>>3)&1)*4 + ((k>>3)&1)*2 + (k&1)`,
  term-for-term Atlas `tile_state`. Bit extraction agrees too: both take the
  16 bits ending at `(stream+257)*bits` from a 64-bit `(hi_word<<32)|lo_word`
  window with word index `% (8*bits)`; for bits 5/6 every `dq4` group of four
  codes spans at most two consecutive words (start bit mod 32 is in
  {21,29,5,13}/{9,17,25,1} for 5 bits and {22,6}/{14,30} for 6 bits, spans of
  31 and 34 bits), so the 64-bit window covers every code it extracts.
- Per (slice, column tile): one `mma.sync.m16n8k16.row.col.f32.f16.f16.f32`
  chain per 16-deep k sub-block, over the tiles of the slice in increasing k,
  starting from +0 (Atlas `acc[sub]`; upstream `frag_c` of `sub_k` group).
- Sub-block reduce (tile_k = 32): upstream `store(1); add(0)` is
  `acc0 + acc1` = Atlas `__fadd_rn(acc[0], acc[1])` (tile_k = 16: no reduce).
- Cross-slice fold: the lock protocol admits a column's chunks from highest k
  to lowest, i.e. slices in descending order, each doing `partial + previous`;
  Atlas folds `total = __fadd_rn(sum, total)` for s = slices..1, skipping empty
  slices. IEEE addition is commutative, so operand order is irrelevant.
  The chunk boundaries are identical (same `total * s / slices` arithmetic,
  same intersection with the column).
- Atlas seeds `total = 0` and adds the first partial (`-0 + +0 = +0`); upstream
  writes the first partial directly. Adaptation 4 adds `+0.0f` to the first
  partial, so even a `-0.0` partial matches.
- All adds are plain `add.rn.f32` (`--fmad=false`; no multiplies are involved
  in the fold), F32 partials are stored unrounded (`c_fp32`), and Atlas's
  profiles all have `half_partial = false`.

The remaining assumption is hardware-level: the same `mma.sync` instruction
with identical A row, B fragment and C input yields identical D regardless of
the other A rows. This holds for the f32-accumulate HMMA (rows are independent
dot products) and is what the GPU identity check will confirm.

## Not covered (fallback, logged once)

- Profiles with tile (16, 128) (reference shape 1) or (16, 256)/(32, 512):
  none of the GLM-5.3 TP2 profiles use them.
- Bits other than 5/6, mcg/3INST codebooks, F16 partials (`half_partial`).
- `logical_offset` or physical width not a multiple of `tile_n`.
- `slices` greater than the device SM count, or a rejected cooperative launch
  (the workspace then stays on the existing kernels).
- The routed-MoE kernels are untouched (phase 2).

## Risks

- Cooperative residency: 92160 B smem/CTA means one CTA per SM; the 48-slice
  profiles need all 48 GB10 SMs co-resident. Concurrent long-running kernels on
  other streams (e.g. communication) delay the launch until 48 SMs are free.
  Registers fit (<= 128/thread at 512 threads, 183 at 256).
- Locks: self-resetting; a kernel aborted mid-run (context fault) leaves
  non-zero locks, which would deadlock a later launch on the same workspace.
  A context fault is fatal for the process anyway.
- CUDA-graph capture of `cuLaunchCooperativeKernel` should record a kernel
  node with the cooperative attribute (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE` is
  documented as valid for graph nodes in the CUDA 13 headers); not yet
  exercised under Atlas's decode graphs.
- The in-kernel lock spin and the logical window are new code paths that only
  the GPU identity check can validate.

## Phase 2: routed-MoE thin experts (`ATLAS_EXL3_COOP_MOE=1`)

Default off; independent of `ATLAS_EXL3_COOP_GEMM`. Replaces only the routed
matmul (`exl3_moe_mul1_matmul_precision_mma[_split]`) for gate, up and down
when the launch is entirely thin (`rows <= thin_capacity`, 32; every row
routes to distinct experts, so `counts[e] <= rows`). Router, input Hadamard,
activation/down-input Hadamard, output Hadamard and the fixed-order
`exl3_moe_combine_f32` are unchanged; launches per projection are unchanged
(the coop matmul replaces the matmul launch one-for-one).

Kernel `exl3_coop_moe_thin_b4_k32_n256` (`exl3_coop_gemm.cu`): per slot
(row, expert) an M=1 GEMM with the thin profile (tile 32x256 = reference shape
3, block 512, 92160 B smem, 8 split-K slices), bits 4, mul1. Two core options
were added to `exllamav3_gemm_inner.cuh` (both off for the dense entries):

- `a_fp32` (adaptation 6): A tile read from the F32 per-slot scratch with
  plain loads, rounded F32->F16 RN into the same swizzled smem slot. Atlas's
  MMA rounds the same scratch with `__float2half_rn` in `pair_bits`.
- `round_partial` (adaptation 7): after every fold add, `total =
  f32(f16_rn(total))` in F32 storage: Atlas `half_partial`, and the value
  sequence upstream's F16-C path produces. Applied after the `+0.0f` seed, so
  the first chunk is `f16_rn(sum + 0)`, as in Atlas.

Scheduling: work items are (slot, slice), ordered slot-major with slices
descending (`item = slot*8 + 7 - slice`); CTA b runs items b, b+grid, ...
(grid = min(items, SMs), cooperative). A slice's lock wait in a column depends
only on higher slices of the same slot, i.e. lower item indices; the lowest
unfinished item's owner is always executing it, so the schedule cannot
deadlock. Locks: one row of `max(hidden, intermediate)/128` ints per slot for
up to `min(capacity, thin_capacity) * topk` slots, per MoE layer, zeroed at
load. Decode (8 slots) = 64 items on 48 CTAs. No atomics: each slot writes its
own F32 row.

B operand for bits 4 (`dq8_aligned_4bits`): lane L reads words L and L-1 (mod
32) and extracts codes `8L..8L+7` at bit offsets that equal Atlas
`tile_state`'s `(stream+257)*4` window (e.g. code 8L is bits 28..43 of
`(w[L-1]:w[L])`), with the same j -> (k, n) fragment mapping as bits 5/6.

Not covered (existing kernel, warned once): fat launches (`rows >
thin_capacity`, prefill), non-4-bit experts, widths not multiples of 256, a
thin profile other than 32/256/8/half, rejected cooperative launch. Slots
whose id is invalid, whose descriptor is not 4-bit or whose count exceeds
thin_capacity are written as NaN, like the existing kernel's guard.

## Phase 3: fused KDA q|k|v (`ATLAS_EXL3_COOP_FUSED_QKV=1`)

Default off; independent of the other two flags. The checkpoint's
`qkv_proj` is one packed matrix; per TP rank the reference runs it as one
handle of width `logical_n = 3n` (n = 4096 per component), and Atlas already
reproduces that partition per component (logical_offset 0, n, 2n) with only
16 of the 48 slices touching each component. Phase 3 launches the three
components as ONE cooperative GEMM over the fused width with the profile's
full 48-slice grid.

- Core option `multi` (adaptation 8): the fused matrix is addressed as
  `count * size_n` columns; column tile `t` maps to component `t / tiles_n`,
  local tile `t % tiles_n` (n is a tile_n multiple, so no tile straddles a
  component). B and C base pointers are chosen per column tile; the split-K
  partition, lock index and fold are the fused width's, i.e. exactly what the
  per-component launches compute after their window clamp. Dense and MoE
  instantiations are unchanged (same ptxas registers).
- Trellises stay in their separate per-component uploads (option b: three
  base pointers, no concatenated copy, no extra weight memory). Each
  component's F32 output goes to its own workspace scratch; each component's
  existing output Hadamard then writes its BF16 plane, so consumers are
  unchanged.
- One input transform (component q's scratch) serves all three: at load the
  three components' input scales are compared byte-for-byte (D2H); a
  mismatch disables the fused path.
- Entries: `exl3_coop_gemm3_b6_{k32_n128,k32_n256,k16_n512}` (the KDA
  profiles use (16,512) for rows <= 8 and (32,256) above). Per-launch lock
  rows: `ceil(capacity/16) * 3n/128` ints, owned by the q projection.
- Wiring: `Exl3FusedCoop` (spark-runtime `native/coop.rs`), attached to the q
  `Glm5Exl3Projection` in the KDA loader; `project_attention_qkv` tries it
  first. Other q/k/v call paths (multi-sequence, paired verify) keep the
  per-component launches (same numerics).

Not covered: MLA `q_a_proj` / `kv_a_proj_with_mqa` and the dense
`gate_proj` / `up_proj` are separate checkpoint tensors whose reference
profiles have logical width == physical width (separate handles, e.g. 48 and
20 slices for q_a / kv_a). Fusing them would either change their split-K
partitions (numerics) or need more than 48 co-resident CTAs, so they keep
separate launches.

## Phase 4: routed-MoE v2 scheduling (`ATLAS_EXL3_COOP_MOE_V2=1`)

Default off; implies the phase-2 resolution (v1 alone stays selectable with
`ATLAS_EXL3_COOP_MOE=1`). Same per-slot arithmetic; only the work split and
launch grouping change.

- Chunk items (core adaptation 9): a slice's work inside one column is an
  independent chunk (own MMA chains, reduce, lock step; `lock_i`/`lock_d`
  depend only on the chunk's k range). v2 items are single (slice, column)
  chunks: gate/up slices are half a column (1 chunk each, 64 tiles), down
  slices cover two columns (2 chunks of 32 tiles). Chunks are ordered
  slices-descending, columns-ascending per (projection, slot), so every lock
  wait points to a lower item index and the v1 no-deadlock argument holds.
- Gate+up in one launch (`exl3_coop_moe2_thin_b4_k32_n256`, projections=2):
  both input transforms are launched first (gate's into the usual xh scratch,
  up's into the down-output scratch `buffers[5]`, unused until the down
  projection), then one cooperative matmul, then the two unchanged output
  Hadamards (up's F32 matmul output follows gate's in `buffers[8]`; needs
  `2 * intermediate <= hidden`). Lock rows per (projection, slot).
- Decode wave count (8 slots, 48 CTAs): gate+up 2 x 2 waves x 64 tiles = 256
  tile-times -> 3 waves x 64 = 192; down 2 waves x 64 = 128 -> 3 waves x 32 =
  96. Total 384 -> 288 (-25%), plus one fewer launch per layer.

Not done: 2 CTAs/SM. The tile_k=32 block is 512 threads at 117-128
registers, which fills the register file; a 256-thread variant would have to
run both k16 sub-blocks per thread (an inner restructure; the fold could be
kept identical, but it is not a scheduling change). The reference
`exl3_moe_kernel.cuh` uses the same inner loop per expert (greedy ticket
scheduling over expert groups, no cross-expert prefetch), so there is no
further inner-loop difference to port.

## Phase 5: routed-MoE v3, two CTAs per SM (`ATLAS_EXL3_COOP_MOE_V3=1`)

Default off; implies v2 scheduling (same items, order, scratch, locks) and
falls back to the v2 kernel if v3 cannot be resolved. v1/v2 stay selectable.

- Core option `both_sub` (adaptation 10, tile_k 32 only): 256 threads per
  CTA, each running both k16 sub-blocks with separate accumulators (acc0,
  acc1) and the same per-sub-block MMA chains as upstream's two 256-thread
  halves; the threadblock reduce becomes the in-register `acc0 + acc1`
  (upstream: sub-block 1 stores, sub-block 0 adds; the same F32 add). Global
  loads were already done by threads 0..255 only. Other instantiations keep
  identical register counts.
- `exl3_coop_moe3_thin_b4_k32_n256`: `__launch_bounds__(256, 2)`, ptxas 124
  registers (allocated as 128), no spills, 16 B stack. Dynamic smem is the
  stage buffers only (no reduce buffer): 6 stages x (1024 + 4096) = 30720 B
  (SH_STAGES 4 -> 6 is pipelining depth only). Per SM: 2 x 256 x 128 = 65536
  registers (the full file), 2 x 30 KB smem (+1 KB reserved each) of the
  ~100 KB sm_12x per-SM limit, 512 threads.
- Grid: `min(items, blocks_per_sm x SMs)` with blocks_per_sm from
  `cuOccupancyMaxActiveBlocksPerMultiprocessor` (after setting the function's
  preferred shared carveout to 100%), via
  `GpuBackend::cooperative_blocks_per_sm`. Expected 2 x 48 = 96 CTAs: decode
  gate+up 128 items -> 1.33 waves (was 2.67), down 128 half-size items ->
  1.33 waves.

Dense coop GEMM: not converted. Its grid is the profile's slice count
(<= 48, one CTA per reference slice), so extra residency only helps if slices
are split into column chunks as in v2; and the KDA decode shape (16, 512)
has a single k16 sub-block per tile at 183 registers x 256 threads, which
cannot reach 2 CTAs/SM without spilling.

## Phase 6: packed shared expert (`ATLAS_EXL3_SHARED_PACKED=1`, numerics-changing)

The flag keeps the MoE shared expert packed (the reference reconstructs it to
BF16), so its profiles are Atlas-chosen, not reference-bound; results are
deterministic but differ from the BF16 path.

- Gate|up as one fused matrix: gate (offset 0) and up (offset n) carry the
  numerics of a logical `[gate | up]` width 2n = 2048, each with its own
  workspace. With both available they run as ONE cooperative launch
  (`exl3_coop_gemm2_b{5,6}_k32_n{128,256}`); otherwise the per-component
  launches compute the identical partition (window clamp), so fallback is
  value-consistent.
- Core option `multi_a` (adaptation 11): gate and up are separate tensors
  with separate input scales, so each component reads its own F16 A; the A
  tile pointer is re-based with B/C when a CTA changes column tile.
  `Exl3FusedCoop` handles 2 components (per-component input transforms) and
  3 (KDA q|k|v, one shared transform, unchanged).
- Profiles: fused gate|up `p(32,128,48)` (16 column tiles x 128 k-tiles ->
  ~43 tiles per slice, ~3 slices folded per column); down
  (K1024 -> N4096) `p(32,128,48)` for all buckets (1024 tiles, ~21 per
  slice, <= 2 slices per column). The previous 32x256 profile for N=1024
  gave 4 column tiles for 48 slices, i.e. ~12 slices serialized through each
  column's lock-ordered fold for ~11 tiles of work each.
- Not done: riding the routed MoE v3 waves. The shared expert has bits 5/6
  (the routed kernel is instantiated for bits 4 only), the plain (non thin,
  non half-rounded) fold, its own input transform and a BF16 output into the
  combine's shared slot; adding it as extra work items needs a runtime-bits
  dispatch inside the routed kernel and a second output path.

## Phase 7: routed-MoE per-CTA throughput (`ATLAS_EXL3_COOP_MOE_V6=1`)

Standalone microbenchmark `tests/exl3_validation/coop_moe_bench/bench.cu`
(decode geometry: 8 slots, gate|up K4096->N1024 x2, down K1024->N4096, 4-bit,
64 experts rotated so weights stream from DRAM; bit-exact cross-check of
every variant against v2; plain-read roofline; side-stream overlap probe).
Spark1, idle, 2026-10-15:

| kernel | gate\|up (33.6 MB) | down (16.8 MB) |
|---|---|---|
| read-only roofline (192 CTAs) | 150 us, 223 GB/s | 75 us, 225 GB/s |
| v2 (512 thr, 1 CTA/SM) | 263 us, 128 GB/s | 137 us, 123 GB/s |
| v3 (both_sub, 2 CTAs/SM) | 201 us, 167 GB/s | 108 us, 155 GB/s |
| v6 (v3 + staged A row) | 186 us, 180 GB/s | 103 us, 163 GB/s |

Findings:
- v3 matches the production trace (200 / 105 us). Time follows the busiest
  SM's bytes: one item (one slice of one column, 64 k-tiles, 256 KB)
  takes ~78 us on an otherwise idle GPU, whether 16, 32 or 48 items run.
- Knocking out decode and MMA (bench-only) changed that by <5%; deeper
  cp.async staging (4..12) did not help either. The cost was the a_fp32 A
  path: per k-tile synchronous global loads that stall the issuing warp and
  then the whole CTA at the next barrier. Core option `a_smem_row`
  (adaptation 12) converts the A row once per item into shared memory (the
  both_sub kernels' unused reduce region); stages copy from there. Same F16
  values. Per-item time 78 -> 70 us; full decode call -15 / -6 us.
- Tail balancing does not pay: splitting chunks into 128-column halves
  (balanced rounds) or mixing full and half items kept the arithmetic exact
  but the 32x128 tile is less efficient per byte, so it was slower or equal.
  Those variants were dropped.
- Overlap probe: a 25 MB memory-bound side kernel on a second stream during
  the routed launches saved ~50 us of a 110 us side cost per layer
  (438 -> 388 us). This is the remaining structural lever: run the shared
  expert concurrently with the routed projections (fork/join events; same
  kernels, bit-exact). Not implemented here.

`exl3_coop_moe6_thin_b4_k32_n256`: v3 + `a_smem_row`, 125 registers, no
spills, 38912 B dynamic smem (6 stages + 8 KB A row), 2 CTAs/SM. Implies v2
scheduling, falls back to v3/v2.

## Phase 8: shared-expert overlap and fused Hadamards (all bit-exact)

Flags (default off):

- `ATLAS_EXL3_SHARED_OVERLAP=1`: the MoE shared expert (BF16 reconstruct or
  `SHARED_PACKED`) is built bound to a per-main-stream non-blocking side
  stream; the layer forks it (event record/wait) after the router and route
  counts, runs the routed projections on the main stream and joins (event
  record/wait) before `buffers[6]` is read. The side stream only reads the
  layer input and writes its own FFN scratch and `buffers[6]`. The classic
  BLAS pool keeps a handle and workspace per stream. Disabled under
  per-stage profiling.
- `ATLAS_EXL3_COOP_MOE_V7=1` (implies V6): gate|up from the raw layer input,
  with the input transform computed into the staged A row (core adaptation
  13: F16 product, 128-point butterfly with hadamard128's per-element op
  order via warp shuffles, x H128_NORM, F16 RN). Drops the two routed
  input-transform launches per layer.
- `ATLAS_EXL3_COOP_FUSED_INPUT=1`: dense coop GEMMs (`exl3_coop_gemmh_*`) and
  KDA q|k|v (`exl3_coop_gemm3h_*`) compute the input transform in every
  CTA's prologue when rows x k F16 fit after the stage and reduce buffers in
  SMEM_MAX (K4096: <= 8 rows at 32x128, <= 5 at 32x256, <= 4 at 16x512).
- `ATLAS_EXL3_COOP_FUSED_OUTPUT=1` (with fused input, BF16 outputs): the CTA
  holding a column tile's last chunk applies the output transform
  (adaptation 14) and writes BF16 (`exl3_coop_gemmf_*`). Not used for the
  F32 head or the multi-component launches.

Standalone (coop_moe_bench, spark1, under /tmp/atlas-bench.lock), all
bit-exact against the unfused Atlas kernels: v7 gate|up 193.7 us vs 199.8
(2 x transform + v6); dense rows=1 32x128 62.2 vs 67.9 us, 32x256 55.3 vs
61.1, 16x512 67.3 vs 69.6 (gemmf vs f16h + gemm + output transform); rows
1-3 checked.

In-model (ab_full.sh, graph env, ab_prose_long, all hashes 80b1eb5fadf8):

| run | binary | tpot p50 |
|---|---|---|
| base (V2+V3) | 596d300f | 0.048915 |
| + SHARED_OVERLAP | 596d300f | 0.048848 |
| + SHARED_OVERLAP + V7 (V6) | 596d300f | 0.046303 |
| + V7 (V6) | 50d0a0bf | 0.048038 |
| + V7 + FUSED_INPUT + FUSED_OUTPUT | 50d0a0bf | 0.048162 |

Single runs; the overlap+V7 vs V7 difference (1.7 ms) is larger than the
overlap-alone effect (0.07 ms) and should be repeated on one binary before
it is attributed.

## Phase 9: multi-row (speculative verify) routed MoE (`ATLAS_EXL3_COOP_MOE_MULTIROW=1`)

At r rows the per-slot kernels stream one expert matrix per (row, expert)
slot, so routed bytes scale with r even when rows share experts. With the
flag (and a v2+ gate|up path), launches with rows >= `ATLAS_EXL3_COOP_MOE_MULTIROW_MIN_ROWS`
(default 4) instead:

1. `exl3_moe_group_slots` (one block): groups slots by expert, ascending
   experts, ascending slots (deterministic, independent of launch timing).
2. `exl3_moe_input_hadamard_f16h`: per-slot transformed gate/up inputs as F16
   (same arithmetic as `exl3_moe_input_hadamard_f16`), both in the matmul
   input scratch; for down, `exl3_f32_to_f16` (exact) of the activation's
   scratch into the down-output scratch.
3. `exl3_coop_moe8_multirow_b4_k32_n256`: items (projection, group, chunk);
   each streams the expert's tiles once for all the group's rows using the
   core gather option (adaptation 15: logical row -> slot row for A and C).
   Fixed co-resident grid (occupancy x SMs); the item count is read on the
   device. FR=1 keeps 128 registers, no spills, 2 CTAs/SM.

Per-row results equal the per-slot kernels' bit for bit (MMA rows are
independent; the thin partition depends only on the matrix shape), so the
grouping does not make a row depend on the other rows' values.

Bench (`tests/exl3_validation/coop_moe_bench/multirow_bench.cu`, spark1,
under the lock; rows draw 8 distinct experts from a 48-expert pool;
bytes = 3 projections x 2 MiB per streamed matrix):

| r | slots | unique | v6 per-slot | v8 grouped (incl. grouping) |
|---|---|---|---|---|
| 1 | 8 | 8 | 289 us, 50 MB | 324 us, 50 MB |
| 2 | 16 | 14 | 507 us, 101 MB | 487 us, 88 MB |
| 4 | 32 | 22 | 829 us, 201 MB | 721 us, 138 MB |
| 8 | 64 | 36 | 1596 us, 403 MB | 1086 us, 227 MB |

With a 256-expert pool (little sharing, 56 unique at r=8) v8 is 1637 vs 1808
us; with a 32-expert pool (30 unique) 919 vs 1601 us. v8 is slower at r=1-2
(grouping launch, F16 staging), hence the default threshold of 4. Per-call
v8 runs at ~205-215 GB/s; v6's higher apparent GB/s at large r comes from
repeated experts hitting L2.

Dense coop GEMMs need no change for r <= 8: weights are streamed once per
launch, K4096 x N4096 6-bit at r = 1/2/4/8 takes 59/59/60/63 us (32x128),
50/50/51/51 us (32x256), 61/61/62/68 us (16x512). The KDA/dense profiles pick
16x512 for rows 3-8 in several geometries; 32x256 would be ~10-16 us faster
per call at r = 5-8 but changes the split-K partition (numerics), so it is
left to an explicitly flagged speculative-lane profile.

## Phase 10: DFlash2 drafter projections (bench; existing low-precision twins)

`tests/exl3_validation/drafter_bench/bench.cu` times the drafter's 7 per-layer
projections (hidden 4096, q 4096, k/v 1024, intermediate 12288) at M = 8 rows
with the kernels the drafter already has (weights rotated over 6 copies so
every launch streams from DRAM; spark1, under the lock):

| format (kernel) | q | k, v | o | gate, up | down | per layer | x5 layers |
|---|---|---|---|---|---|---|---|
| BF16 (`dense_gemv_bf16_batchm`, today) | 167 us | 61 us | 172 us | 454 / 450 us | 682 us (147 GB/s) | 2047 us, 386 MB | 10.23 ms |
| MXFP8 (`mxfp8_gemv_tc8`) | 83 | 18 | 83 | 249 / 250 | 263 | 964 us, 199 MB | 4.82 ms |
| NVFP4 (`w4a16_gemv_tc8`) | 51 | 18 | 52 | 140 / 141 | 141 | 561 us, 109 MB | 2.80 ms |

The BF16 total matches F4's profiled 10.2 ms. BF16 at M = 1 is 8.17 ms: the
batched BF16 GEMV loses throughput at M = 8, worst on `down` (K = 12288) and
k/v. Fusing q|k|v (238 vs 289 us) and gate|up (868 vs 903 us) saves ~0.4 ms
per step; a BF16 `down` at the 225 GB/s read roofline would save ~1.2 ms.

The quantized paths already exist in the shared `dflash_head`:
`ATLAS_DFLASH_MXFP8=1` (`install_mxfp8`: MXFP8 twins of q/o/gate/up/down,
k/v stay BF16) and the NVFP4 twins of all 7 projections served by the
tensor-core tiers (`try_install_nvfp4` + `ATLAS_DFLASH_NVFP4_TC=1`, needs
`ATLAS_W4A16_TC=1` for the tier table). Both are wired into the batched
DFlash2 path (`run_batched_dflash2`) and the paged per-row path. The GLM EXL3
observer disables them: `glm_exl3_dflash.rs` sets
`startup.bf16_weights_only = true`. Projected drafter layer time per step:
MXFP8 ~5.3 ms (k/v BF16), NVFP4 ~2.8 ms, against 10.2 ms today; tau must be
measured in-model.

## Phase 11: verify-scoped profiles; fused routed down chain (`ATLAS_EXL3_COOP_MOE_V9=1`)

`ATLAS_EXL3_VERIFY_PROFILES=1` is now scoped: `Exl3LinearNumerics` carries
`verify_rows_5_8` as a fifth profile slot, selected only while a thread-local
`VerifyScope` is open. `decode_verify_graphed_kgamma_dispatch` opens one, so
5-8-row prefill chunks and serial decode keep the reference profiles with the
flag on. Coop and fused-coop plans resolve all five slots; a verify graph
records the choice made at capture time.

Routed MoE serial gap, findings (standalone, spark1, under the lock):
- The B access pattern is not the limit: a plain read kernel with the routed
  items' layout (64 k-tiles x two 2 KB rows, 8 KB apart) reads as fast as a
  contiguous one (strided 206-227 GB/s vs contiguous 202-218 GB/s).
- Plain reads organized as 128 items on 96 CTAs (2 rounds) take 162 us for
  the gate|up bytes vs 148 us as one round on 128 CTAs; v6 takes 186 us. So
  ~14 us of the gap is the item/round structure itself and ~24 us is the
  kernel's per-item cost. Nsight Compute counters are not accessible to the
  user on the Sparks (ERR_NVGPUCTRPERM), so per-stall attribution was not
  possible.
- v9 removes three launches per layer and their scratch round trips: the
  down kernel's staging computes, per 128-feature group and in the separate
  kernels' per-element op order, the gate/up output transforms
  (butterfly, x H128_NORM, F16 product with svh), the thin activation
  (`exl3_moe_thin.cuh`, shared verbatim with `exl3_moe_precision.cu`, whose
  PTX is unchanged) and the down input transform (adaptation 16). Bench:
  bit-exact (clamp on and off); 88 us vs 102 us for (2 output transforms +
  activation/down transform + v6 down) per layer, ~0.6 ms/token.

In-model (binary a997588f, exact stack: V2/V3/V6/V7 + SHARED_OVERLAP +
ARENA_FILL=skip + graph env, ab_prose_long, all hashes 80b1eb5fadf8):
base 0.045692 / 0.045678, +V9 0.045782 / 0.045818 s/token. V9 is bit-exact
but gives no in-model gain: under decode graphs the three small launches it
removes cost little, and its per-item recomputation of the activation chain
(16 down items per slot) offsets the saving. Leave V9 off.

## Phase 12: streaming persistent routed MoE (`ATLAS_EXL3_COOP_MOE_V10=1`)

`exl3_coop_moe10_thin_b4_k32_n256` is a dedicated kernel (not the vendored
inner loop): every CTA runs ONE cp.async pipeline over the concatenated
k-tiles of its items (v2 item order), so the next item's B/A tiles are in
flight while the current item folds; there is no per-item pipeline ramp or
drain and no per-item A staging. Per item the arithmetic is unchanged: tile
32x256 with both k16 sub-blocks per thread (separate accumulators, MMA chain
over the item's k-tiles ascending), acc0 + acc1, +0.0 seed on a column's first
chunk, lock-ordered fold, F16 rounding of the running total. A comes from F16
per-slot rows (`exl3_moe_input_hadamard_f16h`; down: exact F32 -> F16 of the
activation scratch).

Standalone (coop_moe_bench, spark1, under the lock), all bit-exact vs v6:

| config | regs | CTAs/SM | gate\|up | down |
|---|---|---|---|---|
| v6 | 125 | 2 | 184-186 us | 102-105 us |
| v10 SH 6 | 94 | 2 | 183 | 104 |
| v10 SH 4 (default) | 78 | 3 | 167 | 89 |
| v10 SH 3 | 64 | 4 | 168 | 90 |

The win comes with the third CTA per SM that the smaller register footprint
allows (the streaming alone at 2 CTAs/SM is flat). Nsight Compute
attribution was not used.

In-model (exact stack: V2/V3/V6/V7 + SHARED_OVERLAP + ARENA_FILL=skip +
graphs, `ab_prose_long`, binary `spark.e36ee529`, alternating 3x3 runs,
every run hash `80b1eb5fadf8`):

| run | side | tpot p50 (s/token) |
|---|---|---|
| p12base-20261016T042928Z | base | 0.045639 |
| p12v10-20261016T043656Z | +V10 | 0.046442 |
| p12base-20261016T044423Z | base | 0.045712 |
| p12v10-20261016T045307Z | +V10 | 0.046436 |
| p12base-20261016T050035Z | base | 0.045720 |
| p12v10-20261016T050801Z | +V10 | 0.046506 |

Mean base 0.045690 vs +V10 0.046461: **+0.77 ms/token (+1.7%) slower**,
with no overlap between the two sides' ranges. V10 stays off. The likely
causes, not yet measured: (a) V10 replaces V7's fused input transform with two
separate input-hadamard launches plus an F32->F16 convert before down (extra
launches and scratch traffic per layer); (b) at 3 CTAs/SM the cooperative grid
fills every SM, so the side-stream shared expert (SHARED_OVERLAP) can no
longer co-run with the routed kernel. Follow-ups if pursued: A/B V10 vs base
with SHARED_OVERLAP off on both sides to isolate (b), and fuse the transform
and convert into the v10 A-load to address (a).
