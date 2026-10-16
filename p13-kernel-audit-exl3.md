I have all the evidence needed. Here is the audit report.

---

# EXL3 packed-matmul kernel-family audit (atlas-perf, read-only)

Scope: every `__global__` in the named CUDA files, the vendored inner/map headers, and the Rust that selects them. All citations are `file:line`. Sources: `prefill-kernel-brief-20261016.md`, `prefill-investigation-20261016.md` (trace 1, rank 0, 84.9 s GPU, 1024-token prefill).

Measured attribution used throughout (investigation §"CORRECTION"):
- `exl3_moe_mul1_matmul_precision_mma` **43.90 s / 51.7 %** (756 instances, avg 58.07 ms)
- `exl3_mul1_matmul_mma` **38.30 s / 45.1 %** (3985 instances, median 1.57 ms)
- everything else ≈ 3.2 % → the two hot kernels are **96.8 %** of prefill GPU time.

---

## 1. Entry-point inventory (every `__global__`)

### `kernels/gb10/common/exl3_linear_mma.cu`
| name:line | computes | grid / block | weight+act bytes per useful FLOP | B re-read per M row/tile | dominant cost | prefill share |
|---|---|---|---|---|---|---|
| `exl3_mul1_matmul_mma` :8 | dense packed linears `y=xh·Wᵀ`, one row/CTA, M16 pad, 4 N8 fragments/warp | grid `(ceil(physical_n/128), rows, 1)`, block 128 | W `b/16`, A `≈1/n` (b=5 → 0.3125, b=6 → 0.375 B/FLOP) | **yes, factor = rows** | `tile_state`+`mul1` trellis decode (4 decodes / `mma_step`, L58-61) | **45.1 %** (trace1) / 41.3 % (p0) |
| `exl3_mul1_matmul_mma_split` :50 | bit-identical slice-parallel twin; 16 cols/CTA | grid `(physical_n/16, rows, 1)`, block 128 | same | yes, factor = rows | same decode, spread over warps | 0 (only when `SPLIT_MMA=1`) |

### `kernels/gb10/common/exl3_moe_mma.cu`
| name:line | computes | grid / block | bytes/FLOP | B re-read | dominant cost | share |
|---|---|---|---|---|---|---|
| `exl3_moe_mul1_matmul_precision_mma` :12 | routed expert gate/up/down; per-slot M16-pad; fat (16/128/1/0) or thin (32/256/8/1) chosen per expert by `counts[expert]>thin_cap` (L48-52) | grid `(ceil(n/128), rows*topk, 1)`, block 128 | W `b/16` | **yes, factor = rows·topk** | `tile_state`+`mul1`; fat = 4 decodes/k16 step, no rounding | **51.7 %** (trace1) / 53.6 % |
| `exl3_moe_mul1_matmul_precision_mma_split` :61 | bit-identical slice-parallel twin (fat vs thin) | grid `(n/16, rows*topk, 1)`, block 128 | same | yes, factor = rows·topk | same | 0 unless `SPLIT_MMA=1` |

### `kernels/gb10/common/exl3_moe_precision.cu` (all M=1-per-slot unless noted)
| name:line | computes | grid / block | re-read? | cost | share |
|---|---|---|---|---|---|
| `exl3_moe_route_counts` :21 | histogram ids→counts (drives fat/thin) | `(ceil(experts/128),1,1)` b128 | n/a | O(slots·experts) scan | <0.1 % |
| `exl3_moe_input_hadamard_f16` :33 | per-slot input scale × H128 → F16-in-F32 scratch | `(ceil(k/128), rows*topk,1)` b128 | n/a | 7-stage butterfly | part of ~1 % |
| `exl3_moe_mul1_matmul_precision` :52 | **scalar** non-MMA fallback (1 weight/load/`mul1`) | `(ceil(n/128), rows*topk,1)` b128 | yes, per slot | 1 decode/MAC | 0 (fallback) |
| `exl3_moe_output_hadamard_precision` :75 | gate/up output H128 × svh, thin F16 / fat F32 | `(n/128, rows*topk,1)` b128 | n/a | butterfly | part of ~1 % |
| `exl3_moe_activate_down_hadamard` :95 | SiLU×up, down-input scale, H128 | `(k/128, rows*topk,1)` b128 | n/a | butterfly | ~1 % |
| `exl3_moe_down_weighted_output` :121 | down H128 × svh × routing weight | `(n/128, rows*topk,1)` b128 | n/a | butterfly | ~1 % |
| `exl3_moe_combine_f32` :148 | top-k routed sum + shared, BF16 round | `(ceil(rows·h/256),1,1)` b256 | n/a | BW-bound | ~0.5 % |

### `kernels/gb10/common/exl3_coop_gemm.cu`
| name:line | computes | grid / block | B re-read | cost | share |
|---|---|---|---|---|---|
| `exl3_coop_input_hadamard_f16h` :21 | input transform, F16 stored as `__half` | `(k/128, rows,1)` b128 | n/a | butterfly | ~0 |
| `exl3_coop_gemm_b{5,6}_k{32,16}_n{128,256,512}` :118 (6) | dense GEMM, inner loops `size_m` in **16-row chunks** (L93-116) | `(intersecting_slices,1,1)`, `256·tk/16` | **yes, ceil(rows/16)** | `dq8` BFE/funnel decode + MMA | 0 in trace1 (candidate 45 %) |
| `exl3_coop_gemmh_b{5,6}_…` :137 (6) | + fused input transform (note 13) | same | yes | same | 0 |
| `exl3_coop_gemmf_b{5,6}_…` :154 (6) | + fused output (note 14) | same | yes | same | 0 |
| `exl3_coop_moe_thin_b4_k32_n256` :187 | thin per-slot M=1 GEMM, split-K 8, locks | `(min(items,SMs),1,1)` b512 | yes, per slot | `dq8` + MMA | 0 |
| `exl3_coop_gemm3_b6_k{32,16}_n{…}` :249 (3) | fused q|k|v, 16-row chunks | `(num_slices,1,1)` | yes, ceil(rows/16) | same | KDA ≈0.5 % |
| `exl3_coop_gemm3h_b6_…` :291 (3) | + fused input | same | yes | same | 0 |
| `exl3_coop_gemm2_b{5,6}_k{32,128/256}` :335 (4) | shared-expert gate|up, per-component A | same | yes | same | 0 |
| `exl3_coop_moe2_thin_b4_k32_n256` :389 | thin v2, (slice,column) items | `(min(items,cap),1,1)` | per slot | same | 0 |
| `exl3_coop_moe3…` :485, `moe6…` :575, `moe7…` :666 | thin v3/v6/v7 (both_sub, staged A row, fused input) | `(min(items,cap),1,1)` b256 | per slot | same | **`exl3_coop_moe7_thin_b4_k32_n256` 0.16 s / 0.2 %** |
| `exl3_moe_group_slots` :765 | group slots by expert | `(1,1,1)` b256 | n/a | O(slots·experts) | 0 |
| `exl3_moe_input_hadamard_f16h` :807 | per-slot F16 input transform | `(k/128, rows·topk,1)` | n/a | butterfly | ~0 |
| `exl3_f32_to_f16` :826 | exact F32→F16 convert | `(ceil(n/256),1,1)` b256 | n/a | BW | ~0 |
| `exl3_coop_moe8_multirow_b4_k32_n256` :847 | **grouped M=rows-per-expert thin GEMM** (note 15 gather) | `(occupancy·SMs,1,1)` b256 | **no** within group; **poisoned for `rows>16` L906** | `dq8`+MMA | 0 (unreachable) |
| `exl3_coop_moe9_down_b4_k32_n256` :947 | down with fused activation chain (note 16) | `(min(items,cap),1,1)` b256 | per slot | same | 0 |
| `exl3_coop_moe10_thin_b4_k32_n256` :1121 | streaming persistent thin, one cp.async pipeline | `(min(items,cap),1,1)` b256 | per slot | same | 0 |

`exl3_shared.cu` (`exl3_shared_decode` :11, `exl3_shared_left_hadamard` :25, `exl3_shared_right_hadamard` :44) is the shared-expert twin; same M=1 structure, not hot (shared expert is one expert, small n).

---

## 2. The M=1-per-CTA structure — arithmetic

**Padding fraction.** `mma_step` (exl3_mma.cuh:20-27) issues one `mma.sync.aligned.m16n8k16` = `16·8·16 = 2048` MACs. `accumulate_logical` (L51-54) writes A only for `lane<4`, i.e. **row 0 only**; the comment at exl3_mma.cuh:29-31 states it. So 1 of 16 M rows is real → **15/16 = 93.75 % of tensor-core M-work is padding**, and only 4/32 lanes store output (exl3_linear_mma.cu:41-45, exl3_moe_mma.cu:56-59). Per `mma_step`, useful work = `8 (n8)·16 (k) ·1 row = 128` MACs — **6.25 % efficiency**.

**Decodes/loads per output element.** Per `mma_step` there are 4 `decoded()` calls (L58-61: `b0`=2, `b1`=2). Each `decoded` (L12-16) = `tile_state` + `mul1` and 2 packed-word loads. Output elements per `mma_step` = 8 (n8) × 1 real row = 8. So the kernel does **4 decodes and ~8 word loads per 8 useful output elements per k16** — i.e. ~1 decode and ~2 word loads **per useful output element·k16**, with **zero reuse across the 15 padded rows and across rows**. For one output `y[r,c]` over K, that is `K` decodes that all redundantly re-do work every other row already did.

**Grid scaling.** Dense `grid.y = rows` (native.rs:378 `.grid([.., rows, 1])`); MoE `grid.y = rows*topk` (exl3_moe_mma.cu `slot>=rows*topk`, project.rs:33 grid `[k/128, slots, 1]`). Consequence: total weight bytes read per prefill = `rows · (K·N·b/8)` for dense, `rows·topk · (K·N·b/8)` for MoE. Bytes/useful-FLOP = `b/16` independent of M — the definition of no amortization.

**M-tiling recovery.** If the packed B fragment for a `(k-tile, n-tile)` is decoded once and reused across `M_tile` rows (which the MMA allows: rows are independent — exl3_mma.cuh:29-31, header note 10/15 in exllamav3_gemm_inner.cuh), then:

| M_tile | weight reads per prefill | bytes/useful-FLOP (b=4) | decode count | speedup vs M1 |
|---|---|---|---|---|
| 1 (today) | R·(KN b/8) | 0.250 | R·K·N | 1× |
| 16 | (R/16)·(KN b/8) | 0.0156 | R·K·N/16 | 16× |
| 32 | (R/32)·… | 0.0078 | /32 | 32× |
| 64 | (R/64)·… | 0.0039 | /64 | 64× |
| 128 | (R/128)·… | 0.0020 | /128 | 128× |

for a prompt of `R` tokens. (MoE: replace `R` by `R·topk` slots.) This is exactly the “16× decode amortization” hypothesis in the investigation §"Hypothesis", and it is bounded above by actual rows-per-expert, hence the “≈ number of touched experts” target in the brief.

Cross-check against the measured ~270 GB/s and 43.9 s (investigation): `0.3 s` of weight traffic at M=1 means roughly `0.3 s·270 GB/s ≈ 81 GB / 43.9 s` — the device is not memory-bound; the 8–10× instruction overhead of the decode (§3) is the limiter, and M-tiling removes both the redundant decode and the redundant load together.

---

## 3. The EXL3 trellis decode — instruction count and staging

`decoded` = `mul1(tile_state(words, input&15, output&15, bits))` (exl3_mma.cuh:12-16).

**`tile_state` (exl3_common.cuh:47-59)** per decoded weight:
- stream permutation: `and`, `shl`, `shr+and`, `shl`, `add`, `shr+and`, `shl`, `add`, `shr+and`, `shl`, `add`, `and`, `add` → **~13 ALU** (L49-51).
- `end = (stream+257)*bits`: `add`,`mul` (L52).
- `first`/`last`/`shift`: 2×`add`, 2×`shr`, 1×`sub` → **~5 ALU** (L53-55).
- `first % count`, `last % count` with `count = 8*bits`: **`bits` is a runtime function argument**, so `count` is not a compile-time power of two → this compiles to a full **integer modulo sequence (≈10-15 instructions each, ×2)**. This is a concrete, avoidable per-decode cost.
- 2 global loads of `words[...]` (L57), a 64-bit `<<32|`, 64-bit `>>shift`, `&65535` → **~5 ops + 2 loads**.

**`mul1` (exl3_common.cuh:31-38)**: `mul` 0x83DCD12D, 4 byte-extracts + 3 adds, `ushort_as_half`, `hfma`, `half2float` → **~10 ops**.

Total **≈ 30-45 instructions + 2 global loads per decoded weight**, of which ~13 are pure address permutation and ~20 are the runtime-modulo. Then `pair_bits` (exl3_mma.cuh:8-11) does `__float2half_rn` twice per pair and an `or`/`shl` — a **f32→f16→f32 round trip of every decoded value**.

**Memory vs L1 vs shared.** The native path reads `packed` (the shard’s trellis, a global pointer passed as `trellis`, native.rs matmul arg at :380; exl3_moe_mma.cu:45) with **no explicit staging** — each decode is a raw global load served by L1/L2. Within one row/CTA a weight is read once; the redundancy is entirely **inter-CTA (across rows)**. The vendored coop path by contrast stages the B tile into shared memory (`sh_b`, exllamav3_gemm_inner.cuh:246, 476-486) and decodes 8 weights at a time with `dq8_aligned_4bits` (exllamav3_dq.cuh:230-249): 2 loads + 1 `funnel` + 7 `BFE16` + 4 `decode_3inst_2` (each ~5 ops incl. `dp4a`, exllamav3_codebook.cuh:71-80) ≈ **~34 instructions / 8 weights ≈ 4.3 ops per weight** — **~8-10× fewer instructions per weight than the native `tile_state` path**, and no runtime modulo.

**Concrete fixes (ranked, all bit-preserving):**
1. **Stage the trellis tile in shared memory and decode cooperatively** — lift the coop `dq*` path (or a warp-cooperative decode using `__shfl` on `tile_state` operands) into `exl3_mma.cuh`. Effect: per-weight **30-45 → ~4-5 instructions** (≈8×), and the 2 global loads become 1 shared load per 8 weights. This is the single largest instruction win *within* the M=1 shape.
2. **Precompute the `count=8*bits` modulo**: hoist `8*bits` and replace `x % count` with a compile-time-templated `bits` (instantiate bits∈{4,5,6}) so the divisor is a constant → removes the ~20-instruction modulo per decode.
3. **Remove the f32→f16 round trip**: keep decoded values as `__half2`/`half` and feed `mma_step` directly (the coop `dq` path already does this via `FragB`).
4. **Reorder for reuse**: decode a `(k16, n8)` B fragment once per warp and reuse across M rows (this is the M-tiling change, §2).

Expected bytes/instruction change: (1)+(2) raise decoded-weights/instruction ~8×; combined with M=16+ tiling the **bytes moved per decode rises 16× (rows) on top**.

---

## 4. Can `exllamav3_gemm_inner.cuh` support larger M?

`static_assert(TILESIZE_M == 16, "Invalid kernel params")` at **exllamav3_gemm_inner.cuh:238**. Today it is genuinely M=16-only:

- `TILEBLOCKS_M = TILESIZE_M/16` (L221) exists but `frag_a[FRAG_STAGES]` (L402) is **single-M-block** and the `load_frags` loop `for (int m = 0; m < TILEBLOCKS_M; ++m)` (L527-534) **repeatedly overwrites `frag_a[buf]`**; the disabled `// const int FRAGS_M = TILEBLOCKS_M;` (L224) marks where upstream had it. `matmul()` (L945-948) consumes a single `frag_a[buf]`, so `TILEBLOCKS_M>1` is not wired.
- The chunk loop `exl3_coop_gemm_body` (exl3_coop_gemm.cu:93-116) hard-codes `MIN(size_m_,16)` and `chunk*16*size_k/size_n`.

**What must change for M_tile ∈ {32,64}` (keeping bit-identity):**
- Array the fragments: `frag_a[M_BLOCKS][FRAG_STAGES]`, `frag_b[M_BLOCKS][FRAG_STAGES][FRAGS_N_PER_WARP]`, `frag_c[M_BLOCKS][FRAGS_N_PER_WARP]`; loop `m` in `load_frags`/`matmul`/`clear_frag_c`/`threadblock_reduce` (L527, L945, L556, L610) instead of overwriting.
- `gl_a_stride_m = TILESIZE_M*size_k` (L323) already scales; extend `read_sum_gl`/`write_sum_gl` (L736, L808) `r0,r1` coverage over all M blocks (they only touch rows 0-15 today).
- **Shared-memory staging plan vs `SMEM_MAX`**: `SMEM_MAX = 90*1024` (exllamav3_kernel_map.cuh:last line; `COOP_SMEM_BYTES = 90*1024` coop.rs:29). `sh_a_stage_size = TILESIZE_M*TILESIZE_K` halfs (L226) **grows linearly with M** and is multiplied by `SH_STAGES` in the static_assert (L241-244). For tile `(32,256)` the A share at M=64 quadruples vs M=16, so the variant must reduce `SH_STAGES`/`FRAG_STAGES` or use `TILESIZE_N=128`. Recommend a dedicated instance `TILESIZE_M=64, TILESIZE_K=16, TILESIZE_N=128, SH_STAGES=4, FRAG_STAGES=2` and let the existing static_assert (L241) prove the budget.
- **Split-K fold & locks**: unchanged and M-independent — the lock row is one int per 128-column tile (`lock_stride = size_n/128`, exl3_coop_gemm.cu:96; `locks + chunk*lock_stride` L105) and `lock_i/lock_d` (L1010-1013 in the inner, and moe10 L1200) depend only on `tiles_k`, not M. A chunk = one M-block still self-resets its own lock row.
- **Bit-identity**: per-row MMA is independent, the slice partition is `logical_tiles = tiles_k*logical_tiles_n` (L281-283) — independent of M — and the fold order (`seed_sum_zero` L829, `read_sum_gl` L790, `round_partial` L836) is per-column, so widening M does not change any row’s value sequence. The only M-dependent branch is `size_m<=8` vs `>8` in `threadblock_reduce` (L664/715), which selects *which lanes store*, not the arithmetic.

**Gate:** new kernel names + an explicit profile (`ATLAS_EXL3_GEMM_M=64`), default-off, with a bit-identical-output test vs the M=16 coop kernel for the same geometry (the brief’s P2 acceptance).

---

## 5. Fat-vs-thin numerics split

`exl3_moe_mul1_matmul_precision_mma` computes `fat = counts[expert] > thin_cap` (exl3_moe_mma.cu:48) and selects `selected_k/n/slices/half` (L49-52):
- **fat** (`counts>thin_cap`): `fat_tile_k=16, fat_tile_n=128, fat_slices=1, fat_half_partial=0` → sequential K16 MMA, **unrounded F32** accumulation.
- **thin**: `tile_k=32, tile_n=256, slices=8, half_partial=1` → after **every** k-fold partial the running total is rounded: `if (half_partial) total[j] = __half2float(__float2half_rn(total[j]))` (exl3_mma.cuh:71).

The host policy is bound in `moe/policy.rs:17-42` (`GLM53_TP2_REFERENCE`, validated to exactly `(32,256,8,true)` thin / `(16,128,1,false)` fat, L44-56) and passed as `matmul_args()` (L58-67) → project.rs:70-72.

**Grouped multi-row fat must preserve this per row.** The coop inner already has the exact knob: template `round_partial` (header note 7, exllamav3_gemm_inner.cuh:836-846) applies `f32(f16_rn(total))` after every fold add — set it **false** for fat. The multirow launcher `exl3_coop_moe8_multirow_b4_k32_n256` uses `EXL3_COOP_MOE8_SHAPE = 16,32,256,6,FR` and `round_partial=true` (template arg list at exl3_coop_gemm.cu:930-935 → `<4,true,2,shape,false,true,...,true>` = round_partial true), i.e. **thin only**, and explicitly poisons `rows>16` and `counts>thin_cap` (L906). To do fat multi-row:
- Instantiate the inner with tile `(16,128)`, `slices=1`, `round_partial=false`, `gather=true` (note 15), one CTA per expert group; chunk groups by M_tile.
- The fold for `slices=1` is `seed_sum_zero()` (`__fadd_rn(·,0.0f)`, L829) then `write_sum_gl` (L913) — the exact analogue of native’s `total[j]=0; total[j]=__fadd_rn(sum,total[j])` (exl3_mma.cuh:43,70) with `half_partial=0`. Header note 4 (L58-62) documents that this only maps −0.0→+0.0.
- Reachability: remove/replace the early return at **coop.rs:467-469** (`if rows > thin_capacity { warn; return Ok(false) }`) for fat launches only, and replace the `rows>16` poison at exl3_coop_gemm.cu:906 with M-tile chunking. Rust already groups slots via `exl3_moe_group_slots` (coop.rs:740-748) and passes `row_slots` (L910).

Cite: exl3_moe_mma.cu:48-52; exl3_mma.cuh:70-71; exllamav3_gemm_inner.cuh:836-846 (round_partial) and :790-913 (fold); coop.rs:467-469, 740-748; exl3_coop_gemm.cu:906.

---

## 6. Redundant work

- **Repeated Hadamard launches**: per MoE layer the routed chain issues separate `exl3_moe_input_hadamard_f16` for gate **and** up (project.rs:33 via forward.rs gate/up projections), two output Hadamards (`exl3_moe_output_hadamard_precision`, project.rs:88), `exl3_moe_activate_down_hadamard` (project.rs:110), plus `exl3_moe_down_weighted_output` (project.rs:88 for index 2). The fused v7/v9 tiers (`exl3_coop_moe7`/`moe9`, coop.rs:666/947; `coop_routed_v9` coop.rs:886-985) collapse gate|up+down transform+activation into 3 launches — **already implemented but off in the measured env** (brief env sets V2/V3/V6/V7 but not V9).
- **f32_to_f16 round trips**: `exl3_f32_to_f16` (exl3_coop_gemm.cu:826) converts the down activation scratch F32→F16 on the down path (coop.rs:494-506); `pair_bits` (exl3_mma.cuh:8-11) rounds every decoded f32 back to f16; the native input transform stores F16-representable values *widened into F32* scratch (native.rs doc L7-9).
- **Lock/atomic protocols**: coop thin kernels use per-(projection,slot) lock rows and cooperative co-resident grids (exl3_coop_moe_thin L180-187; moe10 `barrier_acquire/release` L1216-1231); the multirow kernel uses a **fixed** `occupancy·SMs` grid with a device-read item count (coop.rs:62-67, exl3_coop_gemm.cu:849). The native MMA path has no locks but pays the redundant grid instead.
- **Oversized grid**: dense `(ceil(n/128), rows, 1)` (native.rs:378) and MoE `(ceil(n/128), rows·topk, 1)` (project.rs:33) launch one CTA per row while 93.75 % of the M dim is padding; grid.y reaches 65535 (native.rs:130). The decode+load is repeated `rows`/`rows·topk` times.

---

## 7. Ranked proposals by expected prefill win

| # | change | expected win | evidence / gate |
|---|---|---|---|
| **P0** | **Grouped multi-row *fat* MoE with real M-tile (16-64)** — extend `exl3_coop_moe8_multirow` to `round_partial=false, tile(16,128), slices=1, gather`, chunk groups whose count exceeds M_tile instead of poisoning; remove the `rows>thin_cap` early return at coop.rs:467 for fat. | Weight reads `rows·topk → Σ_e ceil(count_e/M_tile)` ≈ #touched experts (~16× at M=16). Targets the **51.7 %** kernel. | exl3_coop_gemm.cu:906, 930-935; exllamav3_gemm_inner.cuh:836-846; coop.rs:467-469, 740-748. Gate: `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW`, bit-identical per-row F32 vs native fat, measured TTFT ≥3 repeats. |
| **P1** | **Dense coop GEMM with TILESIZE_M > 16** (add M-arrayed fragments, chunk loop by M_tile, dedicated `(64,16,128,4,2)` instance). | Dense weight reads `ceil(rows/16) → ceil(rows/64)` (4× over current coop, 64× over native M1). Targets **45.1 %**. | exllamav3_gemm_inner.cuh:238, 402, 527, 945, 241-244; exl3_coop_gemm.cu:93-116. Gate: default-off profile + bit-identity test. |
| **P1** | **Replace native `tile_state`/`mul1` with the staged `dq8` decode (shared B tile + BFE/funnel, no runtime modulo).** | ~**8-10× fewer instructions per decoded weight** (30-45 → ~4.3), removes the per-decode integer modulo. Applies to both hot kernels. | exl3_common.cuh:47-59, 31-38 vs exllamav3_dq.cuh:230-249, exllamav3_codebook.cuh:71-80; exllamav3_gemm_inner.cuh:476-486. Bit-identity: `dq` is the mul1 codebook (`decode_mul1_product_2`, codebook L49-63). |
| **P2** | **Enable v9 fused routed chain (or at least v7 fused input) by default** to delete per-layer Hadamard/activation launches. | 3 fewer launches/layer, identical values. | coop.rs:666, 886-985; exllamav3_gemm_inner.cuh notes 13/16. |
| **P2** | **Drop f32→f16 round trips**: keep decoded values half, and fuse `exl3_f32_to_f16` (down) into the producer. | fewer instructions + one fewer launch. | exl3_mma.cuh:8-11, 58-61; exl3_coop_gemm.cu:826; coop.rs:494-506. |
| **P2** | **Shrink the native grid**: since M-tiling removes the per-row CTA need, fold `grid.y=rows` into a persistent/chunked form (the coop kernels already loop rows internally). | removes CTA-launch and re-decode overhead. | native.rs:378; project.rs:33; exl3_coop_gemm.cu:93-116. |

**Numerics guardrail (all rows):** none of the above may change per-row output. M-tiling is per-row-independent (exl3_mma.cuh:29-31; exllamav3_gemm_inner.cuh note 15). Fat/tile changes must be behind default-off flags (`ATLAS_EXL3_COOP_MOE_FAT_MULTIROW`, `ATLAS_EXL3_GEMM_M`) with a bit-identity test; the thin path already equals the coop fold (round_partial=true ↔ half_partial=1).

---

## Review
- **Correct:** Entry points fully enumerated; numeric split contracts (thin `half_partial=1` at exl3_mma.cuh:71 vs fat `fat_half_partial=0` at exl3_moe_mma.cu:52) confirmed; M=1 padding (93.75 %), decode cost, and the 90 KiB SMEM cap all traced to source; coop thin kernels (0.2 %) and the reachability block (coop.rs:467) match the measured trace.
- **Finding: P0** — MoE fat kernels (`exl3_moe_mul1_matmul_precision_mma`, 51.7 %) run one CTA per slot with 15/16 M padding and no weight reuse (exl3_moe_mma.cu:12, project.rs:33). Smallest fix: enable a fat multi-row coop variant (P0 above).
- **Finding: P1** — dense `exl3_mul1_matmul_mma` (45.1 %) has the same M=1 structure (exl3_linear_mma.cu:8, native.rs:378); coop M is capped at 16 (exllamav3_gemm_inner.cuh:238).
- **Finding: P1** — native decode uses a runtime-`bits` integer modulo per weight (exl3_common.cuh:53-58) and ~30-45 instructions/weight vs ~4.3 in the vendored `dq` path.
- **Merge verdict:** OK with notes (read-only audit; no edits made).