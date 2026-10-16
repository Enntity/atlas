# Prefill performance investigation — 2026-10-16 (branch perf/decode-vllm-parity-20261015)

## Measured baseline (unprofiled, ab_full.sh flags)
- `ab_prose_long` (108 tok in / 1024 out), p12v10 run 2026-10-16T05:08Z:
  - first_output_token_s = **6.08 s** (real prefill+first token), ttft_s (harness) = 9.20 s
  - decoded at 21.5 tok/s; inter-token gap 0.047 s
- `ab_prose_prefill` (1024 tok in / 16 out): baseline run p13base in flight.

## Prefill cost is ~linear in prompt tokens (no amortization)
From p0 traces (2026-10-15), instrumented:
| prompt tokens | TTFT (s) |
|---|---|
| 108 | 11.8 |
| 2573 | 272.3 |
Marginal = (272.3-11.8)/(2573-108) = **0.1057 s/token ≈ 9.5 tok/s**, intercept ~0.4 s.
So prefill ≈ 0.106 s × n_tokens. A batched prefill should be ~constant + small.

## Kernel attribution (p0-short-20261015 rank0, 108-token prefill, busy 11.25 s)
| kernel | time | share |
|---|---|---|
| `exl3_moe_mul1_matmul_precision_mma` | 6.035 s | 53.6% |
| `exl3_mul1_matmul_mma` | 4.644 s | 41.3% |
| `ncclDevKernel_SendRecv` | 0.346 s | 3.1% |
| `grouped_gemm_mla` | 0.0497 s | 0.4% |
| everything else | <0.1 s each | ~1% |
**95% of prefill is two packed-matmul kernels.**

## Root cause: M=1 per block, weight decode not amortized
`kernels/gb10/common/exl3_mma.cuh:29-31`:
> `// All warp lanes participate. One actual row is padded to M16; only lane<4 owns that row's C0/C1 outputs.`

- Both hot kernels launch `grid.y = rows` (or `rows*topk`) — **one CTA per token row**.
- Each CTA decodes the full B fragment (`decoded()` -> `tile_state()` EXL3 trellis decode,
  4 calls per MMA step) for its whole K×N slice, then does an `m16n8k16` MMA in which
  **only 1 of 16 rows is real**.
- Consequence: the expensive trellis decode AND the weight load are repeated once per row,
  while 15/16 of the tensor-core work is padding. Arithmetic intensity does not improve with M.
- The vendored cooperative GEMM (`exl3_coop_gemm_body`, exl3_coop_gemm.cu) already processes
  rows in **16-row chunks** (`CHUNK_ROWS = 16`) and is used for decode/thin paths.

## Why prefill does not take the coop path
`crates/spark-model/src/layers/glm5_exl3_ffn/moe/coop.rs:467`:
```
if rows > self.precision.thin_capacity as usize {
    warn_once("fat (prefill) routed launch");
    return Ok(false);          // -> scalar/MMAv1 per-slot kernel
}
```
`thin_capacity = 32` (moe/policy.rs:20). Confirmed at runtime in the p12v10 container log:
`ATLAS_EXL3_COOP_MOE: fat (prefill) routed launch; using the existing routed EXL3 matmul`.
Numerical reason: thin slots round the partial to FP16; fat slots (>thin_cap rows per expert)
must accumulate unrounded F32 (`exl3_moe_mma.cu`: `fat_tile_k=16, fat_tile_n=128, fat_slices=1, fat_half_partial=0`).

## Hypothesis to validate
Batching the M dimension (16 rows per CTA over the same expert, or per dense weight) amortizes
the trellis decode ~16x and should speed prefill roughly an order of magnitude, without changing
the F32-accumulation numerics of the fat profile (the M-tile rows are independent in the MMA,
exactly as `accumulate_logical` already assumes).
This is consistent with todo.md B open item: "Port the multirow <=16-rows cap (`b030781f`) ...
Above 16 rows per expert in prefill the kernel poisons rows and output degenerates."

## Next measurements
1. Current-stack nsys trace at 108 and 1024 prompt tokens (trace_driver_v2.py) to confirm the
   kernel shares under today's flags, and to measure achieved DRAM bandwidth per kernel.
2. A/B: chunked prefill at <=32 tokens (routes MoE through coop) vs default, to bound the win
   available from the existing code before writing new kernels.

## CORRECTION + current-stack attribution (trace 1, 2026-10-16)

Trace 1 (`p13-trace-20261016/p13trace`, 1024-token prefill + 16 decode, rank 0) ran the
**collector-only descriptor env**, which does NOT include the `ab_full.sh` base flags. It is
therefore the *fallback* stack (no `ATLAS_EXL3_COOP_GEMM`, no `SPLIT_MMA`, no `RDMA_ALLREDUCE`).
`plan.json` confirms: the descriptor env has no `ATLAS_EXL3_COOP_GEMM`. Trace 2
(`p13trace2`) re-runs with the exact baseline env.

Trace 1 kernel attribution (rank 0, total GPU kernel time ≈ 84.9 s):

| kernel | total | % | instances | avg | median |
|---|---|---|---|---|---|
| `exl3_moe_mul1_matmul_precision_mma` | 43.90 s | 51.7 | 756 | 58.07 ms | 58.43 ms |
| `exl3_mul1_matmul_mma` | 38.30 s | 45.1 | 3985 | 9.61 ms | 1.57 ms |
| `exl3_mhc_pre_norm` | 0.87 s | 1.0 | 1890 | 0.46 ms | — |
| `grouped_gemm_mla` | 0.36 s | 0.4 | 132 | 2.76 ms | — |
| `exl3_coop_moe7_thin_b4_k32_n256` | 0.16 s | 0.2 | 630 | 0.26 ms | — |
| everything else | <0.15 s each | ~1.6 | | | |

- **96.8 % of GPU time is the two M=1-per-row packed matmuls**; this holds in both the fallback
  and (expected) the baseline stack.
- No dense `exl3_coop_gemm_b*` kernel executed in trace 1 (the coop module's only executed
  kernels were the thin-MoE variants). That is expected without `COOP_GEMM=1`, and is what
  trace 2 will re-check.
- KDA/SSM prefill (`kda_fla_*`) is ≈0.14 % of GPU time; MLA/indexer ≈0.5 %; mHC ≈1 %.
  **All further prefill effort belongs on the EXL3 packed matmuls.**
- Peak achieved DRAM throughput seen anywhere in the trace: ~270 GB/s on a 2 MB memset
  (GB10 LPDDR5X class). The hot matmuls are instruction-bound on EXL3 trellis decode, not
  bandwidth-bound: 43.9 s for ~0.3 s of weight traffic at that rate.

## Independent kernel audit (subagent, read-only, 2026-10-16)

Full report: `p13-kernel-audit.md` (workflow 67382504, child dc8bd541). Key results:

**Why prefill is instruction-bound, not bandwidth-bound.** The native decode
`mul1(tile_state(words, i, j, bits))` in `kernels/gb10/common/exl3_common.cuh:31-59` costs
**≈30-45 instructions + 2 global loads per decoded weight**, including a **runtime integer
modulo** (`x % (8*bits)` with `bits` a runtime argument, so the divisor is not a compile-time
power of two → ~10-15 instructions each, twice). The vendored coop path
(`exllamav3_dq.cuh:230-249`, `exllamav3_codebook.cuh:71-80`) decodes **≈4.3 instructions per
weight** from a shared-memory staged tile (2 loads + 1 funnel + 7 BFE16 + 4 `decode_3inst_2`) —
**8-10x fewer**. That, not M-tiling, is the single largest instruction win inside the current
M=1 shape.

**M=1 padding is 93.75 % waste.** `mma_step` issues `m16n8k16` (2048 MACs) but
`accumulate_logical` writes A only for `lane<4` → 1 real row of 16, 4 of 32 lanes storing
(`exl3_mma.cuh:29-31`, `exl3_linear_mma.cu:41-45`, `exl3_moe_mma.cu:56-59`). Bytes per useful
FLOP = `b/16`, independent of M — no amortization by construction. grid.y = rows (dense,
`native.rs:378`) or rows*topk (MoE, `project.rs:33`).

**M-tile recovery (weight reads per prompt of R tokens):** M=1: R passes; M=16: R/16; M=64: R/64;
M=128: R/128.

**Larger M in the vendored inner is a bounded change.** `static_assert(TILESIZE_M == 16)` at
`exllamav3_gemm_inner.cuh:238`; the M-block scaling plumbing exists but is unwired
(`TILEBLOCKS_M` L221, `frag_a` overwritten in `load_frags` L527-534, `matmul` L945-948 consumes
one M-block, `read_sum_gl`/`write_sum_gl` L736/L808 only cover rows 0-15). Needs arrayed
`frag_a/frag_b/frag_c` over M blocks, M loop in load/matmul/reduce, and a shared-memory plan
fitted to `SMEM_MAX = 92160 B` (`sh_a_stage_size` grows linearly with M). Split-K fold and the
lock protocol are M-independent, and per-row arithmetic is independent, so bit-identity is
preservable. Suggested first instance: `(M=64, K=16, N=128, SH=4, FR=2)`.

**Fat multi-row MoE (the 51.7 % kernel) is a small delta from existing code.**
`exl3_coop_moe8_multirow_b4_k32_n256` (`exl3_coop_gemm.cu:847`) already groups slots by expert
and calls the GEMM with `size_m = group_rows`, but hard-poisons `rows>16` (L906) and
`counts>thin_cap`, and is unreachable because `moe/coop.rs:467-469` returns early when
`rows > thin_capacity`. For fat: instantiate the inner with tile `(16,128)`, `slices=1`,
`round_partial=false` (the exact knob is documented at `exllamav3_gemm_inner.cuh:836-846`; the
thin mapping `round_partial=true` ↔ `half_partial=1` is `exl3_mma.cuh:70-71`), `gather=true`,
one CTA per expert group, and chunk groups by M_tile instead of poisoning.

**Ranked proposals (audit §7):**
| # | change | expected | target |
|---|---|---|---|
| P0 | grouped multi-row **fat** MoE with M-tile 16-64; remove the `rows>thin_cap` early return for fat | weight reads `rows*topk → Σ_e ceil(count_e/M_tile)` | 51.7 % kernel |
| P1a | dense coop `TILESIZE_M > 16` (arrayed fragments, M loop, `(64,16,128,4,2)` instance) | `ceil(rows/16) → ceil(rows/64)` on coop | 45.1 % kernel |
| P1b | replace native `tile_state`/`mul1` with the staged `dq8` decode (no runtime modulo) | ~8-10x fewer instructions/weight | both hot kernels |
| P2 | enable the v9 fused routed chain; drop f32→f16 round trips; fold `grid.y=rows` into chunked form | fewer launches / round trips | ~3 % |

All of P0/P1a/P1b can preserve per-row numerics: the MMA rows are independent
(`exl3_mma.cuh:29-31`) and the split-K fold is per-column. P1b changes the decode code path, so
it needs a bit-identity test against `exl3_mma.cuh` on the same inputs before any flag is enabled.

## DECISIVE: trace 2 with the exact baseline env (2026-10-16)

`p13trace2-20261016/p13trace2/spark0/nsys-stats.txt`, 1024-token prefill + 16 decode, rank 0,
env = ab_full.sh base flags + V2/V3/V6/V7 (i.e. what p13base measured). Total GPU kernel time 43.44 s.

| kernel | time | % | instances |
|---|---|---|---|
| `exl3_moe_mul1_matmul_precision_mma_split` | **40.845 s** | **93.8** | 756 |
| `grouped_gemm_mla` | 0.363 s | 0.8 | 132 |
| `exl3_mhc_pre_norm` | 0.338 s | 0.8 | 540 |
| `exl3_coop_gemm3_b6_k32_n256` | 0.298 s | 0.7 | 204 |
| `exl3_coop_gemm_b6_k32_n128` | 0.238 s | 0.5 | 814 |
| `exl3_coop_moe7_thin_b4_k32_n256` | 0.168 s | 0.4 | 630 |
| every remaining kernel | <0.15 s each | ~3 | |

Two conclusions that redirect all further work:

1. **The dense side is already solved.** With `COOP_GEMM=1`, `exl3_mul1_matmul_mma` does not run at
   all; the dense linears run the vendored cooperative GEMM (`exl3_coop_gemm_b6_k32_n128/n256`,
   `exl3_coop_gemm3_b6_*`) which loops M in 16-row passes and uses the fast `dq8` decode. Dense is
   now under 2 % of prefill. The audit's P1a/P2 items are therefore not worth doing.
2. **One kernel is 93.8 % of prefill:** `exl3_moe_mul1_matmul_precision_mma_split`, the routed-MoE
   packed matmul that still runs one CTA per (row, expert) slot with the native M=1
   `tile_state`/`mul1` decode (~30-45 instructions + a runtime integer modulo per weight).
   It is the ONLY thing worth optimizing for prefill.

Why this kernel is ~14x slower than it should be, from the audit:
- decode: ~30-45 instr/weight here vs ~4.3 instr/weight in the vendored `dq8` path the coop GEMM
  uses (exllamav3_dq.cuh:230-249) -> ~8x;
- M reuse: one real row per m16 MMA and one CTA per slot, so the decode+load repeats per slot
  (grid.y = rows*topk) -> a further ~3.5x available from grouping slots per expert
  (`Sigma_e ceil(count_e/M_tile)` vs `rows*topk`).
Both fixes land together by routing fat prefill MoE through the vendored coop inner with
`round_partial=false`, tile (16,128), slices=1, gather=true, and chunking groups by M tile
(the audit's P0). Expected prefill win on 93.8 % of the time: ~8-14x, i.e. the 17.8 tok/s
baseline should rise into the 100+ tok/s range if the change lands bit-identically.

Executing that implementation on branch `followup/p13-fat-moe-multirow`.

## Implementation landed (branch followup/p13-fat-moe-multirow, commit b52aef80)

Subagent worker (e20f0ca2, timed out at 30 min before committing; the artifacts were complete, so
they were reviewed and committed by the parent).

What it is: `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1` (default off) routes prefill routed-MoE launches
(`rows > thin_capacity`) through a new grouped multi-row path instead of the per-slot
`exl3_moe_mul1_matmul_precision_mma`:
- new `exl3_moe_group_units` splits slots per accumulation class into M-tile units of <= 16 rows;
- thin units (`counts[e] <= thin_capacity`) -> existing `exl3_coop_moe8_multirow_b4_k32_n256`
  (tile 32x256, 8 slices, F16-rounded fold);
- fat units (`counts[e] > thin_capacity`) -> new `exl3_coop_moe_fat_multirow_b4_k16_n128`
  (tile 16x128, 1 slice, unrounded F32 fold).
Files: `crates/spark-model/src/layers/glm5_exl3_ffn/moe/coop_fat.rs` (new, 389 LoC),
`moe/{moe.rs,forward.rs,setup.rs,coop.rs}`, `kernels/gb10/common/exl3_coop_gemm.cu` (+219) and its
`glm-5.3-flash-exl3` mirror, plus `tests/exl3_validation/coop_moe_bench/{fat_multirow_bench.cu,probe_identity.cu}`.
LoC cap satisfied (389). SPDX header present.

Standalone gate `tests/exl3_validation/coop_moe_bench/fat_multirow_bench.cu`
(rows 128, topk 8, slots 1024, 288 experts, thin_cap 32, iters 50, 2 complete runs, 0 failures):

| routing | gate\|up before -> after | speedup | down before -> after | speedup | per-slot byte compare |
|---|---|---|---|---|---|
| uniform (279 thin / 0 fat) | 111.1 ms -> 5.46 ms | 20.4x | 111.3 ms -> 5.47 ms | 20.4x | IDENTICAL |
| skewed (87 thin / 3 fat) | 111.4 ms -> 1.63 ms | 68.2x | 111.7 ms -> 1.59 ms | 70.2x | IDENTICAL |
| all-fat (0 thin / 64 fat) | 121.8 ms -> 1.97 ms | 62.0x | 121.6 ms -> 1.92 ms | 63.3x | IDENTICAL |

- "PRE-EXISTING ENTRIES WITH CHANGED PTX: 0" — the change is purely additive; existing kernels'
  PTX is byte-unchanged, so with the flag off the stack is untouched.
- Lock rows return to zero after every launch (self-resetting protocol intact).
- Coverage includes counts below and above thin_cap and groups spanning several M tiles.

Also found (latent, pre-existing): `ATLAS_EXL3_COOP_MOE_MULTIROW` sizes its lock buffer at
`min(capacity, thin_capacity) * topk` rows but indexes up to `2 * slots`, so it can run out of
range when nearly every slot draws a distinct expert. The new path allocates `2 * slots` and is
unaffected. Recorded as a narrow follow-up in todo.md.

Owed: model-level A/B (parent) — the same 3-repeat `ab_prose_prefill`/`ab_prose_long` on
`spark.p13fat` with the flag off (neutrality) and on (effect), comparing `first_output_token_s`,
`tpot_s` and the three `output_sha256` values.

## VALIDATED RESULT: 12.4x prefill, bit-identical (2026-10-16)

Same binary `spark.p13fat` (sha256 22875833607658019bebef3a998fb79e14d14cbe65c7dbe1c9c2cd375c32e378)
on both nodes, `ab_full.sh` base env, 3 repeats per workload, bench lock held, no other GPU work.

`first_output_token_s` (real prompt-processing + first token):

| run (binary / flag) | 761-token prompt | 108-token prompt |
|---|---|---|
| p13base (shared binary, flag off) | 42.62 / 42.84 / 42.71 s | 6.07 / 6.07 / 6.06 s |
| p13fat-off (new binary, flag off)  | 42.25 / 42.72 / 42.72 s | 6.078 / 6.085 / 6.073 s |
| **p13fat-on (new binary, flag on)** | **3.427 / 3.410 / 3.422 s** | **0.541 / 0.637 / 0.540 s** |

- **12.4x faster at 761 tokens** (42.6 -> 3.42 s), **11.2x at 108 tokens** (6.07 -> 0.54 s).
- Prefill rate **17.8 -> 222 tok/s** at 761 tokens.
- `ab_prose_prefill` end-to-end 43.5 -> 4.2 s; `ab_prose_long` 51.3 -> 45.8 s (decode unchanged,
  TPOT p50 0.0457 vs 0.0458).
- **`output_sha256` identical flag-off and flag-on** (`1c4328172e81` prefill, `80b1eb5fadf8` long —
  the documented exact-stack hash), so the change is **bit-exact at model level**, not merely
  "within tolerance".
- Engagement proved: flag-off logs one `ATLAS_EXL3_COOP_MOE: fat (prefill) routed launch` fallback;
  flag-on logs **zero** (the grouped path served every prefill launch).
- Neutrality with the flag off: p13fat-off matches p13base within run-to-run variation
  (42.25-42.72 vs 42.62-42.84) and produces the same hashes.

Conclusion: the measured 93.8%-of-prefill kernel was routed through the grouped multi-row path
with per-row bit-identity, and prefill improved by an order of magnitude. The remaining prefill
time (3.42 s for 761 tokens) is now spread across the rest of the stack rather than one kernel.

## Trace 3: attribution AFTER the fix (2026-10-16)

`p13trace3-20261016/p13trace3/spark0/nsys-stats.txt`, same request as trace 2 (1024-token prefill +
16 decode, rank 0), same env plus `ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1`, binary `spark.p13fat`.

**Total GPU kernel time: 43.44 s -> 4.09 s (10.6x).** The old hotspot
(`exl3_moe_mul1_matmul_precision_mma_split`, 40.85 s) does not appear at all.

| kernel | time | % | instances |
|---|---|---|---|
| `exl3_coop_moe8_multirow_b4_k32_n256` (thin grouped) | 1.358 s | 32.4 | 504 |
| `grouped_gemm_mla` | 0.359 s | 8.6 | 132 |
| `exl3_mhc_pre_norm` | 0.339 s | 8.1 | 540 |
| `exl3_coop_gemm3_b6_k32_n256` (fused qkv) | 0.298 s | 7.1 | 204 |
| `exl3_coop_gemm_b6_k32_n128` (dense) | 0.237 s | 5.7 | 814 |
| `exl3_coop_moe7_thin_b4_k32_n256` (decode thin) | 0.167 s | 4.0 | 630 |
| cublas `gemvx` | 0.136 s | 3.3 | 1260 |
| `exl3_coop_moe_fat_multirow_b4_k16_n128` (new fat) | 0.117 s | 2.8 | 504 |
| `exl3_coop_gemm3_b6_k16_n512` (fused qkv) | 0.090 s | 2.2 | 510 |
| `dense_gemm_tc` | 0.090 s | 2.1 | 132 |
| `exl3_moe_input_hadamard_f16h` | 0.068 s | 1.6 | 504 |
| `exl3_moe_group_units` (new) | 0.061 s | 1.5 | 252 |
| `exl3_moe_down_weighted_output` | 0.051 s | 1.2 | 882 |
| rest | ~0.9 s | ~22 | |

Reading of the new profile:
- The cost is now spread; no single kernel dominates.
- The new #1 is the **thin** grouped kernel (32.4 %). Its numeraics contract is expensive: tile
  (32,256), 8 split-K slices with the running fold rounded to F16 after every slice, plus a
  lock-ordered fold. Its cost scales with the number of *units*, and at a 128-token chunk each
  unit holds only ~3.6 rows (1024 slots / 288 experts), so the 16-row M tile is under-filled.
- Consequence: **larger prefill chunks should now help**, which they could not before the change
  (the old per-slot kernel was O(slots) and therefore chunk-independent). A 761-token chunk gives
  ~21 rows/expert instead of 3.6, so the same rows need ~3x fewer unit passes; launch counts also
  drop ~6x.
- Next target after that: `grouped_gemm_mla` (8.6 %, MLA absorption), `exl3_mhc_pre_norm` (8.1 %),
  and the dense coop pair (~13 % combined).

## VOID run (must not be cited): first chunk-size A/B, 2026-10-16

`p13chunk{128,512,4096}` (runs 07:38/07:43/07:48Z) all reported
`chunk_size=128` / `max_prefill_tokens=128` in the container logs and in `meta.json`, i.e. the
`ATLAS_BENCH_DESCRIPTORS` override never reached `bench_tp2.py`. Cause: `ab_full.sh` does
`cd "$(dirname "$0")/../.."`, so invoking it from the `atlas-p13-fatmoe` worktree ran that
worktree's copy of `bench_tp2.py`, which does not carry the override patch (the patch lives in
`atlas-perf`). Results (3.418/3.429/3.538, 3.428/3.426/3.433, 3.533/3.456/3.524 s) are therefore
three repeats of the SAME 128-token configuration and carry no chunk-size information. The
apparent "no gain" is an artifact of the void run, not a finding. Re-run in flight after patching
the fatmoe worktree.

## FINAL VALIDATED (2026-10-16): 42.71 s -> 2.72 s prefill, 15.7x, bit-identical

Median of 3 repeats, `ab_full.sh` env, bench lock held, no other GPU work.

| config | 761-token prompt | 108-token prompt | prefill rate |
|---|---|---|---|
| p13base (shared binary) | 42.711 s | 6.068 s | 17.8 tok/s |
| p13fat-off (new binary, flag off) — control | 42.720 s | 6.078 s | 17.8 tok/s |
| p13fat-on (flag on, chunk 128) | 3.422 s | 0.541 s | 222 tok/s |
| **p13fat-on + `--max-prefill-tokens 512`** | **2.724 s** | — | **279 tok/s** |

- **15.7x faster prefill** than the pre-change stack; 12.4x from the kernel change alone and a
  further 1.26x from the larger chunk.
- `output_sha256` identical between flag-off and flag-on (`1c4328172e81`, `80b1eb5fadf8`):
  bit-exact, so this is a pure speed change.
- The chunk-512 arm needed a second attempt: the first `ATLAS_BENCH_DESCRIPTORS` A/B was void
  because `ab_full.sh` resolves its repo root from `$0`, so it ran the `atlas-p13-fatmoe` worktree's
  unpatched `bench_tp2.py` and silently kept chunk 128 in all three arms. Fixed, then re-run;
  `chunk_size=512` and `max_prefill_tokens=512` are confirmed in the container logs.
- `--max-prefill-tokens 4096` **crashes**: `EP peer lost` after layer construction left only
  6.05 GB free. So 512 is the largest chunk that fits the current `--gpu-memory-utilization 0.84`
  budget; bigger chunks need a memory plan (scratch scales with capacity) before they are usable.
- Why the chunk now matters (it did not before): the grouped route's cost scales with the number of
  `(expert, <=16-row unit)` passes, and a bigger chunk packs more rows per expert per unit
  (761 tokens one chunk = ~21 rows/expert vs 3.6 at 128), so the same rows need fewer passes.

For reference, the vLLM GLM-5.3 container env (`vllm-reference.json`) runs
`MAX_NUM_BATCHED_TOKENS = 7168` with `MAX_MODEL_LEN = 131072`, `TP=2`, `KV_CACHE_DTYPE=fp8`,
`SPEC_METHOD=dflash`; the stopped reference containers are `glm53-exl3-head` /
`glm53-exl3-worker` (image `glm53-flash-mikeroz:dt-carry20`). A same-hardware vLLM prefill
measurement has not been taken and is the remaining comparison owed.
