# GLM-5.3 EXL3 TP=2 decode: vLLM-parity ledger (2026-10-15)

Branch `perf/decode-vllm-parity-20261015`. Target: the vLLM reference's
~25–35 tok/s on this two-Spark ring. Its ~30 tok/s needs DFlash2 speculation;
serial decode is the floor that speculation multiplies (see
`docs/plans/f4-dflash2-exl3-tp2-plan.md`).

## Fingerprint (all rows)

- Image `atlas-exl3-validation:src-4ee96c12b44c-rust1.97.1`
  (`sha256:82173ea6…`). Dev binaries are bind-mounted from
  `~/atlas-dev*/bin/spark`; the build commit is the row's commit.
- Checkpoint `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3@dda99c7e`.
  TP=2/EP=1 over RoCE.
- Harness `perf_baseline.py` 1.1 via `tests/exl3_validation/bench_tp2.py`, C=1,
  greedy, 3 repeats per row.
  - `ab_prose_decode`: 128 in / 128 out.
  - `ab_prose_long`: 128 in / 1024 out.
- Container cpuset 15-19 from the cpuset row onward. 10-19 pins every thread
  to CPU 10 because of `isolcpus`.
- Run directories are under
  `/home/colum/Workspace/atlas-exl3-validation-20261010/perf-ab-20261015/`.

All numbers are **preliminary, single-harness**: each is one 3-repeat run.
The workload changes once, from `ab_prose_decode` to `ab_prose_long`. Rows on
either side of that change are two series, not one A/B.

## Exact series (identity-preserving)

"Exact" means bit-identical logits on all 384 steps (capture compare), or the
same `output_sha256` in graph mode. `8081778d` (decode) and `80b1eb5f` (long)
are the eager reference outputs.

| Step | Commit / flag | Workload | TPOT p50 s | Run |
|---|---|---|---:|---|
| Baseline (instrumentation off) | 4ee96c12 | decode | 0.594 | devoff-…T071348Z |
| Slice-parallel MMA + split mHC | 8fe98293, `SPLIT_MMA`, `SPLIT_MHC` | decode | 0.329 | devsplit-…T072345Z |
| RDMA allreduce | `RDMA_ALLREDUCE=1` | decode | 0.219 | splitrdma-…T073337Z |
| cpuset 15-19 | 37f8c010 | decode | 0.150 | cpuset1519-…T093356Z |
| mHC parallel collapse | 0c4a1e17 | decode | 0.147 | mhcfin-…T143319Z |
| MLA trailing-tile trim | 6910b841 | decode | 0.143 | mlatrim-…T145136Z |
| Coop GEMM port + coop MoE | port merges, `COOP_GEMM`, `COOP_MOE` | decode | 0.120 → 0.0746 | merged-…T152326Z, … |
| Split fold + 16-lane Sinkhorn | 040dc4bc | decode | 0.0708 | sinkpar-…T154808Z |
| Parallel top-k, wide post | baaad527, `PAR_SMALL` | decode | 0.0689 | parsmall-…T155852Z |
| Fused KDA qkv | `COOP_FUSED_QKV` | decode | 0.0616–0.0667 | fusedqkv-…T160439Z |
| MoE v2 | 1a2cbea6, `COOP_MOE_V2` | decode | 0.0586 | moev2-…T164844Z |
| Decode graphs + FLA in graph | d0baaf44, `KDA_FLA_GRAPH`, graph env | decode / long | 0.0537 / 0.0559 | graphs2-…, longgraphs-… |
| MoE v3 | `COOP_MOE_V3` | long | 0.0535 | moev3graphs-…T180143Z |
| KDA decode float4 state | 47f9e45f | long | 0.0518 | kdavec4-…T183813Z |
| mHC finish: hoisted loads | 8d848a77 | long | 0.0497 | mhcv2-…T184655Z |
| Port phase 6 merged (exact path) | cb369b69 | long | 0.0497 | p6exact-…T185912Z |
| Warp-shuffle routing top-k | eeb2dc58 | long | 0.0491 | topkwarp-…T193818Z |
| Rounded (layer-0) mHC site split | e7f8acfe | long | 0.0489 | mhcrounded-…T195940Z |
| Routed MoE v6 (staged A row), port binary | `COOP_MOE_V6=1` | long | 0.0483 | moev6-…T211750Z |
| Skip per-step MLA arena clear | 94b10666, `DECODE_ARENA_FILL=skip` | long | 0.0482 | arenaskip-…T205452Z |
| Combined: v6 + arena skip | 10bec40b | long | **0.0479** | combexact-…T225708Z |
| A: v6 + arena skip (binary 5e1674ba, perf 9ffdb3dd) | same | long | 0.04762 / 0.04754 | p8A-…T000104Z, p8A2-…T013814Z |
| B: A + MoE v7 (input Hadamard in A-row staging) | `COOP_MOE_V7=1` | long | 0.04739 | p8B-…T002158Z |
| C: B + shared expert on a side stream | `SHARED_OVERLAP=1` | long | **0.04574 / 0.04573** | p8C-…T004247Z, p8C2-…T014550Z |
| D: C + fused coop input/output Hadamards | `COOP_FUSED_INPUT/OUTPUT=1` | long | 0.04566 (flat) | p8D-…T011643Z |
| C + MoE V9 (down kernel absorbs output Hadamards, activation, down input Hadamard) | `COOP_MOE_V9=1` | long | 0.04578 / 0.04582 vs base 0.04569 / 0.04568 (slightly slower; leave off) | p11v9-*, p11base-* |
| C + MoE V10 (cross-item streaming, 3 CTAs/SM; standalone −10% gate\|up, −14% down) | `COOP_MOE_V10=1` | long | 0.04646 mean of 3 vs base 0.04569 mean of 3, alternating; +0.77 ms, slower (likely displaces V7 fusion and the SHARED_OVERLAP co-run); leave off | p12base-*, p12v10-* |

`ATLAS_DECODE_ARENA_FILL=poison` refills the arena with 0xFF (NaN) before
every step. With it, the output hash is still `80b1eb5f`
(arenapoison-…T204651Z), so the stack reads no arena scratch it did not write
in that step. This holds for single-sequence decode; multi-sequence and
prefix-cache runs are not yet checked.

These two changes were bit-identical but slower, so they were reverted:
- constant-bitrate templating: 0.188;
- warp-staged tile decode: 0.1557 (staged2-…T151143Z).

## Numerics-changing (flagged, needs requalification)

| Change | Flag | Workload | TPOT p50 s | Output | Run |
|---|---|---|---:|---|---|
| Packed shared expert (untuned profiles) | `EXL3_SHARED_PACKED=1` | long | 0.0517 | `8e069c71` | sharedpacked3-…T181757Z |
| + phase 6 fused gate\|up and tuned profiles | same | long | 0.0466 | `9347843b` | p6packed-…T190652Z |
| FP8 sparse MLA decode split (13 K splits + LSE merge) | `GLM_SPARSE_DECODE_FP8_SPLIT=1` | long | 0.0475 (exact base 0.0489) | `98fc6609` | mlafp8split2-* |
| Combined flagged stack on 10bec40b (packed shared expert + FP8 split + exact stack) | both flags | long | **0.0428** | `564e84bc` | combflagged-…T232705Z |
| Flagged on the current exact stack (+V7 + overlap; binary 5e1674ba) | both flags | long | **0.04254** | `564e84bc` (unchanged: V7/overlap are bit-exact) | p8flagged-…T021707Z |

## Where the time goes (rank 0, packed config, instrumented)

Nsight trace `p0-trace-20261015/runs/p0-p6-20261015`: 47.8 ms/token wall,
45.7 ms kernel-busy, 2.2 ms idle.

| Item | ms/token |
|---|---:|
| Routed MoE v3 | 13.0 |
| Fused KDA qkv | 6.2 |
| Coop GEMMs (o_proj, MLA sites, shared) | ~8.7 |
| NCCL SendRecv in graph | 2.7 |
| mHC pre pair | 2.5 |
| EXL3 input/output Hadamards (~430 launches) | 1.8 |
| KDA recurrent decode | 1.06 |
| Routing top-k (before eeb2dc58) | 0.95 |

Fused qkv is ~79% of its 6-bit weight-bandwidth floor. Routed MoE is the
largest remaining gap (~52% of ideal).

## F4 draft-only acceptance (observer, no verify)

Run `f4-draftonly-20261015T191446Z`: k=7, `ab_prose_long`, τ = 2.689,
prefix acceptance [0.725, 0.422, 0.233, 0.131, 0.076, 0.055, 0.046].
- Both ranks give identical results, and the serial hash is unchanged.
- The drafter costs ~16 ms/step as built.
- On the reference's own workload (`tools/dt_workload.sh`: bridges essay,
  greedy, thinking on), Atlas draft-only gives τ = 3.11 at 200 tokens and
  2.79 at 1024 tokens (f4ref-fp32-…T194917Z). The reference's own greedy runs
  on that workload measured 2.71–2.79, so acceptance is at parity. The
  3.75–4.09 figure came from an unreproducible synthetic long-context probe.
- Verify probe (8 rows, `ATLAS_GLM_EXL3_DFLASH_VERIFY_PROBE`): row argmax
  matches serial on 99.45% of rows; the rest are batch-shape flips. The
  speculative lane is therefore labelled numerics-changing. Its gate is
  self-consistency, determinism and rollback replay, not byte identity to
  serial.
