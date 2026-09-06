# Opt-in exact-row C4 MLA projections

Status: CPU and bounded GPU gates pass; full-model A/B pending.

## Plan and scope

Reuse the existing `mla_batched_gemv_batch_impl` template at exactly four rows,
with a separate default-off `GLM_MLA_BATCH4` launcher switch. Require C4 mode
and a loaded kernel. Keep C1–C3 and K5 dispatch unchanged. The existing batched
MLA chain already accepts row counts: absorb all queries, assemble/write all
independent KV entries, run the corrected 512-dimensional attention kernel,
then extract all values. No new buffer allocation or speculative policy.

Per MLA layer, Q absorption/value extraction fall from eight kernel launches
to two; the whole section including cache assembly/write and attention falls
from 20 to five. There are eleven MLA layers, so this removes 165 launches per
rank per decode step. These are source-derived counts, not a TPS prediction.

The existing C4 guards cover 131,072-byte absorbed-query/attention planes,
65,536-byte expanded-query/extracted-value planes, and 8,192 bytes of assembled
K/V entries. Shapes remain 32 local heads, rank512, original QK256, value256.
All rows retain distinct device lengths, slot metadata and block tables.

Tests first: flag/width dispatch and launcher rejection tests; exact scalar
CUDA parity and independent CPU dots; row permutations, poisoned padding and
allocation guards; finally same-binary full-model quality and throughput A/B.
Do not enable the switch by default on the basis of a microbenchmark.

## Local and GPU receipts

684 model unit tests and three new launcher tests pass. Rust formatting, shell
syntax and whitespace checks pass. An independent review found no blocking
alias, dimension, ABI, or default-dispatch issue.

Root compiled `scripts/dev/bench_glm_mla_batch.cu` with CUDA `-O3 -arch=sm_121a`
on the head Spark while both model containers were stopped. All twenty cases
pass across M2/M3/M4/M5, preserving prior coverage. New M4 cases check every
output against independent double-precision CPU dots, exact scalar parity,
two row permutations, NaN input padding, and output/allocation guards.
The full suite also passes compute-sanitizer memcheck with zero errors.

M4 CUDA-event microtimings, medians of five interleaved rounds, 100 iterations:

| Operation | Four scalar launches | M4 launch | Ratio |
| --- | ---: | ---: | ---: |
| Q absorption, 32 heads, N512 K256 | 44.482 us | 22.583 us | 1.970x |
| Value extraction, 32 heads, N256 K512 | 32.842 us | 16.418 us | 2.000x |

Compiler report: 40 registers, 256 bytes shared memory, zero spill loads/stores
for M4. The executable uses roughly 20 MiB combined host/device storage inside
a 1 GiB no-network container. Raw controller receipts:
`/tmp/atlas-glm53-phase3-20260906/v8-mla-build.log` and `v8-mla-gpu.log`.

These timings are not whole-model throughput. Full-model tests must retain the
2048 context limit, four active/admitted independent rows, non-speculative
execution, current memory guard, and separate grouped-MoE setting.
