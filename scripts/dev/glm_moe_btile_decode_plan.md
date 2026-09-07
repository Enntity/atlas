# B-tile BF16-input decode compatibility experiment

Standalone only; no production registration, loader changes, flags, or promotion.
Root owns native compilation, GPU tests, memcheck, and safe stopped-model windows.

## Contract and hypothesis

Cover existing transposed gate/up GEMV exports for N1/N2/N3 tokens, each with
top8 routes, N2048/K4096, 288 expert pointer entries and block32. Keep BF16
activations, group16 E4M3 scales in `[K/16,N]`, FP32 scale2, and the existing
`acc += a_lo*w_lo + a_hi*w_hi` expression and ordered group/byte loops.
Routed B alone uses `[N/128,K/64,128,32]`; shared gate/up remain transposed.
Remote routes and absent shared weights must write exact zero, as production
does. This differs from the grouped experiment's untouched remote rows.

Two candidates share arithmetic and differ only in B access:

1. Direct byte addressing is the correctness-first baseline; adjacent output
   lanes stride32 bytes, so a performance regression is plausible.
2. Cooperative stage loads the 32-output x32-byte K64 slice coalesced into
   1152 shared bytes (1024 useful, padded rows), then each lane consumes its own row in exactly the same
   ordered accumulation. Reload only every four scale groups; CTA barriers
   protect stage reuse. All active lanes execute every barrier. No async-copy
   ordering or activation quantization changes in this first prototype.

Original production scalar/batch2/batch3 `.cu` files are included unmodified
as the numerical oracle. Candidate shapes are validated on the host; dynamic
N/K remain kernel parameters to avoid gratuitous arithmetic specialization.

## Test-first gates

- Local CPU RED: deliberately wrong B address helper must fail corner mapping
  before implementation; GREEN exhaustive byte bijection/inverse and staged
  slice simulation, legal/illegal route-map and allocation-budget tests.
- One fixture covers widths1/2/3, permuted expert IDs and token inputs,
  nonprefix local experts, repeated experts across tokens, all-remote routes,
  zero input, absent/shared-present projections, and changing tables between
  graph replays. Top8 remains unique within each token.
- Independently poison all routed and shared outputs of all three variants;
  compare every output bit to the included production export. Check CPU double
  oracle columns for all local/shared output rows and exact remote zero.
- Graphs have fixed pointers, refreshed input/routes/weight pointer tables;
  inactive width tails stay poisoned. Inputs, routes, pointer tables, scale2,
  full weight/scales bytes, and allocation guards stay unchanged. Repeat full
  output and immutable-input comparisons after any timing loops.
- Primary compile `--fmad=false` matches production; additionally compile and
  run default FMA mode before promotion, with no tolerance relaxation.
- Native then memcheck for all widths/cases, then three interleaved timing
  repetitions comparing production/direct/staged, never timing-only bypass.

## Memory and scope

Two distinct local expert gate/up pairs have original and tiled packed B:
32 MiB packed plus2 MiB shared immutable scales. A third, distinct shared
expert gate/up pair remains original-only:8 MiB packed plus1 MiB scales.
Total weight storage43 MiB. Fixed max3 BF16 inputs and all twelve output
buffers, metadata and128-byte leading/trailing canaries keep total below
45 MiB; each allocation is charged before cudaMalloc against a hard64 MiB
cap. No model weights or additional kernels run concurrently during GPU gates.

The two local pairs are a small hot working set, not a representative rank's
weights. Scalar staging may add more barrier/instruction cost than it removes;
neither candidate is assumed faster. Shared weights intentionally use their
original layout, and mismatched gate/up ownership is rejected by the fixture.
No down, shared blend, router, collectives, or serving state is modified.

## Commands (native/GPU root only)

```sh
g++ -std=c++17 -O2 -x c++ -DATLAS_BTILE_DECODE_HOST_ONLY scripts/dev/bench_glm_moe_btile_decode.cu -o /tmp/btile-decode-host
/tmp/btile-decode-host --host-test
nvcc -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a scripts/dev/bench_glm_moe_btile_decode.cu -o /tmp/btile-decode
/tmp/btile-decode
compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/btile-decode
/tmp/btile-decode --timing
```

Repeat native correctness/memcheck with `--fmad=false` omitted. Preserve all
paired timing runs, including regressions; no production promotion from a
microbenchmark alone. Validate combined scalar bootstrap/drains/general M64
prefill and full-model C1/K5/C4 before any equal-memory layout replacement.
