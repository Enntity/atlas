# Full-working-set B-tile M64 gate/up benchmark

Prepared; host tests pass. No GPU run or production edit. This adapts the
K128 experiment's full-size synthetic EP2 fixture to compare the existing
production `glm_moe_btile_m64_vecscale_dense` against
`moe_w4a4_grouped_gemm_prequant_t_k64_vecscale`. Both are included unchanged
from the production grouped CUDA source. All code is AGPL-3.0-only.

Each case has 144 different local expert matrices and 144 NULL remote experts.
The 648MiB working set exceeds L2. Both paths use the same E4M3 block-scale
arrays, per-expert global multipliers, FP4 activation bytes, expert offsets,
and sorted token IDs. Only packed-B layout changes. A separate tiled copy is
built before timing; every copied byte is checked on GPU, and CPU tests prove
the complete 4,194,304-byte per-matrix permutation is a bijection.

Rows1024 and2048 use deterministic unique-top8 routing, both uniform hashing
and skew toward eight hot experts. They exercise non-multiple-of64 tails and
multiple M64 tiles. The promoted kernel's bound is **1088 rows per expert**, not
1088 prompt tokens. These fixtures fit it. A legal2048-token top8 route can
nevertheless send2048 rows to one expert: host tests explicitly reject that
case. A serving integration must enforce this bound or preserve a fallback;
passing these fixtures does not qualify every possible2048-token histogram.

This family supports N2048/K4096 gate/up only. It does **not** accelerate the
N4096/K2048 down projection, which is outside its compiled shape guard.

Compile from the engine root in the existing CUDA13 builder:

```sh
nvcc -std=c++17 -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a -Xptxas=-v scripts/dev/bench_glm_moe_btile_full.cu -o /tmp/atlas-moe-btile-full
```

Only after GPU execution is authorized:

```sh
/tmp/atlas-moe-btile-full --run
```

Each case first checks every local output bit against production, finite local
outputs, untouched remote rows with distinct poison patterns, and allocation
guards. CUDA-event timing excludes packing, validation, transfers, and host
generation. Five warmups precede five alternating-order batches of ten launches;
the median milliseconds per projection and ratio are printed. The first2048
case must reach1.2x, otherwise exit3 immediately. The skewed2048 case must also
reach1.2x. Numerical failures exit2, CUDA errors exit1.

The benchmark requires7GiB free and enforces a3GiB allocation ceiling,
preserving4GiB reserve. The original and tiled copies are benchmark-only;
their coexistence is not a proposed production storage policy. No checkpoint
or service state is accessed. Runtime prints compiled register counts, shared
bytes, and occupancy. Expected static shared bytes are23,296 for the original
and20,224 for B-tile M64; actual compiler/resource output is authoritative.

CPU-only validation, passed:

```sh
c++ -std=c++17 -O2 -DATLAS_HOST_ONLY -x c++ scripts/dev/bench_glm_moe_btile_full.cu -o /tmp/atlas-moe-btile-full-host
/tmp/atlas-moe-btile-full-host --host-test
```
