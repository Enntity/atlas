# Bounded native FP4 M128 staging experiment

New standalone files only. No engine dispatch or layout activation.
The source generator pins the exact current production SHA256 and extracts
its M64/N128/K64 prequant vector-scale implementation. Candidate M128 uses
256 threads/eight warps. Each warp still computes 16 rows with exactly the
original K64 MMA sequence and FP32 accumulator/BF16 output conversion.
Only A/scale/token staging doubles; the lower four warps load and transpose
one original B tile, shared by all eight warps. No new quantization.

Expected static shared memory: 23,296 -> 30,208 bytes/CTA. Register count is
not yet measured; if near the original 127/thread, two candidate CTAs versus
four original CTAs should leave the same active warp count. Native compiler
and runtime occupancy reports are authoritative. Large M padding may hurt
at 2048 tokens; 4096 has about 114 routes/expert versus 57 at 2048.

## CPU checks

```sh
python3 scripts/dev/glm_moe_m128/generate.py
c++ -std=c++17 -O2 -DATLAS_HOST_ONLY -x c++ \
  scripts/dev/glm_moe_m128/bench.cu -o /tmp/atlas-moe-m128-host
/tmp/atlas-moe-m128-host --host-test
```

The generator checks the source hash and exact unchanged MMA/pipeline/output
regions. Host tests prove unique complete A/B/scale/output CTA write coverage,
unique top-eight routes, and that both 2048 half histograms sum to 4096.

## Native standalone build and authorized GPU run

```sh
nvcc -std=c++17 -O3 --fmad=false \
  -gencode=arch=compute_121a,code=sm_121a -Xptxas=-v \
  scripts/dev/glm_moe_m128/bench.cu -o /tmp/atlas-moe-m128
/tmp/atlas-moe-m128 --run
```

The runtime preserves a 7 GiB free-memory guard and 3 GiB allocation ceiling
so at least 4 GiB remains. Weights are 144 distinct local expert matrices,
288 global table entries with remote NULLs, and top-eight routing. No repack
or duplicate weight resident copy is needed. Inputs are already quantized;
finite synthetic E4M3 scales and per-expert global scales are identical for
the two kernels within each matched chunk. Gate/up uses gathered tokens;
down uses expanded route-major inputs. All local BF16 output bits must match;
remote rows must remain differently poisoned and allocation guards intact.

Five alternating event batches of ten launches report median kernel time.
For each uniform/skew fixture and gate/down shape, the benchmark reports:

- Direct M128 versus M64 at matching 2048 and 4096 sizes.
- Chunk-only 4096 M64 versus both distinct 2048 M64 halves.
- Effective 4096 M128 versus both distinct 2048 M64 halves.

Acceptance uses all three projections: `(2*gate2048_total + down2048_total)`
divided by `(2*gate4096_M128 + down4096_M128)`. The first complete uniform
fixture stops with exit 3 if this is below 1.5x; skew runs only after that
passes and must also reach 1.5x. This remains a routed-kernel throughput
experiment: routing/quantization/SiLU/collectives/attention are not timed,
and the model's 4096 capacity/semantics still need independent qualification.
