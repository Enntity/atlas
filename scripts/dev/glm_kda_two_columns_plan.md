# Standalone KDA two-column recurrence

Prepared; CPU ownership check passed. CUDA compile, full state/output oracle,
and performance are pending. No serving path selects this kernel.

Current GLM uses heads32/D128 locally, four warps per CTA, one state column per
warp, and grid32×32. Each lane retains four FP32 state elements. For every
token it loads twelve FP32 Q/K/decay values and runs two dependent five-stage
warp reductions. Generic scalar-decay GDN WY is not applicable.

The candidate assigns two adjacent columns to each warp, retaining eight
state elements and sharing those twelve Q/K/decay loads between two independent
recurrences. Grid is32×16, still128 threads. Every column retains the original
four-product partial sum and XOR16/8/4/2/1 reduction order. Preprocessing,
row-specific decay, FP32 state updates, and BF16 output conversion are unchanged.
No shared memory is added. Higher register pressure may reduce occupancy;
compiled register counts and CUDA occupancy are printed, with no predicted gain.

This adapts the AGPL-3.0-only `kda_recurrent_bf16_regresident` body in
`kernels/gb10/common/kda.cu` directly. The benchmark includes that actual
production source for its reference and uses its unchanged preprocessing kernel.
Prepared reference SHA256:
`5f3c4cb3fd7720a15e1e0dd337032ae95ae92fe47ff11169d30d0b8fdc60a609`.
Keep the source snapshot alongside the result if the production kernel changes.

Compile from the engine root in the existing CUDA13 builder:

```sh
nvcc -std=c++17 -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a -Xptxas=-v scripts/dev/bench_glm_kda_two_columns.cu -o /tmp/atlas-kda-two-columns
```

Only after GPU execution is authorized:

```sh
/tmp/atlas-kda-two-columns --run
```

The first case compares every BF16 output and FP32 state bit for2048 tokens,
including a second continuation pass. Distinct output poison catches missing
writes. Allocation guards and finite reference values are checked. The first
case must then reach1.5x median CUDA-event speedup or exit3 immediately.
Following cases exercise1024/2052 tokens, exact-zero Q/K, and strong initial
state/gates. Both paths consume identical production-preprocessed FP32 planes;
this oracle compares recurrence implementations, not the preprocessing math.

Event timing excludes preprocessing, allocations, host generation, state reset,
and transfers. It uses three warmups and five alternating-order batches of ten
launches. Each timing pair resets both state buffers to identical starting
states before either event range. This is a kernel-only experiment with no
model-quality or end-to-end performance claim.

The executable requires4.5GiB free and caps its allocations below512MiB,
preserving4GiB reserve. All allocations are synthetic and process-owned.

CPU-only ownership validation, already passed:

```sh
c++ -std=c++17 -O2 -DATLAS_HOST_ONLY -x c++ scripts/dev/bench_glm_kda_two_columns.cu -o /tmp/atlas-kda-two-columns-host
/tmp/atlas-kda-two-columns-host --host-test
```
