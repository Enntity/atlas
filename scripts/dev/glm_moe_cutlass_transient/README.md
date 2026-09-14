# Standalone transient-SFB grouped CUTLASS benchmark

No serving source or dispatch changes. Production collective, epilogue and
grouped launch assembly are extracted from the pinned source by `generate.py`.
The only launch modification adds a dry-run path and checked workspace writes.
The actual production BF16-to-NVFP4 quantizer and M64 vector-scale kernel are
included directly. CUTLASS receives those same packed activation bytes through
a gather/SFA swizzle; its different BF16 quantizer is deliberately bypassed.

The fixture uses 4100 tokens, top8 of288 experts,144 local expert matrices,
including uniform and skewed routing. Each projection has distinct deterministic
weights, original-native storage for CUTLASS and transposed storage for M64.
Native layout preparation is outside timing, representing alternative resident
layouts. Each timed projection includes BF16 activation quantization; CUTLASS
also includes all activation gathering/SFA packing and a fresh72MiB SFB pack.
The three projection cases are measured separately. Their weighted total removes
one measured gate/up activation preparation to account for its reuse. This is a
projection microbenchmark, not a chained FFN or a model correctness gate.

One512MiB explicit workspace holds activation staging, argument arrays and the
queried CUTLASS workspace. All sizes and `can_implement` are checked in a dry run
before candidate workspace uploads/kernels. A one-byte-short test must reject.
There is no process-global CUTLASS workspace or hidden512MiB wrapper allocation.
The measured required workspace is printed for a later arena integration audit.

Device allocations have128-byte red zones and a combined3GiB ceiling. The largest
case is approximately2556MiB plus small pointer/metadata buffers. Host comparison
temporarily holds512.5MiB of output copies. The initial7GiB free guard retains at
least4GiB beyond the device ceiling. No checkpoint data or model process is used.

The native gate requires exhaustive byte identity of A/SFA/B/SFB, untouched remote
expert rows, finite outputs, relativeL2 <=0.004 and a per-output reduction bound.
An independent FP64 quantized dot oracle samples96 outputs per projection, with
one-half BF16 ULP plus an FP32 accumulation allowance. Output-bit differences are
reported, not silently treated as identical. Median alternating-order CUDA event
timings include preparation. Stop after all3 uniform projections if weighted
speedup is below1.5x; otherwise also require1.5x for skew. A2x result is preferred.

```sh
python3 scripts/dev/glm_moe_cutlass_transient/generate.py
c++ -std=c++17 -O2 -DATLAS_HOST_ONLY -x c++ \
  scripts/dev/glm_moe_cutlass_transient/bench.cu -o /tmp/atlas-cutlass-host
/tmp/atlas-cutlass-host --host-test

nvcc -std=c++17 -O3 --fmad=false --expt-relaxed-constexpr \
  -gencode=arch=compute_121a,code=sm_121a -Xptxas=-v \
  -I/opt/cutlass/include -I/opt/cutlass/tools/util/include \
  scripts/dev/glm_moe_cutlass_transient/bench.cu -o /tmp/atlas-cutlass-transient
# Requires a separately authorized idle GPU window:
/tmp/atlas-cutlass-transient --run
```

Production integration remains unimplemented: it would need transient owner
capacity validation, batched SFB staging, native-original-only expert ownership,
materialized post-SiLU BF16 or direct prequant down input, and shared-cache/MTP
dispatch qualification. A successful microbenchmark alone does not authorize it.

## Native result: rejected

On Spark01, the reviewed native executable passed all three uniform projections:
exhaustive operand-byte equality, unchanged workspace after dry-run rejection,
134,135,808 bit-identical local BF16 outputs in total, and288 independent FP64
dot checks. Combined inclusive time was14.038845ms for production M64 versus
17.487295ms for CUTLASS (0.802803x). The1.5x fail-fast gate exited3 and skipped
skew. Peak explicit device allocation was2556.796MiB. Actual queried workspace
was38,580,480 bytes for gate/up and19,323,136 bytes for down, far below the
explicit512MiB cap. No production integration was made.

The subsequent uniform-only component diagnostic retained the same oracle and
aggregate timing. It measured3.662451ms total SFB packing and12.010595ms total
CUTLASS core plus metadata. Inclusive time was17.518621ms versus14.079632ms.
Even hypothetically free SFB would leave13.856170ms, only1.016x. The transient
pingpong K128 recipe is closed; old source archives and receipts are retained.

## Final cooperative K256 alternative

`cooperative.cu` includes the identical harness with only two collective changes:
TileShape128x128x128 ->128x128x256 and Pingpong -> Cooperative schedule.
StageCountAuto, epilogue, all operands, preparation costs, numerical gates and
memory limits remain unchanged. The generator writes a separate collective
header; the baseline header remains intact. Use the same nvcc command with
`cooperative.cu` and a distinct executable, then `--uniform` to stop after the
first three-projection gate regardless of its result. Exit3 still rejects below
1.5x. This recipe is motivated by the local FlashInfer native probe's compiled
cooperative K256 symbol, not evidence of the matched full-model kernel choice.

Native cooperative K256 also rejected (exit3): all three uniform projections
again passed identical operands,134,135,808 unchanged output bits and288 FP64
dots. Inclusive production14.129487ms versus cooperative16.587219ms =0.851830x.
SFB packing totaled3.631859ms; core plus metadata totaled11.183834ms. Even free
SFB would only reach about1.091x. The family is closed without serving changes.
