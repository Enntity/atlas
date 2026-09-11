# Standalone GLM routed MoE K128 staging experiment

Status: prepared, host routing checks passed; CUDA compilation, numerical checks,
and timings are pending. Nothing selects this kernel in the engine.

`generate.py` extracts the production native-FP4 vector-scale body from
`kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu`, pinned to SHA256
`bad2afc59183a4b46fd094864ef88ae428517a23dee0d00f1d9e87de7503b33b`.
Both sources remain AGPL-3.0-only. Regeneration rejects a changed source hash.

The candidate retains M64/N128 and the exact ordered K64 MMA instructions. Each
ping-pong stage stores two K64 segments. It loads and transposes both segments
before one barrier, then performs segment 0 and segment 1 in their original
accumulator order. This halves the per-K staging/barrier boundaries. It does not
change quantization, expert scales, gather indexing, output rounding, or weights.

Expected static shared memory is 23,296 bytes for the original and 46,336 bytes
for the candidate. On a 100KiB shared-memory SM this reduces the shared-memory
ceiling from four CTAs to two. Register limits can reduce occupancy further;
the program reports compiled shared bytes/registers and actual CUDA occupancy.
The occupancy tradeoff may erase the barrier savings. This is an experiment,
not a predicted speedup.

From the engine root, using the existing CUDA 13 builder:

```sh
python3 scripts/dev/glm_moe_k128/generate.py
nvcc -std=c++17 -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a -Xptxas=-v scripts/dev/glm_moe_k128/bench.cu -o /tmp/atlas-moe-k128
```

Only after GPU execution is authorized:

```sh
/tmp/atlas-moe-k128 --run
```

The program requires at least 7GiB free and enforces a 3GiB allocation ceiling,
leaving at least 4GiB reserve. Typical live allocation is below 1GiB. All memory
is synthetic and process-owned; no model weights, services, or GPU settings are
changed. It uses one stream and releases each case before allocating the next.

Fixtures cover 1024/2048 tokens, 288 global experts, unique top8 routing, uniform
hash and skewed expert loads. Exactly 144 experts have distinct weight storage;
the other 144 have NULL pointers, matching one EP2 rank. Each projection's
648MiB expert working set exceeds L2. Gate/up geometry is N2048/K4096 with a
token gather; down is N4096/K2048 with already-sorted input and no gather.
Weights and already-quantized activations contain deterministic nontrivial FP4
codes, positive finite E4M3 scales, and distinct per-expert global multipliers.

Every case first compares all local output bits against production, checks
finite local outputs, checks that remote rows retain distinct poison values,
and checks allocation guards. This is an equivalence check against the current
production kernel, not an independent mathematical reference or model-quality
qualification. Timing uses five warmups and five alternating-order batches of
ten launches, reporting median CUDA-event milliseconds per projection. Host
allocation, input generation, validation, and transfers are outside timing.

Fail fast: the first 2048-token gate/up case must reach 1.5x, otherwise exit 3.
Other 2048-token cases must also reach 1.5x. Numerical/guard failures exit 2;
CUDA failures exit 1. A successful microbenchmark still requires production
integration review and model canaries. The benchmark does not time the router,
activation quantizer, SiLU, or communication.

CPU-only routing validation (run locally, passed four fixture combinations):

```sh
c++ -std=c++17 -O2 -DATLAS_HOST_ONLY -x c++ scripts/dev/glm_moe_k128/bench.cu -o /tmp/atlas-moe-k128-host
/tmp/atlas-moe-k128-host --host-test
```
