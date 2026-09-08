# Exact ten-row W4A16 specialization: standalone result

2026-09-08. A fixture-local wrapper instantiates the unchanged production
`w4a16_gemv_batchm_impl<10>`. No production export, arithmetic change, launch
bounds tuning, Rust dispatch or serving image change. This follows the
[existing batch16-at-ten-rows experiment](glm-w4a16-m10-standalone-results.md).

## Correctness and resource checks

Root independently reviewed the source and ran the bounded fixture on the head
Spark, with both serving models stopped. Seven deliberate harness-sensitivity
failures returned the expected statuses: output/unused-row corruption 2,
guard corruption 3, allocation budget 4, and three invalid CLI arguments 64.
Corruption occurs inside allocated payload/guard memory; budget and CLI
rejections precede CUDA. These are harness tests, not kernel regressions.

Numerical, memcheck and two timing executions each passed all 36 cases. The
exact-ten candidate matches batch16-at-ten, two M5 calls and ten scalar calls
bitwise across all sixteen allocated output rows. Live rows are finite; six
unused rows and all 128-byte allocation guards remain unchanged. Coverage
retains four shapes, three seeds at two scale2 values, zero scale2, finite E4M3
subnormal scales, signed FP4 impulses with a host algebraic oracle, and two
fixed-address segment/intra-segment permutations. General random profiles do
not have a host-double oracle. K is divisible 16; arrays are contiguous. No
arbitrary K-tail, padded-stride, graph replay or full-model numerical claim.

Memcheck reports `ERROR SUMMARY: 0 errors`. Seven guarded allocations peak at
10,553,088 explicit device bytes, below the 16 MiB fixture ceiling; remaining
bytes are zero. CUDA context/sanitizer overhead is separately contained.
CUDA 13.0 compilation used `-std=c++17 -O3 --fmad=false -arch=sm_121a`.
Exact10 uses 56 registers and 10,624 shared bytes, versus batch16's 56 registers
and 16,960 shared bytes; both have zero stack/spill bytes. Shared storage fell,
but register usage did not. No measured occupancy claim.

## Hot three-arm timings

CUDA-event medians of five rounds, 100 repetitions per arm and ten warmup calls
per arm. Each round includes exact10, batch16-at-ten and two M5 calls. The first
three cyclic orders balance positions; the next two distinct permutations
leave each arm in each position once or twice. Raw logs retain every round's
order and duration. This residual five-round order imbalance is not hidden.

| N/K | Execution | Two M5, µs | Batch16 at M10, µs | Exact10, µs | vs batch16 | vs two M5 |
|---|---|---:|---:|---:|---:|---:|
|4096/4096|Initial|86.726|82.216|79.937|1.029x|1.085x|
|4096/4096|Fresh fixture|86.092|83.486|80.703|1.034x|1.067x|
|512/4096|Initial|20.498|15.412|14.371|1.073x|1.426x|
|512/4096|Fresh fixture|20.510|14.797|14.368|1.030x|1.427x|
|8192/2048|Initial|96.704|94.995|92.219|1.030x|1.049x|
|8192/2048|Fresh fixture|95.465|95.293|93.759|1.016x|1.018x|

The candidate's medians improve over both controls in both executions, though
the larger shapes' gains are modest. Repeated hot weights and operator-only
events exclude model scheduling, collectives and cold-cache behavior. These
measurements neither establish model tok/s gains nor justify a blanket dispatch
change. Preserve this candidate for later paired-verifier integration and
validation; the serving baseline remains unchanged.

## Safety and provenance

All 12 native containers are preserved, exited with expected statuses and
OOM=false. Compile used 4 GiB memory+swap, CPUs 0,1, runc and no GPU visibility;
GPU executions used 2 GiB memory+swap, CPUs 0,1, one head GPU, no network and
300-second timeouts. No compile/GPU/model overlap. Final MemAvailable was
118778/118820 MiB on head/worker; swap used zero on both, no running containers.

Frozen SHA256 identities:

- Production CUDA, unchanged: `eba5c5ee826283bb71e4ffbbf277a1586556cbab249d0de32607a14c84ce8db6`.
- Fixture: `f63615fbfc0b448a0c081ce69bcf4bb8e3474bce9d1614c68981168e0be8efa2`.
- Plan: `81b469be9d0d9ff58939ac68d635bbc0f20d290cb6d7da276ccb96ec96c3128e`.
- Executable: `ac9622cabb3b45d918ce71f40af03de31501bab15ceb65a4fcfb1bd82a6a20d5`.

Controller receipts: `atlas-campaigns/20260908/glm-w4a16-exact10-*`.
Isolated head directory: phase7/`glm-w4a16-exact10.CpQMVo`.
Closed source/binary/log archive `glm-w4a16-exact10-verified-receipts.tar` is
verified on controller and head phase7, SHA256
`4ef9a285a866a8362c4419be558efb5d839d9dbff8aa7cd409aa414beafd287e`.
The earlier batch16 experiment and its archive are preserved unchanged. These
are source-hashed standalone receipts, not serving-matrix or whole-tree CI gates.
