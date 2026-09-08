# Unactivated M10 MLA projection prerequisite for paired GLM verification

2026-09-08. Root owns this CUDA/harness-only slice while the Rust owner builds
Partition A request ownership. This does not enable C2 speculation or change
the serving image. No Rust dispatch, factory, guard or resident B-tile changes.

The dedicated GLM MLA path has exact-row Q-absorption and V-extraction exports
through M5. Its existing `mla_batched_gemv_batch_impl<ROWS>` arithmetic is
row-independent and weight-sharing. Add only the M10 instantiation for later
two-session [5,5] verification, not a ten-token single temporal recurrence.
The eventual MLA caller must separately prove per-row causal lengths, owner
block tables, index metadata, scratch capacity and mHC/FFN handling. None is
established by this standalone linear-operator test.

## Test first

Extend `bench_glm_mla_batch.cu` with an explicit M10 branch referencing the new
export. Capture the native compiler RED for the missing symbol before adding
that export. Preserve M2/M3/M4/M5 coverage. For M10 the primary timing control
is two existing M5 calls over disjoint five-row segments, not ten scalar calls.
Also compare all output bytes against the existing scalar operator, retaining
independent host dot checks, finite checks and allocation/padding canaries.

Use exact GLM Q-absorption `(heads32,N512,K256)` and V-extraction
`(heads32,N256,K512)`, padded/unpadded strides and a small tail shape. Test
swapped five-row segments and an intra-segment row permutation at the same
addresses. Exhaustive independent host dots for M10. No relaxed tolerance on
candidate-versus-legacy BF16 equality; host-double error gate unchanged.

Worst live device allocation: one 8MiB weight plus M10 inputs and three output
buffers with padding/guards, below16MiB. Add a checked total allocation cap
to the fixture. A failed allocation/launch/sync/canary/numerical check stops
the fixture. At most100 timing repetitions per arm, five alternating-order
rounds; zero timing repetitions under memcheck. No models or collectives.

## Root-only native procedure

Before each native phase verify no running serving/GPU/compile containers,
host swap0 and comfortable idle memory on both Sparks. Use the retained
CUDA builder image, unique container and receipt names, no overwrite/removal.
Freeze only this source manifest and transfer an isolated source tree; never
overlay the native engine builder or shared Rust WIP.

CPU-only compile:4GiB memory+swap ceiling, CPUs0,1, no GPU exposure, one nvcc.
First use production kernel flags `-O3 --fmad=false -arch=sm_121a` with C++17;
then an independent default-FMA policy for diagnostic comparison if useful.
Compile and execution cannot overlap. GPU fixture:2GiB memory+swap ceiling,
CPUs0,1, one GPU, outer timeout300s per fixture, and compute-sanitizer memcheck
with error-exitcode99. No reset, clock change, swap or model execution.
If a timeout/fault occurs, preserve all logs and stop new GPU work pending
read-only health checks; do not automatically continue to another variant.

Require production-policy numerical gate and memcheck, then repeated timing
receipts. Check final container exit/OOM state and node swap/memory. Preserve
source hashes, compiler logs and complete outputs. Independent review precedes
the exact-manifest commit. A standalone speedup is not whole-model TPS and
does not qualify the eventual segmented verifier.
