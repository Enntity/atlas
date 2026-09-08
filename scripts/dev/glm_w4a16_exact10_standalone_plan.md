# Exact-template M10 W4A16: unactivated standalone follow-up

2026-09-08. Root read and approved the complete plan at63efaf529debe6e71fc52022881e9af5e07279c28c9e1a44af9203b06e527fd9
before implementation. This follows the existing batch16-at-M10 gate
committed01d2f196. Preserve its source, binary, raw logs and closed archive
unchanged. No production CUDA, Rust, dispatch, factory or serving-image changes.

## One candidate, existing arithmetic

Only modify `scripts/dev/bench_glm_w4a16_m10.cu` after approval. Immediately
after including the unchanged common/w4a16_gemv.cu, add one fixture-local CUDA
wrapper calling `w4a16_gemv_batchm_impl<10>`. Preserve the runtime M/N/K ABI;
the fixture always passes M10. No launch_bounds, changed reduction/FMA order,
new math body, parameter search or production export.

Exact-template10 becomes the primary candidate. Retain existing batch16 at
M10 and two disjoint batch5(M5) calls as timing/correctness controls; retain
ten scalar w4a16_gemv calls as the independent exact CUDA oracle. No single-
warp substitute. Each candidate/control output has its own guarded allocation.

The source predicts shared arrays of64+10*1056=10,624 bytes instead of the
batch16 compilation's16,960 bytes. This is a hypothesis until ptxas reports it;
register count, spills, occupancy and timing may differ. Do not presume a gain,
particularly on8192/2048 where the existing wider tier was effectively neutral.

## Exact preserved coverage and resource bounds

Retain all36 cases: N/K7/80,4096/4096,512/4096,8192/2048; three seeds at
scale2=0.0123 and1 plus zero-scale2, finite E4M3 subnormal and signed-impulse
profiles. Each case runs both same-address segment/intra-segment permutations.
Compare every candidate byte against all three controls, including unused
rows10..15; require finite live outputs and the independent host impulse
oracle. Preserve all128-byte allocation guards, checked allocation/free
accounting, post-timing comparison and remaining0 requirement.

One extra sixteen-row output changes six allocations to seven. The maximum
explicit device allocation becomes10,553,088 bytes at8192/2048, still below
16MiB: the prior10,290,688 plus262,144 output bytes and256 new guard bytes.
This corrects the initial10,552,832 estimate, which omitted the seventh
allocation's guards. CUDA context/sanitizer overhead remains separate. Fixed contiguous ABI,
K divisible16, no padded-stride/arbitrary K-tail/CUDA-graph/model-output claim.

Strict CLI and safe in-bounds fault modes remain unchanged: output/unused
corruption exit2, guard corruption3, budget4, invalid repetition arguments64.
Faults target the new exact10 candidate; corruption is inside allocated payload
or guard storage. Root executes these real comparison/canary sensitivity REDs
before ordinary numerical GREEN. They are intentional harness failures, not
an existing kernel regression. Numerical and memcheck use repetitions0.

## Timing and interpretation

Only profile0 on the three production shapes is timed. Ten warmup calls per
arm, then five rounds with100 repetitions per arm at most. Rotate execution
order among exact10, batch16-at10 and two-M5, using a balanced first-three-round
cycle followed by two distinct remaining permutations. Five rounds cannot
give three arms identical position counts: each arm occupies each position
once or twice. Report this residual order imbalance explicitly, retain all
per-round arm times/order in output, and repeat with a fresh fixture process.
Report medians for all three arms and both control/candidate ratios. Scalar
calls remain correctness-only and outside event timings. No cold-cache or
whole-model throughput interpretation; no shape selection is activated.

## Root-only gates

After implementation, freeze the fixture/this plan and unchanged production
CUDA hashes for independent root source review. Root transfers an isolated tree
and compiles using CUDA13.0 `-std=c++17 -O3 --fmad=false -arch=sm_121a`,
collecting ptxas resource reports. No new production export or native engine
build. Root alone performs expected REDs, numerical, memcheck(error-exit99),
and two timing executions, then archives source/binary/logs under new names.

Preserve prior safeguards: both serving nodes stopped; no compile/GPU/model
overlap; CPU compile4GiB memory+swap, CPUs0,1, runc/no GPU; execution2GiB
memory+swap, CPUs0,1, one head GPU/no network/timeout300s. Check no running
GPU work, swap0 and available memory before/after; preserve containers and
inspect expected exit/OOM states. Unexpected failure/timeout stops further
GPU work pending read-only health checks. No reset, clocks or memory-budget
changes. Author performs no Cargo, Rust edits, node actions or native execution.

Implementation is now source-frozen for root review. The fixture additionally
rejects nonpositive/nonfinite CUDA event durations before reporting ratios.
All36 cases and seven expected-failure CLI modes are retained; no native
compile, numerical, fault-sensitivity or timing PASS is claimed by the author.
