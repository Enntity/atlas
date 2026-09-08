# Existing W4A16 batch16 at M10: standalone reuse gate

2026-09-08. Root approved this scripts-only fixture before implementation.
No production CUDA, Rust dispatch, factory, admission, or serving image changes.
This is a prerequisite experiment for later paired verification, not activation.

Include the existing common/w4a16_gemv.cu unchanged. Compare runtime M10 on
`w4a16_gemv_batch16` against two disjoint M5 calls and ten scalar
`w4a16_gemv` calls. Both batch exports share the production virtual-lane
template; retain exact scalar FMA/reduction arithmetic as the numerical oracle.
Do not substitute the alternate single-warp scalar kernel.

Shapes N/K:4096/4096,512/4096,8192/2048, plus N7/K80 tail. K is divisible16;
the ABI is contiguous, with no arbitrary K-tail or padded-stride claim.
Each shape runs three deterministic random seeds at scale2=0.0123 and1,
plus separate zero-scale, finite E4M3 subnormal-scale, and signed impulse
profiles: nine profiles,36 base cases. Every case also swaps the two five-row
segments and permutes rows within each segment at the same allocations.
All ten live rows must match both controls bitwise and remain finite. Six
unused rows and all allocation guards must remain unchanged. The signed
impulse profile additionally has a host algebraic oracle independent of CUDA.

Six guarded allocations: packed W, GS16 E4M3 scales,16 BF16 activation rows,
and three16-row BF16 output arrays.128-byte guards on both sides, checked
arithmetic/live allocation accounting,16MiB explicit allocation ceiling.
Largest shape8192/2048 uses10,290,688 bytes including guards. Host payloads,
CUDA context and sanitizer allocations are not included in that number.
No full-model weights or collectives. Checked frees and remaining0 required.

Strict CLI validation happens before the first CUDA call. `--repetitions N`
accepts only decimal0..100; default0. Optional `--fault` accepts exactly
`output`, `guard`, `unused`, or `budget`, and requires repetitions0.
Fault modes operate only on the first small case. Output/unused faults mutate
one byte inside the allocated candidate payload; guard mutates one byte inside
the allocated guard, not outside the allocation. They then execute the same
normal comparison/canary checks and must exit2,2,3 respectively (output/unused/
guard). Budget requests a deliberately oversized checked allocation and must
exit4 before any CUDA call. Invalid CLI must exit64 before CUDA work.
These expected failures are harness-sensitivity REDs, not kernel regressions;
root archives each actual execution before normal numerical GREEN.

Timing uses only the three production shapes' first random profile, with
five alternating-order rounds and at most100 repetitions per arm. Primary
control is two M5 calls; scalar is correctness-only. CUDA events measure hot
reused weights, not cold-cache or model throughput. Timing warmup is10 calls
per arm; zero timing/warmup calls when repetitions0, including memcheck.

Root alone freezes/transfers an isolated source tree, compiles with
`-std=c++17 -O3 --fmad=false -arch=sm_121a`, then runs expected REDs,
numerical, compute-sanitizer memcheck and timing gates. Never overlap compile,
GPU fixtures, serving models or other GPU work. Retain existing root safeguards:
4GiB memory+swap/CPUs0,1/no GPU for compilation;2GiB memory+swap/CPUs0,1/one
GPU and timeout300s for execution, memcheck error-exitcode99. Models stopped
on both Sparks; verify swap0 and available memory before/after, preserve
containers/logs and inspect exit/OOM state. Unexpected failure or timeout stops
new GPU work pending read-only health checks. No reset or clock changes.

Independent source review precedes native execution. This author performs no
native compilation or GPU execution; root records RED/GREEN receipts separately.
