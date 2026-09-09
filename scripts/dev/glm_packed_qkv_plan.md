# Direct row-packed GLM KDA QKV: bounded candidate plan

2026-09-09. PLAN ONLY. No CUDA/Rust change, compilation, GPU execution,
dispatch/admission change or serving claim is authorized by this document.
Root must approve each future source slice and owns independent review,
compilation and all native gates. Bootstrap/serial-driver correctness remains
the current priority; this candidate does not widen those authors' scope.

## Primary source and actual baseline

The vLLM GLM implementation at
[`98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5`](https://github.com/vllm-project/vllm/blob/98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5/vllm/models/glm5next/nvidia/kda.py#L165)
merges input projections, splits row-wise QKV, and feeds one channel-wise
convolution. Its BF16 six-projection GEMM is NOT an NVFP4 replacement: Atlas
retains NVFP4 Q/K/V and separate BF16 side projections. Transfer only the
producer/consumer layout principle; do not import weight sharding, arithmetic,
Triton recurrence or claims of bitwise agreement from another implementation.

Atlas `layers/glm5_kda.rs::forward_inner` selects the existing fused producer
only for captured verify intermediates, M5, a nonzero kernel handle and
`ATLAS_GLM_K5_FUSED_QKV=1`. Otherwise Q/K/V are separate projections.
`kernels/gb10/common/w4a16_gemv.cu::w4a16_gemv_batch5_qkv` uses grid-Z to select
three actual weight/scale/scale2 owners and calls the same exact-M5 body,
writing three projection-major planes. `glm5_kda.rs` then unconditionally calls
`kda_pack_qkv` into `ssm_qkvz` before convolution. This pack is not removed by
the existing fused-QKV flag.

The qualified v26 C1 launch receipts
`atlas-campaigns/20260908/v26-idle-{initial,repeat}-launch.log:80` both report
`GLM K5 fused native-FP4 QKV: 1`. Their recipe `run-v26-c1.sh` invokes the frozen
`v25-launcher-a85b1a7b.sh`, which defaults that option to1 and forwards it to
both ranks. Selected live-env extracts for these C1 runs list only a subset
of flags, so do not call them independent full-environment proof. The separate
v26 nonspeculative matrix's full live-config receipts also contain flag1, but
that does not mean those nonspeculative requests execute a K5 producer.
Thus fused QKV is configured in the qualified C1 profile; source establishes
its pack remains on that path. No per-kernel native trace is claimed here.

## First slice: M5 store layout only

Baseline A is the actual fused M5 QKV export PLUS the existing pack kernel.
Candidate B performs those same three NVFP4 projections and directly stores
the final BF16 values into row-packed output. Do not compare only against
three separate GEMVs and attribute their launch savings to this candidate.

Minimal proposed common-body change, subject to separate source approval:
add a compile-time output-row multiplier defaulting to1 to
`w4a16_gemv_batchm_impl<MAX_M>`. The sole arithmetic-body delta is its final
destination index, from `t*N+n` to `t*OUTPUT_ROW_MULTIPLIER*N+n`. Existing
instantiations use1 unchanged; the new fixture-local QKV wrapper uses3 and
passes `C + plane*N`. Keep the existing M5 launch bounds `(256,5)`, grid,
weight choice, group16 unpack, scale2 application, two-phase accumulation,
shuffle/reduction order, BF16 rounding and all synchronization unchanged.
No output-store callback framework, runtime stride API, duplicated math body,
new production export, kernel flag or model caller in the first slice.

Root could instead approve an equally small private specialization, but no
large copied kernel is justified. Snapshot/hash the original body before any
edit and verify all non-template/store text remains identical. Compile ordinary
exports too; preserving default syntax is not sufficient performance evidence.

The layout proof for each plane p in0..3, row t in0..M, column n in0..N is:

- old producer index `(p*M+t)*N+n`, then pack maps it to `(t*3+p)*N+n`;
- new base `p*N` plus store `t*3*N+n` is exactly that destination;
- disjoint (t,p,n) tuples cover precisely M*3*N BF16 elements, without touching
  the reserved inactive rows or side-projection beta/f_a/g_a allocations.

Keep the candidate destination a distinct allocation in the standalone gate.
Future engine integration requires a separate lifetime proof before writing
`ssm_qkvz` earlier, including every caller and capture path. The standalone
test does not prove that arena is dead during all intervening projections.

## Standalone fixture and bounds

Proposed new `scripts/dev/bench_glm_packed_qkv.cu`, <=500 lines, SPDX AGPL.
Reuse the reviewed allocation/CLI/fault/timing patterns from
`bench_glm_w4a16_m10.cu`, not a new general benchmark framework. Preserve all
earlier exact10 fixture/results/archives. Include the common producer and pack
sources; no model, NCCL, host-memory registration or serving image needed.

Start with M5 only, N/K=(4096,4096) actual local KDA projection and (7,80) for
non-multiple-of-four output/grid tails. K remains divisible16 with correctly
aligned rows; this is not arbitrary K-tail or padded-input support. Use nine
profiles per shape: three seeds at two finite scale2 regimes, zero scale2,
finite E4M3 subnormal scales and signed impulse. Three projection owners must
have distinct packed bytes, scales and scale2 values whenever the profile
permits; never reuse one weight pointer three times as the only positive case.

Compare full bytes against both (A) actual fused M5+pack and (C) independent
scalar projections mapped into row-packed order on the host. Preserve the
existing scalar CUDA arithmetic oracle and a host algebraic impulse oracle;
do not claim random-profile host-double parity. Check finite live outputs,
all reserved rows M..15 and 128-byte guards on every allocation. Exercise
same-address Q/K/V-owner permutations and row permutations to expose both
plane and row transpositions; restore the original ordering for timing.
No convolution or recurrent math needs changing to establish layout identity.

Three distinct full QKV weight owners exceed the older fixture's16MiB cap.
Use an explicit32MiB ceiling, not a silently widened allocation budget:
three packed weights + three scale arrays + one sixteen-row input + four
sixteen-row QKV outputs =11 guarded allocations. At N=K=4096 the payload is
`3*(N*K/2+N*K/16) + 16*K*2 + 4*(16*3*N*2)` =30,015,488 bytes;
including11*256 guard bytes gives30,018,304 bytes. scale2 stays a scalar kernel
argument. Host payloads must also be accounted and kept bounded. Any added
device buffer requires a revised exact budget before source approval.
The fourth output is reserved for later M10 controls; M5 may allocate less.
CUDA context/sanitizer overhead is separate from explicit allocation accounting.

Strict CLI must reject malformed/duplicate options and repetitions>100 before
any CUDA call. Safe in-bounds output/unused-row/canary corruption and undersized
budget modes must produce their expected failure exits before normal gates.
These are comparison sensitivity tests, not reproduced kernel bugs. A new
symbol compile RED is mechanical evidence only. A layout-forwarding scaffold
can provide genuine new-contract runtime RED before switching the store mapping;
do not deliberately damage the existing production producer to create a RED.

## M10 only after M5 is closed

Use the same store specialization at MAX_M10, with no new launch bounds or
parameter search. Retain the already-qualified fixture-local exact10 arithmetic
and its independent two-M5/scalar controls. Still no production M10 export or
Rust dispatch. Run the same18 shape/profile cases at each width, not a selective
best-shape subset; M10 also swaps its two five-row segments and reverses rows
within segments at fixed addresses.

Separate two questions: (1) exact10 plane-output+pack versus exact10 row-packed
store isolates packing; (2) two M5+pack versus direct M10 combines row batching
and layout effects. For control2, put each packed five-row segment at its own
correct offset; do not concatenate projection-major segments and apply an
incorrect single ten-row pack. All outputs must match the scalar oracle.
Flat projection rows are independent, but this never permits the downstream
conv/recurrent kernels to treat [5,5] as one ten-token request. Owner maps,
intermediate snapshot capacity and acceptance remain separate Partition C work.

## Timing, safety and promotion boundaries

M5: time A=fused+pack and B=direct, with ten warmups, five rounds, <=100 calls
per arm and alternating arm order; retain the one-round order imbalance and
run a fresh process repeat. Time production shape/profile0 only; scalar oracle
and host checks stay outside event regions. Retain all per-round durations.
For later M10, retain all three arms and explicit rotating order as in the
exact10 plan. Record ptxas registers/shared/spills and reject nonfinite or
nonpositive durations. Numerical/memcheck runs use repetitions0.

At P4096 pack touches122,880 bytes each direction for M5,245,760 for M10.
Removing one launch per M5 and those reads/writes is the proposed structural
saving, not predicted endpoint acceleration or a cold-weight bandwidth claim.
Only after successful standalone gates consider a separately reviewed actual
projection-to-conv/snapshot region measurement with unchanged precision,
kernel flags, graph mode and owner lifetimes. Full target and endpoint tests
remain necessary; do not add this hypothetical gain to earlier M10 ratios.

Root-only future gates use the established CUDA13.0 C++17/O3/--fmad=false/
sm_121a recipe, isolated source hashes and ptxas report; no native engine build.
Both serving models stopped, no compile/GPU/model overlap, compile container
4GiB memory+swap/two CPUs/runc/no GPU; execution2GiB memory+swap/two CPUs/one
head GPU/no network/300s timeout. Check available host memory and swap0 before
and after; preserve exact containers and expected exits/OOM=false. Memcheck
error-exit99, then two timing processes only if all preceding gates pass.
Unexpected failure/timeout stops subsequent execution pending root diagnosis.
No reset, clock changes, wider memory policy or concurrent node work.

Freeze source/plan/unchanged-dependency hashes before root review and execution.
Archive all controls, expected failures and negative timings honestly. Neither
this plan nor a standalone PASS activates M5/M10 in Atlas.

## Root execution amendment: isolated source only

Root reviewed the complete M5 fixture and approves a bounded standalone run.
Apply `glm_packed_qkv_store.patch` only to a copied source tree, leaving the
repository production kernel and all engine dispatch unchanged. The patch is
exactly the template's default multiplier and final store-index change above.
The fixture plus the patch is the reproducible candidate; retain copied source
hashes and compare its two-line delta before compilation. Do not install this
modified common source into a serving image from a standalone result.
Use numerical and memory-sanitizer checks, then initial/fresh timing; no broad
CPU suite is needed for this isolated source experiment. This time-boxed kernel
measurement complements, rather than replaces, the concurrency driver work.
