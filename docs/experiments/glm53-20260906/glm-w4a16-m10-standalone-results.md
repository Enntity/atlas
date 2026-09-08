# Existing W4A16 batch16 at ten rows: standalone reuse qualification

2026-09-08. This scripts-only experiment qualifies an existing projection for
future paired K5 verification. No production CUDA, Rust dispatch, factory,
admission or serving image changed. Numerical reuse is supported for the tested
shapes; a blanket performance preference is not.

## Evidence and limits

Root ran the isolated fixture on head gx10-4386 with both serving nodes stopped.
The candidate is `w4a16_gemv_batch16` with runtime M=10. Controls are two
disjoint existing M5 calls and ten scalar `w4a16_gemv` calls, not the alternate
single-warp scalar export. The source was independently reviewed before native
execution. CUDA13.0 compilation used
`-std=c++17 -O3 --fmad=false -arch=sm_121a`. Candidate batch16 uses56 registers
and16,960 shared bytes; batch5 uses48 registers and5,344 shared bytes. Both have
zero stack/spill bytes. An unrelated compiled dual-M5 export has spills and is
not executed by this fixture.

Seven harness-sensitivity REDs returned the expected status: output and unused
row corruption2, guard corruption3, allocation budget4, and three invalid
repetition arguments64. Corruption stays inside allocated payload/guard memory;
budget and CLI failures precede CUDA. These are deliberately injected harness
failures, not production kernel regressions.

Numerical, memcheck and two timing executions each passed36 cases: four N/K
shapes (7/80,4096/4096,512/4096,8192/2048), each with three random seeds at two
scale2 values, zero scale2, finite E4M3 subnormal scales and a signed FP4 impulse.
All ten live rows match both controls bitwise and remain finite. All six unused
rows and128-byte allocation guards remain unchanged. Two fixed-address row
permutations cover segment swapping and intra-segment order. The signed impulse
also has an independent host algebraic oracle; general random profiles do not
have a host-double oracle. K is divisible16 and arrays are contiguous: no
arbitrary K-tail, padded-stride, graph replay or model-output claim.

Memcheck reports `ERROR SUMMARY: 0 errors`. Peak live explicit device allocation
is10,290,688 bytes, remaining0, below the16MiB fixture ceiling. Context and
sanitizer overhead are excluded from that counter and separately contained.

## Hot standalone timings

CUDA-event medians of five alternating-order rounds,100 repetitions per arm,
after ten warmup calls per arm. Weights are repeatedly reused. No full-model
weights, collectives, scheduling or cold-cache simulation are involved.

| N/K | Execution | Two M5 calls, microseconds | Batch16 at M10, microseconds | Speedup |
|---|---|---:|---:|---:|
|4096/4096|Initial|86.678|82.009|1.057x|
|4096/4096|Fresh fixture|86.278|84.508|1.021x|
|512/4096|Initial|20.500|15.174|1.351x|
|512/4096|Fresh fixture|20.510|15.215|1.348x|
|8192/2048|Initial|94.724|94.492|1.002x|
|8192/2048|Fresh fixture|94.186|95.078|0.991x|

The smaller projection benefits consistently;4096/4096 improves modestly and
8192/2048 is effectively neutral, including a slight regression on repeat.
This does not establish an end-to-end decode gain. Shape-specific selection
and actual paired-verifier validation remain future work.

## Safety and provenance

All12 native containers are preserved, exited with expected statuses and
OOM=false. Compilation used4GiB memory+swap, CPUs0,1, runc and no GPU visibility.
GPU executions used2GiB memory+swap, CPUs0,1, one head GPU, no network and
timeout300s. No compilation/GPU/model overlap. Postflight MemAvailable was
118781/118851MiB on head/worker, swap used0 both, no running containers.

Frozen SHA256 identities:

- Unchanged production CUDA: `eba5c5ee826283bb71e4ffbbf277a1586556cbab249d0de32607a14c84ce8db6`.
- Fixture: `8cd960740ba79a4cf180deeafa74154daada5f87fcbdc04b2e0dc7f35abd41a4`.
- Plan: `76f37cb4d3ec08826b14126831019d97bf979c66a9238515e2082aee60624a8b`.
- Executable: `d0d28e41a8d8af8db25e0092599298b4e9ede18db3b583ec3aea1c221c29189a`.

Persistent controller receipts: `atlas-campaigns/20260908/glm-w4a16-m10-*`.
Isolated head source/build directory: phase7/`glm-w4a16-m10.mKPkSR`.
Closed source/binary/log archive `glm-w4a16-m10-verified-receipts.tar`, verified
on controller and head phase7, SHA256
`2a948a42d2c73ea90b670f0761863af025ee3eca4de7ad8ae164474471a575ae`.
These are source-hashed standalone receipts, not serving-matrix or whole-tree
CI gate records. The v26 serving image and measured model baseline are unchanged.
