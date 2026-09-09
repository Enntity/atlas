# C2 compact MoE: reuse existing numerical fixtures

September 9, 2026. Root approved scripts-only implementation. No production
kernel, Rust dispatcher, engine build, serving activation or native execution
is authorized to this author.

Admit `ATLAS_MOE_TEST_ROWS=2` alongside unchanged default4 and existing5 in
`bench_glm_moe_gate_up_m16.cu` and `bench_glm_moe_down_cost.cu`. Retain their
six maps, eager/refreshed-graph runs, scalar/vector scales, guarded outputs,
remote/unused poison, worklist bounds and sampled independent CPU dot oracle.
Two input rows imply16 routed rows, at most2 per expert. The GU fixture still
compares M64 with the existing standalone M16 implementation; this does not
enable production C2 M16. Run down without `--m16` and both without `--timing`.
Derived guarded device peaks are38,034,888B GU and38,174,728B down, below64MiB.

Add `ATLAS_NVFP4_TEST_ROWS=2` to `bench_glm_nvfp4_m4.cu`, preserving default4.
Use the actual seven-argument `w4a16_gemv_batch2(A,B,S,scale2,C,N,K)` export,
not the eight-argument batch4 ABI. Keep the six shapes, full BF16 comparison
against scalar rows, finite/guard checks and existing CPU oracle tolerance.
The two-row permutation is `[1,0]`; the four-row permutation stays `[3,1,0,2]`.
The shared M2 guarded device peak derived from its fixed six shapes is9,487,616B.
Use positional argument0: correctness only. Add a CUDA-free `--host-test`
build mode to check supported shapes, row permutation and scalar row scales.
No changed timing loops or new CUDA arithmetic.

Focused author gates: preserve actual pre-edit compile refusal for rows2 in
the grouped host-only builds; compile/run both host-only fixtures at2/4/5,
shared at2/4, and verify unsupported row counts fail compilation. Shared
host-only mode is new mechanical exposure, not a reproduced numerical bug.
Root alone compiles GPU binaries and runs correctness plus memcheck after the
active base build ends, with model processes stopped and no GPU overlap.
Use root's bounded container resources, at most2GiB container memory, and
require each explicit device payload below64MiB. The shared fixture's existing
broader safety cap is not increased; its fixed shapes allocate far below64MiB.

These are projection/worklist numerical gates, NOT a whole-pipeline oracle.
Grouped inputs are host-generated packed FP4/scales: neither the actual
BF16-to-FP4 quantizer nor fused SiLU-to-FP4 is numerically tested here. Router,
sort, unpermute, NCCL, shared blend, mHC, model arena aliasing and scheduler
ownership are not exercised by these standalone programs. Preserve actual
CPU dispatcher checks and root's eager whole-model OFF/ON quality, followed
by graph/needle/drain/slot-reuse checks before matched warmed C1..4 timings.
Do not claim bitwise whole-MoE equality after changing routed activation
precision, or derive endpoint speed from these correctness-only fixtures.

## Author host receipts (no CUDA compilation or execution)

Eight host configurations compiled and passed: GU/down each rows2/4/5,
shared rows2/4. All three fixtures reject unsupported rows3 at compile time;
shared host-only mode rejects missing `--host-test`. Logs and host binaries
are under campaign `c2-numerical-host.mFvl7W/`: `host-green.log` and
`original-compile-red-replay.log`. The latter replays the unchanged HEAD
grouped sources' rows2 static-assert failures. Actual pre-edit failures were
also observed directly (tool receipts333e57 and3c40a9). One earlier incorrectly
ordered `g++ -x` invocation did not compile anything and is NOT RED evidence.
Scoped diff whitespace checks pass. No clang-format executable was available;
the changes retain the existing local formatting style. Native bodies remain
uncompiled by this author; root reviews and owns their native gates.

Root compile switches: grouped fixtures `-DATLAS_MOE_TEST_ROWS=2`; shared
fixture `-DATLAS_NVFP4_TEST_ROWS=2`. Use the existing production-compatible
CUDA architecture/arithmetic flags. Correctness commands are GU no arguments,
down no arguments, shared `0`; repeat those under memcheck, with no timing
option and no concurrent GPU/model workload. Expected native case summaries:
GU12 (six eager/six graph), down12 (six eager/six graph), shared6 (each checks
both permutations). These are expected counts, not yet native PASS receipts.
