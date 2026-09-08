# Unactivated ten-row MLA projection: standalone qualification

2026-09-08. The new `mla_batched_gemv_batch10` is an unused instantiation of
the existing row-independent template. It is a prerequisite for later paired
K5 verification, not an enabled C2 verifier. Existing CUDA bodies are unchanged;
there is no Rust dispatcher/factory change and no new serving image.

## Evidence and limits

Root ran only the standalone fixture on head gx10-4386, with both serving
nodes stopped. Genuine compile RED: undefined batch10 export, exit2. After the
five-line export addition, production-policy compile passed with
`-std=c++17 -O3 --fmad=false -arch=sm_121a`. M10 uses56 registers,640 shared
bytes, zero stack/spill bytes; M5 uses47 registers and320 shared bytes.

The numerical execution, memcheck execution and two timing executions each
passed26 cases: five each for M2/M3/M4/M5 and six for M10. M10 compares every
BF16 output/padding byte against two existing M5 calls and an independent
ten-scalar-call control. Exhaustive host-double dots check M10/M4; existing
M2/M3/M5 host checks sample columns. All candidate-versus-legacy BF16 mismatch
counts, max_diff values and guard errors are0; host-double errors satisfy the
unchanged tolerance. M10 includes both GLM shapes, padded/unpadded strides, swapped five-row
segments and intra-segment permutations at fixed addresses, plus N9/K12 tail.
K remains divisible4; no arbitrary K-tail or CUDA-graph replay claim.

Memcheck reports `ERROR SUMMARY: 0 errors`. Peak live explicit device
allocation is9,559,808 bytes, with remaining0. This excludes CUDA context and
sanitizer overhead, which stayed within the separate2GiB container ceiling.

## Hot standalone timings

CUDA-event medians of five alternating-order rounds,100 repetitions per arm.
The M10 control is TWO M5 calls, not ten scalar calls. No model weights,
collectives, request scheduling or cold-cache simulation are involved.

| Shape (heads32) | Launch | Two M5 calls, microseconds | M10, microseconds | Speedup |
|---|---|---:|---:|---:|
| Q absorption N512/K256 | Initial |57.706|53.248|1.084x|
| Q absorption N512/K256 | Fresh fixture |57.361|54.753|1.048x|
| V extraction N256/K512 | Initial |36.956|34.803|1.062x|
| V extraction N256/K512 | Fresh fixture |36.928|34.781|1.062x|

This modest operator improvement cannot be converted into a whole-model TPS
forecast. Event timings use repeatedly reused8MiB weights. Full paired MLA
still needs checked ten-row projection/FFN/mHC scratch, owner-specific block
tables and causal lengths; KDA needs two independent temporal histories and
per-owner acceptance rollback. Partition A request ownership remains in work.

## Safety and provenance

All native containers are preserved and stopped. Expected RED exits2; compile,
numerical, memcheck and both timing containers exit0, OOM=false. Compile used
4GiB memory+swap, CPUs0,1 and runc with GPU visibility disabled; executions
used2GiB memory+swap, CPUs0,1, one head GPU and timeout300s. No build/GPU/model
overlap. Final host MemAvailable head/worker118788/118843MiB; swap used0 both.

Frozen source identities (independently reviewed before GPU execution):

- CUDA `d623cbdc6a817f38a8faac1f31d677a3e0a355f5d90347ca0d1839ce3e7a819c`.
- Fixture `4df348a3cb2dd0175d7bde3b1b66626d6437f9c3902948d3407cc6d299416836`.
- Plan `284274134057fd4e467f17fd4b562f1cc97f89516a46c604b64b7c71652a935c`.
- Executable `f17057c25c4aef359fbd7517e5d6d9284af3e9ac5dbc5855513e06d1017911e2`.

Persistent controller receipts: `atlas-campaigns/20260908/glm-mla-m10-*`.
Isolated head work directory: phase7/`glm-mla-m10.0Ac32p`.
Closed source/binary/log archive `glm-mla-m10-verified-receipts.tar`, verified
on controller and head phase7, SHA256
`907a4a94df8ef1a53e4957a5dea33eaa23178d14cb1fd09b10348f1f80b814a0`.
These are source-hashed standalone experiment receipts, not serving-matrix
or whole-worktree CI gate records.
