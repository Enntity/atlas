# Optional GLM HC TF32 startup warmup

`ATLAS_GLM_HC_TF32_PREWARM=1` initializes the HC cuBLASLt TF32 path before the
server becomes ready. Unset or `0` disables it; other values fail startup.
This is separate from the existing small BF16 cuBLASLt startup warmup.

The option requires CUDA, `glm5_next`, hidden width 4096, HC multiplicity 4,
exactly 4100 allocated batch rows, and `ATLAS_HC_CUBLAS_PREFILL` enabled through
its existing `1`/`true`/`yes` values. An unsupported shape, invalid buffer span,
or requested warmup failure aborts startup. The option does not enable HC
cuBLAS dispatch itself or change any decode/verification path.

The factory runs the helper on both ranks immediately after arena allocation,
before model state construction, worker/scheduler handoff, and the actual-free
snapshot used to size KV. It calls the existing production
`tf32_gemm_act_weight_t` for M=4100, M=4096 and M=3515, N=24, K=16384. The wrapper's
FP32 operands/output, TF32 compute mode, layout, alpha/beta, heuristic selection,
and workspace are unchanged. The rows cover the retained four-chunk prefill
profile; this is not an exhaustive warmup of every possible prompt tail.

No scratch allocation is added. The helper zeroes and reuses dead startup arena
regions:

| Region | Used bytes | Purpose |
|---|---:|---|
| `hc_streams` |268,697,600|FP32 activation for the largest shape|
| `gate_logits_f32` prefix |1,572,864|FP32 dummy weight matrix|
| following disjoint span |393,600|FP32 output for the largest shape|

The last two spans need 1,966,464 bytes together. Capacity, pointer addition,
alignment and allocation non-overlap are checked before any GPU operation.
Only these used spans are zeroed. No real model weight, token, KV state, SSM
state, prediction state, sampling state or sequence ownership is read or
changed. The first real forward overwrites the scratch normally.

The helper synchronizes its stream even after a partial failure. Any library
residency is therefore present in the following KV memory calculation. It
does not change the inference reserve, physical headroom guard, target-cache
floor, context length, speculative depth or concurrency settings. The log
records rank, exact shapes, elapsed startup time and free bytes before/after.

## Qualification

Focused CPU tests cover flag/shape admission, short/overlapping/wrapping spans,
the exact three production-call arguments, zeroing boundaries, no added
scratch allocation, and draining a partial launch failure. Run:

```sh
cargo test --locked -p spark-model --lib hc_tf32_prewarm -- --nocapture
```

CPU tests intercept the cuBLAS boundary; they do not claim CUDA execution.
Before enabling the option in a retained serving profile, run the actual helper
on CUDA, then compare fresh-process disabled/enabled starts with identical
canaries and an uncached first request. Confirm library loading moved before
readiness, output correctness, both ranks' capacity/headroom, and client/server
TTFT. Keep startup cost separate and retain the existing benchmark warmup
description. A faster repeated request alone does not qualify this option.
