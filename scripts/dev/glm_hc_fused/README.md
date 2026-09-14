# Rejected standalone HC producer/consumer screen

Adapted shipped Atlas HC post and finalizer equations; producer newly written
with WMMA TF32 tiling. Engine base f0cf06af7e116a2ce3622b5d64a8b4f06e8f6816
(imported Mango fork https://github.com/Mango-kid/atlas plus retained changes).
The baseline uses the runtime TF32 cuBLASLt layouts and first heuristic, cached
outside timing. No production dispatch changes.

DeepSeek finalizer draft was reviewed as a mechanical replacement of RMS
reduction with supplied inverse RMS. Its producer draft contained missing
coordinates, incomplete weight loads and uninitialized accumulators; rejected
before compile, then parent wrote candidate.cuh independently from reviewed
tile specification. Preserve AGPL notices on all derived source.

Native CUDA13.0 O3 sm121a with fmad=false:106 registers/thread,40,448 shared
bytes,zero spills. Host oracle checks pass. Native M1/3/33 in both alias modes
and M4096 in-place pass exact FP32 highway, explicit TF32/FP64 dot and inverse
RMS oracles, own-raw finalizer oracles and documented changed-chain tolerances.
Synthetic exact TF32 ties, saturated gates, zero rows and random nonuniform
streams/coefficients included. This is not whole-model numerical equivalence.

M4096 median complete chain5.734304 ->4.630464ms (1.238386x), below1.8x
single-route gate. Two warmup and15 interleaved measured pairs; common reset
and cuBLAS setup excluded. Remainingtiming fixtures skipped. No model load
or integration for this candidate. Partial gain retained for future explicit
combined-route assessment; do not call the original gate passed.

Harness initially offset allocations128bytes for redzones, violating cuBLAS
workspace alignment (status7). Corrected to256-byte guardprefix/suffix; native
run passed without changing candidate math. Peak fixture1443.213MiB,2GiBcap
and4GiBfree reserve. Rawreceipts remain in the private research experiment.

Build natively from engine root:

```sh
nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a --fmad=false \
  -Xcompiler=-ffp-contract=off -Xptxas=-v \
  scripts/dev/glm_hc_fused/bench.cu -lcublasLt -o /tmp/hc-fused-bench
/tmp/hc-fused-bench --host-test
/tmp/hc-fused-bench --run
```

Run only on an available GPU. The first timed shape fails fast below1.8x.
The standalone does not qualify cross-layer lookahead, ownership,32K service,
MTP rollback or workload throughput.
