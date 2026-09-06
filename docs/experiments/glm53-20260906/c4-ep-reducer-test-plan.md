# Bounded C4 EP reducer regression

Test only: no checkpoint or distributed process. Compile the production indexed
EP reducer, BF16 sum, and shared-expert blend into one small CUDA executable.
Emulate rank ranges `[0,144)` and `[144,288)` on one GPU, preserving BF16 rounding
after each rank reduction and after their sum. Compare every element with a
CPU oracle, then add a distinct shared contribution exactly once.

Exercise four independent identities and draining/permuted C4/C3/C2/C1 rows;
all routes on either rank, mixed boundary IDs143/144, and identical concentrated
expert choices across rows. Poison unwritten remote expert outputs with NaNs;
also run with invalid remote permutation indices to verify rejection occurs
before memory access. Preserve inactive output rows and 128-byte allocation
guards. Hard allocation cap32MiB; no weights and no NCCL allocations.

This validates reduction/row isolation/shared blending, not FP4 grouped-GEMM
accuracy or distributed collective ordering. Do not require grouped-versus-
scalar model output bit identity: routed activation quantization is lossy.

Root owns compilation and serialized GPU/memcheck execution:

```sh
nvcc -O3 --fmad=false -arch=sm_121a scripts/dev/bench_glm_c4_ep_reduce.cu -o /tmp/bench-glm-c4-ep
/tmp/bench-glm-c4-ep
compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/bench-glm-c4-ep
```

Root execution passed all 40 cases, then repeated all 40 under memcheck with
zero reported errors. Device allocation: 625,408 bytes. Both model containers
were stopped; the test ran in a 1 GiB no-network container on the head Spark.
Raw controller receipt: `/tmp/atlas-glm53-phase3-20260906/v7-ep-reducer-gpu.log`.
