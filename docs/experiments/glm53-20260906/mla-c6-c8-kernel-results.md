# C6/C8 absorbed-MLA projection prerequisite

2026-09-09. Isolated GB10 fixture, not a serving benchmark or integrated image.
Only three exports instantiate the existing row-independent template at6/7/8;
no arithmetic, old export, projection ABI, weight layout or engine dispatch
changed in this kernel slice. GLM already uses the canonical deepseek-v4-flash
kernel source. Host C2..C8 dispatch and admission are separate work in progress.

## Exact source and native evidence

Campaign directory: `/home/abc/storage/models/atlas-campaigns/20260908/`;
receipts `glm-mla-c8-*`. Isolated remote tree:
`/home/mangokid/atlas-glm53-deploy-20260906/phase7/glm-mla-c8.yrStVA/`.
The retained RED and GREEN source archives do not overlay the engine builder.

- Canonical kernel SHA256:
  `2427a82ad49d28b2e78bf3b12de3c91c55ce21a3d089f88d9d5a58ff39dcd14f`.
- Expanded fixture SHA256:
  `b4ab434ce3f0d9e28dad6bca77f97c336035d5baad479450754d4f45d71623ea`.
- Native fixture binary SHA256:
  `0d2123844ff6150fcb28d9bc64d48f432714193c3361a0a0d53dd570f2e26404`.
- Builder image:
  `79bcc40f7d5cf3aa749c41e83ebd0d20c2154920dedf19907b4e3bbb262f4bbb`.

Actual nvcc RED: exactly three undefined6/7/8 symbols, exit2/OOMfalse, before
adding the exports. GREEN: exit0/OOMfalse using C++17, `-O3 --fmad=false
-arch=sm_121a`. Compiler reports zero stack/spill traffic; batch6/7 use48 registers
and batch8 uses56. Compilation used no GPU, CPUs0/1, memory=swap ceiling4GiB.

Numerical and memcheck runs both pass47 cases:26 retained cases plus21 new
cases. Exact BF16 equality with scalar calls includes padding; all new widths
also pass exhaustive host-double dots, finite/guard checks and two same-address
row permutations. Padded/unpadded GLM Q-absorption and V-extraction plus odd-tail
shapes are covered. Existing M10/two-M5 controls are unchanged. Memcheck reports
zero errors. Peak tracked device allocation is9,559,808 bytes, below16MiB;
remaining tracked bytes0. This proves these linear operators, not full attention,
causal metadata, recurrent state or multi-request correctness.

## Repeated standalone latency

Microseconds per projection over32 local heads. Each reported latency is a
median of five alternating-order rounds,100 repetitions per arm after warmup;
allocations and correctness checks are outside event timing. Both timing runs
also pass all numerical/canary checks.

| Rows | Projection (N,K) | Initial scalar / batch µs | Repeat scalar / batch µs | Repeat speedup |
| --- | --- | ---: | ---: | ---: |
| 6 | Q absorption (512,256) | 73.076 /34.096 | 72.851 /32.799 | 2.221× |
| 6 | V extraction (256,512) | 62.006 /22.475 | 61.312 /22.537 | 2.721× |
| 7 | Q absorption (512,256) | 85.172 /37.454 | 85.261 /36.891 | 2.311× |
| 7 | V extraction (256,512) | 71.612 /27.229 | 73.913 /24.602 | 3.004× |
| 8 | Q absorption (512,256) | 97.407 /43.155 | 97.237 /43.207 | 2.251× |
| 8 | V extraction (256,512) | 85.327 /28.698 | 81.736 /29.236 | 2.796× |

GPU runs use isolated runc containers, one GPU, CPUs0/1, memory=swap ceiling2GiB,
timeout300s. No model, compiler or other GPU fixture overlapped. Each numerical,
memcheck and timing container exited0/OOMfalse. Both nodes retained zero used
swap and no GPU applications after each phase; final available memory was
head118757MiB/worker118813MiB. No reset, clock or driver changes.

Next: integrate genuine independent rows through every draining width1..8,
including exact handles, slot ownership and pre-load memory accounting, then
qualify eager/graph full-model coherence, tool calls, retrieval and throughput.
These projection speedups do not establish C6>=64 or C8>=72 aggregate tok/s.
