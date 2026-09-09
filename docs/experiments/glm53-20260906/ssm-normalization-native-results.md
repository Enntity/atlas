# SSM normalization: bounded native racecheck qualification

Status: the standalone GLM-sized kernel gate passes after a single-writer
correction. This is **not** proof of the cause or resolution of the earlier
long-context generation failures, nor full-model qualification.

## Defect and effective scope

Both common FP32/FP16 kernels reduce four warp sums, then let threads0–3 write
the same shared `norm_sums[0]`. Only thread0 holds the complete reduction.
NVIDIA specifies that the final writer is undefined when multiple threads in
a warp perform non-atomic writes to the same shared address.
[CUDA13.0 Programming Guide, SIMT Architecture](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#simt-architecture).

The candidate changes only those two stores to `if (tid == 0)`, plus explanatory
comments, in `kernels/gb10/deepseek-v4-flash/nvfp4/ssm_state_norm.cu`. The common
file is unchanged. GLM's `MODEL.toml` redirects `kernel_source` to DeepSeek;
the build resolves that alias before selecting per-quantization shadows.
Consequently this changes the GLM/DeepSeek NVFP4 artifacts, not other model
targets. DeepSeek's43 layers are all attention and its zero-SSM normalization
dispatch returns before launch; GLM is the active normalization consumer here.

All Model call sites converge on `normalize_ssm_states_dispatch`, which launches
`block.x=v_dim`. The existing four-warp implementation assumes exactly128
threads. This gate covers GLM's `k_dim=v_dim=128`; it does not qualify arbitrary
GDN/Mamba dimensions or silently broaden the correction to other models.

## Exact standalone evidence

Receipts are retained under the external campaign
`atlas-campaigns/20260909/glm-native-controller/`:

| Source | Numerical result | Racecheck | Memcheck |
| --- | --- | --- | --- |
| Original common | FP32 and FP16 PASS |32 displayed hazards/errors; exit99 | Not run in this slice |
| Candidate shadow | FP32 and FP16 PASS inside both sanitizer runs |0 hazards/errors |0 errors |

Raw files: `ssm-norm-red-build.log`, `ssm-norm-red-native.log`,
`ssm-norm-red-racecheck.log`, `ssm-norm-green-build.log`,
`ssm-norm-green-racecheck.log`, `ssm-norm-green-memcheck.log`.
There is no separate `ssm-norm-green-native.log`: its numerical evidence is in
the GREEN sanitizer logs. **The original numerical gate passed too.** The
genuine RED is the actual shared-memory write/write race report, not a claimed
numerical mismatch or source-text test.

| Artifact | SHA256 |
| --- | --- |
| Common source | `804f63d541a1d7713c4162c135c78880e7b0c4a30ec18bbf7ef1ab77ca94aa3e` |
| Shadow source | `fcdb6950d1c25621651ceb15932a7dc1319999ba7021cd13b54e70f199ac4c00` |
| Reproducer | `cd6ae76905186acd78bd269fb80d99e0f17e701a6df40b2ba5068b9f8f8f8b88` |
| RED executable | `925b233da01b5a79f241f4e26ef9f2171b6937878f6d4b3267f3f2992985a7d0` |
| GREEN executable | `cfa4f8db3b89b42faef06ad986ee5869dec8c9d36fe12b9ce966df5cf99d3f83` |

The external `bench-ssm-norm-release.cu` is byte-identical to the portable
`scripts/dev/bench_glm_ssm_state_norm.cu`. It includes the actual source selected
by `ATLAS_SSM_NORM_SOURCE`; data and tolerances are unchanged between builds.
Fixed coverage:32 heads, three pointer-permuted layers,16 freshly reset launches
per storage type, zero/below200/exact200/201 norms, and unequal warp energies
with the dominant warp rotated through each lane of the second reduction.
Every matrix is checked against a CPU double-norm oracle; unclamped values
must remain bitwise unchanged. Clamped FP32 has strict numerical tolerances,
FP16 allows at most one half ULP, and both have independent norm checks.
Layer/table canaries are checked; peak explicit device allocation is under7MiB.

Reproduction command form, inside an independently authorized idle GPU window
with the exact source paths and executable identity pinned by the operator:

```sh
nvcc -std=c++17 -O3 -arch=sm_121a -lineinfo \
  -DATLAS_SSM_NORM_SOURCE='"/ABS/PINNED/ssm_state_norm.cu"' \
  scripts/dev/bench_glm_ssm_state_norm.cu -o /ABS/NEW/ssm-norm-test
compute-sanitizer --tool racecheck --error-exitcode 99 /ABS/NEW/ssm-norm-test
compute-sanitizer --tool memcheck --error-exitcode 99 /ABS/NEW/ssm-norm-test
```

These commands describe the bounded reproducer, not a serving launch or a
replacement for the retained build/output hashes. No Model, NCCL, concurrent
request ownership, speedup or hardware-recovery guarantee follows from them.

## Separate scheduler parity evidence

`normalization-dispatch-red2.log` records two actual scheduler-entry CPU failures
using a real SSM-pool fixture and worker F0 replay: the first head chunk issued
no normalization (`[]`) instead of default stream `[7]`; the continuation head
issued `[1027]` instead of `[7]`. The worker used `[7]`. These are dispatch/order
findings, independent of the kernel's internal reduction race.
`normalization-dispatch-green.log` then records2 PASS/0 FAIL in0.26s, with each
isolated child running its actual entry test. This proves the checked CPU
dispatch behavior, not native SSM arithmetic or long-context quality.

Native full-model quality must be rerun on the final combined source. These
standalone source-hash-pinned checks were performed before the combined commit;
they are not a complete repository benchmark gate. Neither this kernel result
nor the earlier `c853bafa` run qualifies that subsequent serving artifact.
