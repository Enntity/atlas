# Standalone kernel microbenchmarks (GB10)

Small nvcc harnesses for one kernel at a time. Each one times the production
kernel against variants and checks the outputs byte for byte. The bit-exact
decode wins on `perf/decode-vllm-parity-20261015` were found with these:
KDA decode float4, the mHC finish load hoist, warp top-k and the rounded mHC
split.

| Dir | Kernel(s) | Compares |
|---|---|---|
| `kda_decode/` | `kda_fla_recurrent_decode` (`kernels/gb10/common/kda_fla_decode.cuh`) | rows/warps/vector-I/O variants; 34 L2-cold state buffers |
| `mhc_finish/` | `exl3_mhc_pre_partials` + `exl3_mhc_pre_norm_split`; `rbench.cu` = rounded site vs `exl3_mhc_pre_partials_rounded` | `v3.cuh` has `%globaltimer` stage markers; `v4.cuh` drops the residual writeback (timing only) |
| `topk/` | `exl3_topk_sigmoid_par` | `orig.cuh` = the pre-`eeb2dc58` block-tree version; 2000 trials with forced ties |

Build in the spark1 builder container (it has nvcc but no GPU), then run on
the spark1 host, which has the GPU. Always run under the shared bench lock,
because these perturb any concurrent A/B:

```bash
# from this machine; the repo checkout is synced to ~/atlas-dev/src on spark1 by devbuild.sh
flock /tmp/atlas-bench.lock ~/.local/bin/spark1 '
  cd ~/atlas-dev/src/tests/exl3_validation/microbench/mhc_finish &&
  docker exec atlas-dev-builder sh -c "cd /build/tests/exl3_validation/microbench/mhc_finish &&
    nvcc -O3 -arch=sm_121a -std=c++17 --fmad=true -DTQ_PLUS_SIGNS --expt-relaxed-constexpr \
         -I/build/kernels/gb10/common -o /target/mb_mhc bench.cu" &&
  ~/atlas-dev/target/mb_mhc'
```

Compile flags must match the module's production flags:
- `exl3_mhc` builds with `--fmad=true`.
- Every other EXL3/common module builds with `--fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr`.

See `kernels/gb10/*/KERNEL.toml` `[build.module_nvcc_flags]`. A standalone
"bit-exact" claim is only a precondition: confirm it in-model with
`ab_full.sh` and the unchanged `output_sha256`.
