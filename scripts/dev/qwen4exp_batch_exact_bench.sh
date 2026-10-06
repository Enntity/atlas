#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the PTX the qwen4_exp exact-batching bench JIT-loads (production flags,
# as crates/atlas-kernels compiles the qwen3.8-flash-next target), build the
# bench, run it.
#   scripts/dev/qwen4exp_batch_exact_bench.sh check|time
# Run from the repository root on a GB10. See qwen4exp_batch_exact_bench.cu.
set -euo pipefail
out=${QB_BENCH_DIR:-/tmp/qb-exact-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(--ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr --Werror all-warnings)
pids=()
for stem in moe_shared_expert_fused dense_gemv_bf16 dense_gemv_bf16_batchm moe_permute w4a16_gemv w8a16_gemv w8a16_gemv_batch4; do
  "$nvcc" "${flags[@]}" "kernels/gb10/common/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
done
for stem in qwen4exp_moe_rows; do
  "$nvcc" "${flags[@]}" "kernels/gb10/qwen3.8-flash-next/nvfp4/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
done
"$nvcc" -O3 -std=c++17 -arch=sm_121a scripts/dev/qwen4exp_batch_exact_bench.cu -lcuda \
  -o "$out/qwen4exp_batch_exact_bench" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
exec "$out/qwen4exp_batch_exact_bench" "$out" "${1:-check}"
