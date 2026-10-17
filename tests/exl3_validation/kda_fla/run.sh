#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build and run the byte-exact reference-policy KDA kernel tests on a GB10 host.
# Usage (repo root on the GPU host): KDA_ORACLE=<oracle>/inductor bash tests/exl3_validation/kda_fla/run.sh
# Kernels compile to compute_121f PTX and are JIT-loaded by the driver, as in production.
set -euo pipefail
: "${KDA_ORACLE:?set KDA_ORACLE to the stage-oracle inductor directory}"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
OUT=${OUT:-$(mktemp -d)}
status=0
for t in prep kkt wy state decode pipeline; do
  "$NVCC" -O3 -std=c++17 -gencode arch=compute_121f,code=compute_121f \
    -I kernels/gb10/glm-5.3-flash/exl3 -I tests/exl3_validation/kda_fla \
    "tests/exl3_validation/kda_fla/test_${t}.cu" -o "$OUT/test_${t}"
  if "$OUT/test_${t}" > "$OUT/test_${t}.log" 2>&1; then
    echo "PASS test_${t}"
  else
    echo "FAIL test_${t} (log: $OUT/test_${t}.log)"; tail -5 "$OUT/test_${t}.log"; status=1
  fi
done
exit $status
