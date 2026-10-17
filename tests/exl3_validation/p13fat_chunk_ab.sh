#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Chunk-size A/B on the p13fat binary with the grouped fat-MoE route ON.
# Larger --max-prefill-tokens fills the 16-row grouped units better (fewer unit
# passes) and cuts launch counts. Serve args come from the descriptor profile
# selected by ATLAS_BENCH_DESCRIPTORS (bench_tp2.py honours it).
# Env: WL (workload, default ab_prose_prefill), PREFIX (label prefix).
# Chunk size changes the per-expert row counts per launch, hence the thin/fat
# profile split: outputs differ from chunk 128 (numerics-changing, label it).
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=/home/colum/atlas-dev/bin/spark.p13fat
PROF=/home/colum/Workspace/atlas-exl3-validation-20261010/prefill-chunk-profiles
E=(--env ATLAS_EXL3_COOP_MOE_V2=1 --env ATLAS_EXL3_COOP_MOE_V3=1 --env ATLAS_EXL3_COOP_MOE_V6=1
   --env ATLAS_EXL3_COOP_MOE_V7=1 --env ATLAS_EXL3_SHARED_OVERLAP=1 --env ATLAS_DECODE_ARENA_FILL=skip
   --env ATLAS_NO_DECODE_GRAPHS=0 --env ATLAS_NO_DECODE_GRAPHS_MULTISEQ=0 --env ATLAS_DEBUG_NO_GRAPH=0
   --env ATLAS_EP_GRAPHS=1 --env ATLAS_KDA_FLA_GRAPH=1
   --env ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1
   --workload "${WL:-ab_prose_prefill}")
for n in "$@"; do
  echo "### chunk$n"
  ATLAS_BENCH_DESCRIPTORS="$PROF/chunk$n" \
    bash tests/exl3_validation/ab_full.sh "${PREFIX:-p13chunk}$n" "$BIN" "${E[@]}"
done
echo "### done"
