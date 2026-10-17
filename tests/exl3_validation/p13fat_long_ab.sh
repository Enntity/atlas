#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# p13fat long-prompt A/B: same binary (spark.p13fat), grouped fat-MoE flag off
# vs on, on a ~3k-token prompt plus the 761-token prompt as a same-session
# control. Chunk 128 (default descriptor profile), 3 repeats each.
# usage: p13fat_long_ab.sh LABEL_PREFIX
set -euo pipefail
cd "$(dirname "$0")/../.."
P=${1:?label prefix}
BIN=/home/colum/atlas-dev/bin/spark.p13fat
E=(--env ATLAS_EXL3_COOP_MOE_V2=1 --env ATLAS_EXL3_COOP_MOE_V3=1 --env ATLAS_EXL3_COOP_MOE_V6=1
   --env ATLAS_EXL3_COOP_MOE_V7=1 --env ATLAS_EXL3_SHARED_OVERLAP=1 --env ATLAS_DECODE_ARENA_FILL=skip
   --env ATLAS_NO_DECODE_GRAPHS=0 --env ATLAS_NO_DECODE_GRAPHS_MULTISEQ=0 --env ATLAS_DEBUG_NO_GRAPH=0
   --env ATLAS_EP_GRAPHS=1 --env ATLAS_KDA_FLA_GRAPH=1
   --workload ab_prose_prefill_3k --workload ab_prose_prefill)
echo "### $P-off"
bash tests/exl3_validation/ab_full.sh "$P-off" "$BIN" "${E[@]}"
echo "### $P-on"
bash tests/exl3_validation/ab_full.sh "$P-on" "$BIN" "${E[@]}" --env ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1
echo "### done"
