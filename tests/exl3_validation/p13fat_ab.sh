#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# p13fat model A/B: same binary (spark.p13fat) with the new flag off (neutrality
# control) and on (effect), both workloads, 3 repeats each via ab_full.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=/home/colum/atlas-dev/bin/spark.p13fat
E=(--env ATLAS_EXL3_COOP_MOE_V2=1 --env ATLAS_EXL3_COOP_MOE_V3=1 --env ATLAS_EXL3_COOP_MOE_V6=1
   --env ATLAS_EXL3_COOP_MOE_V7=1 --env ATLAS_EXL3_SHARED_OVERLAP=1 --env ATLAS_DECODE_ARENA_FILL=skip
   --env ATLAS_NO_DECODE_GRAPHS=0 --env ATLAS_NO_DECODE_GRAPHS_MULTISEQ=0 --env ATLAS_DEBUG_NO_GRAPH=0
   --env ATLAS_EP_GRAPHS=1 --env ATLAS_KDA_FLA_GRAPH=1
   --workload ab_prose_prefill --workload ab_prose_long)
echo "### p13fat-off (flag off, neutrality control)"
bash tests/exl3_validation/ab_full.sh p13fat-off "$BIN" "${E[@]}"
echo "### p13fat-on (flag on)"
bash tests/exl3_validation/ab_full.sh p13fat-on "$BIN" "${E[@]}" --env ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1
echo "### done"
