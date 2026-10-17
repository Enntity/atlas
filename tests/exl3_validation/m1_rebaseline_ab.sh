#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Post-merge re-baseline (2026-10-16): the merged main binary against the
# last measured binary (spark.p13fat), then the grouped fat-MoE flag on the
# serving stack's numerics flags. ab_full.sh takes the bench lock per run.
set -euo pipefail
cd "$(dirname "$0")/../.."
OLD=/home/colum/atlas-dev/bin/spark.p13fat
NEW=${NEW:-/home/colum/atlas-merge/bin/spark.main-34d94f34}
# The arms use different binaries: require each build identity separately.
: "${OLD_SOURCE_COMMIT:?Set the actual build commit for OLD}"
: "${NEW_SOURCE_COMMIT:?Set the actual build commit for NEW}"
EXACT=(--env ATLAS_EXL3_COOP_MOE_V2=1 --env ATLAS_EXL3_COOP_MOE_V3=1 --env ATLAS_EXL3_COOP_MOE_V6=1
   --env ATLAS_EXL3_COOP_MOE_V7=1 --env ATLAS_EXL3_SHARED_OVERLAP=1 --env ATLAS_DECODE_ARENA_FILL=skip
   --env ATLAS_NO_DECODE_GRAPHS=0 --env ATLAS_NO_DECODE_GRAPHS_MULTISEQ=0 --env ATLAS_DEBUG_NO_GRAPH=0
   --env ATLAS_EP_GRAPHS=1 --env ATLAS_KDA_FLA_GRAPH=1)
FAT=(--env ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1)
FLAGGED=(--env ATLAS_EXL3_SHARED_PACKED=1 --env ATLAS_GLM_SPARSE_DECODE_FP8_SPLIT=1)
W=(--workload ab_prose_long --workload ab_prose_prefill)
run() {
  local source_commit=$NEW_SOURCE_COMMIT
  [[ $2 != "$OLD" ]] || source_commit=$OLD_SOURCE_COMMIT
  echo "### $1"
  ATLAS_BENCH_BINARY_SOURCE_COMMIT=$source_commit bash tests/exl3_validation/ab_full.sh "$@"
}
run m1-ref      "$OLD" "${EXACT[@]}" "${FAT[@]}" "${W[@]}"
run m1-main     "$NEW" "${EXACT[@]}" "${FAT[@]}" "${W[@]}"
run m1-main-off "$NEW" "${EXACT[@]}" --workload ab_prose_long
run m1-srv-off  "$NEW" "${EXACT[@]}" "${FLAGGED[@]}" "${W[@]}"
run m1-srv-on   "$NEW" "${EXACT[@]}" "${FLAGGED[@]}" "${FAT[@]}" "${W[@]}"
echo "### done"
