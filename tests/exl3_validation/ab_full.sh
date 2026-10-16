#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# bench_tp2 with the verified full decode stack; extra args are appended.
# usage: ab_full.sh LABEL [BINARY] [extra bench_tp2 args...]
set -euo pipefail
label=$1; shift
# One bench at a time across every worktree/agent on this host.
exec 9>/tmp/atlas-bench.lock
flock 9
binary=${1:-/home/colum/atlas-dev/bin/spark}; [[ $# -gt 0 ]] && shift
O=/home/colum/Workspace/atlas-exl3-validation-20261010/perf-ab-20261015
cd "$(dirname "$0")/../.."
python3 tests/exl3_validation/bench_tp2.py --image atlas-exl3-validation:src-4ee96c12b44c-rust1.97.1 \
  --binary "$binary" --cpuset 15-19 --label "$label" --out "$O" \
  --env ATLAS_EXL3_SPLIT_MMA=1 --env ATLAS_EXL3_SPLIT_MHC=1 --env ATLAS_RDMA_ALLREDUCE=1 \
  --env ATLAS_EXL3_COOP_GEMM=1 --env ATLAS_EXL3_COOP_MOE=1 --env ATLAS_EXL3_COOP_FUSED_QKV=1 \
  --env ATLAS_EXL3_PAR_SMALL=1 "$@" > "$O/$label.log" 2>&1
d=$(tail -1 "$O/$label.log")
python3 - "$d" <<'PY'
import json, sys
d = sys.argv[1]
for name, w in json.load(open(d + '/summary.json'))['workloads'].items():
    print(name, 'tpot p50', w['tpot_s']['p50'], 'ok', w['n_ok'])
print('hashes', [ (json.loads(l).get('output_sha256') or '')[:12] for l in open(d + '/raw.jsonl')])
PY
echo "$d"
