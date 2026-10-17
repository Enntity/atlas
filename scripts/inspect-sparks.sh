#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Read-only inspection of both Sparks (before any GPU work):
# containers (incl. which image the stopped glm53-exl3-atlas-* containers
# reference), GPU processes / ownership, and host identity. Never mutates
# anything. Answer this before any restart or GPU experiment.
#
# Usage: bash scripts/inspect-sparks.sh [--out DIR]
#   --out DIR   also write spark0.txt / spark1.txt receipts under DIR
#               (use a fresh directory under the durable evidence root)
set -euo pipefail

JUMP="${JUMP:-colum@192.168.1.50}"
SPARK0="${SPARK0:-colum@10.50.0.10}"
SPARK1="${SPARK1:-colum@10.50.0.11}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -J "$JUMP")

OUT=""
if [[ "${1:-}" == "--out" ]]; then
  OUT="${2:?--out needs a directory}"
  mkdir -p "$OUT"
fi

inspect() {
  local label="$1" host="$2"
  {
    echo "== host $label ($host) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    timeout 120 "${SSH[@]}" "$host" 'bash -s' <<'REMOTE'
set -uo pipefail
echo "-- identity"; hostname; uname -r; date -u +%Y-%m-%dT%H:%M:%SZ
echo "-- gpu"; nvidia-smi --query-gpu=driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1
echo "-- gpu processes (ownership: anything running here owns the GPU)"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
echo "-- docker ps -a"
docker ps -a --format '{{.Names}} | {{.Image}} | {{.Status}}' 2>&1
echo "-- glm53 container image binding (Config.Image tag + Image sha256)"
for c in glm53-exl3-atlas-0 glm53-exl3-atlas-1 glm53-exl3-head glm53-exl3-worker; do
  docker inspect --format '{{.Name}} | tag={{.Config.Image}} | sha={{.Image}} | state={{.State.Status}} | restart={{.HostConfig.RestartPolicy.Name}}' "$c" 2>&1 | grep -v '^Error' || echo "$c | absent"
done
echo "-- images"
docker images --format '{{.Repository}}:{{.Tag}} | {{.ID}} | {{.CreatedAt}}' 2>&1 | grep -E 'atlas|glm53|^REPOSITORY' || true
REMOTE
  } | { if [[ -n "$OUT" ]]; then tee "$OUT/${label}.txt"; else cat; fi; }
}

inspect spark0 "$SPARK0"
inspect spark1 "$SPARK1"
echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ); nothing was mutated"
