#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Wait until both Sparks run nothing but dev builders, then ab_full.sh "$@".
set -uo pipefail
for _ in $(seq 1 120); do
  n=$( { "$HOME/.local/bin/spark0" 'docker ps --format "{{.Names}}"'; "$HOME/.local/bin/spark1" 'docker ps --format "{{.Names}}"'; } | grep -cv 'atlas-dev-builder')
  [ "$n" = 0 ] && break
  sleep 40
done
bash "$(dirname "$0")/ab_full.sh" "$@"
