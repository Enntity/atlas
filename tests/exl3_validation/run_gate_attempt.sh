#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Run one external gate collection while preserving its output, captures, and both ranks' logs.
set -u

usage() {
  echo "usage: $0 ENGINE NAME GATE_RUN [GATE_RUN_ARGS...]" >&2
  exit 2
}

[[ $# -ge 3 ]] || usage
engine=$1
name=$2
driver=$3
shift 3
[[ $engine == ref || $engine == cand || $engine == candkda0 ]] || usage
[[ $name =~ ^[A-Za-z0-9_-]+$ ]] || usage
[[ -x $driver ]] || { echo "gate driver is not executable: $driver" >&2; exit 2; }
capture_root=${GATE_CAPTURE_ROOT:-/tmp}
[[ $capture_root == /* && $capture_root =~ ^/[A-Za-z0-9_./-]+$ ]] || {
  echo "capture root must be an absolute path containing only /, letters, digits, dot, underscore, or dash" >&2
  exit 2
}

prefix=ref
[[ $engine == ref ]] || prefix=cand
archive="~/gate/attempts/$name"

# Reserve the attempt directory before running. A reused name must never
# overwrite evidence from an earlier execution or append to an old collection.
for rank in 0 1; do
  spark$rank "test ! -e ~/gate/$name && mkdir -p ~/gate/attempts && mkdir $archive" || exit $?
done

preserve() {
  local status=$?
  local preserve_status=0
  trap - EXIT
  for rank in 0 1; do
    spark$rank "log_status=0; docker logs --timestamps gate-$prefix-$rank > ~/gate/attempts/$name/docker-rank$rank.log 2>&1 || log_status=1; inspect_status=0; docker inspect gate-$prefix-$rank > ~/gate/attempts/$name/inspect-rank$rank.json 2>&1 || inspect_status=1; [ \$log_status -eq 0 ] && [ \$inspect_status -eq 0 ]" || preserve_status=1
  done
  spark0 "if [ -d ~/gate/$name ]; then mkdir -p ~/gate/attempts/$name/artifacts && cp -a ~/gate/$name/. ~/gate/attempts/$name/artifacts/; fi" || preserve_status=1
  for rank in 0 1; do
    spark$rank "for capture in gate-ref gate-cand; do if [ -d $capture_root/\$capture ]; then mkdir -p ~/gate/attempts/$name/captures/\$capture && cp -a $capture_root/\$capture/. ~/gate/attempts/$name/captures/\$capture/ || exit 1; fi; done" || preserve_status=1
  done
  if [[ $status -eq 0 && $preserve_status -ne 0 ]]; then status=1; fi
  exit "$status"
}
trap preserve EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"$driver" "$engine" "$name" "$@"
