#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Run from a reviewed source snapshot AFTER freeing serving GPU/system RAM.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo 'usage: build_image.sh SOURCE_SNAPSHOT_SHA256 SOURCE_COMMIT RUST_STABLE_VERSION BUILD_CPU_ID' >&2
    exit 2
fi
snapshot_sha=$1
source_commit=$2
rust_version=$3
build_cpu=$4
[[ $snapshot_sha =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid snapshot SHA256' >&2; exit 2; }
[[ $source_commit =~ ^[0-9a-f]{40}$ ]] || { echo 'invalid source commit' >&2; exit 2; }
[[ $rust_version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'supply an exact stable Rust release' >&2; exit 2; }
[[ $build_cpu =~ ^[0-9]+$ ]] || { echo 'supply one permitted CPU ID' >&2; exit 2; }
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_root"
devel_image=$(awk '$1 == "FROM" && $NF == "builder" {print $2; exit}' docker/gb10/Dockerfile)
runtime_image=$(awk '$1 == "FROM" && $NF != "builder" {print $2; exit}' docker/gb10/Dockerfile)
[[ $devel_image == *@sha256:* && $runtime_image == *@sha256:* ]] || {
    echo 'main Dockerfile must contain digest-pinned CUDA build/runtime images' >&2
    exit 2
}
validation_image="atlas-exl3-validation:src-${snapshot_sha:0:12}-rust${rust_version}"
docker build --no-cache --progress=plain \
    --build-arg "CUDA_DEVEL_IMAGE=$devel_image" \
    --build-arg "CUDA_RUNTIME_IMAGE=$runtime_image" \
    --build-arg "RUST_TOOLCHAIN=$rust_version" \
    --build-arg "ATLAS_BUILD_CPU=$build_cpu" \
    --build-arg "ATLAS_GIT_SHA=$source_commit" \
    --build-arg "ATLAS_SOURCE_SHA256=$snapshot_sha" \
    -f docker/gb10/glm-5.3-flash-exl3/exl3/Dockerfile \
    -t "$validation_image" .
docker image inspect --format '{{.Id}} {{json .Config.Labels}}' "$validation_image"
