#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Clean --no-cache build of the GLM-5.3-Flash EXL3 validation image from
# committed inputs. Run from a reviewed, clean git worktree AFTER freeing
# serving GPU/system RAM. Never publishes; prints the resolved image ID/labels.
#
# usage: build_image.sh SOURCE_SNAPSHOT_SHA256 RUST_STABLE_VERSION BUILD_CPUS
#        BLAS131_PAIR_DIR=/path/to/pair must name a directory holding
#        libcublas.so.13, libcublasLt.so.13, package-binding-v1.json and
#        dist-info/licenses/License.txt (the nvidia-cublas 13.1.1.3 pair;
#        provenance and TODO in docker/gb10/glm-5.3-flash-exl3/exl3/Dockerfile).
#
# The git commit is derived from the worktree (dirty trees are rejected), so
# the build manifest's git_commit binding is derived, not asserted.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'usage: build_image.sh SOURCE_SNAPSHOT_SHA256 RUST_STABLE_VERSION BUILD_CPU_ID' >&2
    exit 2
fi
snapshot_sha=$1
rust_version=$2
build_cpu=$3
[[ $snapshot_sha =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid snapshot SHA256' >&2; exit 2; }
[[ $rust_version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'supply an exact stable Rust release' >&2; exit 2; }
# One CPU or a taskset list (e.g. 0-15): the affinity also bounds the kernel
# build's internal compiler pool, so size it to the RAM free on the build host.
[[ $build_cpu =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || { echo 'supply a CPU ID or taskset list' >&2; exit 2; }
: "${BLAS131_PAIR_DIR:?set BLAS131_PAIR_DIR to the nvidia-cublas 13.1.1.3 pair directory}"
[[ -d $BLAS131_PAIR_DIR ]] || { echo "no such pair directory: $BLAS131_PAIR_DIR" >&2; exit 2; }
BLAS131_PAIR_DIR=$(cd -- "$BLAS131_PAIR_DIR" && pwd)
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_root"
[[ -z $(git status --porcelain) ]] || {
    echo 'worktree must be clean: the image must build from committed inputs' >&2
    exit 2
}
source_commit=$(git rev-parse HEAD)
devel_image=$(awk '$1 == "FROM" && $NF == "builder" {print $2; exit}' docker/gb10/Dockerfile)
runtime_image=$(awk '$1 == "FROM" && $NF != "builder" {print $2; exit}' docker/gb10/Dockerfile)
[[ $devel_image == *@sha256:* && $runtime_image == *@sha256:* ]] || {
    echo 'main Dockerfile must contain digest-pinned CUDA build/runtime images' >&2
    exit 2
}

# Stage the pinned pair into the build context. cublas131.sha256 is the SSOT
# for the hashes; paths there are install-absolute, rewritten to be relative to
# the pair directory and to blas131/. The pair is verified before and after
# staging (the external recipe checked twice as well).
pins=docker/gb10/glm-5.3-flash-exl3/exl3/cublas131.sha256
install_dir=/opt/atlas/cublas-13.1.1.3
rel_pins=$(mktemp)
trap 'rm -f "$rel_pins"' EXIT
sed "s|  ${install_dir}/|  |" "$pins" > "$rel_pins"
# A format drift would leave absolute names that sha256sum -c resolves on the
# HOST instead of the pair directory; require every name to be relative.
if grep -qvE '^[0-9a-f]{64}  [^/*]' "$rel_pins"; then
    echo "unexpected pin format in $pins (names must rewrite to relative paths)" >&2
    exit 1
fi
(cd "$BLAS131_PAIR_DIR" && sha256sum -c --quiet "$rel_pins") || {
    echo "pair directory does not match $pins" >&2
    exit 1
}
rm -rf blas131
mkdir -p blas131/dist-info/licenses
while read -r _ rel; do
    # -L: stage file contents, never symlinks the image COPY would break.
    cp -L --preserve=mode,timestamps "$BLAS131_PAIR_DIR/$rel" "blas131/$rel"
done < "$rel_pins"
(cd blas131 && sha256sum -c --quiet "$rel_pins") || {
    echo "staged blas131/ does not match $pins" >&2
    exit 1
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
echo "build manifest: /usr/local/share/atlas/fingerprint/build-manifest.json in $validation_image"
echo "runtime attestation: python3 tests/exl3_validation/attest_runtime_maps.py <pid|container>"
