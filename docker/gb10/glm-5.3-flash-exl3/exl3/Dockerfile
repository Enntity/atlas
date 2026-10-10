# SPDX-License-Identifier: AGPL-3.0-only
# Atlas native EXL3 GLM-5.3-Flash validation image (glm-5.3-flash-exl3/exl3).
#
# Builds spark-server AND the exl3_moe_scratch_check example from one source
# snapshot, packages the pinned coherent nvidia-cublas 13.1.1.3 classic+Lt pair
# (libcublas.so.13 + libcublasLt.so.13) at /opt/atlas/cublas-13.1.1.3 with its
# license, puts that directory first in the loader path, and writes
# /usr/local/share/atlas/fingerprint/build-manifest.json binding the git commit,
# source snapshot hash, toolchain, target closure (compiler flags), binary
# SHA256s and embedded PTX SHA256s with byte offsets in both binaries.
#
# Usage (from the repository root; clean --no-cache build):
#
#   BLAS131_PAIR_DIR=/path/to/pair bash tests/exl3_validation/build_image.sh \
#     "$SOURCE_SNAPSHOT_SHA256" "$RUST_STABLE_VERSION" "$BUILD_CPU_ID"
#
# build_image.sh requires a clean committed worktree (it derives the git
# commit itself) and stages the four pair files listed below into ./blas131/,
# verifying each against cublas131.sha256 before `docker build`:
#
#   libcublas.so.13  libcublasLt.so.13  package-binding-v1.json
#   dist-info/licenses/License.txt
#
# cublas131.sha256 (this directory) is the single committed source of truth
# for the pair hashes; it is installed into the image and checked with
# `sha256sum -c` at build time. Provenance: nvidia-cublas 13.1.1.3 extracted
# read-only from reference image
# sha256:6eb42fa20c145cabf9f788d7b850d8d3416706ad1955a003448b48962cb2845b
# (package-binding-v1.json, itself pinned, carries the wheel RECORD digests,
# library byte counts and ELF metadata).
# TODO: no committed input records the upstream nvidia-cublas 13.1.1.3 wheel
# URL/SHA256; extraction from that reference image is the only documented
# channel. Pin the wheel artifact here once the channel is known.
#
# Runtime attestation on the host (any time after the container starts):
#
#   python3 tests/exl3_validation/attest_runtime_maps.py <pid|container-name>
#
# reads /proc/<pid>/maps and proves only the pinned pair is mapped, hashing
# both files. CPU unit tests for the manifest/attestation python:
#
#   uv run --no-project --with pytest python -m pytest \
#     tests/exl3_validation/test_build_manifest.py \
#     tests/exl3_validation/test_attest_runtime_maps.py
#
# Required build args: CUDA_DEVEL_IMAGE / CUDA_RUNTIME_IMAGE (digest-pinned,
# derived from docker/gb10/Dockerfile by build_image.sh), RUST_TOOLCHAIN,
# ATLAS_BUILD_CPU, ATLAS_GIT_SHA, ATLAS_SOURCE_SHA256. This tag is a validation
# candidate; never publish it. See tests/exl3_validation/README.md.
ARG CUDA_DEVEL_IMAGE
ARG CUDA_RUNTIME_IMAGE
FROM ${CUDA_DEVEL_IMAGE} AS builder

ARG RUST_TOOLCHAIN
ARG ATLAS_BUILD_CPU
ARG ATLAS_GIT_SHA
ARG ATLAS_SOURCE_SHA256
RUN test -n "${RUST_TOOLCHAIN}" && test -n "${ATLAS_BUILD_CPU}" && \
    test -n "${ATLAS_GIT_SHA}" && test "${#ATLAS_SOURCE_SHA256}" -eq 64

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev \
      libibverbs-dev librdmacm-dev util-linux python3 && \
    rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_TOOLCHAIN=${RUST_TOOLCHAIN}
ENV CUDA_HOME=/usr/local/cuda
# Cargo/CMake job counts alone do not bound atlas-kernels' nvcc worker pool.
# CPU affinity below also bounds std::thread::available_parallelism().
ENV CARGO_BUILD_JOBS=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV MAKEFLAGS=-j1
ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=glm-5.3-flash-exl3
ENV ATLAS_TARGET_QUANT=exl3
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY kernels/ kernels/
COPY vendor/ vendor/
COPY jinja-templates/ jinja-templates/
COPY tests/fixtures/ tests/fixtures/
COPY tests/exl3_validation/build_manifest.py /build/tools/build_manifest.py
RUN taskset -c "${ATLAS_BUILD_CPU}" cargo build --locked --release -p spark-server && \
    taskset -c "${ATLAS_BUILD_CPU}" cargo build --locked --release -p spark-model \
      --example exl3_moe_scratch_check --features gpu-examples && \
    mkdir /build/fingerprint && \
    rustc --version --verbose > /build/fingerprint/rustc.txt && \
    cargo --version --verbose > /build/fingerprint/cargo.txt && \
    nvcc --version > /build/fingerprint/nvcc.txt && \
    python3 /build/tools/build_manifest.py \
      --release-dir /build/target/release \
      --fingerprint-dir /build/fingerprint \
      --target-key gb10/glm-5.3-flash-exl3/exl3 \
      --git-sha "${ATLAS_GIT_SHA}" \
      --source-snapshot-sha256 "${ATLAS_SOURCE_SHA256}" \
      --out /build/fingerprint/build-manifest.json && \
    printf '%s\n' "${ATLAS_GIT_SHA}" > /build/fingerprint/source-commit.txt && \
    printf '%s\n' "${ATLAS_SOURCE_SHA256}" > /build/fingerprint/source-snapshot.sha256

FROM ${CUDA_RUNTIME_IMAGE}
ARG ATLAS_GIT_SHA
ARG ATLAS_SOURCE_SHA256
ARG RUST_TOOLCHAIN
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.revision="${ATLAS_GIT_SHA}"
LABEL io.atlas.source.sha256="${ATLAS_SOURCE_SHA256}"
LABEL io.atlas.rust.toolchain="${RUST_TOOLCHAIN}"
LABEL io.atlas.validation.target="glm-5.3-flash-exl3/exl3"
LABEL io.atlas.blas.provider="nvidia-cublas 13.1.1.3"
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28"
COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/target/release/examples/exl3_moe_scratch_check /usr/local/bin/exl3_moe_scratch_check
# Pinned coherent cuBLAS 13.1.1.3 classic+Lt pair, first in the loader path.
# ${LD_LIBRARY_PATH:+...} is Dockerfile-interpolated against the base image.
COPY blas131/ /opt/atlas/cublas-13.1.1.3/
ENV LD_LIBRARY_PATH=/opt/atlas/cublas-13.1.1.3${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
COPY docker/gb10/glm-5.3-flash-exl3/exl3/cublas131.sha256 /usr/local/share/atlas/fingerprint/cublas131.sha256
RUN sha256sum -c /usr/local/share/atlas/fingerprint/cublas131.sha256
# Fail the build unless the server's link-time libcublasLt and the classic
# library's own Lt dependency resolve to the pinned directory. The classic
# library is dlopen()ed by name at runtime (resolved through LD_LIBRARY_PATH
# above); runtime attestation (attest_runtime_maps.py) proves the actual
# mappings. The scratch-check example may not link cuBLAS at all (--as-needed),
# so it is not probed here.
RUN ldd /usr/local/bin/spark | grep -F "libcublasLt.so.13 => /opt/atlas/cublas-13.1.1.3/libcublasLt.so.13" && \
    ldd /opt/atlas/cublas-13.1.1.3/libcublas.so.13 | grep -F "libcublasLt.so.13 => /opt/atlas/cublas-13.1.1.3/libcublasLt.so.13"
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY --from=builder /build/fingerprint/ /usr/local/share/atlas/fingerprint/
COPY LICENSE /LICENSE
# The toolkit copy of libcublasLt.so.13 is recorded for contrast only; runtime
# attestation must show it is never mapped.
RUN dpkg-query -W > /usr/local/share/atlas/fingerprint/runtime-packages.txt && \
    sha256sum /usr/lib/aarch64-linux-gnu/libnccl.so.2 \
      /opt/atlas/cublas-13.1.1.3/libcublas.so.13 \
      /opt/atlas/cublas-13.1.1.3/libcublasLt.so.13 \
      /usr/local/cuda/lib64/libcublasLt.so.13 \
      > /usr/local/share/atlas/fingerprint/runtime-libraries.sha256
ENV RUST_LOG=info
EXPOSE 8888
ENTRYPOINT ["spark"]
