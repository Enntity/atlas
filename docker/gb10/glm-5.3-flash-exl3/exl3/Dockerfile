# Atlas native EXL3 GPU validation image. Build from repository root.
# Required base image arguments come from digest-pinned docker/gb10/Dockerfile.
# This image is an isolated validation artifact, with its source snapshot hash.
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
      libibverbs-dev librdmacm-dev util-linux && \
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
RUN taskset -c "${ATLAS_BUILD_CPU}" cargo build --locked --release -p spark-server && \
    mkdir /build/fingerprint && \
    rustc --version --verbose > /build/fingerprint/rustc.txt && \
    cargo --version --verbose > /build/fingerprint/cargo.txt && \
    nvcc --version > /build/fingerprint/nvcc.txt && \
    sha256sum /build/target/release/spark > /build/fingerprint/spark.sha256 && \
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
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28"
COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY --from=builder /build/fingerprint/ /usr/local/share/atlas/fingerprint/
COPY LICENSE /LICENSE
ENV RUST_LOG=info
EXPOSE 8888
ENTRYPOINT ["spark"]
