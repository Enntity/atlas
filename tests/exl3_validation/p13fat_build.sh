#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Tagged dev build for followup/p13-fat-moe-multirow. Syncs this worktree to
# spark1's persistent builder, builds spark-server with the CUDA kernels and
# installs ~/atlas-dev/bin/spark.p13fat on BOTH Sparks. Never touches the
# shared ~/atlas-dev/bin/spark that the running baseline and other agents use.
# Same recipe as devbuild.sh, only the install name differs.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=p13fat
rsync -az --delete -e "ssh -o BatchMode=yes -J colum@192.168.1.50" \
    Cargo.toml Cargo.lock rust-toolchain.toml crates kernels vendor jinja-templates tests \
    colum@10.50.0.11:atlas-dev/src/
"$HOME/.local/bin/spark1" "docker exec -e RUSTUP_TOOLCHAIN=1.97.1 -e ATLAS_PTX_CACHE_DIR=/target/ptx-cache atlas-dev-builder bash -c \"cd /build && rm -f /out/spark.$tag.new && cargo build --locked --release -p spark-server 2>&1 | grep -E \\\"^error|failed|served|Finished|panicked\\\" ; test \\\${PIPESTATUS[0]} -eq 0 && cp /target/release/spark /out/spark.$tag.new\""
"$HOME/.local/bin/spark1" "mv ~/atlas-dev/bin/spark.$tag.new ~/atlas-dev/bin/spark.$tag && sha256sum ~/atlas-dev/bin/spark.$tag"
"$HOME/.local/bin/spark1" "cat ~/atlas-dev/bin/spark.$tag" | "$HOME/.local/bin/spark0" "cat > ~/atlas-dev/bin/spark.$tag.tmp && chmod +x ~/atlas-dev/bin/spark.$tag.tmp && mv ~/atlas-dev/bin/spark.$tag.tmp ~/atlas-dev/bin/spark.$tag && sha256sum ~/atlas-dev/bin/spark.$tag"
