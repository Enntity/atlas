#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Dev build loop (not a qualification build): sync this worktree to spark1's
# persistent builder, incremental cargo build with the persistent PTX cache,
# then install the binary at ~/atlas-dev/bin/spark on BOTH Sparks. Fails fast:
# a failed build never leaves a stale binary looking fresh.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -az --delete -e "ssh -o BatchMode=yes -J colum@192.168.1.50" \
    Cargo.toml Cargo.lock rust-toolchain.toml crates kernels vendor jinja-templates tests \
    colum@10.50.0.11:atlas-dev/src/
"$HOME/.local/bin/spark1" 'docker exec -e RUSTUP_TOOLCHAIN=1.97.1 -e ATLAS_PTX_CACHE_DIR=/target/ptx-cache atlas-dev-builder bash -c "cd /build && rm -f /out/spark.new && cargo build --locked --release -p spark-server 2>&1 | grep -E \"^error|failed|served|Finished|panicked\" ; test \${PIPESTATUS[0]} -eq 0 && cp /target/release/spark /out/spark.new"'
"$HOME/.local/bin/spark1" 'mv ~/atlas-dev/bin/spark.new ~/atlas-dev/bin/spark && sha256sum ~/atlas-dev/bin/spark'
"$HOME/.local/bin/spark1" 'cat ~/atlas-dev/bin/spark' | "$HOME/.local/bin/spark0" 'cat > ~/atlas-dev/bin/spark.tmp && chmod +x ~/atlas-dev/bin/spark.tmp && mv ~/atlas-dev/bin/spark.tmp ~/atlas-dev/bin/spark && sha256sum ~/atlas-dev/bin/spark'
