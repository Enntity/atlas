#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

# Canonical launcher for the fixed two-DGX-Spark GLM-5.3 Flash appliance.
# Keep the historical implementation filename as a compatibility detail.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/start-glm53-ep2.sh" "$@"
