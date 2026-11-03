<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# spark

`spark` is the engine SparkQwen and SparkGLM build from: Atlas-Inf main plus
every open Atlas-Inf pull request from Enntity, merged in. `sync.sh` updates
it; nothing else commits here except this directory. Work happens on a
branch off Atlas-Inf main that becomes an Atlas-Inf PR, then `sync.sh` picks
it up. When the PR merges upstream it drops out on its own.
