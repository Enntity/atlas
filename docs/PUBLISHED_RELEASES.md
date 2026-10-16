<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Enntity Atlas: branches and releases

This fork carries the Atlas engine behind [SparkGLM](https://github.com/Enntity/sparkglm)
(GLM-5.3-Flash) and [SparkQwen](https://github.com/Enntity/sparkqwen)
(Qwen3.8-Flash-Next). The product repositories hold no engine source: each
one pins an exact commit of this fork in `install/atlas-source.json`, and its
`install/build.sh` fetches that commit and refuses any other tree.

## Branches

| Branch | What it is |
|---|---|
| `spark-shared` | The shared engine layer on Atlas-Inf `main`: GLM-5.3-Flash support, Qwen3.8-Flash-Next support (gated to that model) and the model-independent serving work both products use. This is the Atlas-Inf contribution candidate. |
| `sparkglm` | `spark-shared` plus the SparkGLM-only commits: FlashKDA and native sparse-MLA prefill bridges and the GB10 display-carveout KV placement. Not intended for Atlas-Inf. |
| `sparkqwen` | `spark-shared` plus the SparkQwen-only commits, each behind an `ATLAS_*` switch set by SparkQwen's profiles. |
| `main` | A mirror of Atlas-Inf `main`. |
| `enntity/releases` | This index. Its source tree is an older snapshot (SparkGLM RC2, 2026-10-09); build from the product pins, not from this branch. |

The default branch is `spark-shared`.

`sparkglm` and `sparkqwen` always sit directly on `spark-shared`. A change
both products need goes into `spark-shared` and both product branches are
rebased onto it.

## Releases are tags

Each engine a product release pins is a tag of the form
`<product>/atlas-<date>-<name>`; the shared layer it sits on is tagged
`upstream/atlas-shared-<date>[-<name>]`. Tags do not move, so a pinned commit
stays fetchable after the branches move on. The dated branches this fork used
before 2026-10-15 (`upstream/glm53-flash-*`, `sparkglm/atlas-*`,
`sparkqwen/atlas-*`, and the `exp/`, `opt/`, `feat/`, `fix/`, `integ/` and
`archive/` work branches) are kept as tags of the same name at the same
commits.

## Current engines

| | SparkGLM | SparkQwen |
|---|---|---|
| Product `main` pin (2026-10-15) | `sparkglm/atlas-20261015-vision` @ `16315004` ([sparkglm#52](https://github.com/Enntity/sparkglm/pull/52)) | `sparkqwen/atlas-20261015-vision` @ `1954e759` ([sparkqwen#7](https://github.com/Enntity/sparkqwen/pull/7)) |
| Shared layer | `upstream/atlas-shared-20261015-vision` @ `a9464810` | same |
| Atlas-Inf base | `7ffd183f` | same |
| Previous pin | `sparkglm/atlas-20261014-vpc` @ `289cefbc` | `sparkqwen/atlas-20261013-prefill` @ `fcf9735f` |

The pinned engines are the current `sparkglm`, `sparkqwen` and
`spark-shared` branch heads. Their measurements (text output, vision, fill, RigMark and a
same-prompt speed comparison against production) are in each product's
`results/2026-10-15-shared-engine/`.
