<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Published Enntity releases

The released engines below are published Enntity snapshots, based on the
specific Atlas-Inf commits listed in each table. Their contribution branches
remain separate from Atlas-Inf `main`.

The `enntity/releases` branch carries the released GLM source with this
cross-product index. Use each product's `install/atlas-source.json` for the
exact measured build; SparkQwen RC15 uses its own engine branch.

## GLM-5.3-Flash (`sparkglm`)

Product repo: <https://github.com/Enntity/sparkglm> — recipes, source, and
measured results.

| Source field | Published value |
|---|---|
| released branch | `sparkglm/atlas-20261009-rc2` |
| released commit | `f2b805e73d36a7e5527622e24f2f8199052e01e6` |
| equivalent layered branch | `sparkglm/atlas-20261009-rc2-layered` @ `fea1ef6c327df2c1806874c474897fd8bd036d91` |
| tree (both branches) | `513bf0c7187cc202a54980533fb4c5325ec72b2e` |
| upstream candidate | `upstream/glm53-flash-20261009-rc2` @ `a1c9b1bc8b4f11df1aa2d5cd058b6d0ec375254b` |
| Atlas-Inf base | `6a3d24ecb891adac1fac8edeef92de29bb6f3e15` |
| receipts | [`results/2026-10-09-rc2/RESULT.md`](https://github.com/Enntity/sparkglm/blob/main/results/2026-10-09-rc2/RESULT.md) |

The two branches are tree-equivalent: the same sources under two histories.

### Tested scope — GLM

- **Context budget.** 1M is the per-request maximum. It is not four full 1M
  requests at once. What was measured is a shared pool of about 2.36M tokens at
  a `0.91` memory utilization plus carveout plus disk cache.
- **Greedy exactness.** Measured on 4 prompts, comparing C=1 against C=4, run
  twice.
- **Prefill above 8K.** Chunked prefill above 8K tokens is not row-invariant;
  this remains open.
- **DFlash2.** Non-commercial only.

## Qwen3.8-Flash-Next (`sparkqwen`)

Product repo: <https://github.com/Enntity/sparkqwen> — recipes, source, and
measured results.

| Source field | Published value |
|---|---|
| released branch | `sparkqwen/atlas-20261008-rc15` |
| released commit | `de4386b4471811216e8157023be33bd62734278d` |
| tree | `5f06ba8ec93b60f9e866cfcc8098e9ffb6855bf5` |
| upstream candidate | `upstream/qwen38-flash-next-20261006` @ `3322e22109ec002f7efb32ff23fcf87840e19508` |
| Atlas-Inf base | `4405bce359714672c57944fafd8aef9df35de524` |
| receipts | [`results/2026-10-08-rc15/RESULT.md`](https://github.com/Enntity/sparkqwen/blob/main/results/2026-10-08-rc15/RESULT.md) |

### Tested scope — Qwen

- **No prebuilt image.** This release ships as source; there is no published
  container image for it.
- **Output gate.** 8 of 9 gate legs pass.
- **Untested surfaces.** Tool calling, structured output, and vision were not
  tested on this release.

## Pin branches

The product manifests pin immutable commit hashes and source trees. These
pins are preserved when documentation changes. Engine changes require a new
reviewed pin and qualification; they are not included in this documentation
update.

## Historical upstream bases

The tables identify the contribution series and Atlas-Inf base for each
release. Updating a remote branch reference does not rebase or deploy an
engine. Follow the product manifests and dated receipts when comparing builds.
