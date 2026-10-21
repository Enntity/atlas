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
| Product `main` pin | `sparkglm/atlas-20261021-unified` @ `f4d95a10`, 2026-10-21 ([release](https://github.com/Enntity/sparkglm/releases/tag/unified-20261021)) | `sparkqwen/atlas-20261017-vision` @ `92dd5435`, 2026-10-17 ([sparkqwen#12](https://github.com/Enntity/sparkqwen/pull/12)) |
| Shared layer | `upstream/atlas-shared-20261015-vision` @ `a9464810` | same |
| Atlas-Inf base | `7ffd183f` | same |
| Previous pin | `sparkglm/atlas-20261017-fp4local` @ `b3b016ab` | `sparkqwen/atlas-20261016-drafter` @ `1aa8d99f` |

Release tags preserve their exact snapshots as product branches advance. The
shared layer remains `a9464810`. Historical measurements (text output, vision, fill, RigMark and a
same-prompt speed comparison against production) are in each product's
`results/2026-10-15-shared-engine/`. SparkQwen's FP4-Local release adds the
FP4-Local series on the `sparkqwen` layer: NVIDIA's
`nvidia/Qwen3.8-Flash-Next-FP4-Local` checkpoint (the product's new default)
with lossless CSF scale decoding, MXFP8 decode, W8A8 MXFP8 prefill, a packed
NVFP4 PLE table and native NVFP4 MTP experts; the NVFP4 checkpoint loads as
before. Its measurements, including RigMark through the release image, are in
SparkQwen's `results/2026-10-16-fp4local/`. SparkQwen's drafter release
(`sparkqwen/atlas-20261016-drafter`) then gives the MTP drafter the target's own
rows for the prompt and every verified token, queues a long prompt's first
prefill chunk behind the prefills in flight, and carries the W8A8 review
hardening; its model-independent scheduler commits are marked `shared:` and
sit in the `sparkqwen` layer until the shared layer moves. Measurements in
SparkQwen's `results/2026-10-16-drafter/`. SparkQwen's vision admission release
(`sparkqwen/atlas-20261017-vision`, [sparkqwen#10](https://github.com/Enntity/sparkqwen/issues/10))
sizes the Qwen encoder's output window to one prefill chunk's images, admits
image requests from their headers under per-request and in-flight row limits
(400 / 503 with `Retry-After`), bounds decode memory and fixes warm
multi-chunk prompts splicing a cached image's rows; text output is unchanged.
Its commits are mostly `shared:` and keep GLM's admitted sizes; they sit in
the `sparkqwen` layer too. Measurements in SparkQwen's `results/2026-10-17-vision/`.

SparkGLM's FP4-Local release (`sparkglm/atlas-20261017-fp4local` @ `b3b016ab`,
the previous SparkGLM product pin introduced by
[sparkglm#53](https://github.com/Enntity/sparkglm/pull/53); image `ghcr.io/enntity/atlas-sparkglm:c8ced64316bf`) adds the
FP4-Local series on the `sparkglm` layer: NVIDIA's
`nvidia/GLM-5.3-Flash-FP4-Local` checkpoint (QAD; NVFP4 experts with
`csf_32x2_v1` compressed scales, MXFP8 KDA/dense/lm_head, FP8 128x128 block
MLA/shared experts). It carries the lossless `csf_32x2_v1` block-scale decoder
in every safetensors loader, including per-slice decode under expert TP; the
FP4-Local GLM loader with production numerics by default (KDA, shared and dense
re-quantized to NVFP4, MLA through MXFP8 twins, the lm_head split shard from
the checkpoint's own MXFP8); the shared FP4-Local commits from the `sparkqwen`
layer (CSF, MXFP8 twins and GEMV, cuBLASLt MXFP8 W8A8 with pinned, timed kernel
selection); exact prefill switches (`ATLAS_GLM_KDA_CONV_PLANES`,
`_KDA_SIDE_PROJ_LT`, `_HC_PREFILL_FUSE2`, `_SP_GATHER_OVERLAP`; logprob hash unchanged); and a
tools + `response_format` union grammar that fixes fenced JSON when both are
set (ToolEvalBench 88 to 91). Opt-in and off by default: `ATLAS_GLM_FP4L_NATIVE`
(every layer in the checkpoint's own format; ~10-16% slower decode on GB10,
same ToolEvalBench score), the DFlash warm-turn drafter context
(`ATLAS_DFLASH_WARM_CTX`), packed short-prompt prefill
(`ATLAS_GLM_PACKED_PREFILL`, experimental: not yet exact), a TTFT trace, a
tail-split minimum and a MoE prefill row bound. The old
`nvidia/GLM-5.3-Flash-NVFP4` checkpoint still loads with an identical
prompt-logprob hash (`246cd1331b41`). Its `shared:` commits sit in the
`sparkglm` layer until the shared layer moves. Measurements in SparkGLM's
`results/2026-10-17-fp4local/`.

## SparkGLM unified release, 2026-10-21

`sparkglm/atlas-20261021-unified` at `f4d95a100516338c938a4524e23a65a2c4c3206e`
extends the published FP4-Local foundation with Strict and Spark precision
policies, corrected FP4 weight rounding, native MTP with prefix caching and
C4 admission, and TP4 chain/ring transport. The product uses image
`ghcr.io/enntity/atlas-sparkglm:e3792e6e8c51`. The shared and Qwen branches
are unchanged; model-independent additions remain in the GLM product layer
until the next shared refresh.

The accepted export retains the qualified runtime source, with one
documentation-only address substitution. Release packaging preserves the
tested executable, native libraries, launcher and profile bytes. The product
records the source mapping and image hashes, focused checks, earlier TP2
validation and its surgical fixes, and every quick chain/ring repetition.
This is scoped qualification, not a complete upstream CI or independent cold
build claim. See [release evidence](https://github.com/Enntity/sparkglm/blob/main/results/2026-10-21-unified-release/RESULT.md).
