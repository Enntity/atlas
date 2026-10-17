# EXL3 support for GLM-5.3 Flash

Status: native text-model assembly and an experimental EXL3 kernel target are
implemented for pure TP with EP=1. Independent CUDA kernel and two-node NCCL
checks passed on both homelab GB10 hosts. The first full-model Atlas/vLLM TP=2
comparison failed the frozen numerical tolerances and persistent decode text
checks. Numerical localization is in progress; this does not extend the
deployment support matrix.

## Evidence and checkpoint contract

Target: [MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3](https://huggingface.co/MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3/tree/0f46c18259efb8886839b446e25534f9f2e1fb1e),
initial implementation-reference revision
`0f46c18259efb8886839b446e25534f9f2e1fb1e`. GPU validation uses the cached
revision `dda99c7e935920fa6e235ce410e6f806e1c02720` on both engines.
Config declares `quant_method=exl3`, version `1.4.6`, codebook `mul1`,
`out_scales=always`, aggregate `bits=4.05`, and `head_bits=6`.
Atlas already dispatches `glm5_next` to `Glm5WeightLoader`; its NVFP4
projection/expert paths and dense vocabulary projection need EXL3 dispatch.

The first shard's header confirms the following per-linear tensors, where
K is the **individual tensor's** integer bitrate and k/n are input/output widths:

| Suffix | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `.trellis` | I16 | `[k/16, n/16, 16*K]` | Packed 16×16 tiles |
| `.suh` | F16 | `[k]` | Input sign/scale vector |
| `.svh` | F16 | `[n]` | Output sign/scale vector |
| `.mul1` | I32 | `[]` | Codebook multiplier bits `0x83DCD12D` |

Sampled shapes: layer 0 `qkv_proj` uses K6, dense `gate_proj` uses K5,
layer 3 expert 0 `gate_proj` uses K4. Derive the kernel specialization from
each trellis shape, never from the aggregate config bitrate. Keep FP16
scale bytes intact; the generic safetensors loader currently converts F16
to BF16, and does not support I16/I32 store dtypes. A new format-aware
loading boundary is required before using that path.

The reference implementation is pinned to upstream commit
`151539c77abc7ab7425d30da7a4e8e3c5c154e7b`:
[linear representation](https://github.com/turboderp-org/exllamav3/blob/151539c77abc7ab7425d30da7a4e8e3c5c154e7b/exllamav3/modules/quant/exl3.py),
[packing](https://github.com/turboderp-org/exllamav3/blob/151539c77abc7ab7425d30da7a4e8e3c5c154e7b/exllamav3/exllamav3_ext/quant/pack.cu),
[trellis decoder](https://github.com/turboderp-org/exllamav3/blob/151539c77abc7ab7425d30da7a4e8e3c5c154e7b/exllamav3/exllamav3_ext/quant/exl3_dq.cuh),
[codebook](https://github.com/turboderp-org/exllamav3/blob/151539c77abc7ab7425d30da7a4e8e3c5c154e7b/exllamav3/exllamav3_ext/quant/codebook.cuh).
Upstream is MIT licensed. Any substantial code port must preserve its
copyright and permission notice; these host primitives implement the
format independently under Atlas's license.

## Implementation slices and validation gates

1. **Host contract and decoder primitives (this slice).** Add a borrowed,
   validated view of integer K1–8 mul1 linears in `spark-runtime::weights::exl3`.
   Require explicit k/n, exact dtypes/shapes, the codebook marker, and finite
   scales. Preserve original bytes without GPU allocation or filesystem I/O.
   Implement circular trellis state extraction and FP16 mul1 codebook decoding.
   Tests cover malformed layouts, mixed bitrates, bounds, and a 192-byte tile
   from the pinned checkpoint. These tests do not prove full weight reconstruction.
2. **CPU reconstruction oracle (implemented).** Map trellis state order into the
   tensor-core 16×16 layout, then apply normalized H128 on each side and
   input/output sign/scales. Tests compare asymmetric matrices and signed
   scales against an independent dense Walsh-matrix oracle. All integer
   K values are covered for layout decoding; target K4/K5/K6 are covered for
   TP2 linear and FFN equivalence. Upstream real-weight reconstruction
   comparison remains a validation gate.
   Audit all shard headers before claiming checkpoint-wide format coverage;
   bias, legacy sign-only, fractional K, and other codebooks remain explicit
   extensions rather than implicit defaults.
3. **GB10 correctness kernels (implemented, independent GPU checks passed).**
   `kernels/gb10/common/exl3.cu` implements input scale/H128, packed mul1
   matrix multiplication, and output H128/scale. `Exl3LinearWorkspace`
   resolves all three kernels before allocating two F32 activation buffers,
   validates shapes and batch capacity before launch, and binds scratch to
   one explicit compute stream. Both single-token and batched launches retain
   packed weights; no full decoded matrix is allocated. Common kernel
   discovery includes this shared module without a new model registration.
   Validate against the CPU and upstream oracles on actual CUDA hardware.
   The current matrix multiply decodes each coefficient on demand. Optimized
   decode/prefill kernels follow correctness validation. Do not requantize
   EXL3 to NVFP4 as a substitute.
4. **Loading and model integration.** Add format-aware IORouter upload that
   preserves I16 trellises, I32 markers, and F16 scales. Introduce an EXL3
   weight representation separate from `QuantizedWeight`/`Nvfp4Variant`;
   the current `QuantFormat::base_variant` API only describes NVFP4-family
   loading policies. Route GLM dense FFNs, attention/KDA projections, shared
   and routed experts, and vocabulary projection through explicit format
   dispatch. Support groups crossing shard boundaries. Retain ordinary BF16
   parameters. Reject unsupported declared EXL3 before heuristic dispatch.
5. **Placement and release.** Preserve EP filtering for every companion
   tensor and partition on legal trellis/Hadamard boundaries. Check TP scale
   slicing, replicated shared experts, and output reduction. The model card
   lists 153.810 GiB of weights: this exceeds a single 128 GB Spark before
   cache/activation allocations, so multi-device placement or explicit offload
   is necessary. Register an EXL3 target only once the complete kernel closure
   and loader work. Rebuild the Docker image and run the GLM slice of the
   serve suite, including prefill/decode, tool use, thinking and regression
   checks, before a support-matrix change. MTP and vision require their own
   validated EXL3 coverage if enabled.

Changes to weight representations, format dispatch, or build registration
are shared changes and must be flagged in their respective PR descriptions.
Apply the measurement-discipline skill before any performance measurement
or claim; this slice makes no performance claim.

## TP=2 equivalence contract

The current implementation is in `spark-runtime::weights::exl3::tp`:
`TpShardPlan` selects ranges, `OwnedExl3Shard` supports host validation, and
`Exl3DeviceShard::upload` sends only selected checkpoint bytes through
`GpuBackend`. The GPU resource participates in ordered teardown and retains
failed frees for retry. No rank first uploads the full tensor. The indexed
`Exl3CheckpointRouter` maps shard metadata once and borrows groups even when
their companions are in different files, preserving F16/I16/I32 bytes.
After completed uploads, its explicit `evict_cached_pages` boundary releases
mapping residency and advises file-cache eviction on Linux. Call between
reads to avoid retaining checkpoint pages alongside GPU allocations in GB10
unified memory; immutable views remain valid and refault on later access.

| Linear | Trellis axis | Scale sliced | Output completion |
| --- | --- | --- | --- |
| Gate/up, Q/K/V, vocabulary head | output (axis 1) | `svh` | Keep rank-local outputs or gather in global order |
| Down, attention output | input (axis 0) | `suh` | Sum partial outputs once |
| Replicated projection | none | none | Use one identical result; do not sum replicas |

The opposite scale vector and the validated mul1 multiplier are replicated.
Every local width and boundary is a multiple of 128. Even a 16-aligned split
can change weights if it cuts an H128 block: reject it. Fused `[Q,K,V]` must
be split per component, yielding `[Q_rank,K_rank,V_rank]`, not by taking one
contiguous half of the combined output.

The independent [vLLM EXL3 plugin TP tests](https://github.com/vcruz305/vllm-exl3/blob/223e246ff8da552d2ace976c4bcbed804a8ec04c/tests/test_moe_tp_aligned_split.py)
use these same packed axes and companion-scale rules. This is a reference for
partition semantics, not evidence of Atlas/vLLM end-to-end output parity.
CPU tests reconstruct and execute two column shards followed by a gather,
two row shards followed by a sum, fused QKV with component-order assembly,
and gate/up → SiLU product → down with mixed K4/K5/K6 weights. Results are
compared with the unsplit computation using floating-point tolerances.
The CPU oracle keeps F32 arithmetic after FP16 codebook decoding; GPU
intermediate stores, accumulation/reduction order, and final rounding differ.

`Exl3ProjectionSpec::from_rank_local_config` maps GLM KDA, MLA, dense FFN,
shared/routed expert, and vocabulary projections to explicit global checkpoint
dimensions and TP axes. It requires pure TP with EP=1 and reconstructs full
attention widths from topology-adjusted local heads. Fused KDA QKV uses three
separate output segments. `Glm5Exl3Projection` owns one rank-local upload and
native workspace. Input-sharded forward validates the TP rank/world before
launch and performs one stream-ordered BF16 sum; output-sharded forward
returns local-channel ownership, while replicated forward returns full
channels without a sum. Empty batches submit neither kernels nor collectives.
These APIs are building blocks for model assembly, not serving dispatch.

The KDA layer now dispatches `Glm5Projection::Exl3` before all NVFP4/FP8
projection routes, retaining native trellises for decode and prefill. It checks
all four projections' dimensions, TP topology, stream and batch capacity before
attention work. EXL3 O emits a local contribution: the existing KDA layer
owns its single TP sum. Specialized NVFP4 QKV and paired-owner verification
paths remain gated to NVFP4. Sequence-parallel EXL3 KDA is explicitly rejected.

`TpShardPlan::new_output_component` extracts one rank's Q, K or V range from
the fused checkpoint using the same segment validation as fused sharding.
`Glm5Exl3KdaResources::load` uploads those three matrices, O, and the nine
dense companions. `into_weights` transfers projection ownership to real
`Glm5KdaWeights` while returning the dense-resource sidecar for registration.
Fused `conv1d.weight` is sliced per Q/K/V channel segment; F16 side matrices
convert through the existing F16-to-BF16 upload boundary in bounded chunks,
while original BF16 and F32 companions retain their bytes. Checkpoint dtype
and dimensions match the pinned first-shard header. Partial loads roll back
prior allocations; failed frees remain retryable. Model teardown releases
native KDA projections explicitly before its allocation sweep. Factory assembly
must also register the returned dense companion owner.

`Glm5Exl3Ffn` borrows gate/up/down projections and caller-owned BF16 buffers.
It validates matching intermediate ranges, topology, streams, capacities and
live resources before submitting work. Gate/up retain local intermediates;
the down projection performs the sole TP sum. No additional weights or scratch
are allocated by the helper. Activation policy is required explicitly:
unbounded SiLU or a finite positive clamp. The clamped kernel implements
`min(silu(gate), limit) * clamp(up, -limit, limit)`, following the pinned
[upstream activation definition](https://github.com/turboderp-org/exllamav3/blob/151539c77abc7ab7425d30da7a4e8e3c5c154e7b/exllamav3/exllamav3_ext/activation_kernels.cuh).
The target checkpoint declares `swiglu_limit=10.0`; `ModelConfig.text_swiglu_limit`
now retains and validates this separately from the vision policy.
`Exl3FfnActivation::from_glm_config` derives the required activation from that
field. Missing text policy means unclamped per the parsed schema; invalid
declared limits fail rather than silently disabling clamping.
Routing and weighted accumulation across experts remain caller work.

### Serving assembly and validation plan

Native text serving assembly now selects a mmap-backed EXL3 checkpoint before
legacy weight loading. Ordinary residual/mHC/MLA companions upload separately;
packed projection owners upload rank-local trellises and original FP16 scales.
KDA, zero-RoPE MLA, dense/shared/routed FFNs and the vocabulary head dispatch
native kernels. All routed experts remain present as TP slices on both ranks;
shared and routed partial outputs participate in one final TP sum. The router
stays on GPU. Its projection uses BF16 weights and BF16 GEMM output, transported
as FP32 for routing; FP32 correction bias affects expert selection only.
Ordinary F16 KV-B is sharded and converted before per-head attention views.

The experimental `glm-5.3-flash-exl3/exl3` kernel target shares the existing
DeepSeek/GLM CUDA sources through a directory symlink. Native-format selection
occurs before architecture disambiguation, retaining legacy target selection
for other formats. Vision, MTP/speculative decoding, LoRA and EP are rejected
for this initial text validation path. EXL3 is never relabeled as NVFP4.
Routed and ordinary projections retain packed weights; shared experts
reconstruct global BF16 matrices before TP sharding, matching the reference.

The vocabulary head produces BF16 outputs, converts them to F32 transport,
and gathers complete rank-local vocabulary rows on the compute stream,
reorders rank-major batches, and trims checkpoint padding. An opt-in logits
IORouter records full raw F32 matrices before BF16 sampler conversion. Capture
requires explicit eager flags and a fresh directory; existing artifacts are
never overwritten. See `tests/exl3_validation/README.md` for the evidence gate.

Validation is proceeding on homelab `spark0`/`spark1`. Their cached checkpoint
revision is `dda99c7e935920fa6e235ce410e6f806e1c02720`, rather than the earlier
implementation-reference revision; both engines must use this same snapshot.
The production vLLM service uses FP8 KV and speculation, so a separate eager
BF16-KV, TP=2, EP=1 reference profile is required before numeric comparison.
The execution plan is: retain original service fingerprints; run independent
kernel and real two-node collective checks; capture teacher-forced reference
logits; build a source-fingerprinted isolated Atlas image from scratch; run the
same tokens/configuration; compare full vocabulary distributions and tokens
with explicit tolerances and negative controls; restore the original service.
Full-model parity remains unproven until those captures and gates pass.

### Earlier CPU-host verification

The CPU-only `spark-runtime` suite passes (331 passed, 2 ignored), including
38 EXL3 tests. The default CUDA-feature `cargo check -p spark-runtime` passes
with `ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000`; runtime rustdoc and
license-header checks pass. Modified Rust files pass focused rustfmt checks.
These checks used installed Rust 1.97.1 because the repository-pinned 1.93.1
toolchain is missing its Cargo binary. Workspace formatting and clippy fail
on existing unrelated code under that fallback toolchain. `typos` is absent.
A cached CUDA 13 NVRTC compiler successfully compiles the native linear
kernels for `compute_121`; PTX contains FP16 FMA codebook decoding and separate
F32 multiply/add accumulation. `python3 tests/exl3_cuda_oracle.py --cpu-only`
passes independent format/transform checks. Without `--cpu-only`, the harness
requires NumPy/CuPy and a CUDA device and checks intermediate transforms,
all integer bitrates, batches, TP column gathering and row reduction.
The development host lacks a CUDA driver and GPU tools. At this earlier
verification stage, native execution and Docker serving had not been tested;
the remote GPU evidence below supersedes that limitation for kernels and NCCL.
The normal no-CUDA model EXL3 test suite passes 22 tests, including native KDA
prefill/decode dispatch on both TP ranks with recorded GPU operations and exactly
one sum per forward. The broader KDA slice passes 42 tests; core's full library
suite passes 175 tests. The full model library sweep reports 1538 passed, one
ignored and one failure in the unchanged `dense_prefill_bf16_explicit_flag`:
HEAD's parser accepts `2`, while its test treats `2` as invalid. This discrepancy
is outside EXL3 and was left unchanged.

Actual model tests became runnable after adding two missing no-CUDA CUTLASS
ABI stubs and correcting `spark-storage/build.rs` to skip CUDA linking when
the CUDA feature is disabled. The build-script, config schema, projection enum
and model teardown changes are shared changes to flag in PR descriptions.
CUDA-feature `cargo check -p spark-model --tests` also passes with the build
skip flag. Strict clippy still stops on the existing baseline violations;
a source-only clippy audit with capped lint severity completes with no EXL3
diagnostics. Model rustdoc stops on seven existing broken links in unchanged
files (including unescaped `[5,5]` and stale MoE/LoRA method links).
Focused source checks do not establish GPU arithmetic or full-model correctness.

### Current GPU and assembly checks (2026-10-10)

Independent native dense, routing/indirect MoE, activation and F32 vocabulary
kernel oracles passed on both GB10 hosts. A real two-node NCCL F32 vocabulary
gather/reorder check passed on both ranks, including batch rows and padding.
These checks qualify those exercised kernels/collectives, not full serving.
After the explicit packed-input and dense-activation policies, focused CPU
suites pass 39 model EXL3 tests and 43 runtime EXL3 tests. The full runtime
suite passes 336 tests with two ignored. The validation harness passes 28
tests. CUDA-feature server compilation with the build skip gate passes.
These counts bind the preserved local check logs; subsequent source changes
require their own checks.

The isolated candidate `atlas-exl3-validation:src-da61a23b80c1-rust1.97.1`
loaded all 45 layers on both ranks and executed the required-kernel boot gate,
16 teacher-forced full-vocabulary captures and four persistent decode cases.
All 16 captured argmax IDs match the isolated vLLM TP=2 reference. Full-logit
and exact-text comparisons fail; argmax agreement does not establish parity.
Raw captures, request mappings, image/source hashes and actual loaded NCCL
library fingerprints are preserved in the external validation artifact directory
`~/Workspace/atlas-exl3-validation-20261010/`. The original vLLM containers are
preserved for restoration after validation.

The completion-text cleanup independently removed leading whitespace even
without reasoning markers. Regression tests reproduced this behavior; the fix
preserves plain token-decoded text, including whitespace-only tokens and code
indentation. All 115 server library tests pass. This response fix does not resolve
the numerical comparison failure. Same-input expert and layer-stage comparisons
must establish the arithmetic correction before rerunning the frozen gates.
Shared changes now also include config routing groups, the native FfnComponent
variant, stream-ordered communicator gather and kernel-format target selection.

The native sparse selector also passed 26 direct GPU cases. Its EXL3 policy
reserves the incomplete current pool: 2044 historical tokens plus at most
three tail tokens, with a dense cutoff of 2047. This integer-selection check
does not qualify the query/key numerical path. Matching the installed reference
also requires its Hadamard/FP8 index quantization, F32 gate projection, and
F32 positional bias and normalization parameters. Those implementations now
pass the actual installed-reference GPU component oracle, including aliasing
and buffer guards. Their Rust dispatch is implemented but has not yet run
in a rebuilt full-model candidate.

Reference repeat checks found differing full-vocabulary logits within the same
engine process, despite identical selected token IDs. Some differences exceed
the frozen parity limits. Consecutive identical-request controls are being
collected to locate the cause. Preserve the original failed comparison and
reference captures; neither changing the reference nor increasing tolerances
is a substitute for qualification. `tests/exl3_validation/repeat_logits.py`
checks same-engine captures under identical fingerprints and unchanged limits.

The subsequent `src-af89c7cec4ae` candidate fixes completion whitespace:
all 16 teacher-forced one-token texts and argmax IDs match, and three of four
persistent decode texts match exactly. The longer decode case still differs.
Full-logit numerical gates still fail with the original thresholds; its raw
logits are byte-identical to the previous candidate. It contains diagnostics
and response changes, not the later precision corrections. Three identical
arithmetic requests in this native process produced byte-identical full logits.
This control does not establish reproducibility for other paths.

Same-input GPU replay proves the installed packed projection rounds FP16
products before the FP32 Hadamard transform and rounds the transformed input
back to FP16. A new explicit workspace policy implements this for native
packed KDA, MLA, head and dense projections. The dense activation component
now matches the installed CUDA operator byte for byte on captured and synthetic
inputs, including clamp, aliasing, guards and signed zero. Shared experts now
reconstruct original-basis global BF16 weights before selecting TP slices.
Rust ownership and sharding tests pass. Same-input GPU checks reconstruct the
full layer-3 matrices and reproduce the resident TP slices byte for byte on
both ranks. The merged gate/up GEMM, activation and down GEMM also match on
both ranks with the candidate CUDA library at ten rows and actual one-row
decode and five-row prefill inputs. Separate gate/up GEMMs differ at ten rows,
so production uses the reference's merged geometry. These component results
do not establish other layers' resident bytes or rebuilt full-model parity.

The production routed kernel now uses explicit thin and fat arithmetic
profiles. Same-input GPU replay matches all captured transformed projections,
weighted FP32 outputs and BF16 routed outputs byte for byte on both ranks.
Controls at 32 and 33 rows exercise both sides of the occupancy boundary;
the original short layer-3 captures alone exercise only the thin path.
A wrong fat profile and the previous scalar production implementation fail
these controls. Full-model validation of the corrected kernel remains pending.

The first FFN receives different inputs in the two engines. A corrected native
mHC initial normalization now matches all 40,960 captured BF16 reference
values exactly on GB10. The previous separate RMS normalization differed
at 210 values. This is a same-input component result, not full-model parity.
Corrected fused residual mixing and final contraction now match byte for byte
for the captured ten-row input and derived one-, five-, seventeen- and
128-row controls. The small fused projection and the rounded TF32 projection now reproduce
all captured initial coefficients and normalization, fused residuals,
coefficients, normalization and final contraction byte for byte on both
spark0 and spark1. The controls include one, five, ten, seventeen and 128 rows;
independent TF32 raw split checks pass on both hosts as well. These are
same-input component results, not full-model parity. The installed reference
uses the small Tilelang fused projection at up to sixteen rows and TF32
DeepGEMM for initial and larger projections.

The BF16 router GEMM reproduces the reference byte for byte using the
candidate CUDA library at one-row persistent decode and five- and ten-row
prefill. Library symbol ownership and hashes are bound to these reports.
Casting the earlier scalar FP32 dot product to BF16 still differs. Native
BF16 router and BF16 head-producer/F32-transport dispatch are implemented and
pass focused Rust checks. The rebuilt full-model comparison still fails the
frozen numerical gates described below.

Shared-expert merged BF16 GEMM controls pass at one and five rows. The earlier
ten-row evidence covered the router, so it does not qualify the shared expert.
An actual 128-row layer-three input replay fails the merged gate/up,
activation and down byte gates with the candidate cuBLASLt library. Its input
comes from a complete 128-row chunk of the invalid long phase described below;
this is a same-input component diagnostic, not a qualified long-model run.
Independent candidate-only classic cuBLAS replays now isolate the reduction
policy on both physical hosts. With compute 68, algorithm 99 and math mode 16
(disallow reduced-precision reduction), merged gate/up and down from the exact
reference activation match byte for byte. Math mode 0 reproduces the gate/up
failure. These replays bind only the candidate library pair and report version
130000. An earlier mixed-library result resolved a different effective version
and remains excluded from backend selection. Raw output and reference-file
hashes were checked independently for both ranks.

The staged shared-expert implementation selects this explicit classic profile;
the MoE router retains its cuBLASLt profile. Backend resources are owned per
CUDA context and stream. Model teardown fails before freeing operands when
BLAS completion is unknown, and a successful retry releases the real model
owners. Explicit 32 MiB workspace controls now pass at 128, one and five rows
on both physical hosts:
all twelve merged gate/up and independent down outputs are byte-exact with
math mode 16. Both 128-row math-mode-0 gate/up negative controls still fail.
Raw arrays, source hashes, actual library providers/version and workspace
arguments were checked independently. Fresh ten-row shared controls now also
pass on both hosts, using the actual captured layer-three input and resident
weights. The rebuilt implementation and full-model parity remain pending.

Actual captured KDA auxiliary inputs and resident weights now isolate a
separate projection mismatch. The current image's FB/GB projections are
byte-exact at one, five and 128 rows on both hosts. The merged B/FA/GA
projection differs at all three sizes. Candidate classic math mode 0 with the
merged 288-column geometry matches one-row padded-to-three and five-row
outputs exactly; shared-expert math mode 16 does not. At 128 rows, math mode
16 still differs at eleven BF16 elements per rank, and math mode 0 has larger
differences. Independent raw-byte and capture-hash checks preserve these
failures. This evidence does not support a universal classic GEMM policy;
auxiliary row/algorithm qualification remains open, with FB/GB unchanged.

Subsequent same-input controls isolate the BLAS provider version. A coherent
reference 13.1 classic/Lt pair reproduces the merged auxiliary projection
exactly with classic math mode 0 at one padded-to-three row, five rows and
128 rows on both physical hosts. The same pair also passes shared gate/up
and independent down byte comparisons with math mode 16 at one, five, ten
and 128 rows. The same pair's Lt router replay is byte-exact at the four
captured row counts on both hosts; root recomputed the router and auxiliary
comparisons from raw capture bytes, including that captured F32 router logits
are exactly BF16-representable. These component controls do not establish
full-model parity or attest flags used by the resident reference worker.

The native KDA path now follows that geometry. The loader concatenates the
resident BF16 B, FA and GA rows into one owned `[heads + 2 * dim, hidden]`
companion and retires the three sources only after the copy succeeds; the
layer's B/FA/GA handles are interior views of it. Each KDA layer owns a padded
input and merged-output scratch sized from the projection capacity, issues one
`Glm53AuxiliaryClassic` (classic, math mode 0) GEMM, and scatters only the
physical rows into the existing beta/FA/GA buffers. One- and two-row batches
run as a three-row operator with zero padding; only the one-row case is
component-proven, and two rows are unqualified. This branch precedes every
verify-batch GEMV fusion. FB/GB dispatch is unchanged. Scratch is
`2 * max(capacity, 3) * (hidden + heads + 2 * dim)` bytes per layer:
1,131,072 bytes at capacity 129 with 32 local heads, or 38,456,448 bytes
per rank across 34 KDA layers (allocation arithmetic, not a measurement).
On the attested image (source `48624a8605de`), a TP2 replay of the capture
prefixes reproduces the reference layer-0 `normed`, packed Q/K/V and merged
beta/FA/GA bytes exactly at one (decode), five and 128 rows on both ranks, with
only the 13.1 pair mapped. Short16 still fails the frozen gates (all argmax and
text exact; max absolute error up to 2.34, KL up to 0.0588). At layer 0 every
KDA core input is byte-exact and the recurrent core output is the first stage
that differs, at about one BF16 ULP in roughly half the elements.

An actual one-row head replay exposed the scalar packed-matmul difference:
the scalar implementation differs at 99 of 77,440 BF16 channels on rank zero.
The installed reference and its direct extension replay agree byte for byte;
its returned tag and captured autotuner record select K32/N128 tensor-core
arithmetic with 48 splits. Same-input replay of the new production export
matches all 77,440 head outputs byte for byte at both its F32 intermediate
and BF16 output, with either the reference transform or native transformed
input. One-row controls pass on both ranks. Rank-zero controls also pass with
three, five and 128 rows, including a 128-row batch cycling three distinct
captured head inputs. These are component proofs, not full-model parity.

Production packed dispatch now selects the 31 profiles from the source-bound
reference cache across ten local/logical projection geometries. KDA Q/K/V use
the reference's fused logical width and explicit component offsets. MLA Kv-A
uses 20 or 16 splits according to its row bucket; other profiles use 48.
Unsupported topology, geometry or unavailable profiles fail before model
allocation. In that source snapshot, focused model tests (63), runtime tests
(47) and validation-tool tests (70) pass. All 66 source/cache-matched packed component controls pass
their BF16 byte gates and cover all 31 cache profiles. These controls replay
production source through NVRTC; they do not validate embedded NVCC PTX.
The new immutable validation image is built from source snapshot
`1a02497aa6fc6688413f6c43b946ebcf4a1385e50408d3e0726ddd58e1a4ec3d`.
Independent executable checks verify all eight exported NVCC modules are
embedded byte for byte. Six GPU replays using the actual embedded PTX pass
production MMA and BF16-to-F32 transport byte gates: the one-row head,
five-row Q/K/V components, and five- and 128-row Kv-A. Legacy scalar diagnostic
failures remain recorded separately in those reports.

The corrected image's original frozen 16-row comparison still fails the
unchanged numerical tolerances, although all argmax choices and next-token
texts match. Persistent decode matches three of four cases; the longer case
diverges. Comparison metadata copies move only the native numerical-policy
adjunct outside the canonical execution fields, with original and derived
SHA-256 records; raw tensors, requests and thresholds remain unchanged.
The original reference uses a different prefill/context budget, so a fresh
matched-budget reference remains required as a separate comparison. Both ranks
already differ at every captured layer-zero FFN input. This localizes the
difference upstream of that boundary and does not establish an FFN arithmetic
failure. Attention and mHC localization is in progress.

The first 2,047-token long request also exposes an invalid scheduler geometry:
actual captured input rows are 129, fourteen chunks of 128, then 126, despite
a configured prefill limit of 128. The idle scheduler borrows one decode
headroom row from the allocation arena. Subsequent requests were prevented;
the failed geometry and raw captures remain preserved. A native EXL3 budget
cap and rebuilt GPU validation are required. This image is not qualified for
release.

A subsequent diagnostic image is built from source snapshot
`af4e1f2c96941ab11682290d30bff71fd7b2362199cf3984589f9d9a745147d5`.
It adds the configured chunk-budget capability and typed layer-zero upstream
captures, with no numerical kernel changes. Independent checks verify its
3,530 build inputs and all ten embedded PTX modules; the previous eight PTX
blobs are unchanged. GPU validation of this image remains pending.
For these captures, set `ATLAS_EXL3_UPSTREAM_DUMP_DIR` to an explicit per-rank
destination and `ATLAS_EXL3_UPSTREAM_DUMP_LAYERS=0`, disable graphs, and create
the directory's exclusive `capture-enabled` marker only for an isolated serial
capture phase. Remove it afterward. Capture I/O uses the diagnostic router.
Enabled captures reject sequence-parallel geometry and unsupported
layer selection. The captures cover incoming highway, attention normalization,
projections, recurrence output, gated normalization, local/reduced output and
post-attention/FFN mHC boundaries.

The fresh vLLM reference uses the matched 4,096-token context and 128-token
prefill budget. Its source-bound capture hook targets the actual v2
`GPUModelRunner.prepare_inputs` provider. Actual-image CPU checks and the first
runtime request pass on both ranks, including full five-row intermediate
tensors and one physical-vocabulary head row. All 16 registered logit rows
were subsequently collected, with the private 31-profile tuning cache
unchanged. The 16 registered long-context requests also pass reference chunk
and position validation on both ranks: 269 prepared chunks covering 32,820
tokens, with nonterminal chunks of 128 rows and terminal head/context bindings.
Both registered long prefixes also pass reference position checks through
three persistent one-row decode steps. The matched short comparison
against the earlier candidate still fails the frozen numerical gates;
successful capture does not establish numerical parity.

The independent `CODE_REVIEW_2026-10-10.md` identifies repeated routed MoE
scratch allocation, vocabulary staging and missing FFN stage profiling.
The current launch derives a 129-row arena from the configured prefill/decode
budget. Even at this bound, 42 routed layers retain roughly 2.365 GiB per rank
of MoE scratch. A model-owned shared workspace is implemented with the
same geometry, capacity and stream contract, retaining distinct buffers within
each layer invocation. Tests exercise two loaded layers borrowing the same
workspace, failed allocation/free cleanup and completion checks. Actual GPU
numerical validation of sharing remains pending. KV sizing already observes
actual free memory after these allocations. Opt-in FFN profiling uses the
existing profiling switch, reports actual loader layer identity and stage
precision policy, and synchronizes only in profiling mode. Tests verify that
disabled profiling adds no synchronization or tensor reads. The F32
vocabulary transport remains the current reference
contract; a BF16 transport optimization requires separate qualification.

The reference API behavior slice collected 11 raw HTTP receipts and explicitly
skipped the dependent tool-result followup when no valid initial tool ID was
available. Its strict behavior checks fail on reasoning usage and tool
completion fields. These preserved reference diagnostics neither qualify the
Atlas API nor establish an Atlas regression.

Three complete reference repeats capture all 45 FFN layers and both ranks.
The first captured routed FP32 difference is at layer 3; the first captured
post-TP BF16 output difference is at layer 16. Head inputs also vary. These
observations do not establish stability of uncaptured intermediate state or
excuse the failed frozen comparison. Original outputs and tolerances remain
unchanged.

Full SHA-256 reads of all 23 indexed checkpoint shards completed on both
Sparks. The manifests match, and index and file metadata remain unchanged.
This establishes the checkpoint identity used in these runs; it does not
change the failed numerical qualification verdict.

The shared kernel build now supports explicit module compiler flags in
`MODEL.toml` or `KERNEL.toml`. The EXL3 model declares `--fmad=true` only for
`exl3_mhc`, matching the reference's libdevice sigmoid contraction. Effective
flags enter the compilation cache key and source attestation; other modules
retain their existing flags. The EXL3 target shares canonical quant sources
through per-file relative symlinks; the kernel-structure gate passes and all
resolved source bytes are unchanged by this layout correction. This build-script
change and the optional parsed
mHC post-multiplier field in `ModelConfig` need cross-cutting review. The
configured-prefill-budget capability also changes the shared `Model` trait;
ordinary models retain the existing idle arena expansion.
The typed GEMM profile/capability, backend resource quiescence and model
teardown boundary also require cross-cutting review. Existing direct cuBLASLt
dispatch remains unchanged.
The shared routed workspace adds a model-entry stream completion check before
any layer resources are freed. Teardown also defers the final allocation sweep
when an owner release fails, retaining failed pointers for retry. These shared
teardown changes require review beyond the EXL3 numerical paths.

The independent kernel optimization review and its prior-art cross-reference
inform the next profiling work. The immediate additions are an audit of the
exact embedded EXL3 PTX for register spills and targeted installed-reference
index-quantization checks around representable power-of-two scale boundaries.
The production mHC host policy already validates rounded split geometry before
dispatch. Source inspection alone does not establish spills, kernel latency or
the proposed speedups. The cross-reference's corrected estimates remain
hypotheses for Atlas.

Assembly of the unchanged diagnostic image's generic and routed EXL3 PTX
with the builder's CUDA 13.0 assembler reports zero register spill loads/stores,
but disassembly contains local-memory accumulator accesses, including accesses
inside a repeated MMA region. Those counters therefore do not establish that
the accumulator stays in registers. The archived PTX, assembler flags and SASS
make decoder specialization a concrete followup. This is offline assembly
evidence, not a measurement of the driver's runtime code or kernel speed.

After numerical qualification, profile the actual native serving path before
changing packed split-K, coefficient decoding, mHC parallelization or expert
grouping. Each change must retain the existing component byte gates and
explicit row-dependent arithmetic policy. Packed shared experts, compressed
KV or collectives, and BF16 vocabulary transport need separate qualification.
Reference repeat variability remains diagnostic; it does not relax the frozen
comparison gates. Host compaction and fabric changes also require independent
measurements and are excluded from the current parity run.
