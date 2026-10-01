# Experimental GLM native sparse prefill bridge

Selected source checkpoint for resuming the private native-attention experiment.
This directory does not establish public model qualification or an end-to-end
performance result. It contains no weights, compiled objects/libraries, raw model
responses, host configuration, or private deployment locators.

`SOURCE_MAP.json` maps every copied source to its experiment-relative path and
SHA256. Nine files are byte-identical; the two Python checkers gained
an SPDX comment, merge/split copies had an extra trailing blank line removed,
and `native-bridge.cpp`/`native-init.cpp` later gained origin comments only.
No CUDA math, launch geometry, wrapper ABI, or test logic was changed during
this copy. DeepSeek supplied supervised mechanical prep/split/
merge drafts; reviewed corrections and wrappers were independently checked.
Raw assignment receipts remain in the private experiment archive, outside this
source checkpoint.

## Required retained artifact and source limitation

The build requires this exact existing object at
`/native/csrc_sparse_mla_sm120_prefill.cuda.o`:

```text
9e372b5a47ade0a8332a73878ec3f6b917ae137447fea2335d2003dd146d549a
```

`build-unified.sh` rejects any other object hash. This is the NVIDIA sparse MLA
SM120/SM121 prefill object from the retained FlashInfer0.6.15/121a cache dated
September4. The available copied NVIDIA source snapshot is newer; exact source
identity for the retained object has not been established. Do not claim that
this directory rebuilds that object from source. Preserve the verified object
in the private artifact store to resume; it is intentionally not committed here.
SparkGLM's installer (`install/build-native.sh`) instead rebuilds the object from
source at FlashInfer `8eccd0c1`; the retained object's revision is still
unresolved.
The enum header is the exact copied NVIDIA declaration needed for the host ABI.

## Build and mounting

Mount this directory read/write at `/eval`, and the retained artifact directory
read-only at `/native`. Use a CUDA13.0 compiler environment with `nvcc`, `g++`,
`readelf`, and `sha256sum`; no GPU is needed for compilation/linking:

```sh
bash /eval/build-unified.sh
```

The script uses explicit `compute_121a`/`sm_121a`, precise division, disabled
FTZ/FMA contraction, and links only the one retained object plus CUDA/C++
runtime dependencies. `--no-undefined` and `-z defs` reject unresolved linkage;
DT_NEEDED checks reject Torch/TVM/Python dependencies in the resulting library.
Output is `/eval/libatlas_glm_sparse_native.so`, with its hash printed.

The original checked builder image digest was
`sha256:b1ff6a353c269287e3e77a8f812fbb07a2d33112a23bc2d87b1d00048b13f701`.
The Python validation runtime digest was
`sha256:1c5d50d6717d4fa958f8dcb701b635d78c77c6b3e1cb4281a3b3c0344ca30c9e`.
These are retained-environment identifiers, not published image availability
or a complete reproducible upstream build recipe.

## ABI, ownership and precision

`atlas-glm-sparse-native.h` defines ABI1:112-byte arguments,8-byte alignment,
ten disjoint device spans, caller-owned device/context/stream and capacities.
Call `atlas_glm_sparse_native_init()` on the serving context **before KV free
memory sizing**. It forces the selected kernel/module setup without attention
execution or buffer allocation. The process retains the shared-library mapping.
Run is asynchronous and allocates nothing; every span remains live through
stream completion. Any failure after submission aborts that forward pass; do
not retry through a different attention implementation.

This route accepts only ordinary continued GLM prefills with2048..4100rows,
seq_start>=2048, end<=32768,32heads and BF16 physical16-token cache blocks.
Q is copied from512 to576 channels with zero RoPE padding. The KV pack uses
E4M3 plus four FP32 scales and zero RoPE, so it changes attention operand
precision relative to the original BF16 attention kernel. The reviewed prep
scale multiplies by the rounded FP32 reciprocal of448 to match the pinned
Torch CPU-scalar division convention; the original draft's true division failed
byte parity. No normalization or query quantization is added by preparation.

The ID split retains all2048 main IDs and the causal0..3-token tail. Zero-tail
native calls use a harmless scratch slot, discarded by the merge's exact
zero-tail branch. Caller must supply valid block-table and selected-ID contents;
host span checks cannot prove device index validity. The linked upstream object
can abort on certain CUDA_CHECK failures; this shim cannot make those errors
recoverable without rebuilding the object.

## Validation entry points

CPU-only checks:

```sh
bash -n /eval/build-unified.sh
python3 -m py_compile /eval/native-prep-check.py /eval/unified-check.py
```

In an explicitly authorized idle GPU window, first check initialization:

```sh
g++ -std=c++17 /eval/native-init-check.cpp -I/usr/local/cuda/include \
  -L/eval -latlas_glm_sparse_native -Wl,-rpath,/eval \
  -L/usr/local/cuda/lib64 -lcudart -o /tmp/native-init-check
timeout 30s /tmp/native-init-check
```

For the prep byte oracle, separately build `native-prep.cu` using the compile
command in its header to `/eval/libatlas_native_prep.so`, then run:

```sh
timeout 180s python3 /eval/native-prep-check.py
```

This gate previously passed4Q and24KV cases, including32K tails, finite extremes,
exact input/guard preservation, and a1GiB Torch cap. Raw evidence remains in the
private experiment archive; that component pass is not a model-quality claim.

The combined gate requires the previously reviewed reference artifacts
`/eval/libatlas_sparse.so`, `/eval/libatlas_sparse_merge.so`, and
`/native/sparse_mla_sm120.so`, with Torch and TVM FFI available in the validation
runtime. They are reference-test dependencies, not dependencies of the unified
library. Their sources/receipts remain in the private experiment archive; these
libraries are not included here. Then run:

```sh
timeout 240s python3 /eval/unified-check.py
```

The checker verifies ABI, rejected-call poison, exact metadata/Q/KV preparation,
and output/LSE byte identity against separately invoked reference components at
2048/3515/4100/4096rows. It checks guards and input hashes under a2GiB Torch cap,
then times4096rows with3warmups/9rotating pairs. The whole-preparation-inclusive
speed gate is1.3x versus the original Atlas wrapper. This source checkpoint does
not assert that this combined gate or subsequent model qualification passed.

## Licenses

Original bridge/adaptation/checker files use AGPL-3.0-only. The copied NVIDIA
enum header retains its original BSD-3-Clause notice and SPDX declaration,
unaltered. BSD-3-Clause code can be included in an AGPL distribution while
retaining its conditions; it is not relabeled as exclusively AGPL.
`NATIVE-BRIDGE-NOTICE.txt` retains the NVIDIA copyright, three conditions and
disclaimer required with redistribution of the linked object. The enum header's
original first-line copyright intentionally differs from the repository's
AGPL-first-line convention; preserve the third-party notice if a generic header
checker flags it. No repository-wide license-check exception was changed here.
The NVIDIA sparse-MLA SM120 prefill source this links against is BSD-3-Clause
(Copyright 2026 NVIDIA CORPORATION & AFFILIATES) inside Apache-2.0 FlashInfer
(https://github.com/flashinfer-ai/flashinfer). Licenses of the built image:
see Enntity/sparkglm `docs/LICENSING.md`.
