// SPDX-License-Identifier: AGPL-3.0-only

// Qwen3.8-Flash-Next: the glm-5.3-flash `mxfp8_gemv.cu` kernels verbatim.
// The FP4-Local checkpoint (nvidia/Qwen3.8-Flash-Next-FP4-Local) ships the
// GDN, QSA attention and shared-expert projections as MXFP8, E4M3 [n, k] +
// UE8M0 [n, k/32], which is exactly the row-major layout these tiers read, so
// decode reads the checkpoint's own bytes (`weight_map::mxfp8_load`).
// Including (not copying) keeps one source for both targets, the precedent
// glm-5.3-flash's `w4a16_gemm.cu` set for deepseek-v4-flash's.
#include "../../glm-5.3-flash/nvfp4/mxfp8_gemv.cu"
