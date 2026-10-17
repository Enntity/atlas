// SPDX-License-Identifier: AGPL-3.0-only
// Explicit source-reproduction harness only. Embedded runtime PTX replay in
// native_gated_norm_replay.py is preferred for arithmetic attribution.
// Compile with the frozen builder toolchain/production flags and repository
// common include path; this file deliberately contains no copied arithmetic.
#include <cuda_runtime.h>
#include "kda.cu"

extern "C" cudaError_t atlas_replay_gated_norm(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int tokens, float epsilon, cudaStream_t stream
) {
    if (tokens == 0 || tokens > 0xffffffffu / 32u ||
        !(epsilon > 0.0f) || !isfinite(epsilon)) return cudaErrorInvalidValue;
    kda_sigmoid_gated_rms_norm<<<tokens * 32u, 128u, 0, stream>>>(
        input, gate, weight, output, 32u, 128u, epsilon);
    return cudaGetLastError();
}
