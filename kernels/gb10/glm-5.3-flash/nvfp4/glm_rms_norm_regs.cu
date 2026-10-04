// SPDX-License-Identifier: AGPL-3.0-only

// GLM-only register twin of rms_norm_vanilla. It reuses the common file's
// helpers; the legacy exports of this include are contained in this new
// module, and every serving caller of rms_norm_vanilla keeps its module.
// The row body lives in glm_rms_norm_regs.cuh (shared with the step-fuse
// seam finalizer).
#include "glm_rms_norm_regs.cuh"

// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1).
extern "C" __global__ void rms_norm_vanilla_regs(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    atlas_pdl_enter();
    unsigned int token = blockIdx.x;
    rms_norm_vanilla_regs_row(input + token * hidden_size, weight, output + token * hidden_size,
                              hidden_size, eps);
}
