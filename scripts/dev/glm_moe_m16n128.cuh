// SPDX-License-Identifier: AGPL-3.0-only

// Test-only down wrappers. Include AFTER the production native-FP4 .cu,
// which provides the validated shared M16 helper and gate/up exports.
// No production down registration or dispatch uses these symbols.
#pragma once

#define GLM_M16_ARGS \
    const unsigned char* A_packed, const unsigned char* A_scale, \
    const unsigned long long* B_packed_ptrs, const unsigned long long* B_scale_ptrs, \
    const float* scale2_vals, __nv_bfloat16* C, const int* expert_offsets, \
    const int* sorted_token_ids, unsigned int num_experts, unsigned int N, unsigned int K

#define GLM_M16_CALL(VEC) \
    glm_moe_m16n128_impl<VEC>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, \
        scale2_vals, C, expert_offsets, sorted_token_ids, num_experts, N, K, \
        blockIdx.z, blockIdx.y, blockIdx.x)

extern "C" __global__ void glm_moe_down_m16n128(GLM_M16_ARGS) {
    if (N != 4096 || K != 2048 || sorted_token_ids != nullptr) return;
    GLM_M16_CALL(false);
}
extern "C" __global__ void glm_moe_down_m16n128_vecscale(GLM_M16_ARGS) {
    if (N != 4096 || K != 2048 || sorted_token_ids != nullptr) return;
    GLM_M16_CALL(true);
}
#undef GLM_M16_ARGS
#undef GLM_M16_CALL
