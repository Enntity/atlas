// SPDX-License-Identifier: AGPL-3.0-only

// F4 DFlash2 draft-only observer capture variant (ATLAS_GLM_EXL3_DFLASH_CAPTURE
// =bf16mean). Reference-shaped contraction of the FP32 mHC highway: round each
// stream to BF16 first (vLLM materializes the post streams in BF16), then
// mean the streams in F32 and round once more, into rows `out_stride` BF16
// elements apart. The default capture (hc_contract_strided) means the F32
// streams and rounds once.
#include <cuda_bf16.h>

// Grid (rows, 1, 1), block (256, 1, 1).
extern "C" __global__ void exl3_dflash_contract_bf16_streams(
    const float* __restrict__ streams,
    __nv_bfloat16* __restrict__ out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int out_stride
) {
    const unsigned int t = blockIdx.x;
    const float* row = streams + (size_t)t * hc_mult * hidden_size;
    __nv_bfloat16* dst = out + (size_t)t * out_stride;
    for (unsigned int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < hc_mult; ++i) {
            sum += __bfloat162float(__float2bfloat16(row[(size_t)i * hidden_size + d]));
        }
        dst[d] = __float2bfloat16(sum / (float)hc_mult);
    }
}
