// SPDX-License-Identifier: AGPL-3.0-only

// Native EXL3 index precision. Public cache addressing remains BF16; pooled
// keys store FP8 values dequantized by their explicit power-of-two scale.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "exl3_common.cuh"
#include "glm_indexer_score.cuh"

namespace {
__device__ __forceinline__ float glm_index_exp2_approx_ftz(float argument) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(argument));
    return result;
}
__device__ __forceinline__ float glm_index_quant_row(
    float value, float* scratch, bool dequantize, float& scale) {
    value = __bfloat162float(__float2bfloat16_rn(hadamard128(value, scratch)));
    const unsigned int lane = threadIdx.x;
    scratch[lane] = fabsf(value);
    __syncthreads();
    for (unsigned int distance = 64; distance; distance >>= 1) {
        if (lane < distance) scratch[lane] = fmaxf(scratch[lane], scratch[lane + distance]);
        __syncthreads();
    }
    const float maximum = fmaxf(scratch[0], 1e-4f);
    scale = exp2f(ceilf(log2f(__fmul_rn(maximum, 1.0f / 448.0f))));
    const float normalized = fminf(448.0f, fmaxf(-448.0f, __fdividef(value, scale)));
    const __nv_fp8_e4m3 quantized(normalized);
    const float raw = static_cast<float>(quantized);
    return dequantize ? __fmul_rn(raw, scale) : raw;
}
}

// Grid=(rows,1,1), block=(128,1,1). Each row is exactly one H128 vector.
// input and output may alias exactly; scales must occupy separate storage.
// Query stores raw FP8 values as BF16; caller folds scales into F32 weights.
extern "C" __global__ void glm_index_query_quant_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    float* __restrict__ scales, unsigned int rows) {
    if (blockIdx.x >= rows || blockDim.x != 128) return;
    __shared__ float scratch[128];
    const unsigned long long index = (unsigned long long)blockIdx.x * 128 + threadIdx.x;
    float scale;
    const float quantized = glm_index_quant_row(__bfloat162float(input[index]), scratch, false, scale);
    output[index] = __float2bfloat16_rn(quantized);
    if (threadIdx.x == 0) scales[blockIdx.x] = scale;
}

// Same paged tail/cache ABI as the legacy finalizer, except APE is F32.
// Grid=(num_tokens,1,1), block=(128,1,1); only pool-ending slots write.
extern "C" __global__ void glm_index_kpool_finalize_quant_bf16(
    const __nv_bfloat16* __restrict__ tail,
    const unsigned int* __restrict__ tail_map,
    const float* __restrict__ ape,
    __nv_bfloat16* __restrict__ cache,
    const long long* __restrict__ slots,
    unsigned int num_tokens, unsigned int block_size,
    unsigned int pool_size, unsigned int head_dim,
    unsigned long long tail_block_stride_bytes,
    unsigned long long values_block_stride_bytes) {
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens || pool_size != 4 || head_dim != 128
        || block_size == 0 || blockDim.x != 128) return;
    const long long slot = slots[token];
    if (slot < 0) return;
    const unsigned int physical = (unsigned int)(slot / block_size);
    const unsigned int offset = (unsigned int)(slot % block_size);
    if (offset % 4 != 3) return;
    const unsigned int tail_index = tail_map ? tail_map[physical] : physical;
    if (tail_index == 0xffffffffu) return;
    const __nv_bfloat16* block_tail = (const __nv_bfloat16*)((const char*)tail
        + (unsigned long long)tail_index * tail_block_stride_bytes);
    const unsigned int lane = threadIdx.x;
    const unsigned long long start = (unsigned long long)(offset - 3) * 128 + lane;
    const __nv_bfloat16* keys = block_tail + start;
    const __nv_bfloat16* gates = block_tail + (unsigned long long)block_size * 128 + start;
    float scores[4], maximum = -INFINITY;
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        scores[i] = __fadd_rn(__bfloat162float(gates[i * 128]), ape[i * 128 + lane]);
        maximum = fmaxf(maximum, scores[i]);
    }
    float numerator = 0.0f, denominator = 0.0f;
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        // Triton's exp uses the CUDA exp2 approximation after an F32 multiply.
        const float probability = glm_index_exp2_approx_ftz(__fmul_rn(__fsub_rn(scores[i], maximum), 1.4426950408889634f));
        denominator = __fadd_rn(denominator, probability);
        numerator = __fmaf_rn(__bfloat162float(keys[i * 128]), probability, numerator);
    }
    const float pooled = __bfloat162float(__float2bfloat16_rn(__fdividef(numerator, denominator)));
    __shared__ float scratch[128];
    float scale;
    const float quantized = glm_index_quant_row(pooled, scratch, true, scale);
    __nv_bfloat16* destination = (__nv_bfloat16*)((char*)cache
        + (unsigned long long)physical * values_block_stride_bytes)
        + (unsigned long long)(offset / 4) * 128;
    destination[lane] = __float2bfloat16_rn(quantized);
}

// Grid=(rows,1,1), block=(128,1,1). In-place BF16 values with F32 parameters.
extern "C" __global__ void glm_index_layernorm_f32_params_bf16(
    __nv_bfloat16* __restrict__ values,
    const float* __restrict__ weight, const float* __restrict__ bias,
    unsigned int rows, unsigned int dim, float eps) {
    if (blockIdx.x >= rows || dim != 128 || blockDim.x != 128) return;
    const unsigned int lane = threadIdx.x;
    const unsigned long long index = (unsigned long long)blockIdx.x * 128 + lane;
    const float value = __bfloat162float(values[index]);
    __shared__ float scratch[128];
    scratch[lane] = value;
    __syncthreads();
    for (unsigned int distance = 64; distance; distance >>= 1) {
        if (lane < distance) scratch[lane] = __fadd_rn(scratch[lane], scratch[lane + distance]);
        __syncthreads();
    }
    const float mean = __fmul_rn(scratch[0], 1.0f / 128.0f);
    const float centered = __fsub_rn(value, mean);
    __syncthreads();
    scratch[lane] = __fmul_rn(centered, centered);
    __syncthreads();
    for (unsigned int distance = 64; distance; distance >>= 1) {
        if (lane < distance) scratch[lane] = __fadd_rn(scratch[lane], scratch[lane + distance]);
        __syncthreads();
    }
    const float inverse = rsqrtf(__fadd_rn(__fmul_rn(scratch[0], 1.0f / 128.0f), eps));
    values[index] = __float2bfloat16_rn(__fmaf_rn(__fmul_rn(centered, inverse), weight[lane], bias[lane]));
}

// Grid=(ceil(heads/128),rows,1), block=(128,1,1). Matrix is [heads,hidden].
extern "C" __global__ void glm_index_weight_gemv_f32(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ matrix, float* __restrict__ output,
    unsigned int rows, unsigned int heads, unsigned int hidden) {
    const unsigned int head = blockIdx.x * 128 + threadIdx.x;
    if (blockIdx.y >= rows || head >= heads) return;
    float sum = 0.0f;
    for (unsigned int k = 0; k < hidden; ++k)
        sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(input[(unsigned long long)blockIdx.y * hidden + k]),
            __bfloat162float(matrix[(unsigned long long)head * hidden + k])));
    output[(unsigned long long)blockIdx.y * heads + head] = sum;
}

// Grid=(ceil(count/256),1,1), block=(256,1,1); explicit F32 normalization.
extern "C" __global__ void glm_index_weight_scale_f32(
    float* __restrict__ weights, const float* __restrict__ query_scales,
    unsigned long long count, float normalization) {
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    weights[index] = __fmul_rn(__fmul_rn(weights[index], query_scales[index]), normalization);
}

// F32 weights already include query scale and head/dimension normalization.
// Same paged argument ABI as the legacy scorer except the weights pointer.
// Grid=(ceil(logits_stride/8),rows,1), block=(256,1,1), shared=0.
extern "C" __global__ void glm_index_logits_quant_bf16(
    const __nv_bfloat16* __restrict__ query,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows, unsigned int seq_len_start,
    unsigned int logits_stride, unsigned int index_heads, unsigned int head_dim,
    unsigned int pool_size, unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes) {
    if (head_dim != 128 || pool_size != 4 || cache_block_size == 0 || blockDim.x != 256) return;
    glm_index_logits_impl<float, false>(query, weights, index_cache, logits, block_table,
        rows, seq_len_start + blockIdx.y + 1, logits_stride, index_heads, head_dim,
        pool_size, cache_block_size, index_block_stride_bytes);
}

// Grid=(ceil(logits_stride/8),1,1), block=(256,1,1), shared=0.
extern "C" __global__ void glm_index_logits_quant_bf16_dynamic(
    const __nv_bfloat16* __restrict__ query,
    const float* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows, const unsigned int* __restrict__ seq_len,
    unsigned int logits_stride, unsigned int index_heads, unsigned int head_dim,
    unsigned int pool_size, unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes, unsigned int dense_threshold) {
    if (rows != 1 || blockIdx.y != 0 || blockDim.x != 256) return;
    const unsigned int length = seq_len[0];
    const bool valid = head_dim == 128 && pool_size == 4 && cache_block_size > 0
        && length > dense_threshold
        && (unsigned long long)length <= (unsigned long long)logits_stride * pool_size;
    if (!valid) {
        const unsigned int pool = blockIdx.x * 8 + (threadIdx.x >> 5);
        if ((threadIdx.x & 31) == 0 && pool < logits_stride) logits[pool] = -INFINITY;
        return;
    }
    glm_index_logits_impl<float, false>(query, weights, index_cache, logits, block_table,
        1, length, logits_stride, index_heads, head_dim, pool_size, cache_block_size,
        index_block_stride_bytes);
}
