// SPDX-License-Identifier: AGPL-3.0-only

// embed_from_argmax: GPU-side token embedding after argmax.
//
// Reads the argmax result (u32 token ID) from device memory and copies
// the corresponding row from the BF16 embedding table to the output buffer.
// Eliminates the D2H sync that was required when the CPU read the token ID.
//
// Also writes the token ID to a secondary output for deferred CPU readback.
//
// Grid: (ceil(hidden_size/256), 1, 1)  Block: (256, 1, 1)
// 256 threads × 4 bf16 per thread = 1024 elements per iteration.
// hidden_size=8192 → 32 blocks × 256 threads, each loads 1 element.

#include <cuda_bf16.h>

extern "C" __global__ void embed_from_argmax(
    const unsigned int* __restrict__ argmax_result,   // [1] token ID on GPU
    const __nv_bfloat16* __restrict__ embed_table,    // [vocab_size, hidden_size]
    __nv_bfloat16* __restrict__ output,               // [hidden_size]
    unsigned int* __restrict__ token_id_out,           // [1] copy of token ID for deferred readback
    unsigned int hidden_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread 0 of block 0 writes the token ID for deferred CPU readback
    if (idx == 0) {
        token_id_out[0] = argmax_result[0];
    }
    if (idx < hidden_size) {
        unsigned int token_id = argmax_result[0];
        output[idx] = embed_table[token_id * hidden_size + idx];
    }
}

// Batched embedding: gather N rows from the embedding table.
//
// token_ids[i] → embed_table[token_ids[i], :] → output[i, :]
// Replaces N individual D2D copies with a single kernel launch.
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void batched_embed(
    const unsigned int* __restrict__ token_ids,        // [num_tokens] on device
    const __nv_bfloat16* __restrict__ embed_table,     // [vocab_size, hidden_size]
    __nv_bfloat16* __restrict__ output,                // [num_tokens, hidden_size]
    unsigned int hidden_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int token_id = token_ids[token_idx];
    const __nv_bfloat16* src = embed_table + (unsigned long long)token_id * hidden_size;
    __nv_bfloat16* dst = output + (unsigned long long)token_idx * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// FP32 output variant: reads BF16 embedding, writes FP32 to the residual stream.
extern "C" __global__ void batched_embed_f32(
    const unsigned int* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_table,     // [vocab_size, hidden_size] BF16
    float* __restrict__ output,                         // [num_tokens, hidden_size] FP32
    unsigned int hidden_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int token_id = token_ids[token_idx];
    const __nv_bfloat16* src = embed_table + (unsigned long long)token_id * hidden_size;
    float* dst = output + (unsigned long long)token_idx * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// FP32 output variant of embed_from_argmax for decode with FP32 residual stream.
extern "C" __global__ void embed_from_argmax_f32(
    const unsigned int* __restrict__ argmax_result,
    const __nv_bfloat16* __restrict__ embed_table,
    float* __restrict__ output,                          // FP32
    unsigned int* __restrict__ token_id_out,
    unsigned int hidden_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        token_id_out[0] = argmax_result[0];
    }
    if (idx < hidden_size) {
        unsigned int token_id = argmax_result[0];
        output[idx] = __bfloat162float(embed_table[token_id * hidden_size + idx]);
    }
}


// FP8-table batched embedding gather: rows stored as FP8 E4M3 BYTES with a
// per-row f32 dequant scale (exactly the `quantize_bf16_to_fp8` output
// layout from dense_gemv_fp8w.cu). Built for the n-gram embedding tables
// (LongCat / Qwen3.8-Flash-Next family) whose ~10M-row tables are
// FP8-quantized at load to halve their footprint; the gather dequantizes
// on read and hands BF16 rows to the projection GEMM.
//
// DECODE CONVENTION: software E4M3 decode on raw bytes, copied from
// dense_gemv_fp8w.cu's scl_fp8 — the hardware `__nv_fp8_e4m3` cast is
// non-standard under SCALE/gfx1151 and would mismatch the encoder there.
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
__device__ __forceinline__ float ngram_fp8_decode(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
extern "C" __global__ void batched_embed_fp8(
    const unsigned int* __restrict__ token_ids,   // [num_tokens] on device
    const unsigned char* __restrict__ embed_table, // [rows, hidden_size] FP8 bytes
    const float* __restrict__ row_scale,          // [rows] f32 dequant scale
    __nv_bfloat16* __restrict__ output,           // [num_tokens, hidden_size]
    unsigned int hidden_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int row = token_ids[token_idx];
    const unsigned char* src = embed_table + (unsigned long long)row * hidden_size;
    const float scale = row_scale[row];
    __nv_bfloat16* dst = output + (unsigned long long)token_idx * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = __float2bfloat16(ngram_fp8_decode(src[i]) * scale);
    }
}

// NVFP4 (ModelOpt, group 16) table gather: each row is PACKED as
//   [head_dim/2 B: E2M1 code pairs, LOW nibble = even element]
//   [head_dim/16 B: E4M3 block scales, one per 16 elements]
//   [zero pad up to row_stride]
// — the layout `spark_storage::ngram_pack` writes for the Qwen3.8-Flash-Next-
// FP4-Local PLE table, whose checkpoint keeps the codes and the scales in two
// tensors. One global f32 (`weight_scale_2`) multiplies every row.
//
// value = (e2m1 * e4m3_scale) * global, in f32, then RNE to BF16: the order of
// flashinfer's oracle (flashinfer/experimental/b12x/sequence/ple_embedding/
// reference.py `_dequantize_selected_nvfp4`, Apache-2.0). e2m1 x e4m3 is exact
// in f32 (at most 6 significant bits), so only the global multiply rounds;
// `--fmad=false` (common KERNEL.toml) keeps the compiler from fusing anything.
// The scale decode is `ngram_fp8_decode`, which maps E4M3's NaN to 0 where the
// reference would propagate NaN — a valid checkpoint has no NaN scale.
//
// Grid: (num_rows, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void batched_embed_nvfp4(
    const unsigned int* __restrict__ token_ids,    // [num_rows] slot per row
    const unsigned char* __restrict__ embed_table, // [slots, row_stride] packed
    __nv_bfloat16* __restrict__ output,            // [num_rows, head_dim]
    float global_scale,
    unsigned int head_dim,
    unsigned int row_stride
) {
    const unsigned int row = token_ids[blockIdx.x];
    const unsigned char* src = embed_table + (unsigned long long)row * row_stride;
    const unsigned char* scales = src + head_dim / 2;
    __nv_bfloat16* dst = output + (unsigned long long)blockIdx.x * head_dim;
    for (unsigned int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const unsigned int b = src[i >> 1];
        const unsigned int code = (i & 1u) ? (b >> 4) : (b & 0xFu);
        // E2M1: 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign. Code 8 is -0,
        // as in flashinfer's Triton kernel and `atlas_core::numeric::FP4_E2M1`
        // (the oracle's table says +0; no product can tell them apart).
        const unsigned int mag = code & 7u;
        float v = mag < 4u ? 0.5f * (float)mag
                           : (float)(2u << ((mag - 4u) >> 1)) * ((mag & 1u) ? 1.5f : 1.0f);
        if (code & 8u) v = -v;
        const float s = ngram_fp8_decode(scales[i >> 4]);
        dst[i] = __float2bfloat16((v * s) * global_scale);
    }
}
