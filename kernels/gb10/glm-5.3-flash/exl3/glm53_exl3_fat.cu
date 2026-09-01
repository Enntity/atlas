// SPDX-License-Identifier: AGPL-3.0-only

// GB10-native, device-compacted adaptation of Mia AI Lab's MIT-licensed E2
// direct-trellis fat GEMM. The original notice is in MIA_EXL3_FAT_LICENSE.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "exl3_vendor/util.h"
#include "exl3_vendor/util.cuh"
#include "exl3_vendor/ptx.cuh"
#include "exl3_vendor/quant/exl3_dq.cuh"
#include "exl3_vendor/quant/hadamard_inner.cuh"

namespace {

constexpr int FAT_THREADS = 256;
constexpr int FAT_TILE_M = 128;
constexpr int FAT_TILE_K = 16;
constexpr int FAT_TILE_N = 128;
constexpr int FAT_M_BLOCKS = FAT_TILE_M / 16;
constexpr int FAT_N_BLOCKS = FAT_TILE_N / 16;
constexpr int FAT_PACKED_WORDS = 4 * 16;
constexpr int FAT_SLOTS = 288;
constexpr float HAD_SCALE = 0.088388347648f;

__device__ __forceinline__ bool fat_descriptor(
    const int64_t* expert_count,
    const int4* descriptors,
    int num_experts,
    int slot,
    int4& descriptor) {
  if (slot >= FAT_SLOTS || slot >= expert_count[num_experts]) return false;
  descriptor = descriptors[slot];
  return true;
}

__device__ __forceinline__ void fat_had_ff_128(
    float4& values,
    const half* scale,
    int lane) {
  float s0 = values.x + values.y;
  float d0 = values.x - values.y;
  float s1 = values.z + values.w;
  float d1 = values.z - values.w;
  values.x = s0 + s1;
  values.y = d0 + d1;
  values.z = s0 - s1;
  values.w = d0 - d1;
  shuffle_had_f2x32(values.x, values.y, lane);
  shuffle_had_f2x32(values.z, values.w, lane);
  values.x *= HAD_SCALE * __half2float(scale[lane * 4]);
  values.y *= HAD_SCALE * __half2float(scale[lane * 4 + 1]);
  values.z *= HAD_SCALE * __half2float(scale[lane * 4 + 2]);
  values.w *= HAD_SCALE * __half2float(scale[lane * 4 + 3]);
}

__device__ __forceinline__ void fat_gather_had_128(
    const half* input,
    half* output,
    const half* scale) {
  int lane = threadIdx.x & 31;
  half4 packed = reinterpret_cast<const half4*>(input)[lane];
  half4 scales = reinterpret_cast<const half4*>(scale)[lane];
  float h0 = __low2float(packed.x) * __low2float(scales.x);
  float h1 = __high2float(packed.x) * __high2float(scales.x);
  float h2 = __low2float(packed.y) * __low2float(scales.y);
  float h3 = __high2float(packed.y) * __high2float(scales.y);
  float s0 = h0 + h1;
  float d0 = h0 - h1;
  float s1 = h2 + h3;
  float d1 = h2 - h3;
  h0 = s0 + s1;
  h1 = d0 + d1;
  h2 = s0 - s1;
  h3 = d0 - d1;
  shuffle_had_f4x32(h0, h1, h2, h3, lane);
  packed.x = __floats2half2_rn(h0 * HAD_SCALE, h1 * HAD_SCALE);
  packed.y = __floats2half2_rn(h2 * HAD_SCALE, h3 * HAD_SCALE);
  reinterpret_cast<half4*>(output)[lane] = packed;
}

__device__ __forceinline__ void fat_had_pre_ff_128(
    float4& values,
    const half* scale,
    int lane) {
  values.x *= __half2float(scale[lane * 4]);
  values.y *= __half2float(scale[lane * 4 + 1]);
  values.z *= __half2float(scale[lane * 4 + 2]);
  values.w *= __half2float(scale[lane * 4 + 3]);
  float s0 = values.x + values.y;
  float d0 = values.x - values.y;
  float s1 = values.z + values.w;
  float d1 = values.z - values.w;
  values.x = s0 + s1;
  values.y = d0 + d1;
  values.z = s0 - s1;
  values.w = d0 - d1;
  shuffle_had_f2x32(values.x, values.y, lane);
  shuffle_had_f2x32(values.z, values.w, lane);
  values.x *= HAD_SCALE;
  values.y *= HAD_SCALE;
  values.z *= HAD_SCALE;
  values.w *= HAD_SCALE;
}

template <bool scatter>
__device__ __forceinline__ void fat_gemm_body(
    const half* a,
    const uint16_t* packed,
    float* out,
    const half* svh,
    const int64_t* token_idx,
    const half* route_weight,
    int route_start,
    int size_m,
    int size_k,
    int size_n,
    int out_stride,
    int out_column,
    int m_tile,
    int n_tile) {
  extern __shared__ unsigned char shared_raw[];
  half* sh_a = reinterpret_cast<half*>(shared_raw);
  uint16_t* sh_b = reinterpret_cast<uint16_t*>(sh_a + FAT_TILE_M * FAT_TILE_K);
  float* sh_c = reinterpret_cast<float*>(sh_b + FAT_N_BLOCKS * FAT_PACKED_WORDS);
  int t = threadIdx.x;
  int warp = t / 32;
  int lane = t & 31;
  int m_base = m_tile * FAT_TILE_M;
  int n_base = n_tile * FAT_TILE_N;
  int tiles_n = size_n / 16;

  FragC frag_c[FAT_M_BLOCKS][2];
#pragma unroll
  for (int mb = 0; mb < FAT_M_BLOCKS; ++mb) {
    frag_c[mb][0] = {};
    frag_c[mb][1] = {};
  }

  for (int k_block = 0; k_block < size_k / FAT_TILE_K; ++k_block) {
    int a_row = t / 2;
    int a_col8 = t & 1;
    int a_dst_col8 = a_col8 ^ ((a_row >> 2) & 1);
    int4 a_value = {};
    if (m_base + a_row < size_m) {
      const int4* source = reinterpret_cast<const int4*>(
          a + (route_start + m_base + a_row) * size_k +
          k_block * FAT_TILE_K);
      a_value = source[a_col8];
    }
    reinterpret_cast<int4*>(sh_a)[a_row * 2 + a_dst_col8] = a_value;
    if (t < 64) {
      const int4* source = reinterpret_cast<const int4*>(
          packed + (k_block * tiles_n + n_base / 16) * FAT_PACKED_WORDS);
      reinterpret_cast<int4*>(sh_b)[t] = source[t];
    }
    __syncthreads();

    FragB frag_b0;
    FragB frag_b1;
    const uint32_t* warp_b = reinterpret_cast<const uint32_t*>(
        sh_b + warp * FAT_PACKED_WORDS);
    dq_dispatch<4, 1>(warp_b, lane << 3, frag_b0, frag_b1);
#pragma unroll
    for (int mb = 0; mb < FAT_M_BLOCKS; ++mb) {
      FragA frag_a;
      int row = (lane % 8) + 8 * ((lane / 8) % 2) + mb * 16;
      int base_col = lane / 16;
      int swizzled_col = base_col ^ ((row >> 2) & 1);
      ldsm4(frag_a, reinterpret_cast<int4*>(sh_a) + row * 2 + swizzled_col);
      ptx_mma_m16n8k16(frag_a, frag_b0, frag_c[mb][0]);
      ptx_mma_m16n8k16(frag_a, frag_b1, frag_c[mb][1]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int mb = 0; mb < FAT_M_BLOCKS; ++mb) {
    int rows = min(16, size_m - (m_base + mb * 16));
    if (rows <= 0) break;
    int row0 = lane / 4;
    int row1 = row0 + 8;
    int col = (lane % 4) * 2;
    int n0 = warp * 16;
    if (row0 < rows) {
      float* dst = sh_c + row0 * FAT_TILE_N + n0 + col;
      dst[0] = frag_c[mb][0][0];
      dst[1] = frag_c[mb][0][1];
      dst[8] = frag_c[mb][1][0];
      dst[9] = frag_c[mb][1][1];
    }
    if (row1 < rows) {
      float* dst = sh_c + row1 * FAT_TILE_N + n0 + col;
      dst[0] = frag_c[mb][0][2];
      dst[1] = frag_c[mb][0][3];
      dst[8] = frag_c[mb][1][2];
      dst[9] = frag_c[mb][1][3];
    }
    __syncthreads();
    for (int row = warp; row < rows; row += 8) {
      float4 values = reinterpret_cast<float4*>(
          sh_c + row * FAT_TILE_N)[lane];
      fat_had_ff_128(values, svh + n_base, lane);
      reinterpret_cast<float4*>(sh_c + row * FAT_TILE_N)[lane] = values;
    }
    __syncthreads();
    for (int i = t; i < rows * FAT_TILE_N; i += FAT_THREADS) {
      int row = i / FAT_TILE_N;
      int col_out = i % FAT_TILE_N;
      int local_row = m_base + mb * 16 + row;
      float value = sh_c[i];
      if constexpr (scatter) {
        int route = route_start + local_row;
        int64_t destination = token_idx[route];
        value *= __half2float(route_weight[route]);
        atomicAdd(out + destination * out_stride + n_base + col_out, value);
      } else {
        int route = route_start + local_row;
        out[route * out_stride + out_column + n_base + col_out] = value;
      }
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" __global__ void glm53_exl3_fat_gather(
    const half* __restrict__ hidden,
    half* __restrict__ transformed,
    const half** __restrict__ suh_table,
    const int64_t* __restrict__ expert_count,
    const int4* __restrict__ descriptors,
    const int64_t* __restrict__ token_sorted,
    int hidden_size,
    int max_rows,
    int num_experts) {
  int warp = threadIdx.x / 32;
  int row_tiles = (max_rows + 7) / 8;
  for (int task = blockIdx.x; task < FAT_SLOTS * row_tiles;
       task += gridDim.x) {
    int slot = task / row_tiles;
    int row_tile = task % row_tiles;
    int4 descriptor;
    if (!fat_descriptor(expert_count, descriptors, num_experts, slot,
                        descriptor)) continue;
    int route_row = row_tile * 8 + warp;
    if (route_row >= descriptor.z) continue;
    int route = descriptor.y + route_row;
    int64_t token = token_sorted[route];
    const half* suh = suh_table[descriptor.x];
    for (int chunk = 0; chunk < hidden_size / 128; ++chunk) {
      fat_gather_had_128(
          hidden + token * hidden_size + chunk * 128,
          transformed + route * hidden_size + chunk * 128,
          suh + chunk * 128);
    }
  }
}

extern "C" __global__ __launch_bounds__(FAT_THREADS)
void glm53_exl3_fat_gemm_gate_up(
    const half* __restrict__ transformed,
    const uint16_t** __restrict__ trellis_table,
    float* __restrict__ gate_up,
    const half** __restrict__ svh_table,
    const int64_t* __restrict__ expert_count,
    const int4* __restrict__ descriptors,
    int size_k,
    int size_n,
    int out_stride,
    int out_column,
    int max_rows,
    int num_experts) {
  int m_tiles = (max_rows + FAT_TILE_M - 1) / FAT_TILE_M;
  int n_tiles = size_n / FAT_TILE_N;
  int tasks_per_slot = m_tiles * n_tiles;
  for (int task = blockIdx.x; task < FAT_SLOTS * tasks_per_slot;
       task += gridDim.x) {
    int slot = task / tasks_per_slot;
    int tile = task % tasks_per_slot;
    int m_tile = tile / n_tiles;
    int n_tile = tile % n_tiles;
    int4 descriptor;
    if (!fat_descriptor(expert_count, descriptors, num_experts, slot,
                        descriptor) || m_tile * FAT_TILE_M >= descriptor.z) {
      continue;
    }
    fat_gemm_body<false>(transformed, trellis_table[descriptor.x], gate_up,
                         svh_table[descriptor.x], nullptr, nullptr,
                         descriptor.y, descriptor.z, size_k, size_n,
                         out_stride, out_column, m_tile, n_tile);
  }
}

extern "C" __global__ void glm53_exl3_fat_activate_down_had(
    const float* __restrict__ gate_up,
    half* __restrict__ transformed,
    const half** __restrict__ down_suh_table,
    const int64_t* __restrict__ expert_count,
    const int4* __restrict__ descriptors,
    int intermediate_size,
    int gate_up_stride,
    float activation_limit,
    int max_rows,
    int num_experts) {
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  int row_tiles = (max_rows + 7) / 8;
  for (int task = blockIdx.x; task < FAT_SLOTS * row_tiles;
       task += gridDim.x) {
    int slot = task / row_tiles;
    int row_tile = task % row_tiles;
    int4 descriptor;
    if (!fat_descriptor(expert_count, descriptors, num_experts, slot,
                        descriptor)) continue;
    int route_row = row_tile * 8 + warp;
    if (route_row >= descriptor.z) continue;
    int route = descriptor.y + route_row;
    const float* row = gate_up + route * gate_up_stride;
    const half* scale = down_suh_table[descriptor.x];
    for (int chunk = 0; chunk < intermediate_size / 128; ++chunk) {
      float4 values;
      float* value = reinterpret_cast<float*>(&values);
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        int channel = chunk * 128 + lane * 4 + part;
        float gate = fminf(row[channel], activation_limit);
        float up = fminf(fmaxf(row[intermediate_size + channel],
                               -activation_limit), activation_limit);
        value[part] = gate / (1.0f + expf(-gate)) * up;
      }
      fat_had_pre_ff_128(values, scale + chunk * 128, lane);
      half* destination = transformed + route * intermediate_size +
                          chunk * 128 + lane * 4;
      destination[0] = __float2half(values.x);
      destination[1] = __float2half(values.y);
      destination[2] = __float2half(values.z);
      destination[3] = __float2half(values.w);
    }
  }
}

extern "C" __global__ __launch_bounds__(FAT_THREADS)
void glm53_exl3_fat_gemm_down_scatter(
    const half* __restrict__ transformed,
    const uint16_t** __restrict__ trellis_table,
    float* __restrict__ output,
    const half** __restrict__ svh_table,
    const int64_t* __restrict__ expert_count,
    const int4* __restrict__ descriptors,
    const int64_t* __restrict__ token_sorted,
    const half* __restrict__ weight_sorted,
    int size_k,
    int size_n,
    int max_rows,
    int num_experts) {
  int m_tiles = (max_rows + FAT_TILE_M - 1) / FAT_TILE_M;
  int n_tiles = size_n / FAT_TILE_N;
  int tasks_per_slot = m_tiles * n_tiles;
  for (int task = blockIdx.x; task < FAT_SLOTS * tasks_per_slot;
       task += gridDim.x) {
    int slot = task / tasks_per_slot;
    int tile = task % tasks_per_slot;
    int m_tile = tile / n_tiles;
    int n_tile = tile % n_tiles;
    int4 descriptor;
    if (!fat_descriptor(expert_count, descriptors, num_experts, slot,
                        descriptor) || m_tile * FAT_TILE_M >= descriptor.z) {
      continue;
    }
    fat_gemm_body<true>(transformed, trellis_table[descriptor.x], output,
                        svh_table[descriptor.x], token_sorted, weight_sorted,
                        descriptor.y, descriptor.z, size_k, size_n,
                        size_n, 0, m_tile, n_tile);
  }
}
