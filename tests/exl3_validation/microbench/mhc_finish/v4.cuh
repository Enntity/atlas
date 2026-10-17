// SPDX-License-Identifier: AGPL-3.0-only
// Variant of exl3_mhc_pre_norm_split: identical arithmetic, loads hoisted.
extern "C" __global__ void __launch_bounds__(256) exl3_mhc_pre_norm_split_v4(
    float* highway,
    const float* __restrict__ fn,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ normed,
    float* __restrict__ post,
    float* __restrict__ comb,
    unsigned int hidden, unsigned int iterations,
    float rms_eps, float norm_eps, float hc_eps, float post_multiplier,
    unsigned int splits, unsigned int rounded_splits
) {
    (void)fn;
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x;
    const unsigned int width = HC * hidden;
    float* residual = highway + static_cast<unsigned long long>(row) * width;
    __shared__ float scratch[2 * THREADS];
    __shared__ float mixes[MIX];
    __shared__ float pre[HC];
    __shared__ float inverse;

    if (rounded_splits == 0 || rounded_splits > 64 || rounded_splits > width / 64) return;
    // Each thread owns GROUPS float4 feature groups (hidden <= 4096, a multiple
    // of 128, host-checked). Their residual and norm-weight loads are issued
    // here so their latency overlaps the fold, gates and Sinkhorn below.
    constexpr unsigned int GROUPS = 4096 / (4 * THREADS);
    float4 streams[GROUPS][HC];
    uint2 weights[GROUPS];
#pragma unroll
    for (unsigned int g = 0; g < GROUPS; ++g) {
        const unsigned int feature = 4u * (lane + g * THREADS);
        if (feature >= hidden) continue;
#pragma unroll
        for (unsigned int stream = 0; stream < HC; ++stream)
            streams[g][stream] = *reinterpret_cast<const float4*>(residual + stream * hidden + feature);
        weights[g] = *reinterpret_cast<const uint2*>(norm_weight + feature);
    }
    __shared__ float totals[MIX + 1];
    if (lane <= MIX) {
        const float* partials = reinterpret_cast<const float*>(normed + static_cast<unsigned long long>(row) * hidden);
        float total = 0.0f;
        for (unsigned int split = 0; split < splits; ++split)
            total = __fadd_rn(total, partials[lane * splits + split]);
        totals[lane] = total;
    }
    __syncthreads();
    if (lane == 0) inverse = rsqrtf(__fadd_rn(totals[MIX] / static_cast<float>(width), rms_eps));
    __syncthreads();
    if (lane < MIX) mixes[lane] = __fmul_rn(totals[lane], inverse);
    __syncthreads();
    if (lane == 0) {
        for (unsigned int i = 0; i < HC; ++i) {
            pre[i] = __fadd_rn(sigmoid(__fmaf_rn(mixes[i], scale[0], base[i])), hc_eps);
            post[static_cast<unsigned long long>(row) * HC + i] = __fmul_rn(
                sigmoid(__fmaf_rn(mixes[HC + i], scale[1], base[HC + i])), post_multiplier);
        }
    }
    __syncthreads();
    __shared__ float sink[HC * HC];
    if (lane >= 32 && lane < 32 + HC * HC) {
        const unsigned int at = lane - 32, i = at / HC, j = at % HC;
        float value = __fmaf_rn(mixes[2 * HC + at], scale[2], base[2 * HC + at]);
        sink[at] = value;
        __syncwarp(0xffffu);
        const float* r = sink + i * HC;
        const float maximum = fmaxf(fmaxf(r[0], r[1]), fmaxf(r[2], r[3]));
        value = expf(value - maximum);
        __syncwarp(0xffffu);
        sink[at] = value;
        __syncwarp(0xffffu);
        value = __fadd_rn(__fdiv_rn(value, sum4(r[0], r[1], r[2], r[3])), hc_eps);
        for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
            if (iteration != 0) {
                __syncwarp(0xffffu);
                sink[at] = value;
                __syncwarp(0xffffu);
                value = __fdiv_rn(value, __fadd_rn(sum4(r[0], r[1], r[2], r[3]), hc_eps));
            }
            __syncwarp(0xffffu);
            sink[at] = value;
            __syncwarp(0xffffu);
            value = __fdiv_rn(value, __fadd_rn(sum4(sink[j], sink[HC + j], sink[2 * HC + j], sink[3 * HC + j]), hc_eps));
        }
        comb[static_cast<unsigned long long>(row) * HC * HC + at] = value;
    }
    __shared__ float collapsed_values[4096];
#pragma unroll
    for (unsigned int g = 0; g < GROUPS; ++g) {
        const unsigned int feature = 4u * (lane + g * THREADS);
        if (feature >= hidden) continue;
        float collapsed[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (unsigned int stream = 0; stream < HC; ++stream) {
            float4& s = streams[g][stream];
            s.x = bf16(s.x); s.y = bf16(s.y); s.z = bf16(s.z); s.w = bf16(s.w);
            collapsed[0] = __fmaf_rn(pre[stream], s.x, collapsed[0]);
            collapsed[1] = __fmaf_rn(pre[stream], s.y, collapsed[1]);
            collapsed[2] = __fmaf_rn(pre[stream], s.z, collapsed[2]);
            collapsed[3] = __fmaf_rn(pre[stream], s.w, collapsed[3]);
        }
#pragma unroll
        for (unsigned int c = 0; c < 4; ++c) collapsed_values[feature + c] = collapsed[c];
    }
    __syncthreads();
    float square_slots[16] = {};
    if (lane < 64) {
#pragma unroll
        for (unsigned int block = 0; block < 4096; block += 1024) {
            if (block >= hidden) break;
#pragma unroll
            for (unsigned int slot = 0; slot < 16; ++slot) {
                const unsigned int feature = block + (slot / 8) * 512 + lane * 8 + slot % 8;
                if (feature >= hidden) continue;
                const float collapsed = collapsed_values[feature];
                square_slots[slot] = __fmaf_rn(collapsed, collapsed, square_slots[slot]);
            }
        }
    }
    float square = 0.0f;
#pragma unroll
    for (unsigned int slot = 0; slot < 16; ++slot)
        square = __fadd_rn(square, square_slots[(slot & 1) * 8 + slot / 2]);
    scratch[lane] = square;
    __syncthreads();
    float collapsed_sum = 0.0f;
    if (lane < 64) {
        collapsed_sum = __fadd_rn(square, scratch[lane ^ 32]);
        for (unsigned int offset = 16; offset > 0; offset >>= 1)
            collapsed_sum = __fadd_rn(collapsed_sum, __shfl_xor_sync(0xffffffffu, collapsed_sum, offset));
    }
    if (lane == 0) inverse = rsqrtf(__fadd_rn(collapsed_sum / static_cast<float>(hidden), norm_eps));
    __syncthreads();
    // normed = bf16(bf16(collapsed) * inverse * weight); bf16(collapsed) is the
    // value the original pass stored to and re-read from `normed`.
#pragma unroll
    for (unsigned int g = 0; g < GROUPS; ++g) {
        const unsigned int feature = 4u * (lane + g * THREADS);
        if (feature >= hidden) continue;
        const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&weights[g]);
        __align__(8) __nv_bfloat16 out[4];
#pragma unroll
        for (unsigned int c = 0; c < 4; ++c)
            out[c] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf16(collapsed_values[feature + c]), inverse),
                                                   __bfloat162float(w[c])));
        *reinterpret_cast<uint2*>(normed + static_cast<unsigned long long>(row) * hidden + feature) =
            *reinterpret_cast<const uint2*>(out);
    }
}
