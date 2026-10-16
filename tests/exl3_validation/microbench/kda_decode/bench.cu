// SPDX-License-Identifier: AGPL-3.0-only
// Microbenchmark: kda_fla_recurrent_decode vs schedule variants (bit-exact check).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "kda_fla_decode.cuh"

using namespace kda_fla;

// Variant: ROWS value rows per warp, WARPS warps per block, VEC = float4 state I/O.
template <int ROWS, int WARPS, bool VEC>
__global__ void __launch_bounds__(32 * WARPS) kda_var(
    const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ state, __nv_bfloat16* __restrict__ output,
    unsigned int heads) {
    using namespace kda_fla_decode;
    const unsigned int head = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int v0 = (blockIdx.x * WARPS + warp) * ROWS;
    const int k0 = lane * 4;
    const float scale = __int_as_float(0x3DB504F3);
    float* H = state + (size_t)head * D * D;
    float h[ROWS][4];
    if constexpr (VEC) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r4 = 0; r4 < ROWS / 4; ++r4) {
                const float4 t = *reinterpret_cast<const float4*>(H + (size_t)(k0 + j) * D + v0 + 4 * r4);
                h[4 * r4 + 0][j] = fadd(t.x, 0.0f);
                h[4 * r4 + 1][j] = fadd(t.y, 0.0f);
                h[4 * r4 + 2][j] = fadd(t.z, 0.0f);
                h[4 * r4 + 3][j] = fadd(t.w, 0.0f);
            }
    } else {
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int j = 0; j < 4; ++j) h[r][j] = fadd(H[(size_t)(k0 + j) * D + v0 + r], 0.0f);
    }
    const __nv_bfloat16* qp = qkv + (size_t)head * D;
    const __nv_bfloat16* kp = qkv + (size_t)(heads + head) * D;
    const __nv_bfloat16* vp = qkv + (size_t)(2 * heads + head) * D;
    float q[4], k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        q[j] = __bfloat162float(qp[k0 + j]);
        k[j] = __bfloat162float(kp[k0 + j]);
    }
    const float eps = __int_as_float(0x358637BD);
    const float qs = sqrt_approx_ftz(fadd(dot4_reduce(q, q), eps));
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = div_full(q[j], qs);
    const float ks = sqrt_approx_ftz(fadd(dot4_reduce(k, k), eps));
#pragma unroll
    for (int j = 0; j < 4; ++j) k[j] = div_full(k[j], ks);
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = fmul(scale, q[j]);
    const float neg_a = -tl_exp(a_log[head]);
    float decay[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gk = fadd(__bfloat162float(raw_gate[(size_t)head * D + k0 + j]), dt_bias[(size_t)head * D + k0 + j]);
        const float gate = div_full(-5.0f, fadd(tl_exp(ffma(neg_a, gk, 0.0f)), 1.0f));
        decay[j] = tl_exp(gate);
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int j = 0; j < 4; ++j) h[r][j] = fmul(h[r][j], decay[j]);
    const float beta = inductor_sigmoid(__bfloat162float(raw_beta[head]));
    float o[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float hk = dot4_reduce(k, h[r]);
        const float v = fmul(fsub(__bfloat162float(vp[v0 + r]), hk), beta);
#pragma unroll
        for (int j = 0; j < 4; ++j) h[r][j] = ffma(k[j], v, h[r][j]);
        o[r] = dot4_reduce(q, h[r]);
    }
    if (lane < ROWS) {
        float mine = o[0];
#pragma unroll
        for (int r = 1; r < ROWS; ++r)
            if (lane == r) mine = o[r];
        output[(size_t)head * D + v0 + lane] = to_bf16(mine);
    }
    if constexpr (VEC) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r4 = 0; r4 < ROWS / 4; ++r4)
                *reinterpret_cast<float4*>(H + (size_t)(k0 + j) * D + v0 + 4 * r4) =
                    make_float4(h[4 * r4][j], h[4 * r4 + 1][j], h[4 * r4 + 2][j], h[4 * r4 + 3][j]);
    } else {
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int j = 0; j < 4; ++j) H[(size_t)(k0 + j) * D + v0 + r] = h[r][j];
    }
}

static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static float frand() {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((rng >> 40) / float(1 << 24)) * 2.0f - 1.0f;
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef void (*KFn)(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, float*, __nv_bfloat16*, unsigned int);

struct Var { const char* name; KFn fn; int rows; int warps; };

int main(int argc, char** argv) {
    const unsigned heads = argc > 1 ? atoi(argv[1]) : 32;
    const int layers = 34, iters = 50;
    const int D = 128;
    const size_t sn = (size_t)heads * D * D;
    std::vector<__nv_bfloat16> hqkv(3 * heads * D), hgate(heads * D), hbeta(heads);
    std::vector<float> halog(heads), hdt(heads * D), hstate(sn);
    for (auto& x : hqkv) x = __float2bfloat16(frand() * 2.0f);
    for (auto& x : hgate) x = __float2bfloat16(frand() * 3.0f);
    for (auto& x : hbeta) x = __float2bfloat16(frand() * 2.0f);
    for (auto& x : halog) x = frand();
    for (auto& x : hdt) x = frand();
    for (auto& x : hstate) x = frand() * 0.3f;
    __nv_bfloat16 *qkv, *gate, *beta, *out;
    float *alog, *dt, *st0, *stl;
    CK(cudaMalloc(&qkv, hqkv.size() * 2)); CK(cudaMalloc(&gate, hgate.size() * 2)); CK(cudaMalloc(&beta, heads * 2));
    CK(cudaMalloc(&out, heads * D * 2)); CK(cudaMalloc(&alog, heads * 4)); CK(cudaMalloc(&dt, hdt.size() * 4));
    CK(cudaMalloc(&st0, sn * 4)); CK(cudaMalloc(&stl, sn * 4 * layers));
    CK(cudaMemcpy(qkv, hqkv.data(), hqkv.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gate, hgate.data(), hgate.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(beta, hbeta.data(), heads * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(alog, halog.data(), heads * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dt, hdt.data(), hdt.size() * 4, cudaMemcpyHostToDevice));
    for (int l = 0; l < layers; ++l) CK(cudaMemcpy(stl + sn * l, hstate.data(), sn * 4, cudaMemcpyHostToDevice));

    Var vars[] = {
        {"ref r8w4", nullptr, 8, 4},
        {"r8w4 vec", kda_var<8, 4, true>, 8, 4},
        {"r8w4 scl", kda_var<8, 4, false>, 8, 4},
        {"r4w4 vec", kda_var<4, 4, true>, 4, 4},
        {"r4w8 vec", kda_var<4, 8, true>, 4, 8},
        {"r4w2 vec", kda_var<4, 2, true>, 4, 2},
        {"r8w2 vec", kda_var<8, 2, true>, 8, 2},
        {"r8w1 vec", kda_var<8, 1, true>, 8, 1},
        {"r4w1 vec", kda_var<4, 1, true>, 4, 1},
    };
    std::vector<float> ref_state(sn), st(sn);
    std::vector<__nv_bfloat16> ref_out(heads * D), o(heads * D);
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (auto& v : vars) {
        auto launch = [&](float* s) {
            if (!v.fn) {
                kda_fla_recurrent_decode<<<dim3(D / 32, heads), 128>>>(qkv, gate, beta, alog, dt, s, out, heads);
            } else {
                v.fn<<<dim3(D / (v.rows * v.warps), heads), 32 * v.warps>>>(qkv, gate, beta, alog, dt, s, out, heads);
            }
        };
        // correctness: 3 chained steps from the same initial state
        CK(cudaMemcpy(st0, hstate.data(), sn * 4, cudaMemcpyHostToDevice));
        for (int i = 0; i < 3; ++i) launch(st0);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(st.data(), st0, sn * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o.data(), out, heads * D * 2, cudaMemcpyDeviceToHost));
        bool same = true;
        if (!v.fn) { ref_state = st; ref_out = o; }
        else same = !memcmp(st.data(), ref_state.data(), sn * 4) && !memcmp(o.data(), ref_out.data(), heads * D * 2);
        // timing: rotate over `layers` distinct state buffers (L2-cold like decode)
        for (int l = 0; l < layers; ++l) launch(stl + sn * l);
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i)
            for (int l = 0; l < layers; ++l) launch(stl + sn * l);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        printf("%-10s heads=%u  %7.2f us/call  %s\n", v.name, heads, ms * 1e3 / (iters * layers), v.fn ? (same ? "BIT-EXACT" : "MISMATCH") : "reference");
    }
    return 0;
}
