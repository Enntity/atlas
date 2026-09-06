// SPDX-License-Identifier: AGPL-3.0-only
// Synthetic EP invariants only; no weights, NCCL, model, or performance claim.
// nvcc -O3 --fmad=false -arch=sm_121a scripts/dev/bench_glm_c4_ep_reduce.cu -o /tmp/bench-glm-c4-ep
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include "../../kernels/gb10/common/moe_permute.cu"
#include "../../kernels/gb10/common/bf16_add.cu"
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)
static void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); std::exit(2); }
}
static size_t allocated = 0;
template<class T> struct Buffer {
    T *base, *ptr; size_t n;
    explicit Buffer(size_t count) : n(count) {
        size_t bytes = n * sizeof(T) + 256;
        allocated += bytes;
        require(allocated < 32ULL * 1024 * 1024, "32MiB allocation bound");
        CHECK(cudaMalloc(&base, bytes)); CHECK(cudaMemset(base, 0xa5, bytes));
        ptr = base + 128 / sizeof(T);
    }
    ~Buffer() { cudaFree(base); }
    void put(const std::vector<T>& x) {
        require(x.size() == n, "upload length");
        CHECK(cudaMemcpy(ptr, x.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> x(n);
        CHECK(cudaMemcpy(x.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost)); return x;
    }
    void guards() const {
        unsigned char a[128], b[128];
        CHECK(cudaMemcpy(a, base, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b, ptr + n, 128, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 128; ++i) require(a[i] == 0xa5 && b[i] == 0xa5, "guard overwrite");
    }
};
static float f(__nv_bfloat16 x) { return __bfloat162float(x); }
static __nv_bfloat16 b(float x) { return __float2bfloat16(x); }
static void equal(const std::vector<__nv_bfloat16>& got,
                  const std::vector<__nv_bfloat16>& want, const char* label) {
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(f(got[i])) || std::memcmp(&got[i], &want[i], 2)) {
            std::fprintf(stderr, "%s element=%zu got=%g want=%g\n", label, i, f(got[i]), f(want[i]));
            std::exit(2);
        }
    }
}
int main() {
    constexpr unsigned H = 4096, K = 8, R = 4, E = R * K;
    Buffer<__nv_bfloat16> experts0(E * H), experts1(E * H), out0(R * H), out1(R * H), shared(R * H);
    Buffer<int> ids(E), map0(E), map1(E);
    Buffer<float> weights(E);
    const __nv_bfloat16 poison = b(NAN), sentinel = b(-101.0f);
    unsigned cases = 0;
    const std::vector<std::vector<int>> orders = {{0,1,2,3}, {3,1,0,2}, {3,0,2}, {2,0}, {1}};
    for (const auto& order : orders) for (int pattern = 0; pattern < 4; ++pattern)
    for (int invalid_remote = 0; invalid_remote < 2; ++invalid_remote) {
        unsigned rows = unsigned(order.size()), routes = rows * K;
        std::vector<int> eid(E, 0), p0(E, 0), p1(E, 0), perm(routes);
        std::vector<float> w(E, 0);
        std::vector<__nv_bfloat16> x0(E * H, poison), x1(E * H, poison), sh(R * H, b(0));
        std::vector<__nv_bfloat16> ref0(R * H, sentinel), ref1(R * H, sentinel);
        std::iota(perm.begin(), perm.end(), 0);
        std::reverse(perm.begin(), perm.end()); // deliberately not identity token_to_perm
        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned k = 0; k < K; ++k) {
                unsigned s = row * K + k;
                // Concentrated pattern: all identities select the same 8 experts.
                eid[s] = pattern == 0 ? int(k) : pattern == 1 ? int(280 + k)
                    : pattern == 2 ? int(140 + k) : int((k % 2) * 144 + k / 2);
                bool rank0 = eid[s] < 144;
                p0[s] = invalid_remote && !rank0 ? 0x7fffffff : perm[s];
                p1[s] = invalid_remote && rank0 ? 0x7fffffff : perm[s];
                w[s] = float(k + 1) / 16.0f; // exact dyadic, non-normalized weights
                auto& x = rank0 ? x0 : x1;
                for (unsigned c = 0; c < H; ++c) {
                    float value = float((order[row] + 1) * 8 + int(k) - 4) / 16.0f;
                    value *= c % 3 == 0 ? -1.0f : 1.0f;
                    x[size_t(perm[s]) * H + c] = b(value);
                }
            }
            for (unsigned c = 0; c < H; ++c) {
                float a0 = 0, a1 = 0;
                for (unsigned k = 0; k < K; ++k) {
                    unsigned s = row * K + k;
                    if (eid[s] < 144) a0 += w[s] * f(x0[size_t(perm[s]) * H + c]);
                    else a1 += w[s] * f(x1[size_t(perm[s]) * H + c]);
                }
                ref0[row * H + c] = b(a0); ref1[row * H + c] = b(a1);
                sh[row * H + c] = b(float(order[row] + 1) / 4.0f);
            }
        }
        experts0.put(x0); experts1.put(x1); ids.put(eid); map0.put(p0); map1.put(p1); weights.put(w); shared.put(sh);
        out0.put(std::vector<__nv_bfloat16>(R * H, sentinel));
        out1.put(std::vector<__nv_bfloat16>(R * H, sentinel));
        moe_unpermute_reduce_indexed_ep<<<rows,256>>>(experts0.ptr,out0.ptr,map0.ptr,ids.ptr,weights.ptr,H,rows,K,0,144);
        CHECK(cudaGetLastError());
        moe_unpermute_reduce_indexed_ep<<<rows,256>>>(experts1.ptr,out1.ptr,map1.ptr,ids.ptr,weights.ptr,H,rows,K,144,288);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        equal(out0.get(), ref0, "rank0"); equal(out1.get(), ref1, "rank1");
        bf16_add_inplace<<<(rows * H + 255) / 256,256>>>(out0.ptr,out1.ptr,rows * H);
        CHECK(cudaGetLastError());
        // GLM has no shared gate; normed input is unused with null gate weight.
        moe_batched_blend<<<rows,256>>>(out0.ptr,shared.ptr,shared.ptr,nullptr,H,rows);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        for (unsigned i = 0; i < rows * H; ++i) ref0[i] = b(f(b(f(ref0[i]) + f(ref1[i]))) + f(sh[i]));
        equal(out0.get(), ref0, "EP sum plus shared exactly once");
        experts0.guards(); experts1.guards(); out0.guards(); out1.guards(); shared.guards();
        ids.guards(); map0.guards(); map1.guards(); weights.guards(); ++cases;
    }
    std::printf("PASS cases=%u device_bytes=%zu rows=4,3,2,1 remoteNaN+invalidmap CPUoracle exact shared_once\n", cases, allocated);
}
