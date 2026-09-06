// SPDX-License-Identifier: AGPL-3.0-only

// nvcc -O3 --fmad=false -arch=sm_121a scripts/dev/bench_glm_kda_temporal.cu -o /tmp/bench-glm-kda-temporal
// Repeat without --fmad=false for common-kernel consumers. See glm_kda_temporal_plan.md.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../../kernels/gb10/common/causal_conv1d.cu"
#include "../../kernels/gb10/common/kda.cu"
#include "glm_kda_legacy_reference.cuh"

#define CHECK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); \
} } while (0)
static void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(2); }
}
static size_t device_bytes = 0;
template<class T> struct Buffer {
    static constexpr size_t guard = 128 / sizeof(T);
    T *allocation, *ptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) {
        device_bytes += (n + 2 * guard) * sizeof(T);
        require(device_bytes < 16ULL * 1024 * 1024, "16MiB allocation ceiling exceeded");
        CHECK(cudaMalloc(&allocation, (n + 2 * guard) * sizeof(T)));
        CHECK(cudaMemset(allocation, 0xa5, (n + 2 * guard) * sizeof(T)));
        ptr = allocation + guard;
    }
    ~Buffer() { cudaFree(allocation); }
    void upload(const std::vector<T>& v, cudaStream_t stream) const {
        require(v.size() == count, "upload extent mismatch");
        CHECK(cudaMemcpyAsync(ptr, v.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    std::vector<T> read() const {
        std::vector<T> v(count);
        CHECK(cudaMemcpy(v.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
    void guards() const {
        unsigned char a[128], b[128];
        CHECK(cudaMemcpy(a, allocation, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b, ptr + count, 128, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 128; ++i)
            require(a[i] == 0xa5 && b[i] == 0xa5, "allocation canary overwritten");
    }
};
template<class T> static void exact(const std::vector<T>& a, const std::vector<T>& b,
                                    const char* label) {
    require(a.size() == b.size(), "comparison size mismatch");
    for (size_t i = 0; i < a.size(); ++i) if (std::memcmp(&a[i], &b[i], sizeof(T))) {
        unsigned av = 0, bv = 0;
        std::memcpy(&av, &a[i], sizeof(T)); std::memcpy(&bv, &b[i], sizeof(T));
        std::fprintf(stderr, "FAIL %s element=%zu reference=%.12g actual=%.12g bits=%08x/%08x\n",
                     label, i, double(float(a[i])), double(float(b[i])), av, bv);
        std::exit(2);
    }
}
static float value(size_t i, unsigned seed) {
    unsigned x = unsigned(i) ^ seed;
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return float(x & 65535) / 65535.0f - 0.5f;
}
static std::vector<float> floats(size_t n, unsigned seed, float amplitude) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = value(i, seed) * amplitude;
    return v;
}
static std::vector<__nv_bfloat16> bf16s(size_t n, unsigned seed, float amplitude) {
    std::vector<__nv_bfloat16> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = __float2bfloat16(value(i, seed) * amplitude);
    return v;
}
__global__ void temporal_pack(const __nv_bfloat16* strided, __nv_bfloat16* packed,
                               unsigned tokens, unsigned channels, unsigned stride) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(tokens) * channels) packed[i] = strided[(i / channels) * stride + i % channels];
}
static void output_shape(const std::vector<__nv_bfloat16>& v, unsigned tokens,
                          unsigned channels, unsigned stride) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (i / stride < tokens && i % stride < channels)
            require(std::isfinite(float(v[i])), "nonfinite live BF16 output");
        else {
            unsigned short bits; std::memcpy(&bits, &v[i], 2);
            require(bits == 0x5a5a, "unused row or stride padding overwritten");
        }
    }
}
int main() {
    constexpr unsigned heads = 32, dim = 128, p = heads * dim, channels = 3 * p;
    constexpr unsigned maximum_tokens = 17, width = 4, maximum_stride = channels + 32;
    constexpr size_t h_count = size_t(heads) * dim * dim, c_count = channels * width;
    cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Buffer<float> old_h(h_count), new_h(h_count), old_c(c_count), new_c(c_count);
    Buffer<float> a_log(heads), dt_bias(p), conv_bias(channels);
    Buffer<__nv_bfloat16> input(maximum_tokens * maximum_stride), weight(c_count);
    Buffer<__nv_bfloat16> gate(maximum_tokens * p), beta(maximum_tokens * heads);
    Buffer<__nv_bfloat16> old_conv(input.count), new_conv(input.count);
    Buffer<__nv_bfloat16> old_packed(maximum_tokens * channels), new_packed(old_packed.count);
    Buffer<__nv_bfloat16> old_out(maximum_tokens * p), new_out(old_out.count);
    std::printf("device_bytes=%zu temporal_single_history heads32 D128\n", device_bytes);
    unsigned cases = 0;
    for (unsigned tokens : {1U, 2U, 3U, 5U, 17U}) for (unsigned profile = 0; profile < 4; ++profile) {
        const unsigned input_stride = channels + (profile % 2 ? 16 : 0);
        const unsigned output_stride = channels + (profile % 2 ? 32 : 0);
        const auto bias = profile % 2 ? conv_bias.ptr : nullptr;
        const auto launch = [&](bool legacy) {
            if (legacy)
                frozen_causal_conv1d_update_prefill<<<channels / 256, 256, 0, stream>>>(
                    old_c.ptr, input.ptr, weight.ptr, bias, old_conv.ptr, channels, width,
                    tokens, input_stride, output_stride);
            else
                causal_conv1d_update_prefill<<<channels / 256, 256, 0, stream>>>(
                    new_c.ptr, input.ptr, weight.ptr, bias, new_conv.ptr, channels, width,
                    tokens, input_stride, output_stride);
            temporal_pack<<<(tokens * channels + 255) / 256, 256, 0, stream>>>(
                legacy ? old_conv.ptr : new_conv.ptr, legacy ? old_packed.ptr : new_packed.ptr,
                tokens, channels, output_stride);
            if (legacy)
                frozen_kda_recurrent_bf16<<<heads, 128, 0, stream>>>(old_packed.ptr, gate.ptr,
                    beta.ptr, a_log.ptr, dt_bias.ptr, old_h.ptr, old_out.ptr, tokens, heads, dim, -5.0f);
            else
                kda_recurrent_bf16<<<heads, 128, 0, stream>>>(new_packed.ptr, gate.ptr,
                    beta.ptr, a_log.ptr, dt_bias.ptr, new_h.ptr, new_out.ptr, tokens, heads, dim, -5.0f);
            CHECK(cudaGetLastError());
        };
        cudaGraph_t graph; cudaGraphExec_t executable;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(false);
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (unsigned replay = 0; replay < 3; ++replay, ++cases) {
            const unsigned seed = 137 + cases * 101;
            std::printf("START tokens=%u profile=%u mode=%s seed=%u\n", tokens, profile,
                        replay ? "graph" : "eager", seed);
            std::fflush(stdout);
            const bool strong = profile == 2, zero_history = profile == 1, zero_qk = profile == 3;
            auto h = floats(h_count, seed, zero_history ? 0.0f : strong ? 32.0f : 0.02f);
            auto c = floats(c_count, seed + 1, zero_history ? 0.0f : strong ? 2.0f : 0.02f);
            auto x = bf16s(input.count, seed + 2, strong ? 16.0f : 1.0f);
            auto w = bf16s(weight.count, seed + 3, 1.0f);
            auto g = bf16s(gate.count, seed + 4, strong ? 64.0f : 1.0f);
            auto b = bf16s(beta.count, seed + 5, strong ? 32.0f : 1.0f);
            auto a = floats(heads, seed + 6, 0.02f), dt = floats(p, seed + 7, 0.02f);
            auto cb = floats(channels, seed + 8, 0.02f);
            if (zero_qk) {
                for (unsigned ch = 0; ch < 2 * p; ++ch) {
                    cb[ch] = 0.0f;
                    for (unsigned k = 0; k < width; ++k) { c[ch * width + k] = 0; w[ch * width + k] = __float2bfloat16(0); }
                }
                for (unsigned t = 0; t < maximum_tokens; ++t)
                    for (unsigned ch = 0; ch < 2 * p; ++ch) x[t * input_stride + ch] = __float2bfloat16(0);
            }
            old_h.upload(h, stream); new_h.upload(h, stream); old_c.upload(c, stream); new_c.upload(c, stream);
            input.upload(x, stream); weight.upload(w, stream); gate.upload(g, stream); beta.upload(b, stream);
            a_log.upload(a, stream); dt_bias.upload(dt, stream); conv_bias.upload(cb, stream);
            for (auto* out : {&old_conv, &new_conv, &old_packed, &new_packed, &old_out, &new_out})
                CHECK(cudaMemsetAsync(out->ptr, 0x5a, out->count * 2, stream));
            launch(true);
            if (replay == 0) launch(false); else CHECK(cudaGraphLaunch(executable, stream));
            CHECK(cudaStreamSynchronize(stream));
            const auto final_h = old_h.read(), final_c = old_c.read();
            exact(final_h, new_h.read(), "complete temporal H"); exact(final_c, new_c.read(), "complete temporal conv state");
            const auto convolved = old_conv.read(), packed = old_packed.read(), output = old_out.read();
            exact(convolved, new_conv.read(), "all temporal conv outputs");
            exact(packed, new_packed.read(), "all packed temporal QKV");
            exact(output, new_out.read(), "all temporal recurrent outputs");
            for (float v : final_h) require(std::isfinite(v), "nonfinite final H");
            for (float v : final_c) require(std::isfinite(v), "nonfinite final conv state");
            output_shape(convolved, tokens, channels, output_stride);
            output_shape(packed, tokens, channels, channels); output_shape(output, tokens, p, p);
            if (zero_qk) for (unsigned t = 0; t < tokens; ++t) {
                for (unsigned j = 0; j < 2 * p; ++j) require(float(packed[t * channels + j]) == 0, "zero Q/K precondition failed");
                for (unsigned j = 0; j < p; ++j) require(float(output[t * p + j]) == 0, "zero Q must yield zero output");
            }
            exact(x, input.read(), "read-only input"); exact(w, weight.read(), "read-only conv weight");
            exact(g, gate.read(), "read-only gate"); exact(b, beta.read(), "read-only beta");
            exact(a, a_log.read(), "read-only a_log"); exact(dt, dt_bias.read(), "read-only dt_bias");
            exact(cb, conv_bias.read(), "read-only conv bias");
            old_h.guards(); new_h.guards(); old_c.guards(); new_c.guards();
            input.guards(); weight.guards(); gate.guards(); beta.guards(); a_log.guards(); dt_bias.guards(); conv_bias.guards();
            old_conv.guards(); new_conv.guards(); old_packed.guards(); new_packed.guards(); old_out.guards(); new_out.guards();
            std::printf("PASS tokens=%u profile=%u mode=%s seed=%u complete_H/conv/output BITEXACT\n",
                        tokens, profile, replay ? "graph" : "eager", seed);
        }
        CHECK(cudaGraphExecDestroy(executable)); CHECK(cudaGraphDestroy(graph));
    }
    CHECK(cudaStreamDestroy(stream));
    std::printf("PASS all %u frozen-scalar temporal cases\n", cases);
}
