// SPDX-License-Identifier: AGPL-3.0-only

// nvcc -O3 --fmad=false -arch=sm_121a scripts/dev/bench_glm_kda_batch.cu -o /tmp/bench-glm-kda-batch
// Standalone state-indexed decode gate. No model/collectives; GPU budget64MiB.
// See glm_kda_batch_plan.md. All state/output comparisons are bitwise.
// Indexed exports compile from production; reference recurrence/serial conv
// remain frozen independently, and token-parallel conv remains unchanged.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../../kernels/gb10/common/causal_conv1d.cu"
#include "../../kernels/gb10/common/kda.cu"
#include "glm_kda_legacy_reference.cuh"

#define CHECK(call) do { const auto error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(1); } } while (0)
static void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(2); }
}
static size_t device_bytes = 0;
template<class T> struct Buffer {
    static constexpr size_t guard = 128 / sizeof(T);
    T* allocation;
    T* ptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) {
        device_bytes += (count + 2 * guard) * sizeof(T);
        require(device_bytes <= 64ULL * 1024 * 1024, "device allocation budget exceeded");
        CHECK(cudaMalloc(&allocation, (count + 2 * guard) * sizeof(T)));
        CHECK(cudaMemset(allocation, 0xa5, (count + 2 * guard) * sizeof(T)));
        ptr = allocation + guard;
    }
    ~Buffer() { cudaFree(allocation); }
    std::vector<T> read() const {
        std::vector<T> host(count);
        CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
    void guards() const {
        unsigned char a[128], b[128];
        CHECK(cudaMemcpy(a, allocation, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b, ptr + count, 128, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 128; ++i)
            require(a[i] == 0xa5 && b[i] == 0xa5, "outer allocation guard overwritten");
    }
};
__device__ float random_value(size_t i, unsigned seed) {
    unsigned x = unsigned(i) ^ seed;
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return float(x & 65535) / 65535.0f - 0.5f;
}
__global__ void initialize_bf16(__nv_bfloat16* dst, size_t count, unsigned seed,
                               float amplitude = 1.0f) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2bfloat16(random_value(i, seed) * amplitude);
}
__global__ void initialize_state(float* dst, size_t count, size_t live, size_t stride,
                                 unsigned seed, bool zero, float amplitude = 0.02f) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = i % stride < live
        ? (zero ? 0.0f : random_value(i, seed) * amplitude) : __uint_as_float(0x7fc12345);
}
template<class T> static void exact(const std::vector<T>& a, const std::vector<T>& b,
                                    const char* label) {
    require(a.size() == b.size(), "comparison size mismatch");
    size_t mismatches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(T))) {
            unsigned bits_a = 0, bits_b = 0;
            std::memcpy(&bits_a, &a[i], sizeof(T));
            std::memcpy(&bits_b, &b[i], sizeof(T));
            if (mismatches < 8)
                std::fprintf(stderr, "FAIL: %s element=%zu reference=%.12g candidate=%.12g "
                             "bits_reference=0x%08x bits_candidate=0x%08x\n", label, i,
                             double(float(a[i])), double(float(b[i])), bits_a, bits_b);
            ++mismatches;
        }
    }
    if (mismatches) {
        std::fprintf(stderr, "FAIL: %s differing_elements=%zu/%zu\n", label, mismatches, a.size());
        std::exit(2);
    }
}
static bool valid_rows(const std::vector<int>& ids, unsigned slots) {
    if (ids.empty() || ids.size() > 4) return false;
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] >= 0 && unsigned(ids[i]) < slots)
            for (size_t j = 0; j < i; ++j) if (ids[i] == ids[j]) return false;
    return true;
}
static void state_isolation(const std::vector<float>& before, const std::vector<float>& after,
                            const std::vector<int>& ids, unsigned slots, size_t live, size_t stride) {
    for (unsigned slot = 0; slot < slots; ++slot) {
        const bool active = std::find(ids.begin(), ids.end(), int(slot)) != ids.end();
        for (size_t j = 0; j < stride; ++j) {
            const size_t i = slot * stride + j;
            if (!active || j >= live)
                require(std::memcmp(&before[i], &after[i], 4) == 0, "inactive slot/padding modified");
            if (j < live) require(std::isfinite(after[i]), "nonfinite live FP32 state");
        }
    }
}
static void output_isolation(const std::vector<__nv_bfloat16>& out,
                             const std::vector<int>& ids, unsigned slots, size_t stride) {
    for (unsigned row = 0; row < 4; ++row) {
        const bool active = row < ids.size() && ids[row] >= 0 && unsigned(ids[row]) < slots;
        for (size_t j = 0; j < stride; ++j) {
            const auto value = out[row * stride + j];
            if (active) require(std::isfinite(__bfloat162float(value)), "nonfinite output");
            else {
                unsigned short bits; std::memcpy(&bits, &value, 2);
                require(bits == 0x5a5a, "inactive/invalid output row modified");
            }
        }
    }
}

int main(int argc, char** argv) {
    const bool timing = argc == 2 && std::strcmp(argv[1], "--timing") == 0;
    require(argc == 1 || timing, "usage: bench-glm-kda-batch [--timing]");
    constexpr unsigned heads = 32, dim = 128, p = heads * dim, channels = 3 * p;
    constexpr unsigned slots = 8, conv_width = 4;
    constexpr size_t h_live = heads * dim * dim, h_stride = h_live + 64;
    constexpr size_t c_live = channels * conv_width, c_stride = c_live + 64;
    constexpr float lower_bound = -5.0f;
    require(!valid_rows({}, slots) && !valid_rows({2, 2}, slots)
            && !valid_rows({0, 1, 2, 3, 4}, slots) && valid_rows({7, 0, 5, 2}, slots),
            "host duplicate/width safety gate regression");
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Buffer<float> old_h(slots * h_stride), new_h(old_h.count);
    Buffer<float> old_c(slots * c_stride), new_c(old_c.count), serial_c(old_c.count);
    Buffer<__nv_bfloat16> input(4 * channels), weight(c_live), gate(4 * p), beta(4 * heads);
    Buffer<float> a_log(heads), dt_bias(p);
    Buffer<__nv_bfloat16> old_conv(input.count), new_conv(input.count), serial_conv(input.count);
    Buffer<__nv_bfloat16> old_out(4 * p), new_out(old_out.count);
    Buffer<int> indices(4);
    auto init_state = [&](Buffer<float>& dst, size_t live, size_t stride, unsigned seed,
                          float amplitude = 0.02f) {
        initialize_state<<<(dst.count + 255) / 256, 256, 0, stream>>>(dst.ptr, dst.count,
            live, stride, seed, false, amplitude); CHECK(cudaGetLastError());
    };
    init_state(old_h, h_live, h_stride, 17); init_state(new_h, h_live, h_stride, 17);
    for (auto* c : {&old_c, &new_c, &serial_c}) init_state(*c, c_live, c_stride, 37);
    init_state(a_log, heads, heads, 57); init_state(dt_bias, p, p, 67);
    initialize_bf16<<<(weight.count + 255) / 256, 256, 0, stream>>>(weight.ptr, weight.count, 77);
    CHECK(cudaGetLastError()); CHECK(cudaStreamSynchronize(stream));
    const auto original_weight = weight.read();
    const auto original_a = a_log.read(), original_bias = dt_bias.read();
    std::printf("device_bytes=%zu slots=%u heads=%u D=%u state_FP32\n", device_bytes, slots, heads, dim);
    const auto candidate = [&](unsigned rows) {
        glm_kda_conv_indexed<<<dim3((channels + 255) / 256, rows), 256, 0, stream>>>(
            new_c.ptr, input.ptr, weight.ptr, nullptr, new_conv.ptr, channels, conv_width,
            rows, channels, channels, indices.ptr, slots, c_stride);
        CHECK(cudaGetLastError());
        glm_kda_recurrent_indexed<<<dim3(heads, rows), 128, 0, stream>>>(
            new_conv.ptr, gate.ptr, beta.ptr, a_log.ptr, dt_bias.ptr, new_h.ptr, new_out.ptr,
            rows, heads, dim, lower_bound, indices.ptr, slots, h_stride);
        CHECK(cudaGetLastError());
    };
    if (timing) {
        // Separate submission timing, not a correctness substitute. Reset all
        // state outside each interval; no extra GPU allocation beyond events.
        const std::vector<int> table = {7, 2, 0, 5};
        CHECK(cudaMemcpyAsync(indices.ptr, table.data(), 16, cudaMemcpyHostToDevice, stream));
        for (auto* x : {&input, &gate, &beta})
            initialize_bf16<<<(x->count + 255) / 256, 256, 0, stream>>>(x->ptr, x->count, 123);
        CHECK(cudaGetLastError());
        cudaEvent_t begin, end;
        CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
        for (unsigned rows : {2U, 3U, 4U}) {
            const auto reference = [&]() {
                for (unsigned row = 0; row < rows; ++row) {
                    const int slot = table[row];
                    causal_conv1d_update_prefill_tp<<<dim3(channels / 32, 1), dim3(32, 8), 0, stream>>>(
                        old_c.ptr + slot * c_stride, input.ptr + row * channels, weight.ptr, nullptr,
                        old_conv.ptr + row * channels, channels, conv_width, 1, channels, channels);
                    frozen_kda_recurrent_bf16<<<heads, 128, 0, stream>>>(old_conv.ptr + row * channels,
                        gate.ptr + row * p, beta.ptr + row * heads, a_log.ptr, dt_bias.ptr,
                        old_h.ptr + slot * h_stride, old_out.ptr + row * p, 1, heads, dim, lower_bound);
                }
                CHECK(cudaGetLastError());
            };
            for (unsigned i = 0; i < 10; ++i) { reference(); candidate(rows); }
            float medians[2];
            std::vector<float> samples[2];
            for (unsigned trial = 0; trial < 5; ++trial) {
                for (unsigned order = 0; order < 2; ++order) {
                    const unsigned variant = (trial + order) % 2;
                    init_state(old_h, h_live, h_stride, 17); init_state(new_h, h_live, h_stride, 17);
                    init_state(old_c, c_live, c_stride, 37); init_state(new_c, c_live, c_stride, 37);
                    CHECK(cudaEventRecord(begin, stream));
                    for (unsigned i = 0; i < 100; ++i) {
                        if (variant) candidate(rows); else reference();
                    }
                    CHECK(cudaEventRecord(end, stream)); CHECK(cudaEventSynchronize(end));
                    float ms; CHECK(cudaEventElapsedTime(&ms, begin, end));
                    samples[variant].push_back(ms * 10.0f); // 1000 us/ms / 100 iterations.
                }
            }
            for (unsigned variant = 0; variant < 2; ++variant) {
                std::sort(samples[variant].begin(), samples[variant].end());
                medians[variant] = samples[variant][2];
            }
            std::printf("TIMING rows=%u per_row_us=%.3f indexed_us=%.3f ratio=%.3f "
                        "CUDA_events_eager_submission median5x100 interleaved reset_excluded\n",
                        rows, medians[0], medians[1], medians[0] / medians[1]);
        }
        CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end));
        CHECK(cudaStreamDestroy(stream));
        std::printf("Timing only: run default correctness and memcheck separately.\n");
        return 0;
    }
    cudaGraph_t graphs[4]; cudaGraphExec_t executable[4];
    for (unsigned rows = 1; rows <= 4; ++rows) {
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        candidate(rows);
        CHECK(cudaStreamEndCapture(stream, &graphs[rows - 1]));
        CHECK(cudaGraphInstantiate(&executable[rows - 1], graphs[rows - 1], nullptr, nullptr, 0));
    }
    const std::vector<std::vector<int>> cases = {{0}, {6, 2}, {5, 1, 7}, {7, 2, 0, 5},
        {5, 7, 2, 0}, {7, 0, 5}, {7, 5}, {5}, {2, 5}, {2, 0, 7, 5}, {-1, 5, 99}, {7, -1}, {-1}};
    unsigned step = 0;
    for (unsigned round = 0; round < 7; ++round) for (unsigned scenario = 0; scenario < cases.size(); ++scenario, ++step) {
        const auto& ids = cases[scenario];
        const char* profile = round < 3 ? "standard" : round == 3 ? "zero-Q"
            : round == 4 ? "zero-K" : round == 5 ? "zero-QK" : "strong";
        std::printf("START step=%u rows=%zu mode=%s profile=%s\n", step, ids.size(),
                    round ? "graph" : "eager", profile);
        std::fflush(stdout);
        require(valid_rows(ids, slots), "duplicate live slots must never reach CUDA");
        if (round == 6 && scenario == 0) {
            for (auto* h : {&old_h, &new_h}) init_state(*h, h_live, h_stride, 137, 32.0f);
            for (auto* c : {&old_c, &new_c, &serial_c}) init_state(*c, c_live, c_stride, 157, 2.0f);
        }
        if (scenario == 8) {
            // Slot2 was absent throughout the drain. Reuse it for a fresh zero-history sequence.
            for (auto* h : {&old_h, &new_h})
                CHECK(cudaMemsetAsync(h->ptr + 2 * h_stride, 0, h_live * 4, stream));
            for (auto* c : {&old_c, &new_c, &serial_c})
                CHECK(cudaMemsetAsync(c->ptr + 2 * c_stride, 0, c_live * 4, stream));
        }
        for (auto* x : {&input, &gate, &beta}) {
            initialize_bf16<<<(x->count + 255) / 256, 256, 0, stream>>>(
                x->ptr, x->count, 97 + step * 101 + unsigned(x->count),
                round != 6 ? 1.0f : x == &gate ? 64.0f : x == &beta ? 32.0f : 16.0f);
            CHECK(cudaGetLastError());
        }
        if (round >= 3 && round <= 5) {
            const size_t first = round == 4 ? p : 0;
            const size_t count = round == 5 ? 2 * p : p;
            // Exact zero Q/K requires both current input AND all conv history
            // to be zero. V and recurrent H remain nonzero and independently checked.
            for (unsigned row = 0; row < 4; ++row)
                CHECK(cudaMemsetAsync(input.ptr + row * channels + first, 0, count * 2, stream));
            for (auto* c : {&old_c, &new_c, &serial_c})
                for (unsigned slot = 0; slot < slots; ++slot)
                    CHECK(cudaMemsetAsync(c->ptr + slot * c_stride + first * conv_width,
                                          0, count * conv_width * 4, stream));
        }
        std::vector<int> table(4, -1); std::copy(ids.begin(), ids.end(), table.begin());
        CHECK(cudaMemcpyAsync(indices.ptr, table.data(), table.size() * 4, cudaMemcpyHostToDevice, stream));
        for (auto* out : {&old_conv, &new_conv, &serial_conv, &old_out, &new_out})
            CHECK(cudaMemsetAsync(out->ptr, 0x5a, out->count * 2, stream));
        CHECK(cudaStreamSynchronize(stream));
        const auto before_h = old_h.read(), before_c = old_c.read();
        for (unsigned row = 0; row < ids.size(); ++row) {
            const int slot = ids[row];
            if (slot < 0 || unsigned(slot) >= slots) continue;
            causal_conv1d_update_prefill_tp<<<dim3(channels / 32, 1), dim3(32, 8), 0, stream>>>(
                old_c.ptr + slot * c_stride, input.ptr + row * channels, weight.ptr, nullptr,
                old_conv.ptr + row * channels, channels, conv_width, 1, channels, channels);
            frozen_causal_conv1d_update_prefill<<<channels / 256, 256, 0, stream>>>(
                serial_c.ptr + slot * c_stride, input.ptr + row * channels, weight.ptr, nullptr,
                serial_conv.ptr + row * channels, channels, conv_width, 1, channels, channels);
            frozen_kda_recurrent_bf16<<<heads, 128, 0, stream>>>(old_conv.ptr + row * channels,
                gate.ptr + row * p, beta.ptr + row * heads, a_log.ptr, dt_bias.ptr,
                old_h.ptr + slot * h_stride, old_out.ptr + row * p, 1, heads, dim, lower_bound);
            CHECK(cudaGetLastError());
        }
        if (round == 0) candidate(unsigned(ids.size()));
        else CHECK(cudaGraphLaunch(executable[ids.size() - 1], stream));
        CHECK(cudaStreamSynchronize(stream));
        const auto after_h = old_h.read(), after_c = old_c.read();
        exact(after_h, new_h.read(), "complete recurrent H");
        exact(after_c, new_c.read(), "complete convolution state");
        exact(after_c, serial_c.read(), "token-parallel/serial convolution state");
        const auto convolved = old_conv.read(), output = old_out.read();
        exact(convolved, new_conv.read(), "convolution BF16 output");
        exact(convolved, serial_conv.read(), "token-parallel/serial convolution output");
        exact(output, new_out.read(), "recurrent BF16 output");
        if (round >= 3 && round <= 5) {
            const size_t first = round == 4 ? p : 0;
            const size_t count = round == 5 ? 2 * p : p;
            for (unsigned row = 0; row < ids.size(); ++row) {
                if (ids[row] < 0 || unsigned(ids[row]) >= slots) continue;
                for (size_t j = 0; j < count; ++j)
                    require(float(convolved[row * channels + first + j]) == 0.0f,
                            "zero Q/K test precondition failed");
                if (round != 4)
                    for (unsigned j = 0; j < p; ++j)
                        require(float(output[row * p + j]) == 0.0f, "zero Q must produce zero output");
            }
        }
        state_isolation(before_h, after_h, ids, slots, h_live, h_stride);
        state_isolation(before_c, after_c, ids, slots, c_live, c_stride);
        output_isolation(convolved, ids, slots, channels); output_isolation(output, ids, slots, p);
        require(indices.read() == table, "device slot indices modified");
        old_h.guards(); new_h.guards(); old_c.guards(); new_c.guards(); serial_c.guards();
        input.guards(); gate.guards(); beta.guards(); weight.guards(); a_log.guards(); dt_bias.guards();
        old_conv.guards(); new_conv.guards(); serial_conv.guards(); old_out.guards(); new_out.guards(); indices.guards();
        std::printf("PASS step=%u rows=%zu mode=%s ids=[", step, ids.size(), round ? "graph" : "eager");
        for (int slot : ids) std::printf("%d,", slot);
        std::printf("] full_H/conv/output BITEXACT; inactive slots/guards intact\n");
    }
    exact(original_weight, weight.read(), "read-only convolution weights");
    exact(original_a, a_log.read(), "read-only a_log"); exact(original_bias, dt_bias.read(), "read-only dt_bias");
    for (unsigned i = 0; i < 4; ++i) {
        CHECK(cudaGraphExecDestroy(executable[i])); CHECK(cudaGraphDestroy(graphs[i]));
    }
    CHECK(cudaStreamDestroy(stream));
    std::printf("PASS all %u state-indexed KDA steps\n", step);
}
