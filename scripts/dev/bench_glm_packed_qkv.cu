// SPDX-License-Identifier: AGPL-3.0-only
// M5-only standalone fixture; requires separately reviewed output-multiplier template.
// No model/conv/recurrence, production export, native execution or speed claim.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "../../kernels/gb10/common/w4a16_gemv.cu"
#include "../../kernels/gb10/common/kda.cu"

extern "C" __global__ __launch_bounds__(256, 5) void fixture_qkv_packed5(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ qw, const unsigned char* __restrict__ qs, const float q2,
    const unsigned char* __restrict__ kw, const unsigned char* __restrict__ ks, const float k2,
    const unsigned char* __restrict__ vw, const unsigned char* __restrict__ vs, const float v2,
    __nv_bfloat16* __restrict__ C, unsigned M, unsigned N, unsigned K
) {
    const unsigned p = blockIdx.z;
    const auto* w = p == 0 ? qw : (p == 1 ? kw : vw);
    const auto* s = p == 0 ? qs : (p == 1 ? ks : vs);
    const float scale2 = p == 0 ? q2 : (p == 1 ? k2 : v2);
    w4a16_gemv_batchm_impl<5, 3>(A, w, s, scale2, C + size_t(p) * N, M, N, K);
}

#define CHECK(call) do { const auto status = (call); if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    std::exit(1); } } while (0)
static constexpr size_t cap = 32 * 1024 * 1024;
// Fixed host readback scratch and two guard arrays; no full weight-readback clone.
static constexpr size_t host_fixed = 32768 + 256;
static size_t device_live = 0, device_peak = 0, host_live = host_fixed, host_peak = host_fixed;
static unsigned char scratch[32768], guard_before[128], guard_after[128];
static void require(bool ok, const char* why, int code = 2) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(code); }
}
static size_t add(size_t a, size_t b) {
    require(a <= std::numeric_limits<size_t>::max() - b, "size addition overflow", 4);
    return a + b;
}
static size_t mul(size_t a, size_t b) {
    require(!b || a <= std::numeric_limits<size_t>::max() / b, "size product overflow", 4);
    return a * b;
}
static void charge(size_t bytes, size_t& live, size_t& peak) {
    require(live <= cap && bytes <= cap - live, "32MiB explicit payload budget", 4);
    live += bytes; peak = std::max(live, peak);
}
template<class T> struct Host {
    T* ptr; size_t count, bytes;
    explicit Host(size_t n) : count(n), bytes(mul(n, sizeof(T))) {
        charge(bytes, host_live, host_peak);
        ptr = static_cast<T*>(std::malloc(bytes)); require(ptr != nullptr, "host allocation", 4);
    }
    ~Host() { std::free(ptr); host_live -= bytes; }
    Host(const Host&) = delete; Host& operator=(const Host&) = delete;
};
template<class T> struct Device {
    T *allocation, *ptr; size_t count, payload, bytes;
    explicit Device(size_t n) : count(n), payload(mul(n, sizeof(T))), bytes(add(payload, 256)) {
        charge(bytes, device_live, device_peak);
        CHECK(cudaMalloc(&allocation, bytes));
        ptr = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(allocation) + 128);
        CHECK(cudaMemset(allocation, 0xa5, bytes));
    }
    ~Device() { CHECK(cudaFree(allocation)); device_live -= bytes; }
    Device(const Device&) = delete; Device& operator=(const Device&) = delete;
    void upload(const Host<T>& h) const {
        require(h.count == count, "upload extent");
        CHECK(cudaMemcpy(ptr, h.ptr, payload, cudaMemcpyHostToDevice));
    }
    void read(Host<T>& h) const {
        require(h.count == count, "read extent");
        CHECK(cudaMemcpy(h.ptr, ptr, payload, cudaMemcpyDeviceToHost));
    }
    void guards() const {
        CHECK(cudaMemcpy(guard_before, allocation, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(guard_after, ptr + count, 128, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 128; ++i)
            require(guard_before[i] == 0xa5 && guard_after[i] == 0xa5, "allocation canary", 3);
    }
    void unchanged(const Host<T>& h) const {
        require(h.count == count, "immutable owner extent");
        for (size_t offset = 0; offset < payload; offset += sizeof(scratch)) {
            const size_t bytes_now = std::min(sizeof(scratch), payload - offset);
            CHECK(cudaMemcpy(scratch, reinterpret_cast<const unsigned char*>(ptr) + offset,
                             bytes_now, cudaMemcpyDeviceToHost));
            require(!std::memcmp(scratch, reinterpret_cast<const unsigned char*>(h.ptr) + offset,
                                 bytes_now), "input/weight/scale owner modified");
        }
        guards();
    }
};
static unsigned short bits(__nv_bfloat16 v) {
    unsigned short b; std::memcpy(&b, &v, 2); return b;
}
static __nv_bfloat16 bf_bits(unsigned short b) {
    __nv_bfloat16 v; std::memcpy(&v, &b, 2); return v;
}
struct Lcg {
    uint64_t state;
    uint32_t next() { state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return uint32_t(state >> 32); }
    float value() { return (float(next() & 65535) / 65535.0f - .5f) * 3.0f; }
};
enum class Fault { None, Output, Guard, Unused, Budget };
struct Options { unsigned repetitions = 0; Fault fault = Fault::None; };
static Options options(int argc, char** argv) {
    Options out; bool reps = false, fault = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--repetitions")) {
            require(!reps && ++i < argc, "invalid/repeated repetitions", 64); reps = true;
            const char* p = argv[i]; require(*p != '\0', "empty repetitions", 64);
            for (; *p; ++p) {
                require(*p >= '0' && *p <= '9', "repetitions must be decimal", 64);
                out.repetitions = out.repetitions * 10 + unsigned(*p - '0');
                require(out.repetitions <= 100, "repetitions exceeds100", 64);
            }
        } else if (!std::strcmp(argv[i], "--fault")) {
            require(!fault && ++i < argc, "invalid/repeated fault", 64); fault = true;
            if (!std::strcmp(argv[i], "output")) out.fault = Fault::Output;
            else if (!std::strcmp(argv[i], "guard")) out.fault = Fault::Guard;
            else if (!std::strcmp(argv[i], "unused")) out.fault = Fault::Unused;
            else if (!std::strcmp(argv[i], "budget")) out.fault = Fault::Budget;
            else require(false, "unknown fault", 64);
        } else require(false, "unknown option", 64);
    }
    require(out.fault == Fault::None || out.repetitions == 0, "fault requires repetitions0", 64);
    return out;
}
static void flip(void* p) {
    unsigned char b; CHECK(cudaMemcpy(&b, p, 1, cudaMemcpyDeviceToHost)); b ^= 1;
    CHECK(cudaMemcpy(p, &b, 1, cudaMemcpyHostToDevice));
}

static void run(unsigned n, unsigned k, unsigned profile, const Options& opt) {
    require(n > 0 && k % 16 == 0 && profile < 9, "unsupported fixture geometry", 64);
    const size_t nk = mul(n, k), in_count = mul(16, k), out_count = mul(48, n);
    const size_t weights = mul(3, add(nk / 2, nk / 16));
    const size_t device_expected = add(add(weights, mul(in_count, 2)), add(mul(out_count, 8), 11 * 256));
    const size_t host_expected = add(add(weights, mul(in_count, 4)), add(mul(out_count, 10), host_fixed));
    require(device_expected <= cap && host_expected <= cap, "geometry exceeds explicit cap", 4);
    if (opt.fault == Fault::Budget) require(device_expected <= 1, "injected undersized budget", 4);
    Host<unsigned char> hw[3] = {Host<unsigned char>(nk / 2), Host<unsigned char>(nk / 2), Host<unsigned char>(nk / 2)};
    Host<unsigned char> hs[3] = {Host<unsigned char>(nk / 16), Host<unsigned char>(nk / 16), Host<unsigned char>(nk / 16)};
    Host<__nv_bfloat16> input(in_count), permuted(in_count), hc(out_count), hb(out_count),
        hp(out_count), hsout(out_count), canonical(out_count);
    const unsigned seeds[] = {1, 99, 12345};
    Lcg rng{uint64_t(seeds[profile < 6 ? profile / 2 : 0]) * 0x9e3779b97f4a7c15ULL};
    const bool impulse = profile == 8;
    float scale2[3];
    for (size_t i = 0; i < in_count; ++i) input.ptr[i] = bf_bits(0x7fc1);
    for (unsigned t = 0; t < 5; ++t)
        for (unsigned d = 0; d < k; ++d)
            input.ptr[size_t(t) * k + d] = __float2bfloat16(impulse ?
                (d == t * 7 % k ? (t % 2 ? -1.f : 1.f) : 0.f) : rng.value());
    for (unsigned p = 0; p < 3; ++p) {
        scale2[p] = profile == 6 ? 0.f : float(p + 1) * (profile < 6 && profile % 2 == 0 ? .0123f : 1.f);
        for (size_t i = 0; i < hw[p].count; ++i) {
            const unsigned lo = 1 + unsigned((i + p) % 7) + ((i + p) % 2 ? 8 : 0);
            const unsigned hi = 1 + unsigned((i + p + 3) % 7) + ((i + p) % 2 ? 0 : 8);
            hw[p].ptr[i] = impulse ? static_cast<unsigned char>(lo | hi << 4) : static_cast<unsigned char>(rng.next());
        }
        for (size_t i = 0; i < hs[p].count; ++i)
            hs[p].ptr[i] = impulse ? 0x38 + 8 * p : (profile == 7 ? 1 + rng.next() % 7 : 0x30 + rng.next() % 24);
        // Deterministic distinctness, not a probabilistic whole-array assumption.
        if (!impulse) { hw[p].ptr[0] = 0x21 + 0x11 * p; hs[p].ptr[0] = profile == 7 ? 1 + p : 0x30 + p; }
    }
    Device<unsigned char> w[3] = {Device<unsigned char>(nk / 2), Device<unsigned char>(nk / 2), Device<unsigned char>(nk / 2)};
    Device<unsigned char> s[3] = {Device<unsigned char>(nk / 16), Device<unsigned char>(nk / 16), Device<unsigned char>(nk / 16)};
    Device<__nv_bfloat16> a(in_count), candidate(out_count), planes(out_count), baseline(out_count), scalar(out_count);
    require(device_live == device_expected && host_live == host_expected, "explicit accounting mismatch", 4);
    const dim3 grid((n + 3) / 4, 1, 3), one((n + 3) / 4);
    std::array<unsigned, 3> owners{{0, 1, 2}};
    std::array<unsigned, 5> rows{{0, 1, 2, 3, 4}};
    float active_scale2[3];
    const auto upload = [&] {
        std::memcpy(permuted.ptr, input.ptr, input.bytes);
        for (unsigned t = 0; t < 5; ++t)
            std::memcpy(permuted.ptr + size_t(t) * k, input.ptr + size_t(rows[t]) * k, mul(k, 2));
        a.upload(permuted);
        for (unsigned p = 0; p < 3; ++p) {
            w[p].upload(hw[owners[p]]); s[p].upload(hs[owners[p]]); active_scale2[p] = scale2[owners[p]];
        }
        // All init, uploads, kernels and readbacks use the same legacy stream.
        CHECK(cudaDeviceSynchronize());
    };
    const auto reset = [&] {
        for (auto* b : {&candidate, &planes, &baseline, &scalar}) CHECK(cudaMemset(b->ptr, 0xff, b->payload));
    };
    const auto launch = [&](unsigned arm) {
        if (arm == 0) {
            w4a16_gemv_batch5_qkv<<<grid, 256>>>(a.ptr, w[0].ptr, s[0].ptr, active_scale2[0],
                w[1].ptr, s[1].ptr, active_scale2[1], w[2].ptr, s[2].ptr, active_scale2[2], planes.ptr, 5, n, k);
            CHECK(cudaGetLastError());
            kda_pack_qkv<<<(5 * 3 * n + 255) / 256, 256>>>(planes.ptr, baseline.ptr, 5, n);
            CHECK(cudaGetLastError());
        } else if (arm == 1) {
            fixture_qkv_packed5<<<grid, 256>>>(a.ptr, w[0].ptr, s[0].ptr, active_scale2[0],
                w[1].ptr, s[1].ptr, active_scale2[1], w[2].ptr, s[2].ptr, active_scale2[2], candidate.ptr, 5, n, k);
            CHECK(cudaGetLastError());
        } else {
            for (unsigned p = 0; p < 3; ++p) for (unsigned t = 0; t < 5; ++t) {
                w4a16_gemv<<<one, 256>>>(a.ptr + size_t(t) * k, w[p].ptr, s[p].ptr, active_scale2[p],
                    scalar.ptr + size_t(p * 5 + t) * n, n, k);
                CHECK(cudaGetLastError());
            }
        }
    };
    const auto guards = [&] {
        a.unchanged(permuted);
        for (unsigned p = 0; p < 3; ++p) { w[p].unchanged(hw[owners[p]]); s[p].unchanged(hs[owners[p]]); }
        candidate.guards(); planes.guards(); baseline.guards(); scalar.guards();
    };
    const auto validate = [&] {
        candidate.read(hc); baseline.read(hb); planes.read(hp); scalar.read(hsout);
        require(!std::memcmp(hc.ptr, hb.ptr, hc.bytes), "direct versus fused5+pack full bytes");
        require(!std::memcmp(hp.ptr, hsout.ptr, hp.bytes), "fused5 planes versus scalar full bytes");
        for (size_t i = 0; i < out_count; ++i) {
            if (i < size_t(15) * n) {
                const unsigned t = i / (3 * n), p = (i / n) % 3, col = i % n;
                require(bits(hc.ptr[i]) == bits(hsout.ptr[size_t(p * 5 + t) * n + col]), "host scalar row-pack mapping");
                require(std::isfinite(__bfloat162float(hc.ptr[i])), "nonfinite live output");
            } else {
                require(bits(hc.ptr[i]) == 0xffff && bits(hb.ptr[i]) == 0xffff &&
                    bits(hp.ptr[i]) == 0xffff && bits(hsout.ptr[i]) == 0xffff, "unused output rows modified");
            }
        }
        guards();
    };
    upload(); reset(); launch(0); launch(1); launch(2); CHECK(cudaDeviceSynchronize());
    if (opt.fault == Fault::Output) flip(candidate.ptr);
    if (opt.fault == Fault::Unused) flip(candidate.ptr + size_t(15) * n);
    if (opt.fault == Fault::Guard) flip(candidate.allocation);
    validate(); std::memcpy(canonical.ptr, hc.ptr, hc.bytes);
    if (impulse) {
        const float lut[] = {0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
        for (unsigned t = 0; t < 5; ++t) for (unsigned p = 0; p < 3; ++p) for (unsigned col = 0; col < n; ++col) {
            const unsigned d = t * 7 % k;
            const unsigned packed = hw[p].ptr[size_t(col) * (k / 2) + d / 2];
            const unsigned nibble = d % 2 ? packed >> 4 : packed & 15;
            const float expected = lut[nibble] * float(1u << p) * scale2[p] * (t % 2 ? -1.f : 1.f);
            require(bits(canonical.ptr[size_t(t * 3 + p) * n + col]) == bits(__float2bfloat16(expected)), "independent signed impulse oracle");
        }
    }
    for (unsigned permutation = 0; permutation < 3; ++permutation) {
        owners = permutation == 1 ? std::array<unsigned, 3>{{0,1,2}} : std::array<unsigned, 3>{{2,0,1}};
        rows = permutation == 0 ? std::array<unsigned, 5>{{0,1,2,3,4}} : std::array<unsigned, 5>{{4,3,2,1,0}};
        upload(); reset(); launch(0); launch(1); launch(2); CHECK(cudaDeviceSynchronize()); validate();
        for (unsigned t = 0; t < 5; ++t) for (unsigned p = 0; p < 3; ++p)
            require(!std::memcmp(hc.ptr + size_t(t * 3 + p) * n,
                canonical.ptr + size_t(rows[t] * 3 + owners[p]) * n, mul(n, 2)), "same-address owner/row permutation");
    }
    owners = {{0,1,2}}; rows = {{0,1,2,3,4}};
    upload(); reset(); launch(0); launch(1); launch(2); CHECK(cudaDeviceSynchronize()); validate();
    require(!std::memcmp(hc.ptr, canonical.ptr, hc.bytes), "restored canonical order");
    if (opt.repetitions && n == 4096 && profile == 0) {
        for (unsigned i = 0; i < 10; ++i) { launch(0); launch(1); }
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
        float us[5][2];
        for (unsigned round = 0; round < 5; ++round) for (unsigned slot = 0; slot < 2; ++slot) {
            const unsigned arm = (round + slot) % 2;
            CHECK(cudaEventRecord(start));
            for (unsigned i = 0; i < opt.repetitions; ++i) launch(arm);
            CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
            float ms; CHECK(cudaEventElapsedTime(&ms, start, end));
            require(std::isfinite(ms) && ms > 0, "invalid event duration");
            us[round][arm] = ms * 1000 / opt.repetitions;
        }
        CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
        launch(2); CHECK(cudaDeviceSynchronize()); validate();
        require(!std::memcmp(hc.ptr, canonical.ptr, hc.bytes), "post-timing canonical output");
        std::printf("timing_policy rounds=5 warmups_per_arm=10 repetitions=%u arms=0:fused5+pack,1:direct5 residual_order_imbalance=3_vs_2 fresh_process_repeat_required=1\n", opt.repetitions);
        for (unsigned r = 0; r < 5; ++r) std::printf("timing_round=%u order=%u,%u fused_pack_us=%.6f direct_us=%.6f ratio=%.6f\n",
            r, r % 2, (r + 1) % 2, us[r][0], us[r][1], us[r][0] / us[r][1]);
    }
    guards();
    std::printf("M=5 N=%u K=%u profile=%u scale2=%.9g,%.9g,%.9g exact_fused_pack=1 exact_scalar=1 finite=1 unused_rows=11 permutations=3 impulse_host=%u device_bytes=%zu host_payload_bytes=%zu PASS\n",
        n, k, profile, scale2[0], scale2[1], scale2[2], unsigned(impulse), device_live, host_live);
}

int main(int argc, char** argv) {
    const Options opt = options(argc, argv); // Entire CLI parsed before any CUDA call.
    unsigned cases = 0;
    for (const auto shape : {std::array<unsigned, 2>{{7,80}}, std::array<unsigned, 2>{{4096,4096}}})
        for (unsigned profile = 0; profile < 9; ++profile) {
            run(shape[0], shape[1], profile, opt); ++cases;
            require(opt.fault == Fault::None, "requested fault escaped detection", 5);
        }
    require(cases == 18 && device_live == 0 && host_live == host_fixed, "case/allocation closure", 4);
    std::printf("cases=%u peak_device_bytes=%zu peak_host_payload_bytes=%zu device_remaining=%zu host_fixed_bytes=%zu cap_bytes=%zu PASS\n",
        cases, device_peak, host_peak, device_live, host_live, cap);
    return 0;
}
