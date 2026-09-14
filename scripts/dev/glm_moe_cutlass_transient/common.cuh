// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(2); }
}
struct Routing {
    std::array<int, 289> offsets{};
    std::vector<int> ids;
    int max_rows = 0;
};
static unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b;
    return x ^ (x >> 16);
}
static Routing routing(int rows, bool skew, int first = 0) {
    std::array<std::vector<int>, 288> groups;
    for (int t = 0; t < rows; ++t) {
        int token = t + first;
        std::array<bool, 288> seen{};
        for (int r = 0; r < 8; ++r) {
            unsigned e = (skew && r == 0) ? unsigned(token % 8) : mix(token * 19 + r * 4001) % 288;
            while (seen[e]) e = (e + 1) % 288;
            seen[e] = true; groups[e].push_back(t);
        }
    }
    Routing out;
    for (int e = 0; e < 288; ++e) {
        out.max_rows = std::max(out.max_rows, int(groups[e].size()));
        out.ids.insert(out.ids.end(), groups[e].begin(), groups[e].end());
        out.offsets[e + 1] = int(out.ids.size());
    }
    return out;
}

#ifndef ATLAS_HOST_ONLY
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
static size_t live = 0, peak = 0;
constexpr size_t GiB = size_t(1) << 30;
template<class T> struct Buffer {
    T* raw = nullptr; T* p; size_t count, bytes;
    explicit Buffer(size_t n):count(n),bytes(n*sizeof(T)+256) {
        require(bytes <= 3*GiB && live <= 3*GiB-bytes, "3GiB benchmark allocation ceiling");
        CHECK(cudaMalloc(&raw, bytes)); p = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(raw)+128);
        CHECK(cudaMemset(raw, 0xa5, bytes)); live += bytes; peak = std::max(peak, live);
    }
    ~Buffer() { cudaFree(raw); live -= bytes; }
    Buffer(const Buffer&) = delete;
    void upload(const T* data) { CHECK(cudaMemcpy(p, data, count*sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> read() const {
        std::vector<T> out(count); CHECK(cudaMemcpy(out.data(), p, count*sizeof(T), cudaMemcpyDeviceToHost)); return out;
    }
    void guards() const {
        std::array<unsigned char,128> a,b;
        CHECK(cudaMemcpy(a.data(), raw, 128, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b.data(), p+count, 128, cudaMemcpyDeviceToHost));
        for (int i=0;i<128;++i) require(a[i]==0xa5 && b[i]==0xa5, "device guard overwritten");
    }
};
__global__ void fill(unsigned char* out, size_t bytes, bool scale, unsigned seed) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<bytes; i+=size_t(gridDim.x)*blockDim.x) {
        unsigned x=unsigned(i)^seed; x^=x>>16; x*=0x7feb352d; x^=x>>15;
        out[i]=scale ? static_cast<unsigned char>(0x20+(x%24)) : static_cast<unsigned char>(x);
    }
}
template<class F> static float event_time(F launch, cudaStream_t stream) {
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a,stream));
    for(int i=0;i<10;++i) launch();
    CHECK(cudaEventRecord(b,stream)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); return ms/10;
}

#endif
