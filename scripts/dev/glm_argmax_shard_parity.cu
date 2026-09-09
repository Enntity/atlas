// SPDX-License-Identifier: AGPL-3.0-only
// Root-only standalone qualification; no Model, NCCL, or serving claim.
// nvcc -std=c++17 -O3 -arch=sm_121a -DATLAS_ARGMAX_SOURCE='"/ABS/argmax_bf16.cu"' \
//   argmax-merge-microtest.cu -o argmax-merge-microtest
// compute-sanitizer --tool memcheck --error-exitcode 99 ./argmax-merge-microtest
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef ATLAS_ARGMAX_SOURCE
#error "Pin ATLAS_ARGMAX_SOURCE to the exact retained repository kernel file"
#endif
#include ATLAS_ARGMAX_SOURCE

static constexpr unsigned shard = 77428, width = 2 * shard, rows = 2;
static constexpr unsigned guard = 32, marker = 0xa5a5a5a5u;
static constexpr float floor_value = -1e30f;
static constexpr size_t bytes = size_t(rows) * width * 2 + (12 + 2 * guard) * 4;
static_assert(bytes < 1024 * 1024, "device allocation cap");
#define CUDA(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)
static void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); std::exit(2); }
}
static float unpack(uint32_t bits) {
    float value; std::memcpy(&value, &bits, 4); return value;
}
static float bf16(uint16_t bits) { return unpack(uint32_t(bits) << 16); }

// Exact pre-fix Rust merge expression from Glm5MtpHead::forward_one.
// This mirror is an explicit legacy control, not a linked Rust invocation.
static unsigned legacy_merge(const uint32_t* p) {
    return unpack(p[2]) > unpack(p[0]) ? p[3] + shard : p[1];
}
// Proposed merge: BF16 cannot equal the FP32 sentinel, so a tied sentinel
// denotes two invalid shards. Preserve the canonical kernel's fallback0.
static unsigned proposed_merge(const uint32_t* p) {
    const float v0 = unpack(p[0]), v1 = unpack(p[2]);
    return v1 > v0 || (v1 == v0 && v1 > floor_value) ? p[3] + shard : p[1];
}
static unsigned oracle(const uint16_t* input, unsigned n) {
    float best = floor_value; unsigned index = 0;
    for (unsigned i = 0; i < n; ++i) {
        const float value = bf16(input[i]);
        if (value > best || (value == best && i > index)) {
            best = value; index = i;
        }
    }
    return index;
}

static const char* names[] = {
    "finite_unique", "same_lane_tie", "cross_lane_tie", "shard_boundary_tie",
    "last_index_tie", "nan_and_finite", "rank0_invalid", "rank1_invalid",
    "all_nan", "all_negative_infinity", "all_below_floor", "positive_infinity_tie"
};
static std::vector<uint16_t> make(unsigned test) {
    std::vector<uint16_t> input(size_t(rows) * width, 0xbf80); // -1
    auto at = [&](unsigned row, unsigned i, uint16_t bits) { input[row * width + i] = bits; };
    for (unsigned row = 0; row < rows; ++row) {
        // Distinct owner placement, including the other physical shard.
        const unsigned base = row * shard;
        switch (test) {
        case 0: at(row, base + 17, 0x40e0); break;
        case 1: at(row, base + 17, 0x40e0); at(row, base + 1041, 0x40e0); break;
        case 2: at(row, base + 1024, 0x40e0); at(row, base + 2047, 0x40e0); break;
        case 3: at(row, shard - 1 - row, 0x40e0); at(row, shard + row, 0x40e0); break;
        case 4: at(row, row, 0x40e0); at(row, width - 1 - row, 0x40e0); break;
        case 5:
            for (unsigned i = 0; i < width; ++i) at(row, i, 0x7fc1);
            at(row, base + 17, 0x4000); at(row, base + 1041, 0x4000); break;
        case 6: case 7: {
            const unsigned invalid = test - 6;
            for (unsigned i = 0; i < shard; ++i) at(row, invalid * shard + i, 0x7fc1);
            at(row, (1 - invalid) * shard + row + 3, 0xc000); // -2, background-1 wins
            at(row, (1 - invalid) * shard + row + 2047, 0x3f80); break;
        }
        case 8: case 9: case 10:
            for (unsigned i = 0; i < width; ++i)
                at(row, i, test == 8 ? uint16_t(0x7fc1 + row) : test == 9 ? 0xff80 : 0xf280);
            break;
        case 11: at(row, base + 17, 0x7f80); at(row, base + 1041, 0x7f80); break;
        default: require(false, "case index");
        }
    }
    return input;
}

int main() {
    uint32_t sentinel; std::memcpy(&sentinel, &floor_value, 4);
    require((sentinel & 0xffffu) != 0, "sentinel must not be representable in BF16");
    __nv_bfloat16* logits = nullptr;
    uint32_t* allocation = nullptr;
    CUDA(cudaMalloc(&logits, size_t(rows) * width * 2));
    CUDA(cudaMalloc(&allocation, (12 + 2 * guard) * 4));
    uint32_t* output = allocation + guard;
    unsigned failures = 0, legacy_mismatches = 0;
    std::printf("source=%s rows=%u shard=%u vocab=%u device_bytes=%zu sentinel=%08x\n",
                ATLAS_ARGMAX_SOURCE, rows, shard, width, bytes, sentinel);
    for (unsigned test = 0; test < sizeof(names) / sizeof(names[0]); ++test) {
        const auto input = make(test);
        CUDA(cudaMemcpy(logits, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
        CUDA(cudaMemset(allocation, 0xa5, (12 + 2 * guard) * 4));
        for (unsigned row = 0; row < rows; ++row) {
            argmax_bf16<<<1, 1024>>>(logits + row * width, output + row, width);
            CUDA(cudaGetLastError());
            for (unsigned rank = 0; rank < 2; ++rank) {
                argmax_bf16_value<<<1, 1024>>>(logits + row * width + rank * shard,
                                             output + 4 + row * 4 + rank * 2, shard);
                CUDA(cudaGetLastError());
            }
        }
        argmax_bf16_batch<<<rows, 1024>>>(logits, output + 2, width, width);
        CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
        std::array<uint32_t, 12 + 2 * guard> actual{};
        CUDA(cudaMemcpy(actual.data(), allocation, actual.size() * 4, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < guard; ++i)
            require(actual[i] == marker && actual[guard + 12 + i] == marker, "output canary");
        const auto* got = actual.data() + guard;
        for (unsigned row = 0; row < rows; ++row) {
            const unsigned expected = oracle(input.data() + row * width, width);
            const auto* pair = got + 4 + row * 4;
            const unsigned combined = proposed_merge(pair), legacy = legacy_merge(pair);
            bool okay = got[row] == expected && got[2 + row] == expected && combined == expected;
            for (unsigned rank = 0; rank < 2; ++rank) {
                const auto* local = input.data() + row * width + rank * shard;
                const unsigned index = oracle(local, shard);
                const float value = bf16(local[index]);
                const float maximum = value > floor_value ? value : floor_value;
                okay = okay && pair[rank * 2 + 1] == index && unpack(pair[rank * 2]) == maximum;
            }
            failures += !okay; legacy_mismatches += legacy != expected;
            std::printf("%s case=%s row=%u expected=%u scalar=%u batch=%u proposed=%u legacy=%u "
                        "shards=(%08x,%u),(%08x,%u)\n", okay ? "PASS" : "FAIL", names[test], row,
                        expected, got[row], got[2 + row], combined, legacy,
                        pair[0], pair[1], pair[2], pair[3]);
        }
        std::vector<uint16_t> unchanged(input.size());
        CUDA(cudaMemcpy(unchanged.data(), logits, unchanged.size() * 2, cudaMemcpyDeviceToHost));
        require(unchanged == input, "input bytes changed");
    }
    CUDA(cudaFree(allocation)); CUDA(cudaFree(logits));
    std::printf("summary cases=24 failures=%u legacy_merge_mismatches=%u\n", failures, legacy_mismatches);
    return failures ? 2 : 0;
}
