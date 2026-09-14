# SPDX-License-Identifier: AGPL-3.0-only
"""Standalone follow-up: end up-accumulator lifetime using retired MMA storage.

Preserves generated.cuh and generate.py. Run --host-test without CUDA or the
new epilogue; otherwise requires parent-reviewed epilogue-shared.cuh.
"""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PIN = "ea5dae666b4f811d09e148b0f5b059c50874c7a17b911d0f3fbccbadc286b50b"

STORAGE = """    // Union lifetime: MMA reads finish before the BF16 up tile is written.
    // Gate values remain in registers; no additional shared allocation.
    struct alignas(16) AtlasMmaStorage {
        unsigned char smem_Ap_pq[2][M_TILE][K_STEP_T64 / 2 + 16];
        unsigned char smem_As_pq[2][M_TILE][K_STEP_T64 / GROUP_SIZE];
        unsigned char smem_BpT_pq[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
        unsigned char smem_Bp_pq[N_TILE_LG][K_STEP_T64 / 2 + 16];
        unsigned char smem_Bs_pq[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG];
        int smem_tok_pq[M_TILE];
    };
    union alignas(16) AtlasSharedStorage {
        AtlasMmaStorage mma;
        unsigned short up_bits[M_TILE][N_TILE_LG];
    };
    static_assert(M_TILE == 64 && N_TILE_LG == 128 && K_STEP_T64 == 64);
    static_assert(GROUP_SIZE == 16 && BP_PAD == 16);
    static_assert(sizeof(AtlasMmaStorage) == 23296);
    static_assert(sizeof(AtlasSharedStorage) == sizeof(AtlasMmaStorage));
    static_assert(alignof(AtlasSharedStorage) >= 16);
    static_assert(sizeof(AtlasSharedStorage::up_bits) == 16384);
    static_assert(offsetof(AtlasMmaStorage, smem_Ap_pq) % 16 == 0);
    static_assert(offsetof(AtlasMmaStorage, smem_As_pq) % 16 == 0);
    static_assert(offsetof(AtlasMmaStorage, smem_BpT_pq) % 16 == 0);
    static_assert(offsetof(AtlasMmaStorage, smem_Bp_pq) % 16 == 0);
    static_assert(offsetof(AtlasMmaStorage, smem_Bs_pq) % 16 == 0);
    __shared__ AtlasSharedStorage atlas_shared;
    auto& smem_Ap_pq = atlas_shared.mma.smem_Ap_pq;
    auto& smem_As_pq = atlas_shared.mma.smem_As_pq;
    auto& smem_BpT_pq = atlas_shared.mma.smem_BpT_pq;
    auto& smem_Bp_pq = atlas_shared.mma.smem_Bp_pq;
    auto& smem_Bs_pq = atlas_shared.mma.smem_Bs_pq;
    auto& smem_tok_pq = atlas_shared.mma.smem_tok_pq;
"""

OLD_EPILOGUE = """        atlas_fused_epilogue(gate_values, acc, scale2, packed_out, scale_out,
            cta_m, cta_n, cta_m_local, M_expert, N);"""
NEW_EPILOGUE = """        // Some warps may still read the final shared MMA operands.
        __syncthreads();
        // Complete all 64x128 stores, including padding, exactly once. The
        // FP32 accumulator is dead after this phase; preserve its BF16 rounding.
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const unsigned c = nt * 8 + tid * 2;
            const unsigned r = warp_m_offset + group_id;
            atlas_shared.up_bits[r][c] = __bfloat16_as_ushort(__float2bfloat16(acc[nt][0] * scale2));
            atlas_shared.up_bits[r][c + 1] = __bfloat16_as_ushort(__float2bfloat16(acc[nt][1] * scale2));
            atlas_shared.up_bits[r + 8][c] = __bfloat16_as_ushort(__float2bfloat16(acc[nt][2] * scale2));
            atlas_shared.up_bits[r + 8][c + 1] = __bfloat16_as_ushort(__float2bfloat16(acc[nt][3] * scale2));
        }
        __syncthreads();
        atlas_fused_epilogue(gate_values,
            reinterpret_cast<const __nv_bfloat16*>(&atlas_shared.up_bits[0][0]),
            packed_out, scale_out, cta_m, cta_n, cta_m_local, M_expert, N);"""


def transform(source):
    start = source.index("template<bool PQ_VEC_SCALES>")
    body = source[start:]
    a = body.index("    // A is already compact FP4.")
    b = body.index("    if (threadIdx.x < M_TILE)", a)
    body = body[:a] + STORAGE + "\n" + body[b:]
    assert body.count(OLD_EPILOGUE) == 1
    body = body.replace(OLD_EPILOGUE, NEW_EPILOGUE)
    # Compute and loads must remain byte-for-byte unchanged, including MMA order.
    def mma(text):
        return text[text.index("    #define PQ4_ISSUE_LOADS"):text.index("    if (projection == 0)")]
    assert mma(source) == mma(body)
    gate = "    if (projection == 0)"
    assert source[source.index(gate):source.index(OLD_EPILOGUE)] == body[body.index(gate):body.index(NEW_EPILOGUE)]
    assert body.count("__shared__") == 1
    return body


def host_checks():
    counts = [0] * (64 * 128)
    for thread in range(128):
        warp, lane = divmod(thread, 32)
        group, tid = divmod(lane, 4)
        for nt in range(16):
            for dr in (0, 8):
                for dc in (0, 1):
                    counts[(warp * 16 + group + dr) * 128 + nt * 8 + tid * 2 + dc] += 1
    assert all(n == 1 for n in counts)
    offsets, end = [], 0
    for extent in (2*64*48, 2*64*4, 2*32*144, 128*48, 2*4*128, 64*4):
        offsets.append(end)
        end += extent
    assert all(n % 16 == 0 for n in offsets)
    assert end == 23296 and 64*128*2 <= end
    print("PASS host: 8192 unique up stores, aligned storage layout, unchanged MMA/gate body")


source_bytes = (HERE / "generated.cuh").read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == PIN, "frozen register candidate changed"
body = transform(source_bytes.decode())
host_checks()
if sys.argv[1:] == ["--host-test"]:
    raise SystemExit(0)
assert not sys.argv[1:], "usage: generate-shared.py [--host-test]"
epilogue = (HERE / "epilogue-shared.cuh").read_text()
assert "const __nv_bfloat16* up_tile" in epilogue
assert "const float acc[16][4]" not in epilogue
output = "// SPDX-License-Identifier: AGPL-3.0-only\n// Generated shared-staging standalone; original remains frozen.\n#include <cstddef>\n" + epilogue + "\n" + body
(HERE / "generated-shared.cuh").write_text(output)
print(hashlib.sha256(output.encode()).hexdigest())
