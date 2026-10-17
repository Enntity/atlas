# SPDX-License-Identifier: AGPL-3.0-only
"""Native EXL3 correctness checks, independent bitstream/F32 CPU oracle.

Run `python3 tests/exl3_cuda_oracle.py --cpu-only` for oracle self-checks.
Run `python3 tests/exl3_cuda_oracle.py` on a CUDA host with NumPy and CuPy
installed to validate the three kernels at every integer bitrate. No model
weights are downloaded. Synthetic cases exercise tile boundaries, signed FP16
scales, batches, and both TP=2 split axes. This is not a serving parity gate.
"""

import argparse
from pathlib import Path

import numpy as np


def bf16_round(values):
    """Round finite F32 values to BF16 ties-to-even, retaining F32 storage."""
    raw = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (raw + np.uint32(0x7FFF) + ((raw >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def hadamard(values):
    result = np.array(values, dtype=np.float32, copy=True)
    blocks = result.reshape(-1, 128)
    distance = 1
    while distance < 128:
        pairs = blocks.reshape(-1, 2, distance)
        left, right = pairs[:, 0].copy(), pairs[:, 1].copy()
        pairs[:, 0], pairs[:, 1] = left + right, left - right
        distance *= 2
    result *= np.float32(1 / np.sqrt(128))
    return result


def codebook():
    states = np.arange(65536, dtype=np.uint32)
    product = states * np.uint32(0x83DCD12D)
    sums = sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
    half_values = (sums + 0x6400).astype(np.uint16).view(np.float16)
    inverse, bias = np.array([0x1EEE, 0xC931], dtype=np.uint16).view(np.float16)
    # Both operands are exact dyadics. F64 preserves the fused result before
    # the single FP16 conversion; F32 intermediate rounding is avoided.
    return (half_values.astype(np.float64) * float(inverse) + float(bias)).astype(np.float16).astype(np.float32)


CODEBOOK = codebook()


def silu_oracle(gate, up, limit, clamp):
    # F64 transcendental evaluation is independent of CUDA expf. The limits
    # apply to the activated gate and to up, before final BF16 rounding.
    activated = gate.astype(np.float64) / (1 + np.exp(-gate.astype(np.float64)))
    other = up.astype(np.float64)
    if clamp:
        activated = np.minimum(activated, limit)
        other = np.clip(other, -limit, limit)
    return bf16_round((activated * other).astype(np.float32))


def rotated_weights(words, k, n, bits):
    weights = np.empty((k, n), dtype=np.float32)
    tiles = words.reshape(k // 16, n // 16, 8 * bits)
    for ti in range(k // 16):
        for tj in range(n // 16):
            # Unpack bits MSB first in each little-endian u32 word. Read each
            # circular window bit by bit, independent of CUDA's u64 shifts.
            stream_bits = [(int(word) >> bit) & 1 for word in tiles[ti, tj] for bit in range(31, -1, -1)]
            for stream in range(256):
                end = (stream + 257) * bits
                state = 0
                for bit in range(end - 16, end):
                    state = (state << 1) | stream_bits[bit % len(stream_bits)]
                row = ((stream >> 3) & 3) * 2 + (stream & 1) + ((stream >> 1) & 1) * 8
                col = (stream >> 5) + ((stream >> 2) & 1) * 8
                weights[ti * 16 + row, tj * 16 + col] = CODEBOOK[state]
    return weights


def oracle(x, suh, svh, words, bits):
    rows, k = x.shape
    n = len(svh)
    xh = hadamard(x * suh.astype(np.float32))
    weights = rotated_weights(words, k, n, bits)
    yh = np.zeros((rows, n), dtype=np.float32)
    for feature in range(k):
        # Separate multiply and add, matching the Rust oracle and CUDA.
        yh += xh[:, feature, None] * weights[feature]
    return xh, yh, bf16_round(hadamard(yh) * svh.astype(np.float32))


def cpu_self_checks():
    eye = np.eye(128, dtype=np.float32)
    np.testing.assert_allclose(hadamard(hadamard(eye)), eye, atol=2e-7)
    assert bf16_round(np.array([1.00390625, 1.01171875], dtype=np.float32)).tolist() == [1.0, 1.015625]
    for state, expected in ((0, 0xC2E8), (1, 0x3921), (2, 0xB72D), (42, 0xBA06), (255, 0x3AB3), (256, 0xB3DB), (32767, 0x3FCA), (32768, 0xBAF2), (65535, 0xB936)):
        assert np.float16(CODEBOOK[state]).view(np.uint16) == expected
    for bits in range(1, 9):
        weights = rotated_weights(np.zeros(8 * 8 * 8 * bits, dtype=np.uint32), 128, 128, bits)
        np.testing.assert_array_equal(weights, np.full((128, 128), CODEBOOK[0]))
    gate = np.array([-10, 0, 10], dtype=np.float32)
    up = np.array([20, 20, 20], dtype=np.float32)
    np.testing.assert_array_equal(silu_oracle(gate, up, 1, True)[1:], [0, 1])
    assert silu_oracle(gate, up, 1, False)[2] == 200
    assert silu_oracle(gate, up, 1, True)[0] < 0
    print("CPU oracle self-checks passed")


def gpu_checks():
    import cupy as cp

    source = (Path(__file__).resolve().parents[1] / "kernels/gb10/glm-5.3-flash/exl3/exl3.cu").read_text()
    include = Path(__file__).resolve().parents[1] / "kernels/gb10/glm-5.3-flash/exl3"
    module = cp.RawModule(code=source, options=("--std=c++17", "-I" + str(include)))
    input_h = module.get_function("exl3_input_hadamard")
    input_h_f16 = module.get_function("exl3_input_hadamard_f16")
    matmul = module.get_function("exl3_mul1_matmul")
    output_h = module.get_function("exl3_output_hadamard")
    silu_mul = module.get_function("exl3_silu_mul")
    output_f32 = module.get_function("exl3_output_hadamard_f32")
    reorder = module.get_function("exl3_vocab_gather_reorder")
    convert = module.get_function("exl3_logits_to_bf16")

    # True rank-major gathered layouts with multiple rows and padded vocab.
    gathered = np.arange(2 * 3 * 128, dtype=np.float32).reshape(2, 3, 128) / 32
    for active_vocab in (250, 256):
        full = cp.full(3 * active_vocab + 4, -999, dtype=cp.float32)
        out = full[1:1 + 3 * active_vocab]
        reorder(((3 * active_vocab + 255) // 256,), (256,),
                (cp.asarray(gathered), out, np.uint32(3), np.uint32(128), np.uint32(active_vocab)))
        expected = np.concatenate([gathered[0], gathered[1]], axis=1)[:, :active_vocab]
        np.testing.assert_array_equal(cp.asnumpy(out).reshape(3, active_vocab), expected)
        np.testing.assert_array_equal(cp.asnumpy(full)[[0, -3, -2, -1]], [-999] * 4)
        bf16_out = cp.empty(3 * active_vocab, cp.uint16)
        convert(((3 * active_vocab + 255) // 256,), (256,),
                (out, bf16_out, np.uint64(3 * active_vocab)))
        got = (cp.asnumpy(bf16_out).astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(got.reshape(3, active_vocab), bf16_round(expected))

    # Odd element counts and sentinel tails catch launch-boundary errors.
    cases = np.array([-100, -20, -10.0625, -10, -9.9375, -1, 0, 1,
                      9.9375, 10, 10.0625, 20, 100], dtype=np.float32)
    for count in (1, 255, 256, 257, 1025):
        gate = np.resize(cases, count)
        up = np.resize(cases[::-1], count)
        gate_raw = (gate.view(np.uint32) >> 16).astype(np.uint16)
        du = cp.asarray((up.view(np.uint32) >> 16).astype(np.uint16))
        for clamp, limit in ((False, 1.0), (True, 1.0), (True, 10.0)):
            expected = silu_oracle(gate, up, limit, clamp)
            separate_raw = None
            for inplace in (False, True):
                # Keep sentinels around both sides of each output buffer,
                # including the production output=gate aliasing contract.
                gate_buffer = cp.full(count + 4, 0x3F80, dtype=cp.uint16)
                gate_buffer[1:count + 1] = cp.asarray(gate_raw)
                dg = gate_buffer[1:count + 1]
                out_buffer = gate_buffer if inplace else cp.full(count + 4, 0x3F80, dtype=cp.uint16)
                out = out_buffer[1:count + 1]
                silu_mul(((count + 255) // 256,), (256,),
                         (dg, du, out, np.uint64(count), np.float32(limit), np.uint32(clamp)))
                raw = cp.asnumpy(out_buffer)
                np.testing.assert_array_equal(raw[[0, count + 1, count + 2, count + 3]], [0x3F80] * 4)
                actual = (raw[1:count + 1].astype(np.uint32) << 16).view(np.float32)
                np.testing.assert_allclose(actual, expected, rtol=0.008, atol=1e-7)
                if inplace:
                    np.testing.assert_array_equal(raw[1:count + 1], separate_raw)
                else:
                    separate_raw = raw[1:count + 1]
                    np.testing.assert_array_equal(cp.asnumpy(dg), gate_raw)
                np.testing.assert_array_equal(cp.asnumpy(du), (up.view(np.uint32) >> 16).astype(np.uint16))

    def run(x, suh, svh, words, bits):
        rows, k = x.shape
        n = len(svh)
        dx = cp.asarray((x.view(np.uint32) >> 16).astype(np.uint16))
        dsuh, dsvh, dw = cp.asarray(suh), cp.asarray(svh), cp.asarray(words)
        xh, yh = cp.empty((rows, k), cp.float32), cp.empty((rows, n), cp.float32)
        out = cp.empty((rows, n), cp.uint16)
        u32 = np.uint32
        input_h((k // 128, rows), (128,), (dx, dsuh, xh, u32(rows), u32(k)))
        f16_buffer = cp.full(rows * k + 2, -999, cp.float32)
        f16_xh = f16_buffer[1:-1].reshape(rows, k)
        input_h_f16((k // 128, rows), (128,), (dx, dsuh, f16_xh, u32(rows), u32(k)))
        product = (x.astype(np.float16) * suh).astype(np.float16).astype(np.float32)
        expected_f16_xh = hadamard(product).astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(cp.asnumpy(f16_xh), expected_f16_xh)
        np.testing.assert_array_equal(cp.asnumpy(f16_buffer)[[0, -1]], [-999, -999])
        matmul(((n + 127) // 128, rows), (128,), (xh, dw, yh, u32(rows), u32(n), u32(k), u32(bits)))
        output_h((n // 128, rows), (128,), (yh, dsvh, out, u32(rows), u32(n)))
        actual = (cp.asnumpy(out).astype(np.uint32) << 16).view(np.float32)
        expected_xh, expected_yh, expected = oracle(x, suh, svh, words, bits)
        f32_out = cp.empty((rows, n), cp.float32)
        output_f32((n // 128, rows), (128,), (yh, dsvh, f32_out, u32(rows), u32(n)))
        expected_f32 = hadamard(expected_yh) * svh.astype(np.float32)
        np.testing.assert_array_equal(cp.asnumpy(f32_out), expected_f32)
        # These kernels deliberately use the oracle's F32 operation order.
        np.testing.assert_array_equal(cp.asnumpy(xh), expected_xh)
        np.testing.assert_array_equal(cp.asnumpy(yh), expected_yh)
        np.testing.assert_array_equal(actual, expected)
        return actual

    rng = np.random.default_rng(20261009)
    for bits in range(1, 9):
        for rows, k, n in ((1, 128, 256), (3, 256, 128), (17, 256, 256)):
            x = bf16_round(rng.uniform(-0.5, 0.5, (rows, k)).astype(np.float32))
            suh = rng.uniform(-1, 1, k).astype(np.float16)
            svh = rng.uniform(-1, 1, n).astype(np.float16)
            words = rng.integers(0, 2**32, (k // 16, n // 16, 8 * bits), dtype=np.uint32)
            whole = run(x, suh, svh, words, bits)
            if n == 256:
                gathered = np.concatenate([run(x, suh, svh[start:start + 128], words[:, start // 16:(start + 128) // 16].copy(), bits) for start in (0, 128)], axis=1)
                np.testing.assert_array_equal(gathered, whole)
            if k == 256:
                parts = [run(x[:, start:start + 128].copy(), suh[start:start + 128], svh, words[start // 16:(start + 128) // 16].copy(), bits) for start in (0, 128)]
                # Rank-local BF16 output and different summation grouping
                # need tolerances. Exact equivalence is asserted above only
                # for each rank against its own CPU operation order.
                reduced = bf16_round(parts[0] + parts[1])
                np.testing.assert_allclose(reduced, whole, rtol=0.02, atol=0.04)
    print("CUDA oracle checks passed: K=1..8, batched, TP gather/reduce, bounded/unbounded SwiGLU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    cpu_self_checks()
    if not args.cpu_only:
        gpu_checks()
