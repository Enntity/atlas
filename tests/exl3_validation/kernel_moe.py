# SPDX-License-Identifier: AGPL-3.0-only
"""Correctness gate for native indirect EXL3, routing, and rank-local combine.

Default requires a real CUDA device and CuPy; missing prerequisites fail.
--cpu-only checks the independent routing oracle, never hardware qualification.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exl3_cuda_oracle import bf16_round, oracle


def route(logits, bias, topk, normalize, scale):
    scores = (1 / (1 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
    ids = np.argsort(-(scores + bias), axis=1, kind="stable")[:, :topk].astype(np.int32)
    weights = np.take_along_axis(scores, ids, axis=1)
    if normalize:
        total = np.zeros(logits.shape[0], dtype=np.float32)
        for slot in range(topk):
            total += weights[:, slot]
        weights /= total[:, None]
    return ids, weights * np.float32(scale)


def cpu_checks():
    ids, weights = route(np.zeros((2, 4), np.float32), np.zeros(4, np.float32), 2, True, 2.5)
    np.testing.assert_array_equal(ids, [[0, 1], [0, 1]])
    np.testing.assert_array_equal(weights, [[1.25, 1.25], [1.25, 1.25]])
    ids, weights = route(np.zeros((1, 4), np.float32), np.array([0, 1, 0, 2], np.float32), 2, False, 2.5)
    np.testing.assert_array_equal(ids, [[3, 1]])
    np.testing.assert_array_equal(weights, [[1.25, 1.25]])
    print("CPU routing oracle checks passed; no GPU execution performed")


def gpu_checks():
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("A real CUDA device is required")
    common = Path(__file__).resolve().parents[2] / "kernels/gb10/glm-5.3-flash/exl3"
    # Header resolution is explicit; compiler failure is not a skipped test.
    module = cp.RawModule(code=(common / "exl3_moe.cu").read_text(),
                          options=("--std=c++17", "-I" + str(common)))
    funcs = {name: module.get_function("exl3_" + name) for name in
             ("moe_input_hadamard", "moe_mul1_matmul", "moe_output_hadamard", "moe_combine", "router", "topk_sigmoid")}
    rng = np.random.default_rng(20261009)
    rows, k, n, topk, experts = 3, 128, 256, 2, 4
    ids = np.array([[0, 1], [3, 2], [1, 3]], dtype=np.int32)
    dids = cp.asarray(ids)
    payloads = []
    descriptors = np.zeros((experts, 4), dtype=np.uint64)
    refs = []
    for expert, bits in enumerate((1, 3, 6, 8)):
        suh = rng.uniform(-1, 1, k).astype(np.float16)
        svh = rng.uniform(-1, 1, n).astype(np.float16)
        words = rng.integers(0, 2**32, (k // 16, n // 16, 8 * bits), dtype=np.uint32)
        dw, dsuh, dsvh = cp.asarray(words), cp.asarray(suh), cp.asarray(svh)
        payloads.extend((dw, dsuh, dsvh))  # Keep owning arrays alive through launches.
        descriptors[expert] = [dw.data.ptr, dsuh.data.ptr, dsvh.data.ptr, bits]
        refs.append((suh, svh, words, bits))
    table = cp.asarray(descriptors)
    slots = rows * topk
    for expanded in (False, True):
        x = bf16_round(rng.uniform(-0.5, 0.5, (slots if expanded else rows, k)).astype(np.float32))
        dx = cp.asarray((x.view(np.uint32) >> 16).astype(np.uint16))
        xh = cp.empty((slots, k), cp.float32)
        yh = cp.empty((slots, n), cp.float32)
        out = cp.empty((slots, n), cp.uint16)
        args = tuple(map(np.uint32, (rows, k, topk, expanded, experts)))
        funcs["moe_input_hadamard"]((k // 128, slots), (128,), (dx, dids, table, xh, *args))
        args = tuple(map(np.uint32, (rows, n, k, topk, experts)))
        funcs["moe_mul1_matmul"]((n // 128, slots), (128,), (xh, dids, table, yh, *args))
        args = tuple(map(np.uint32, (rows, n, topk, experts)))
        funcs["moe_output_hadamard"]((n // 128, slots), (128,), (yh, dids, table, out, *args))
        actual = (cp.asnumpy(out).astype(np.uint32) << 16).view(np.float32)
        for slot, expert in enumerate(ids.flat):
            suh, svh, words, bits = refs[expert]
            row = slot if expanded else slot // topk
            expected_xh, expected_yh, expected = oracle(x[row:row + 1], suh, svh, words, bits)
            np.testing.assert_array_equal(cp.asnumpy(xh[slot]), expected_xh[0])
            np.testing.assert_array_equal(cp.asnumpy(yh[slot]), expected_yh[0])
            np.testing.assert_array_equal(actual[slot], expected[0])
        weights = rng.uniform(0.1, 1, (rows, topk)).astype(np.float32)
        shared = bf16_round(rng.uniform(-0.5, 0.5, (rows, n)).astype(np.float32))
        dshared = cp.asarray((shared.view(np.uint32) >> 16).astype(np.uint16))
        for has_shared in (False, True):
            combined = cp.empty((rows, n), cp.uint16)
            funcs["moe_combine"](((rows * n + 255) // 256,), (256,),
                                 (out, cp.asarray(weights), dshared, combined,
                                  *map(np.uint32, (rows, n, topk, has_shared))))
            expected = np.zeros((rows, n), np.float32)
            for slot in range(topk):
                expected += actual.reshape(rows, topk, n)[:, slot] * weights[:, slot, None]
            if has_shared:
                expected += shared
            got = (cp.asnumpy(combined).astype(np.uint32) << 16).view(np.float32)
            np.testing.assert_array_equal(got, bf16_round(expected))
        # Negative IDs must produce NaNs, never dereference pointer-table OOB.
        for invalid_id in (-1, experts):
            badids = cp.full_like(dids, invalid_id)
            funcs["moe_input_hadamard"]((k // 128, slots), (128,),
                                       (dx, badids, table, xh, *map(np.uint32, (rows, k, topk, expanded, experts))))
            funcs["moe_mul1_matmul"]((n // 128, slots), (128,),
                                   (xh, badids, table, yh, *map(np.uint32, (rows, n, k, topk, experts))))
            funcs["moe_output_hadamard"]((n // 128, slots), (128,),
                                        (yh, badids, table, out, *map(np.uint32, (rows, n, topk, experts))))
            assert np.isnan(cp.asnumpy(xh)).all()
            assert np.isnan(cp.asnumpy(yh)).all()
            raw = (cp.asnumpy(out).astype(np.uint32) << 16).view(np.float32)
            assert np.isnan(raw).all()
    x = bf16_round(rng.uniform(-1, 1, (rows, k)).astype(np.float32))
    gate = rng.uniform(-0.1, 0.1, (experts, k)).astype(np.float32)
    dx = cp.asarray((x.view(np.uint32) >> 16).astype(np.uint16))
    logits = cp.empty((rows, experts), cp.float32)
    funcs["router"](((experts + 127) // 128, rows), (128,),
                    (dx, cp.asarray(gate), logits, *map(np.uint32, (rows, experts, k))))
    expected = np.zeros((rows, experts), np.float32)
    for feature in range(k):
        expected += x[:, feature, None] * gate[:, feature]
    np.testing.assert_array_equal(cp.asnumpy(logits), expected)
    for tie in (False, True):
        if tie:
            logits.fill(0)
        bias = np.zeros(experts, np.float32) if tie else np.array([0.5, -0.25, 0, 0.1], np.float32)
        for normalize in (False, True):
            route_ids, route_weights = cp.empty((rows, topk), cp.int32), cp.empty((rows, topk), cp.float32)
            funcs["topk_sigmoid"]((rows,), (256,), (logits, cp.asarray(bias), route_ids, route_weights,
                                                  *map(np.uint32, (rows, experts, topk, normalize)), np.float32(2.5)))
            expected_ids, expected_weights = route(cp.asnumpy(logits), bias, topk, normalize, 2.5)
            np.testing.assert_array_equal(cp.asnumpy(route_ids), expected_ids)
            np.testing.assert_allclose(cp.asnumpy(route_weights), expected_weights, rtol=2e-6, atol=2e-7)
    cp.cuda.runtime.deviceSynchronize()
    print("Native CUDA MoE routing, mixed-bitrate indirect slots, combine and negative-ID checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    cpu_checks()
    if not args.cpu_only:
        gpu_checks()
