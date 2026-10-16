#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Golden vectors for the NVFP4-CSF block-scale decoder
(`crates/spark-runtime/src/weights/nvfp4_csf.rs`).

WHAT IT DOES
------------
Range-reads a few real `*.weight_scale.nvfp4_csf_{fixed,exceptions}` pairs out
of `nvidia/Qwen3.8-Flash-Next-FP4-Local` over HTTP (a few KB each, never a
5 GB shard), decodes each with the vLLM reference decoder, and writes:

    test_data/nvfp4_csf/<tag>.fixed.bin        raw U8 fixed stream
    test_data/nvfp4_csf/<tag>.exceptions.bin   raw U32 (LE) exception words
    test_data/nvfp4_csf/golden.json            shapes + sha256 of the DECODED
                                               E4M3 matrix + a byte sample

The Rust test reads the two inputs, decodes them, and must be byte-identical
(sha256 and sample) to what the reference produced here.

THE REFERENCE
-------------
`decode_nvfp4_csf_scale` below is copied verbatim (minus logging) from vLLM
PR 60646, `vllm/model_executor/model_loader/csf_scales.py`, as carried by
`eugr/spark-vllm-docker:docker/patches/qwen3.8/vllm/14-pr60646.patch`
(upstream commit 00b5ad0712629374e6352e2b31de0460be0fdf17).
Copyright contributors to the vLLM project, Apache-2.0. It is deliberately the
unmodified reference rather than a re-derivation, so the golden tests the Rust
decoder against someone else's reading of the format.

USAGE
-----
    python3 scripts/nvfp4_csf_golden.py            # (re)generate the fixtures
    python3 scripts/nvfp4_csf_golden.py --check    # re-decode, compare to golden.json

Needs `torch` (CPU is fine) and network access to huggingface.co.
"""

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

import torch

REPO = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
REV = "b9ce18a6887e18c089fdea8509a33e63429a3300"
OUT = Path(__file__).resolve().parent.parent / "test_data" / "nvfp4_csf"

# (tag, tensor stem). One of each projection shape — gate/up are [640, 160]
# scale matrices, down is [2560, 40] — from both ends of the layer and expert
# ranges, so a decoder that only works on one geometry or one shard fails.
CASES = [
    ("l0_e0_gate", "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale"),
    ("l0_e0_down", "model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale"),
    ("l47_e511_up", "model.language_model.layers.47.mlp.experts.511.up_proj.weight_scale"),
]


# ---- vLLM reference (Apache-2.0, see module docstring) -----------------------
def decode_nvfp4_csf_scale(fixed: torch.Tensor, exceptions: torch.Tensor) -> torch.Tensor:
    """Rebuild one ``[rows, cols]`` float8_e4m3fn block-scale matrix."""
    if fixed.dtype != torch.uint8 or fixed.dim() != 2 or fixed.shape[1] % 16:
        raise ValueError(f"Invalid CSF fixed stream {fixed.dtype} {fixed.shape}")
    if exceptions.dtype != torch.uint32:
        raise ValueError(f"Invalid CSF exceptions dtype {exceptions.dtype}")
    rows = fixed.shape[0] * 16
    cols = (fixed.shape[1] // 16 - 1) * 2
    fixed = fixed.cpu()
    bases = fixed[:, :16].reshape(rows, 1).to(torch.int16)
    if bool((bases > 240).any()):
        raise ValueError("CSF row bases must be in 0..240")
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(rows, cols)
    out = (bases + codes.to(torch.int16)).reshape(-1)
    words = exceptions.cpu().reshape(-1).view(torch.int32).to(torch.int64)
    words &= 0xFFFFFFFF
    positions = words & 0xFFFFFF
    if positions.numel() and int(positions.max()) >= rows * cols:
        raise ValueError("CSF exception position out of range")
    out[positions] = (words >> 24).to(torch.int16)
    return out.to(torch.uint8).view(torch.float8_e4m3fn).reshape(rows, cols)


# ---- HTTP range reads --------------------------------------------------------
def _get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req).read()


def _index():
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors.index.json"
    return json.loads(urllib.request.urlopen(url).read())["weight_map"]


_headers = {}


def read_tensor(weight_map, name):
    shard = weight_map[name]
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{shard}"
    if shard not in _headers:
        n = struct.unpack("<Q", _get(url, 0, 7))[0]
        _headers[shard] = (8 + n, json.loads(_get(url, 8, 7 + n)))
    base, header = _headers[shard]
    info = header[name]
    a, b = info["data_offsets"]
    data = _get(url, base + a, base + b - 1)
    assert len(data) == b - a, f"short read for {name}"
    return data, info["dtype"], info["shape"]


def decode(fixed_bytes, fixed_shape, exc_bytes):
    fixed = torch.frombuffer(bytearray(fixed_bytes), dtype=torch.uint8).reshape(fixed_shape)
    exc = torch.frombuffer(bytearray(exc_bytes), dtype=torch.uint32)
    return decode_nvfp4_csf_scale(fixed, exc).view(torch.uint8).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="re-decode the fixtures on disk")
    args = ap.parse_args()

    if args.check:
        golden = json.loads((OUT / "golden.json").read_text())
        for case in golden["cases"]:
            f = (OUT / f"{case['tag']}.fixed.bin").read_bytes()
            e = (OUT / f"{case['tag']}.exceptions.bin").read_bytes()
            out = decode(f, case["fixed_shape"], e).numpy().tobytes()
            ok = hashlib.sha256(out).hexdigest() == case["decoded_sha256"]
            print(f"{case['tag']}: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                sys.exit(1)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    weight_map = _index()
    cases = []
    for tag, stem in CASES:
        fb, fdt, fshape = read_tensor(weight_map, f"{stem}.nvfp4_csf_fixed")
        eb, edt, eshape = read_tensor(weight_map, f"{stem}.nvfp4_csf_exceptions")
        assert (fdt, edt) == ("U8", "U32"), (fdt, edt)
        weight = read_tensor_meta(weight_map, stem.removesuffix("_scale"))
        out = decode(fb, fshape, eb)
        rows, cols = out.shape
        # The scale matrix must line up with its NVFP4 weight: [n, k/2] packed
        # E2M1 bytes carry k/16 scales per row.
        assert rows == weight[0] and cols == weight[1] * 2 // 16, (out.shape, weight)
        raw = out.numpy().tobytes()
        (OUT / f"{tag}.fixed.bin").write_bytes(fb)
        (OUT / f"{tag}.exceptions.bin").write_bytes(eb)
        cases.append(
            {
                "tag": tag,
                "tensor": stem,
                "fixed_shape": fshape,
                "exceptions_shape": eshape,
                "decoded_shape": [rows, cols],
                "decoded_sha256": hashlib.sha256(raw).hexdigest(),
                # First 32 decoded bytes and the bytes at the first three
                # exception positions, so a failing test can say WHERE.
                "head": list(raw[:32]),
                "exception_samples": [
                    [int(w) & 0xFFFFFF, raw[int(w) & 0xFFFFFF]]
                    for w in torch.frombuffer(bytearray(eb), dtype=torch.uint32)[:3].tolist()
                ],
                "exceptions": len(eb) // 4,
            }
        )
        print(f"{tag}: fixed {fshape} + {len(eb) // 4} exceptions -> {rows}x{cols}")
    golden = {
        "source": f"https://huggingface.co/{REPO}/tree/{REV}",
        "reference": "vLLM PR 60646 csf_scales.decode_nvfp4_csf_scale (Apache-2.0)",
        "codec": "byte-window4-fixed-stream-u24-exceptions/1",
        "cases": cases,
    }
    (OUT / "golden.json").write_text(json.dumps(golden, indent=1) + "\n")


def read_tensor_meta(weight_map, stem):
    """Shape of `<stem>.weight` from the cached shard header (no data read)."""
    name = f"{stem}.weight" if not stem.endswith(".weight") else stem
    shard = weight_map[name]
    if shard not in _headers:
        read_tensor(weight_map, name)
    return _headers[shard][1][name]["shape"]


if __name__ == "__main__":
    main()
