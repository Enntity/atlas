#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare the native GLM-5.3 vision dump with pinned vLLM math.

The Rust test writes a small manifest, the exact post-processor F32 patch
input, and the native BF16 output.  This driver reads only the visual tensors
from the prepared safetensors index; it never instantiates or loads the text
model.  The reference follows the native CUDA kernel write boundaries so a
failure points at encoder math or checkpoint layout rather than a dtype
conversion made only by the oracle.

Example (on a CUDA host after the ignored Rust test):

    python tools/glm5_vision_oracle.py \
      --model-dir /private/vision-only-checkpoint \
      --dump-dir /private/glm-vision-oracle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REFERENCE_REVISION = "487ecf187d3dfe74d2cf6119a92881dba403c219"


def bf16(value):
    """Round a tensor to BF16 and expose it as F32 for subsequent math."""

    import torch

    return value.to(torch.bfloat16).to(torch.float32)


class VisualWeights:
    """Lazy visual-only safetensors reader with a bounded tensor cache."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        self.model_dir = model_dir
        index_path = next(
            (
                model_dir / name
                for name in (
                    "model.safetensors.index.json",
                    "consolidated.safetensors.index.json",
                )
                if (model_dir / name).is_file()
            ),
            None,
        )
        if index_path is None:
            raise FileNotFoundError(
                "a prepared model.safetensors.index.json or "
                "consolidated.safetensors.index.json is required"
            )
        index = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = index.get("weight_map", {})
        if not self.weight_map:
            raise ValueError(f"{index_path} has no weight_map")
        self._safe_open = safe_open
        self._files: dict[str, Any] = {}
        self._cache: dict[str, Any] = {}

        patch_candidates = [
            name
            for name in self.weight_map
            if name.endswith(".patch_embed.proj.weight")
        ]
        if len(patch_candidates) != 1:
            raise ValueError(
                "visual-only index must contain exactly one patch embed weight; "
                f"found {patch_candidates}"
            )
        self.prefix = patch_candidates[0][: -len(".patch_embed.proj.weight")]
        allowed = (
            f"{self.prefix}.",
        )
        non_visual = [name for name in self.weight_map if not name.startswith(allowed)]
        if non_visual:
            raise ValueError(
                "prepared oracle index contains non-visual tensors, first "
                f"entries: {non_visual[:4]}"
            )

    def _handle(self, shard: str):
        handle = self._files.get(shard)
        if handle is None:
            handle = self._safe_open(
                str(self.model_dir / shard), framework="pt", device="cpu"
            )
            self._files[shard] = handle
        return handle

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str):
        import torch

        if name not in self.weight_map:
            raise KeyError(f"missing visual tensor {name}")
        value = self._cache.get(name)
        if value is None:
            value = self._handle(self.weight_map[name]).get_tensor(name)
            if value.dtype != torch.bfloat16:
                raise TypeError(f"{name} is {value.dtype}; GLM visual weights must be BF16")
            self._cache[name] = value
        return value

    def direct_or_concat(self, base: str, pieces: tuple[str, ...]):
        import torch

        direct = f"{base}.weight"
        if self.has(direct):
            weight = self.get(direct)
        else:
            weight = torch.cat([self.get(f"{piece}.weight") for piece in pieces], dim=0)
        direct_bias = f"{base}.bias"
        if self.has(direct_bias):
            bias = self.get(direct_bias)
        else:
            bias = torch.cat([self.get(f"{piece}.bias") for piece in pieces], dim=0)
        return weight, bias

    def direct_or_concat_no_bias(self, base: str, pieces: tuple[str, ...]):
        import torch

        direct = f"{base}.weight"
        if self.has(direct):
            return self.get(direct)
        return torch.cat([self.get(f"{piece}.weight") for piece in pieces], dim=0)


def linear(x, weight, bias=None):
    """Dense BF16 GEMM with the native kernel's BF16 output store."""

    y = x @ weight.to(dtype=x.dtype).transpose(0, 1)
    if bias is not None:
        y = y + bias.to(dtype=x.dtype)
    return bf16(y)


def rms_norm(x, weight, eps: float):
    import torch

    x = x.to(torch.float32)
    inv = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return bf16(x * inv * weight.to(torch.float32))


def qk_rms_float(x, weight, eps: float):
    """Q/K norm before RoPE; the CUDA kernel keeps this result in F32."""

    import torch

    x = x.to(torch.float32)
    inv = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * inv * weight.to(torch.float32)


def layer_norm(x, weight, bias, eps: float):
    import torch

    x = x.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) * (x - mean)).mean(dim=-1, keepdim=True)
    y = (x - mean) * torch.rsqrt(variance + eps)
    return bf16(y * weight.to(torch.float32) + bias.to(torch.float32))


def rope_table(grid_h: int, grid_w: int, head_dim: int):
    """Build the BF16 table emitted by glm/forward.rs."""

    import torch

    rotary_dim = head_dim // 2
    half = rotary_dim // 2
    inv = [1.0 / (10000.0 ** (2.0 * i / rotary_dim)) for i in range(half)]
    cos_rows: list[list[float]] = []
    sin_rows: list[list[float]] = []
    for block_h in range(grid_h // 2):
        for block_w in range(grid_w // 2):
            for inner_h in range(2):
                for inner_w in range(2):
                    for position in (block_h * 2 + inner_h, block_w * 2 + inner_w):
                        cos_rows.append(
                            [
                                math.cos(position * inv[d if d < half else d - half])
                                for d in range(rotary_dim)
                            ]
                        )
                        sin_rows.append(
                            [
                                math.sin(position * inv[d if d < half else d - half])
                                for d in range(rotary_dim)
                            ]
                        )
    # Each patch gets H-axis rotary values followed by W-axis values.
    cos = torch.tensor(
        [cos_rows[2 * i] + cos_rows[2 * i + 1] for i in range(grid_h * grid_w)],
        dtype=torch.bfloat16,
    ).to(torch.float32)
    sin = torch.tensor(
        [sin_rows[2 * i] + sin_rows[2 * i + 1] for i in range(grid_h * grid_w)],
        dtype=torch.bfloat16,
    ).to(torch.float32)
    return cos, sin


def apply_rope(x, cos, sin):
    import torch

    seq, heads, head_dim = x.shape
    axis_dim = head_dim // 2
    half = axis_dim // 2
    local = torch.arange(axis_dim, device=x.device)
    partner = torch.where(local < half, local + half, local - half)
    rotated = x.clone()
    # The native kernel applies the two independent axes over the first and
    # second half of head_dim and rotates only half of each axis's channels.
    for axis in range(2):
        start = axis * axis_dim
        stop = start + axis_dim
        values = x[:, :, start:stop]
        paired = values.index_select(-1, partner)
        c = cos[:, start:stop].unsqueeze(1)
        s = sin[:, start:stop].unsqueeze(1)
        signs = torch.where(
            local < half,
            -torch.ones_like(local, dtype=x.dtype),
            torch.ones_like(local, dtype=x.dtype),
        )
        rotated[:, :, start:stop] = values * c + signs * paired * s
    return rotated


def attention(qkv, q_norm, k_norm, grid_h: int, grid_w: int, heads: int, head_dim: int):
    import torch

    seq = qkv.shape[0]
    hidden = heads * head_dim
    q = qkv[:, :hidden].view(seq, heads, head_dim)
    k = qkv[:, hidden : 2 * hidden].view(seq, heads, head_dim)
    v = qkv[:, 2 * hidden :].view(seq, heads, head_dim)
    # The pinned implementation hard-codes q/k RMS epsilon to 1e-5.
    q = qk_rms_float(q, q_norm, 1.0e-5)
    k = qk_rms_float(k, k_norm, 1.0e-5)
    cos, sin = rope_table(grid_h, grid_w, head_dim)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    scores = torch.einsum("qhd,khd->hqk", q, k) * (head_dim**-0.5)
    probs = torch.softmax(scores, dim=-1)
    return bf16(torch.einsum("hqk,khd->qhd", probs, v).reshape(seq, hidden))


def clamped_swiglu(x, limit: float):
    import torch

    hidden = x.shape[-1] // 2
    gate = x[..., :hidden].clamp(-limit, limit)
    up = x[..., hidden:].clamp(-limit, limit)
    return bf16(torch.nn.functional.silu(gate) * up)


def run_encoder(weights: VisualWeights, pixels, grid_h: int, grid_w: int, vision: dict[str, Any]):
    import torch

    hidden = int(vision["hidden_size"])
    heads = int(vision["num_heads"])
    head_dim = hidden // heads
    intermediate = int(vision["intermediate_size"])
    out_hidden = int(vision["out_hidden_size"])
    projection_intermediate = int(vision["projection_intermediate_size"])
    eps = float(vision["rms_norm_eps"])
    limit = float(vision["swiglu_limit"])
    prefix = weights.prefix

    x = bf16(pixels)
    patch_weight = weights.get(f"{prefix}.patch_embed.proj.weight").reshape(hidden, -1)
    patch_bias = weights.get(f"{prefix}.patch_embed.proj.bias")
    x = linear(x, patch_weight, patch_bias)

    for index in range(int(vision["depth"])):
        base = f"{prefix}.blocks.{index}"
        norm1 = weights.get(f"{base}.norm1.weight")
        qkv_weight, qkv_bias = weights.direct_or_concat(
            f"{base}.attn.qkv",
            (f"{base}.attn.q", f"{base}.attn.k", f"{base}.attn.v"),
        )
        q_norm = weights.get(f"{base}.attn.q_norm.weight")
        k_norm = weights.get(f"{base}.attn.k_norm.weight")
        proj_weight = weights.get(f"{base}.attn.proj.weight")
        proj_bias = weights.get(f"{base}.attn.proj.bias")
        norm2 = weights.get(f"{base}.norm2.weight")
        gate_weight, gate_bias = weights.direct_or_concat(
            f"{base}.mlp.gate_up_proj",
            (f"{base}.mlp.gate_proj", f"{base}.mlp.up_proj"),
        )
        down_weight = weights.get(f"{base}.mlp.down_proj.weight")
        down_bias = weights.get(f"{base}.mlp.down_proj.bias")

        residual = x
        qkv = linear(rms_norm(x, norm1, eps), qkv_weight, qkv_bias)
        attn = attention(qkv, q_norm, k_norm, grid_h, grid_w, heads, head_dim)
        x = bf16(residual + linear(attn, proj_weight, proj_bias))

        residual = x
        wide = linear(rms_norm(x, norm2, eps), gate_weight, gate_bias)
        act = clamped_swiglu(wide, limit)
        x = bf16(residual + linear(act, down_weight, down_bias))

    x = rms_norm(x, weights.get(f"{prefix}.post_layernorm.weight"), eps)
    merged = (grid_h // 2) * (grid_w // 2)
    # The input patch order is already the [merge block, four members] order
    # emitted by the GLM processor. Conv2d therefore consumes four contiguous
    # rows for every output token.
    x = x.view(merged, 4, hidden)
    conv_weight = weights.get(f"{prefix}.downsample.weight")
    conv_bias = weights.get(f"{prefix}.downsample.bias")
    conv_weight = conv_weight.reshape(out_hidden, hidden, 4)
    conv = torch.einsum("mic,oci->mo", x, conv_weight) + conv_bias
    conv = bf16(conv)

    merger = f"{prefix}.merger"
    y = linear(conv, weights.get(f"{merger}.proj.weight"))
    y = layer_norm(
        y,
        weights.get(f"{merger}.post_projection_norm.weight"),
        weights.get(f"{merger}.post_projection_norm.bias"),
        eps,
    )
    y = bf16(0.5 * y * (1.0 + torch.erf(y * (2.0**-0.5))))
    gate_up = weights.direct_or_concat_no_bias(
        f"{merger}.gate_up_proj",
        (f"{merger}.gate_proj", f"{merger}.up_proj"),
    )
    y = clamped_swiglu(linear(y, gate_up), limit)
    return linear(y, weights.get(f"{merger}.down_proj.weight"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_case(weights: VisualWeights, dump_dir: Path, case: dict[str, Any], vision: dict[str, Any], atol: float, rtol: float):
    import torch

    pixels_path = dump_dir / case["pixels_file"]
    output_path = dump_dir / case["output_file"]
    if sha256_file(pixels_path) != case["input_sha256"]:
        raise ValueError(f"{pixels_path} does not match manifest hash")
    if sha256_file(output_path) != case["output_sha256"]:
        raise ValueError(f"{output_path} does not match manifest hash")
    pixels = torch.frombuffer(
        bytearray(pixels_path.read_bytes()), dtype=torch.float32
    ).clone().view(int(case["grid_h"]) * int(case["grid_w"]), int(case["patch_dim"]))
    native = torch.frombuffer(
        bytearray(output_path.read_bytes()), dtype=torch.bfloat16
    ).clone().to(torch.float32)
    expected_shape = (int(case["output_rows"]), int(case["output_hidden"]))
    native = native.view(*expected_shape)
    reference = run_encoder(
        weights, pixels, int(case["grid_h"]), int(case["grid_w"]), vision
    )
    if tuple(reference.shape) != expected_shape:
        raise AssertionError(
            f"{case['name']}: reference shape {tuple(reference.shape)} != {expected_shape}"
        )
    if not torch.isfinite(native).all() or not torch.isfinite(reference).all():
        raise AssertionError(f"{case['name']}: non-finite native or reference output")
    delta = (reference - native).abs()
    scale = native.abs().clamp_min(1.0e-8)
    max_abs = float(delta.max())
    max_rel = float((delta / scale).max())
    passed = bool(torch.allclose(reference, native, atol=atol, rtol=rtol))
    return {
        "name": case["name"],
        "shape": list(expected_shape),
        "native_sha256": case["output_sha256"],
        "max_abs": max_abs,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="vision-only checkpoint directory (or ATLAS_GLM_VISION_ORACLE_MODEL_DIR)",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="Rust oracle output directory (or ATLAS_GLM_VISION_ORACLE_OUT)",
    )
    parser.add_argument("--atol", type=float, default=0.10)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()

    import os

    model_dir = args.model_dir or Path(os.environ["ATLAS_GLM_VISION_ORACLE_MODEL_DIR"])
    dump_dir = args.dump_dir or Path(os.environ["ATLAS_GLM_VISION_ORACLE_OUT"])
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    if manifest.get("reference_revision") != REFERENCE_REVISION:
        raise ValueError(
            "manifest reference revision differs from the pinned vLLM oracle: "
            f"{manifest.get('reference_revision')}"
        )
    vision = manifest["vision"]
    weights = VisualWeights(model_dir)
    summaries = [
        compare_case(weights, dump_dir, case, vision, args.atol, args.rtol)
        for case in manifest["cases"]
    ]
    result = {
        "reference_revision": REFERENCE_REVISION,
        "model_dir": str(model_dir),
        "dump_dir": str(dump_dir),
        "cases": summaries,
        "passed": all(summary["passed"] for summary in summaries),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
