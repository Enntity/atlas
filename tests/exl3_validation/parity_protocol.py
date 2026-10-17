# SPDX-License-Identifier: AGPL-3.0-only
"""Shared artifact validation for teacher-forced EXL3 comparisons."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def token_sha(cases):
    return hashlib.sha256(canonical(cases)).hexdigest()


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "prompt_ids", "continuation_ids"}:
            raise ValueError("case requires exactly id/prompt_ids/continuation_ids")
        name = case["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in ids:
            raise ValueError("case ID must be unique and filename safe")
        ids.add(name)
        for field in ("prompt_ids", "continuation_ids"):
            tokens = case[field]
            if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("tokens must be explicit nonnegative integers")
        if not case["prompt_ids"]:
            raise ValueError("prompt must contain at least one token")
    return cases


def validate_provenance(value):
    required = {"engine", "engine_build", "checkpoint_sha256", "checkpoint_revision",
                "tokenizer_sha256", "config_sha256", "execution", "serve_flags",
                "environment", "hardware", "hostname", "utc"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"provenance requires {sorted(required)}")
    if value["engine"] not in ("atlas", "vllm"):
        raise ValueError("engine must be atlas or vllm")
    for field in required - {"execution", "environment"}:
        if not value[field]:
            raise ValueError(f"missing provenance {field}")
    for field in ("checkpoint_sha256", "tokenizer_sha256", "config_sha256"):
        hashes = value[field]
        if not isinstance(hashes, dict) or not hashes or any(
                not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                for k, v in hashes.items()):
            raise ValueError(f"{field} requires filename -> SHA256 map")
    build = value["engine_build"]
    if not isinstance(build, dict) or any(not build.get(k) for k in ("commit", "binary_sha256", "source_sha256", "image")):
        raise ValueError("engine_build requires commit/binary_sha256/source_sha256/image")
    for field in ("binary_sha256", "source_sha256"):
        if not isinstance(build[field], str) or not re.fullmatch(r"[0-9a-f]{64}", build[field]):
            raise ValueError("build binary/source hashes must be SHA256")
    hardware = value["hardware"]
    if not isinstance(hardware, dict) or any(not hardware.get(k) for k in ("gpu_uuids", "driver", "cuda", "nccl")) or not isinstance(hardware["gpu_uuids"], list) or len(hardware["gpu_uuids"]) != 2 or not all(isinstance(v, str) and v for v in hardware["gpu_uuids"]) or len(set(hardware["gpu_uuids"])) != 2:
        raise ValueError("hardware requires two GPU UUIDs and driver/CUDA/NCCL versions")
    execution = value["execution"]
    required_execution = {"tp": 2, "ep": 1, "kv_dtype": "bf16", "activation_dtype": "bf16",
                          "speculative": False, "eager": True, "prefix_caching": False}
    if not isinstance(execution, dict) or any(
            type(execution.get(k)) is not type(v) or execution[k] != v
            for k, v in required_execution.items()):
        raise ValueError("comparison requires TP2/EP1/BF16 KV and activations/eager/no speculation/no prefix cache")
    if not isinstance(value["serve_flags"], list) or not all(isinstance(s, str) for s in value["serve_flags"]):
        raise ValueError("serve_flags must preserve the complete argument list")
    if not isinstance(value["environment"], dict):
        raise ValueError("environment must be an explicit mapping")
    return value


def validate_thresholds(value):
    if not isinstance(value, dict) or set(value) != {"abs", "rel", "kl", "cross_entropy", "top_k", "argmax"}:
        raise ValueError("thresholds require abs/rel/kl/cross_entropy/top_k/argmax")
    for key in ("abs", "rel", "kl", "cross_entropy"):
        if type(value[key]) not in (int, float) or not np.isfinite(value[key]) or value[key] < 0:
            raise ValueError("thresholds must be finite nonnegative numbers")
    if type(value["top_k"]) is not int or value["top_k"] <= 0 or value["argmax"] != "exact":
        raise ValueError("thresholds require positive top_k and exact argmax")
    return value


def validate_run_pair(reference, candidate, input_fields):
    for field in ("checkpoint_sha256", "checkpoint_revision", "tokenizer_sha256", "config_sha256", "execution"):
        if reference["provenance"][field] != candidate["provenance"][field]:
            raise ValueError(f"comparison provenance differs: {field}")
    if reference["provenance"]["engine"] == candidate["provenance"]["engine"]:
        raise ValueError("comparison requires distinct Atlas/vLLM engines")
    for field in input_fields:
        if reference[field] != candidate[field]:
            raise ValueError(f"comparison inputs differ: {field}")


def read_raw(sidecar):
    sidecar = Path(sidecar)
    meta = json.loads(sidecar.read_text())
    if meta.get("dtype") != "f32-le":
        raise ValueError("raw capture dtype must be f32-le")
    rows, vocab = meta.get("rows"), meta.get("vocab")
    if type(rows) is not int or type(vocab) is not int or min(rows, vocab) <= 0:
        raise ValueError("raw capture shape must be positive rows/vocab")
    filename = meta.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename or not re.fullmatch(r"logits-\d{6}\.f32", filename):
        raise ValueError("raw filename must be local sequential logits-NNNNNN.f32")
    if sidecar.stem != Path(filename).stem:
        raise ValueError("capture sidecar/data sequence mismatch")
    path = sidecar.parent / filename
    if path.stat().st_size != rows * vocab * 4:
        raise ValueError("raw capture byte count mismatch")
    array = np.fromfile(path, dtype="<f4").reshape(rows, vocab)
    if not np.isfinite(array).all():
        raise ValueError("raw capture contains nonfinite logits")
    return array[-1].copy(), meta


def load_artifact(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "atlas-exl3-teacher-forced-v1":
        raise ValueError("unsupported comparison manifest")
    thresholds = validate_thresholds(manifest["thresholds"])
    registration = root / "pre_registration.json"
    if manifest.get("pre_registration_sha256") != hashlib.sha256(registration.read_bytes()).hexdigest() or json.loads(registration.read_text())["thresholds"] != thresholds:
        raise ValueError("threshold preregistration mismatch")
    validate_provenance(manifest["provenance"])
    cases = validate_cases(manifest["cases"])
    if manifest.get("tokens_sha256") != token_sha(cases):
        raise ValueError("token fingerprint mismatch")
    if manifest.get("request_policy") != {"temperature": 0, "seed": 42, "max_tokens": 1}:
        raise ValueError("request policy mismatch")
    if manifest.get("logits_sha256") != hashlib.sha256((root / "logits.npz").read_bytes()).hexdigest():
        raise ValueError("logits archive fingerprint mismatch")
    with np.load(root / "logits.npz", allow_pickle=False) as archive:
        if set(archive.files) != {c["id"] for c in cases}:
            raise ValueError("artifact case IDs mismatch")
        arrays = {name: archive[name].copy() for name in archive.files}
    vocab = manifest["vocab"]
    if type(vocab) is not int or vocab <= 1:
        raise ValueError("invalid artifact vocabulary")
    for case in cases:
        array = arrays[case["id"]]
        if array.dtype != np.float32 or array.shape != (len(case["continuation_ids"]) + 1, vocab):
            raise ValueError("artifact logits full shape/dtype mismatch")
        if not np.isfinite(array).all() or max(case["prompt_ids"] + case["continuation_ids"]) >= vocab:
            raise ValueError("nonfinite logits or token outside vocabulary")
    records = manifest.get("captures")
    if not isinstance(records, list) or len(records) != sum(len(c["continuation_ids"]) + 1 for c in cases):
        raise ValueError("capture records do not cover every prefix")
    position = 0
    generated_text = {}
    for case in cases:
        generated_text[case["id"]] = []
        for step, row in enumerate(arrays[case["id"]]):
            record = records[position]
            position += 1
            prefix = case["prompt_ids"] + case["continuation_ids"][:step]
            if record.get("case") != case["id"] or record.get("step") != step or record.get("prefix_sha256") != hashlib.sha256(canonical(prefix)).hexdigest() or record.get("row_sha256") != hashlib.sha256(row.astype("<f4").tobytes()).hexdigest():
                raise ValueError("capture token/row mapping mismatch")
            expected_response = f"{case['id']}-{step:06d}.json"
            if record.get("response") != expected_response:
                raise ValueError("response mapping filename mismatch")
            http = json.loads((root / "responses" / expected_response).read_text())
            expected_request = {"model": manifest["model"], "prompt": prefix, "max_tokens": 1,
                                "temperature": 0, "seed": 42, "stream": False}
            if http.get("request") != expected_request:
                raise ValueError("recorded HTTP request differs from prescribed prefix")
            choices = http.get("response", {}).get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0].get("text"), str):
                raise ValueError("HTTP response requires exactly one string text choice")
            generated_text[case["id"]].append(choices[0]["text"])
    manifest["generated_text"] = generated_text
    return manifest, arrays
