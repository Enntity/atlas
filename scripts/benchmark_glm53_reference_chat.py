#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded Atlas chat run using the published Apollo coding prompt.

This is NOT engine parity: Atlas does not implement the reference's ignore_eos
field. All early stops are retained; no watchdog or thinking-budget overrides.
No generated code is executed. API key, if needed: ATLAS_BENCH_API_KEY.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import time

from benchmark_glm53_concurrency import summarize_batch

REFERENCE = (
    "https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/"
    "3407023e0b8109a1dd12e8a5544e106ca6912afe/benchmark.py"
)
# Short benchmark input from the above reference; client implementation is Atlas's.
PROMPT = ("Write a compact Python function that validates a topological ordering. "
          "Return code only.")


def make_payload(model: str) -> dict:
    return {
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256, "temperature": 0, "reasoning_effort": "low",
        "stream": True, "stream_options": {"include_usage": True},
    }


def read_stream(lines, started: float, clock=time.perf_counter) -> dict:
    first = None
    usage = None
    finish = None
    done = False
    content = []
    reasoning = []
    for line in lines:
        if not line or not line.startswith("data: "):
            continue
        if line[6:] == "[DONE]":
            done = True
            break
        value = json.loads(line[6:])
        if "error" in value:
            raise ValueError(f"server stream error: {value['error']}")
        if value.get("usage"):
            usage = value["usage"]
        for choice in value.get("choices", []):
            delta = choice.get("delta") or {}
            if first is None and any(delta.get(name) for name in
                                     ("content", "reasoning", "reasoning_content", "tool_calls")):
                first = clock()
            if delta.get("content"):
                content.append(delta["content"])
            for name in ("reasoning", "reasoning_content"):
                if delta.get(name):
                    reasoning.append(delta[name])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    ended = clock()
    if not done or usage is None or first is None or finish is None:
        raise ValueError("incomplete stream: require content/reasoning, usage, finish and DONE")
    for name in ("prompt_tokens", "completion_tokens"):
        if type(usage.get(name)) is not int or usage[name] <= 0:
            raise ValueError(f"invalid {name}")
    if not all(math.isfinite(t) for t in (started, first, ended)) or not started <= first < ended:
        raise ValueError("invalid stream timestamps")
    if usage["completion_tokens"] > 256:
        raise ValueError("server exceeded output cap")
    return {
        "started": started, "first_text": first, "ended": ended,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "ttft_ms": (first - started) * 1000,
        "decode_tps": (usage["completion_tokens"] - 1) / (ended - first),
        "finish_reason": finish, "usage": usage,
        "content": "".join(content), "reasoning": "".join(reasoning),
    }


def summarize_wave(rows: list[dict], started: float, ended: float) -> dict:
    result = summarize_batch(rows, 256)
    if (not all(math.isfinite(t) for t in (started, ended))
            or started > min(row["started"] for row in rows)
            or ended < max(row["ended"] for row in rows)):
        raise ValueError("wave must enclose all request timestamps")
    result.update({
        "reference_wave_seconds": ended - started,
        "reference_aggregate_e2e_tps": round(
            sum(row["completion_tokens"] for row in rows) / (ended - started), 3),
        "reference_mean_stream_decode_tps": statistics.mean(row["decode_tps"] for row in rows),
        "forced_output_cap": False,
        "raw_requests": rows,
    })
    return result


def request(args, payload: dict) -> dict:
    import requests

    headers = {}
    key = os.environ.get("ATLAS_BENCH_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    started = time.perf_counter()
    with requests.post(args.base_url.rstrip("/") + "/v1/chat/completions",
                       json=payload, headers=headers, stream=True, timeout=args.timeout) as response:
        response.raise_for_status()
        return read_stream(response.iter_lines(decode_unicode=True), started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="server origin, without /v1")
    parser.add_argument("--model", default="/var/tmp/models/glm53-flash-nvfp4")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--context-limit", type=int, choices=[2048], required=True)
    args = parser.parse_args()
    if (any(n not in (1, 2, 3, 4) for n in args.concurrency)
            or not 1 <= args.repetitions <= 10
            or not math.isfinite(args.timeout) or not 0 < args.timeout <= 120):
        parser.error("require widths1..4, repetitions1..10, timeout in (0,120]")
    payload = make_payload(args.model)
    message_bytes = json.dumps(payload["messages"], sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
    metadata = {
        "reference": REFERENCE, "payload": payload, "context_limit": args.context_limit,
        "message_sha256": hashlib.sha256(message_bytes).hexdigest(),
        "hash_encoding": "UTF-8 sorted-key compact JSON messages array",
        "eos_difference": "reference ignore_eos=true; Atlas server default (field unsupported)",
        "warmup_batches_per_width": 1, "measured_batches_per_width": args.repetitions,
        "timeout_seconds": args.timeout,
        "timeout_semantics": "HTTP connect/read inactivity, not a total wave deadline",
        "quality_scope": "Throughput only; generated output is not executed or graded",
    }
    for width in args.concurrency:
        for repetition in range(-1, args.repetitions):
            wave_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
                rows = list(pool.map(lambda _: request(args, payload), range(width)))
            result = summarize_wave(rows, wave_start, time.perf_counter())
            if any(row["prompt_tokens"] + 256 > args.context_limit for row in rows):
                raise ValueError("observed token usage exceeds the declared context envelope")
            print(json.dumps({"metadata": metadata, "warmup": repetition == -1,
                              "repetition": repetition, "result": result}), flush=True)


if __name__ == "__main__":
    main()
