#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Measure per-session and aggregate GLM-5 decode throughput."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time

import requests

from benchmark_glm53 import make_prompt


def one_request(
    base_url: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    timeout: float,
) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 1,
    }
    started = time.perf_counter()
    first_text = None
    usage = None
    with requests.post(
        f"{base_url}/v1/completions", json=body, stream=True, timeout=timeout
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            if first_text is None and any(c.get("text") for c in event.get("choices", [])):
                first_text = time.perf_counter()
            if event.get("usage"):
                usage = event["usage"]
    ended = time.perf_counter()
    if usage is None or first_text is None:
        raise RuntimeError("stream ended without text or usage")
    return {
        "started": started,
        "first_text": first_text,
        "ended": ended,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "ttft_ms": float(usage["time_to_first_token_ms"]),
        "decode_tps": float(usage["response_token/s"]),
    }


def batch(args: argparse.Namespace, concurrency: int, prompt: list[int]) -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                one_request,
                args.base_url,
                args.model,
                prompt,
                args.output_tokens,
                args.timeout,
            )
            for _ in range(concurrency)
        ]
        rows = [future.result() for future in futures]
    first = min(row["first_text"] for row in rows)
    last = max(row["ended"] for row in rows)
    completion_tokens = sum(row["completion_tokens"] for row in rows)
    return {
        "concurrency": concurrency,
        "prompt_tokens_each": rows[0]["prompt_tokens"],
        "completion_tokens_total": completion_tokens,
        "median_ttft_ms": round(statistics.median(row["ttft_ms"] for row in rows), 3),
        "median_session_decode_tps": round(
            statistics.median(row["decode_tps"] for row in rows), 3
        ),
        "sum_session_decode_tps": round(sum(row["decode_tps"] for row in rows), 3),
        "aggregate_decode_window_tps": round(completion_tokens / (last - first), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="/var/tmp/models/glm53-flash-nvfp4")
    parser.add_argument("--prompt-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--min-concurrency", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if not 1 <= args.min_concurrency <= args.max_concurrency:
        parser.error("require 1 <= --min-concurrency <= --max-concurrency")
    args.base_url = args.base_url.rstrip("/")
    prompt = make_prompt(args.base_url, args.prompt_tokens, args.timeout)

    results = []
    for concurrency in range(args.min_concurrency, args.max_concurrency + 1):
        batch(args, concurrency, prompt)  # warm-up
        measured = [batch(args, concurrency, prompt) for _ in range(args.repetitions)]
        result = {
            "concurrency": concurrency,
            "median_sum_session_decode_tps": round(
                statistics.median(row["sum_session_decode_tps"] for row in measured), 3
            ),
            "median_aggregate_decode_window_tps": round(
                statistics.median(
                    row["aggregate_decode_window_tps"] for row in measured
                ),
                3,
            ),
            "median_session_decode_tps": round(
                statistics.median(row["median_session_decode_tps"] for row in measured),
                3,
            ),
            "median_ttft_ms": round(
                statistics.median(row["median_ttft_ms"] for row in measured), 3
            ),
            "runs": measured,
        }
        results.append(result)
        print(json.dumps(result), file=sys.stderr, flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
