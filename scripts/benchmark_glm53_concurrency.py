#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Measure per-session and aggregate GLM-5 decode throughput."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path


def one_request(
    base_url: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    timeout: float,
    allow_repetition: bool = False,
    barrier: threading.Barrier | None = None,
) -> dict:
    import requests

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 1,
    }
    if allow_repetition:
        body["repetition_detection"] = {
            "min_pattern_size": 2, "max_pattern_size": 64,
            "min_count": output_tokens + 1,
        }
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.perf_counter()
    first_text = None
    usage = None
    finish_reason = None
    text_chunks = []
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
            for choice in event.get("choices", []):
                if choice.get("text"):
                    text_chunks.append(choice["text"])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
            if event.get("usage"):
                usage = event["usage"]
    ended = time.perf_counter()
    if usage is None or first_text is None:
        raise RuntimeError("stream ended without text or usage")
    if int(usage["prompt_tokens"]) != len(prompt):
        raise RuntimeError("server prompt token count does not match supplied prompt")
    # Text identity, not token-ID or semantic equivalence. Hash outside the
    # timed window, preserving the request and all historical rate formulas.
    completion_text = "".join(text_chunks).encode("utf-8")
    return {
        "started": started,
        "first_text": first_text,
        "ended": ended,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "ttft_ms": float(usage["time_to_first_token_ms"]),
        "decode_tps": float(usage["response_token/s"]),
        "finish_reason": finish_reason,
        "completion_text_sha256": hashlib.sha256(completion_text).hexdigest(),
        "completion_text_bytes": len(completion_text),
    }


def summarize_batch(rows: list[dict], requested_output_tokens: int) -> dict:
    if not rows or requested_output_tokens < 1:
        raise ValueError("require a nonempty batch and positive output cap")
    if any(
        not all(math.isfinite(row[key]) for key in ("started", "first_text", "ended"))
        or not row["started"] <= row["first_text"] < row["ended"]
        for row in rows
    ):
        raise ValueError("invalid request timing order")
    if any(not 1 <= row["completion_tokens"] <= requested_output_tokens for row in rows):
        raise ValueError("completion count must be positive and within requested cap")
    if any(row["prompt_tokens"] != rows[0]["prompt_tokens"] for row in rows):
        raise ValueError("all requests must use the same prompt length")
    started = min(row["started"] for row in rows)
    first = min(row["first_text"] for row in rows)
    last = max(row["ended"] for row in rows)
    completion_tokens = sum(row["completion_tokens"] for row in rows)
    return {
        "concurrency": len(rows),
        "prompt_tokens_each": rows[0]["prompt_tokens"],
        "completion_tokens_total": completion_tokens,
        "output_tokens_requested_total": requested_output_tokens * len(rows),
        "all_outputs_reached_cap": all(
            row["completion_tokens"] == requested_output_tokens for row in rows
        ),
        "batch_wall_seconds": round(last - started, 6),
        "aggregate_e2e_tps": round(completion_tokens / (last - started), 3),
        "median_client_ttft_ms": round(statistics.median(
            (row["first_text"] - row["started"]) * 1000 for row in rows
        ), 3),
        "median_ttft_ms": round(statistics.median(row["ttft_ms"] for row in rows), 3),
        "median_session_decode_tps": round(
            statistics.median(row["decode_tps"] for row in rows), 3
        ),
        "sum_session_decode_tps": round(sum(row["decode_tps"] for row in rows), 3),
        "aggregate_decode_window_tps": round(completion_tokens / (last - first), 3),
        # Preserve the historical metric above. This separately named client
        # rate removes each stream's first completion token from the numerator.
        # First text can contain multiple tokens; this is not a GPU timer.
        "aggregate_post_first_token_tps": round(
            (completion_tokens - len(rows)) / (last - first), 3
        ),
        "requests": [
            {
                "start_offset_ms": round((row["started"] - started) * 1000, 3),
                "first_text_offset_ms": round((row["first_text"] - started) * 1000, 3),
                "end_offset_ms": round((row["ended"] - started) * 1000, 3),
                "completion_tokens": row["completion_tokens"],
                "finish_reason": row.get("finish_reason"),
                "completion_text_sha256": row.get("completion_text_sha256"),
                "completion_text_bytes": row.get("completion_text_bytes"),
            }
            for row in rows
        ],
    }


def batch(args: argparse.Namespace, concurrency: int, prompt: list[int]) -> dict:
    barrier = threading.Barrier(concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(one_request, args.base_url, args.model, prompt,
                        args.output_tokens, args.timeout, args.allow_repetition, barrier)
            for _ in range(concurrency)
        ]
        rows = [future.result() for future in futures]
    return summarize_batch(rows, args.output_tokens)


def summarize_runs(concurrency: int, allow_repetition: bool, measured: list[dict]) -> dict:
    if not measured or any(row["concurrency"] != concurrency for row in measured):
        raise ValueError("require measured batches with the stated concurrency")
    result = {
        "concurrency": concurrency,
        "allow_repetition": allow_repetition,
        "forced_output_cap": False,  # EOS and other server stop conditions remain active.
        "all_outputs_reached_cap": all(row["all_outputs_reached_cap"] for row in measured),
        "runs": measured,
    }
    for output, source in (
        ("median_aggregate_e2e_tps", "aggregate_e2e_tps"),
        ("median_client_ttft_ms", "median_client_ttft_ms"),
        ("median_sum_session_decode_tps", "sum_session_decode_tps"),
        ("median_aggregate_decode_window_tps", "aggregate_decode_window_tps"),
        ("median_aggregate_post_first_token_tps", "aggregate_post_first_token_tps"),
        ("median_session_decode_tps", "median_session_decode_tps"),
        ("median_ttft_ms", "median_ttft_ms"),
    ):
        result[output] = round(statistics.median(row[source] for row in measured), 3)
    return result


def workload_metadata(args: argparse.Namespace, prompt: list[int], literal: str | None = None) -> dict:
    """Hash the exact token sequence with an unambiguous, reproducible encoding."""
    canonical = json.dumps(prompt, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    result = {
        "model": args.model,
        "prompt_source": "literal" if literal is not None else "generated",
        "prompt_tokens": len(prompt),
        "prompt_token_sha256": hashlib.sha256(canonical).hexdigest(),
        "prompt_token_hash_encoding": "ASCII compact JSON integer array",
        "output_tokens_requested_per_stream": args.output_tokens,
        "forced_output_cap": False,
        "allow_repetition": args.allow_repetition,
        "temperature": 0,
        "seed": 1,
        "warmup_batches_per_concurrency": 1,
        "measured_batches_per_concurrency": args.repetitions,
    }
    if literal is not None:
        result["literal_prompt_sha256"] = hashlib.sha256(literal.encode("utf-8")).hexdigest()
        result["literal_prompt_hash_encoding"] = "UTF-8 text sent to /tokenize"
    else:
        result["generated_prompt_tokens_requested"] = args.prompt_tokens
    return result


def main() -> None:
    from benchmark_glm53 import make_prompt, post_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="/var/tmp/models/glm53-flash-nvfp4")
    parser.add_argument("--prompt-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--min-concurrency", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--allow-repetition", action="store_true",
                        help="raise the repetition-watchdog threshold for each request only")
    parser.add_argument("--prompt-file", type=Path,
                        help="literal UTF-8 prompt; replaces the generated --prompt-tokens workload")
    args = parser.parse_args()
    if not 1 <= args.min_concurrency <= args.max_concurrency:
        parser.error("require 1 <= --min-concurrency <= --max-concurrency")
    if not 1 <= args.output_tokens <= 1_000_000 or args.repetitions < 1:
        parser.error("require positive repetitions and output cap in 1..1000000")
    args.base_url = args.base_url.rstrip("/")
    literal = None
    if args.prompt_file is not None:
        literal = args.prompt_file.read_text(encoding="utf-8")
        if not literal.strip():
            parser.error("prompt file is empty")
        prompt = post_json(f"{args.base_url}/tokenize", {"prompt": literal}, args.timeout)["tokens"]
    else:
        prompt = make_prompt(args.base_url, args.prompt_tokens, args.timeout)

    workload = workload_metadata(args, prompt, literal)

    results = []
    for concurrency in range(args.min_concurrency, args.max_concurrency + 1):
        warmup = batch(args, concurrency, prompt)
        measured = [batch(args, concurrency, prompt) for _ in range(args.repetitions)]
        result = summarize_runs(concurrency, args.allow_repetition, measured)
        # Keep warmup evidence separate; all historical medians/cap flags above
        # still describe only the measured waves, including normal early stops.
        result["warmup_runs"] = [warmup]
        result["warmup_all_outputs_reached_cap"] = warmup["all_outputs_reached_cap"]
        result["workload"] = workload
        results.append(result)
        print(json.dumps(result), file=sys.stderr, flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
