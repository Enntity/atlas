#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run a low-concurrency Atlas prefill/decode receipt for GLM-5.3."""

from __future__ import annotations

import argparse
import json
import time

import requests


SEED_TEXT = (
    "Atlas is benchmarking one deterministic request on two DGX Sparks. "
    "The prompt is deliberately repetitive so tokenizer construction is local "
    "and the measured work remains reproducible. Continue the following account: "
)


def post_json(url: str, body: dict, timeout: float) -> dict:
    response = requests.post(url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def make_prompt(base_url: str, count: int, timeout: float) -> list[int]:
    text = (SEED_TEXT * (count // 12 + 8)) + "\nThe next result is"
    result = post_json(f"{base_url}/tokenize", {"prompt": text}, timeout)
    tokens = result["tokens"]
    if len(tokens) < count:
        raise RuntimeError(f"tokenizer produced only {len(tokens)} tokens")
    return tokens[:count]


def run(args: argparse.Namespace) -> dict:
    base_url = args.base_url.rstrip("/")
    prompt = make_prompt(base_url, args.prompt_tokens, args.timeout)
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 1,
    }

    started = time.perf_counter()
    first_chunk_at = None
    usage = None
    output = []
    with requests.post(
        f"{base_url}/v1/completions",
        json=body,
        stream=True,
        timeout=args.timeout,
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
            choices = event.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
                if first_chunk_at is None and text:
                    first_chunk_at = time.perf_counter()
                output.append(text)
            if event.get("usage"):
                usage = event["usage"]
    ended = time.perf_counter()

    if usage is None:
        raise RuntimeError("stream ended without a usage receipt")
    ttft_ms = float(usage["time_to_first_token_ms"])
    prompt_count = int(usage["prompt_tokens"])
    receipt = {
        "prompt_tokens": prompt_count,
        "completion_tokens": int(usage["completion_tokens"]),
        "time_to_first_token_ms": round(ttft_ms, 3),
        "prefill_tokens_per_second": round(prompt_count / (ttft_ms / 1000.0), 3),
        "decode_tokens_per_second": round(float(usage["response_token/s"]), 3),
        "wall_seconds": round(ended - started, 3),
        "client_first_text_ms": (
            round((first_chunk_at - started) * 1000.0, 3)
            if first_chunk_at is not None
            else None
        ),
        "output_preview": "".join(output)[:240],
    }
    if prompt_count != args.prompt_tokens:
        raise RuntimeError(
            f"server reported {prompt_count} prompt tokens, expected {args.prompt_tokens}"
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="/var/tmp/models/glm53-flash-nvfp4")
    parser.add_argument("--prompt-tokens", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
