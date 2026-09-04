#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run an exact-token GLM-5.3 needle-in-a-haystack receipt."""

from __future__ import annotations

import argparse
import json
import time

import requests


FILLER = (
    "The archive records routine observations about maps, gardens, weather, "
    "workshops, rivers, libraries, and scientific instruments. Each entry is "
    "independent and contains no secret project identifiers. "
)


def tokenize(base_url: str, prompt: str, timeout: float) -> list[int]:
    response = requests.post(
        f"{base_url}/tokenize", json={"prompt": prompt}, timeout=timeout
    )
    response.raise_for_status()
    return response.json()["tokens"]


def repeat_to_length(source: list[int], length: int) -> list[int]:
    if not source:
        raise RuntimeError("filler tokenization was empty")
    repeats = (length + len(source) - 1) // len(source)
    return (source * repeats)[:length]


def make_prompt(
    base_url: str,
    target: int,
    needle: str,
    position: float,
    timeout: float,
) -> list[int]:
    prefix = tokenize(
        base_url,
        "Read the archive carefully. Remember any explicit important fact.\n\n",
        timeout,
    )
    needle_tokens = tokenize(
        base_url, f"\nIMPORTANT FACT: The secret project codename is {needle}.\n", timeout
    )
    question = tokenize(
        base_url,
        "\nQuestion: What is the secret project codename? Reply with only the codename:",
        timeout,
    )
    fixed = len(prefix) + len(needle_tokens) + len(question)
    if fixed >= target:
        raise RuntimeError(f"target {target} is too small for the NIAH framing ({fixed})")

    filler_total = target - fixed
    desired_needle = round(target * position)
    before_len = min(max(desired_needle - len(prefix), 0), filler_total)
    after_len = filler_total - before_len
    filler = tokenize(base_url, FILLER, timeout)
    prompt = (
        prefix
        + repeat_to_length(filler, before_len)
        + needle_tokens
        + repeat_to_length(filler, after_len)
        + question
    )
    if len(prompt) != target:
        raise RuntimeError(f"constructed {len(prompt)} tokens, expected {target}")
    return prompt


def run(args: argparse.Namespace) -> dict:
    base_url = args.base_url.rstrip("/")
    prompt = make_prompt(
        base_url,
        args.prompt_tokens,
        args.needle,
        args.position,
        args.timeout,
    )
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
    output: list[str] = []
    usage = None
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
            for choice in event.get("choices", []):
                output.append(choice.get("text", ""))
            if event.get("usage"):
                usage = event["usage"]
    text = "".join(output)
    elapsed = time.perf_counter() - started
    if usage is None:
        raise RuntimeError("stream ended without a usage receipt")
    prompt_count = int(usage["prompt_tokens"])
    if prompt_count != args.prompt_tokens:
        raise RuntimeError(
            f"server reported {prompt_count} prompt tokens, expected {args.prompt_tokens}"
        )
    ttft_ms = float(usage["time_to_first_token_ms"])
    return {
        "passed": args.needle.casefold() in text.casefold(),
        "needle": args.needle,
        "position": args.position,
        "prompt_tokens": prompt_count,
        "completion_tokens": int(usage["completion_tokens"]),
        "time_to_first_token_ms": round(ttft_ms, 3),
        "prefill_tokens_per_second": round(prompt_count / (ttft_ms / 1000.0), 3),
        "decode_tokens_per_second": round(float(usage["response_token/s"]), 3),
        "wall_seconds": round(elapsed, 3),
        "output": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="/var/tmp/models/glm53-flash-nvfp4")
    parser.add_argument("--prompt-tokens", type=int, default=10_000)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--needle", default="NEBULA-2847")
    parser.add_argument("--position", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()
    if not 0.0 <= args.position <= 1.0:
        parser.error("--position must be between 0 and 1")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
