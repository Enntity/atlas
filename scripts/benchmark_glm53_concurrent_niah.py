#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Independent-needle concurrency check; mixed output caps exercise batch drain.

This is a behavioral check, not proof that every decode step was co-batched.
Correlate with server multi-sequence traces when validating scheduler coverage.
The client barrier precedes tokenization, not the completion POST or GPU launch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import threading

NEEDLE_NAMES = ("AURORA", "NEBULA", "ORBIT", "QUASAR", "PULSAR", "COMET", "ZENITH", "LYRA")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="/var/tmp/models/glm53-flash-nvfp4")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[2047, 2048, 2049])
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[32, 64, 96])
    parser.add_argument("--position", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--context-limit", type=int, help=(
        "server's per-request context cap; required for four or more requests; reject "
        "prompt+output beyond this limit before making any HTTP requests"
    ))
    args = parser.parse_args(argv)
    count = len(args.prompt_tokens)
    if not 1 <= count <= len(NEEDLE_NAMES) or len(args.output_tokens) != count:
        parser.error("supply one to eight prompt lengths and matching output caps")
    if min(args.output_tokens + args.prompt_tokens) < 1 or args.repetitions < 1:
        parser.error("token counts and repetitions must be positive")
    if not 0 <= args.position <= 1:
        parser.error("position must be between zero and one")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if count >= 4 and args.context_limit is None:
        parser.error("four or more requests require --context-limit matching the server cap")
    if args.context_limit is not None:
        if args.context_limit < 1:
            parser.error("context-limit must be positive")
        if any(prompt + output > args.context_limit for prompt, output in
               zip(args.prompt_tokens, args.output_tokens)):
            parser.error("each prompt length plus output cap must fit context-limit")
    return args


def needles_for(count: int, repetition: int) -> list[str]:
    if not 1 <= count <= len(NEEDLE_NAMES) or repetition < 0:
        raise ValueError("needles require one to eight rows and a nonnegative repetition")
    return [f"{name}-{6193 + repetition * 17}" for name in NEEDLE_NAMES[:count]]


def main() -> None:
    args = parse_args()
    # Keep argument validation CPU-only and usable without HTTP dependencies.
    from benchmark_glm53_niah import run

    count = len(args.prompt_tokens)

    all_passed = True
    for repetition in range(args.repetitions):
        needles = needles_for(count, repetition)
        barrier = threading.Barrier(count)

        def request(index: int) -> dict:
            row_args = argparse.Namespace(
                base_url=args.base_url, model=args.model,
                prompt_tokens=args.prompt_tokens[index],
                max_tokens=args.output_tokens[index], needle=needles[index],
                position=args.position, timeout=args.timeout,
            )
            barrier.wait(timeout=30)
            receipt = run(row_args)
            receipt["foreign_needles"] = [
                needle for needle in needles if needle != needles[index]
                and needle.casefold() in receipt["output"].casefold()
            ]
            receipt["passed"] = receipt["passed"] and not receipt["foreign_needles"]
            return receipt

        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            rows = list(pool.map(request, range(count)))
        passed = all(row["passed"] for row in rows)
        all_passed = all_passed and passed
        print(json.dumps({"repetition": repetition, "passed": passed,
                          "context_limit": args.context_limit,
                          "scope": "Barrier precedes tokenization; server traces establish decode occupancy.",
                          "requests": rows}), flush=True)
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
