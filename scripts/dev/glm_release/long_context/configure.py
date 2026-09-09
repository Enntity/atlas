#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve local workload paths/hashes into a new runner JSON; no node contact."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("duplicate concurrency")
    for path in (args.input, args.output, args.python):
        if not path.is_absolute() or path != path.resolve():
            parser.error("canonical absolute local paths without symlinks required")
    if args.output.exists() or args.output.is_relative_to(HERE.parents[3]):
        parser.error("new output outside checkout required")
    spec = importlib.util.spec_from_file_location("long_context_runner", HERE / "long-context-runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if args.input.stat().st_size > 65536:
        parser.error("input exceeds64KiB")
    config = json.loads(args.input.read_text(), object_pairs_hook=runner.unique)
    if config.get("workload") != "GENERATED_LOCAL_WORKLOAD":
        parser.error("template workload must be GENERATED_LOCAL_WORKLOAD; no hidden overrides")
    runner.digest(args.python_sha256)
    if runner.file_hash(args.python) != args.python_sha256:
        parser.error("explicit Python executable digest mismatch")
    suite, quality, boundary = [HERE / name for name in
        ("release-context-suite.py", "release-context-quality.py", "release-context-boundaries.py")]
    pins = {str(p): runner.file_hash(p) for p in (args.python, suite, quality, boundary)}
    config["workload"] = {
        "argv": [str(args.python), str(suite), "--base-url", config["api_url"],
                 "--model", runner.MODEL, "--context-limit", str(config["context"]),
                 "--concurrency", *map(str, args.concurrency), "--output-root",
                 str(Path(config["output_directory"]) / "quality-receipts"),
                 "--python", str(args.python), "--python-sha256", args.python_sha256,
                 "--quality-sha256", pins[str(quality)],
                 "--boundaries-sha256", pins[str(boundary)], "--total-seconds", "1800"],
        "files_sha256": pins, "timeout_seconds": 1800}
    runner.validate(config)
    data = (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
    if len(data) > 65536:
        parser.error("generated input exceeds64KiB")
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        os.fchmod(output.fileno(), 0o600)
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    print(args.output)


if __name__ == "__main__":
    main()
