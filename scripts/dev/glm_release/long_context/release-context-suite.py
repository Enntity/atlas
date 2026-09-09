#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-fast quality then boundary supplement in one unchanged server profile.

The outer runner must independently pin this wrapper, both sibling scripts and
Python, and bind these explicit arguments to the actual launched profile. No
server launch/configuration or tool execution. Child stdout/stderr are inherited
by the outer bounded runner log; raw HTTP receipts occupy fresh child directories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def pinned(path, expected):
    if (not path.is_absolute() or len(expected) != 64
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise ValueError("pinned executable/script mismatch: " + str(path))


def stop(child):
    if child is None or child.poll() is not None:
        return
    # The group belongs solely to this exact unreaped, start_new_session child.
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("base-url", "model", "output-root", "python", "python-sha256",
                 "quality-sha256", "boundaries-sha256"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--context-limit", type=int, choices=(4096, 8192, 16384), required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), nargs="+", required=True)
    parser.add_argument("--total-seconds", type=int, required=True)
    args = parser.parse_args()
    if not 1501 <= args.total_seconds <= 1800 or len(set(args.concurrency)) != len(args.concurrency):
        parser.error("total-seconds must be1501..1800 and concurrency entries unique")
    here, python = Path(__file__).resolve().parent, Path(args.python)
    quality, boundary = here / "release-context-quality.py", here / "release-context-boundaries.py"
    pins = ((python, args.python_sha256), (quality, args.quality_sha256),
            (boundary, args.boundaries_sha256))
    for path, sha in pins:
        pinned(path, sha)
    root = Path(args.output_root)
    if not root.is_absolute():
        parser.error("absolute new output-root required")
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = time.monotonic()
    # Reserve four seconds for scoped child termination/reaping before the
    # outer total deadline. Never start another HTTP-generating child afterward.
    end = started + args.total_seconds - 4
    result = {"passed": False, "context_limit": args.context_limit,
              "concurrency": args.concurrency, "total_seconds": args.total_seconds,
              "pins": {str(path): sha for path, sha in pins}, "children": []}
    child = None
    def interrupted(signum, _frame):
        raise InterruptedError("suite received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for name, script in (("quality", quality), ("boundaries", boundary)):
            left = int(end - time.monotonic())
            if left < 1:
                raise TimeoutError("whole-suite deadline; no further child issued")
            budget = min(1500 if name == "quality" else 300, left)
            for path, sha in pins:
                pinned(path, sha)
            budget = min(budget, int(end - time.monotonic()))
            if budget < 1:
                raise TimeoutError("deadline during pin validation; no child issued")
            command = [str(python), str(script), "--base-url", args.base_url,
                       "--model", args.model, "--context-limit", str(args.context_limit),
                       "--output-dir", str(root / name), "--concurrency",
                       *map(str, args.concurrency), "--timeout", str(min(600 if name == "quality" else 180, budget)),
                       "--deadline", str(budget)]
            if name == "boundaries":
                command += ["--quality-sha256", args.quality_sha256]
            row = {"phase": name, "argv": command, "budget_seconds": budget}
            result["children"].append(row)
            print(json.dumps({"suite_phase": name, "budget_seconds": budget}), flush=True)
            began = time.monotonic()
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True,
                                     env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"})
            row["exit_code"] = child.wait(timeout=min(budget, end - time.monotonic()))
            row["seconds"] = round(time.monotonic() - began, 6)
            child = None
            if row["exit_code"] != 0:
                raise RuntimeError(name + " failed; later HTTP phase not issued")
            if time.monotonic() >= end:
                raise TimeoutError("child completion crossed whole-suite deadline")
        result["passed"] = True
    except BaseException as error:
        result["error"] = repr(error)
    finally:
        # Preserve the primary failure if scoped cleanup itself fails.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            stop(child)
        except BaseException as error:
            result["passed"] = False
            result["cleanup_error"] = repr(error)
        result["seconds"] = round(time.monotonic() - started, 6)
        with open(root / "suite-summary.json", "x", encoding="utf-8") as output:
            json.dump(result, output, allow_nan=False)
        print(json.dumps(result, allow_nan=False), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
