#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Strict offline C1 receipt correlation; no inference or metric replacement."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
START = re.compile(r"Chunked prefill start: (\d+) prompt tokens, chunk_size=(\d+), max_tokens=(\d+)")
DONE = re.compile(r"Done: (\d+) tokens \(([^)]+)\)")
TIMING = re.compile(r"MTP verify timing \[(\d+) steps, seq_len=(\d+)\]:")
FLOAT_FIELDS = ("serial", "mtp", "p1", "mean_na", "tok_step")
COUNT_FIELDS = ("regime_reprobes", "depth_switches")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=1):
    require(type(value) is int and minimum <= value <= 2**63 - 1,
            f"invalid {label}: require bounded integer >= {minimum}")
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"invalid finite {label}")
    return value


def validate_benchmark(document):
    require(isinstance(document, list) and len(document) == 1,
            "require exactly one C1 benchmark result")
    result = document[0]
    require(isinstance(result, dict) and type(result.get("concurrency")) is int
            and result["concurrency"] == 1, "only C1 benchmark supported")
    workload = result.get("workload")
    require(isinstance(workload, dict), "missing workload metadata")
    prompt = integer(workload.get("prompt_tokens"), "prompt_tokens")
    cap = integer(workload.get("output_tokens_requested_per_stream"), "output cap")
    warmups = integer(workload.get("warmup_batches_per_concurrency"), "warmup count", 0)
    repetitions = integer(workload.get("measured_batches_per_concurrency"), "repetition count")
    integer(warmups + repetitions, "total request count")
    runs = result.get("runs")
    require(isinstance(runs, list) and len(runs) == repetitions,
            "measured run count does not match metadata")
    caps = []
    for index, run in enumerate(runs):
        require(isinstance(run, dict) and type(run.get("concurrency")) is int
                and run["concurrency"] == 1, f"run {index} must be C1")
        require(run.get("prompt_tokens_each") == prompt
                and run.get("output_tokens_requested_total") == cap,
                f"run {index} workload does not match metadata")
        requests = run.get("requests")
        require(isinstance(requests, list) and len(requests) == 1
                and isinstance(requests[0], dict), f"run {index} must contain one request")
        count = integer(requests[0].get("completion_tokens"), "completion count")
        require(count <= cap and run.get("completion_tokens_total") == count,
                f"run {index} completion count mismatch")
        reached = count == cap
        require(run.get("all_outputs_reached_cap") is reached, f"run {index} cap integrity mismatch")
        finite(run.get("aggregate_e2e_tps"), "run fullwall rate")
        caps.append(reached)
    require(result.get("all_outputs_reached_cap") is all(caps), "summary cap integrity mismatch")
    finite(result.get("median_aggregate_e2e_tps"), "original fullwall median")
    metrics = {key: value for key, value in result.items() if key.startswith("median_")}
    for key, value in metrics.items():
        finite(value, key)
    metrics["all_outputs_reached_cap"] = result["all_outputs_reached_cap"]
    return result, workload, metrics, prompt, cap, warmups, repetitions


def timing_record(line, line_number):
    match = TIMING.search(line)
    require(match is not None and int(match[1]) == 25,
            f"unsupported/malformed timing at line {line_number}")
    record = {"line": line_number, "steps": 25, "seq_len": int(match[2])}
    for field in ("fwd", "propose"):
        values = re.findall(rf"\b{field}=([^\s(]+)ms\(x([^\s)]+)\)", line)
        require(len(values) == 1, f"missing/duplicate timing {field} at line {line_number}")
        try:
            value, calls = (float(x) for x in values[0])
        except ValueError as exc:
            raise ValueError(f"invalid timing {field} at line {line_number}") from exc
        record[field + "_ms"] = finite(value, f"timing {field}")
        record[field + "_calls_per_step"] = finite(calls, f"timing {field} calls")
    return record


def done_diagnostics(line, line_number):
    result = {}
    for key in (*FLOAT_FIELDS, *COUNT_FIELDS, "depth"):
        values = re.findall(rf"\b{key}=([^\s,]+)", line)
        require(len(values) == 1, f"missing/duplicate diagnostic {key} at line {line_number}")
        raw = values[0]
        if key == "depth":
            require(bool(re.fullmatch(r"k\d+", raw)), f"unsupported diagnostic depth {raw}")
            result[key] = raw
        elif key in COUNT_FIELDS:
            require(raw.isascii() and raw.isdecimal(), f"invalid diagnostic {key}")
            result[key] = integer(int(raw), f"diagnostic {key}", 0)
        else:
            try:
                result[key] = finite(float(raw), f"diagnostic {key}")
            except ValueError as exc:
                raise ValueError(f"invalid diagnostic {key} at line {line_number}") from exc
            if key in ("serial", "mtp", "p1"):
                require(result[key] <= 1, f"invalid diagnostic fraction {key}")
    return result


def parse_windows(log_text, prompt, cap):
    windows, active = [], None
    for number, raw in enumerate(log_text.splitlines(), 1):
        line = ANSI.sub("", raw)
        if "Chunked prefill start:" in line:
            require(active is None, f"interleaved request starts at line {number}")
            match = START.search(line)
            require(match is not None, f"malformed request start at line {number}")
            p, chunk, c = (integer(int(x), "log request shape") for x in match.groups())
            active = {"start_line": number, "prompt_tokens": p, "output_cap": c,
                      "chunk_size": chunk, "timings": [], "matched": (p, c) == (prompt, cap)}
        elif "MTP verify timing [" in line:
            require(active is not None, f"orphan timing at line {number}")
            if active["matched"]:
                active["timings"].append(timing_record(line, number))
        elif "Done:" in line:
            require(active is not None, f"orphan Done at line {number}")
            match = DONE.search(line)
            require(match is not None, f"malformed Done at line {number}")
            count = integer(int(match[1]), "log completion count")
            require(count <= active["output_cap"], f"log completion count exceeds cap at line {number}")
            active.update(end_line=number, completion_tokens=count, finish_reason=match[2])
            if active["matched"]:
                active.update(done_diagnostics(line, number))
            windows.append(active)
            active = None
    require(active is None, "incomplete request at end of log")
    return windows


def summarize(document, log_text):
    result, workload, metrics, prompt, cap, warmups, repetitions = validate_benchmark(document)
    windows = parse_windows(log_text, prompt, cap)
    matching = [row for row in windows if row["matched"]]
    require(len(matching) == warmups + repetitions,
            f"matching request count: expected {warmups + repetitions}, found {len(matching)}; no last-N guessing")
    requests = []
    for ordinal, window in enumerate(matching):
        warmup = ordinal < warmups
        record = {key: value for key, value in window.items() if key not in ("matched", "timings")}
        record.update(ordinal=ordinal, warmup=warmup,
                      discarded_timing_windows=min(1, len(window["timings"])),
                      clean_timing_windows=window["timings"][1:])
        if not warmup:
            run_index = ordinal - warmups
            run = result["runs"][run_index]
            require(window["completion_tokens"] == run["completion_tokens_total"],
                    f"measured run {run_index} log/benchmark completion count mismatch")
            record["benchmark_run_index"] = run_index
            record["benchmark_aggregate_e2e_tps"] = run["aggregate_e2e_tps"]
        requests.append(record)
    clean = [window for request in requests if not request["warmup"]
             for window in request["clean_timing_windows"]]
    return {
        "correlation": "C1 prompt-length/output-cap/order only; not prompt/token identity proof",
        "timing_policy": "discard first 25-step window per request; host diagnostics, not GPU event timings",
        "workload": dict(workload), "benchmark_metrics": metrics,
        "unmatched_complete_requests": len(windows) - len(matching),
        "requests": requests,
        "clean_measured_timing_window_count": len(clean),
        "diagnostic_median_fwd_ms": statistics.median(w["fwd_ms"] for w in clean) if clean else None,
        "diagnostic_median_propose_ms": statistics.median(w["propose_ms"] for w in clean) if clean else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument("rank0_log", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(json.loads(args.benchmark_json.read_text()), args.rank0_log.read_text())
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
