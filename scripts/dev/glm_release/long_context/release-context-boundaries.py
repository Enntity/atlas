#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded over-context/cancel/reuse supplement; no server launch or reconfigure.

Run separately in the SAME frozen nonspec 4096/8192/16384 profile as the main
quality driver. This imports its exact SHA-pinned adjacent client/validators;
the launcher must pin BOTH Python files and the Python executable. No code/tool
execution. All cancellation generation is capped at256, and client-close is
recorded only after actual streamed text with a response ID, before termination.
Successful subsequent answers/tools are reuse evidence, NOT proof of immediate
GPU memory release or forced cancellation completion. Server traces remain the
authority for occupancy and teardown.
"""

import argparse
import concurrent.futures
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import threading
import time


def load_quality(expected):
    path = Path(__file__).resolve().with_name("release-context-quality.py")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(expected) != 64 or actual != expected:
        raise ValueError("quality helper hash mismatch")
    spec = importlib.util.spec_from_file_location("context_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exchange(client, label, body, expected_status, cancel=False):
    """Bounded special-status/SSE request; raw bytes saved on EVERY outcome."""
    started = time.monotonic()
    end = min(client.end, started + client.args.timeout)
    conn = http.client.HTTPConnection(client.host, client.port,
                                     timeout=min(client.args.timeout, client.remaining()))
    response, raw, pending, observed = None, bytearray(), b"", []
    receipt = {"label": label, "request": body, "passed": False}
    try:
        conn.request("POST", "/v1/completions", json.dumps(body).encode(),
                     {"Content-Type": "application/json"})
        q.require(time.monotonic() < end, "deadline before response")
        transport = conn.sock
        transport.settimeout(max(.001, end - time.monotonic()))
        response = conn.getresponse()
        receipt["status"] = response.status
        q.require(response.status == expected_status, "unexpected HTTP status")
        if cancel:
            q.require(response.getheader("Content-Type", "").startswith("text/event-stream"),
                      "cancellation requires actual SSE response")
        while True:
            q.require(time.monotonic() < end, "deadline before read")
            if transport.fileno() >= 0:
                transport.settimeout(max(.001, end - time.monotonic()))
            part = response.read1(4096)
            q.require(time.monotonic() < end, "deadline after read")
            if not part:
                break
            raw.extend(part)
            q.require(len(raw) <= 131072, "special response byte cap")
            if not cancel:
                continue
            pending += part
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if not line.startswith(b"data: "):
                    continue
                q.require(line != b"data: [DONE]", "stream completed before cancellation")
                event = q.strict_json(line[6:])
                q.require("error" not in event, "SSE error before cancellation")
                for choice in event.get("choices", []):
                    q.require(choice.get("finish_reason") is None,
                              "terminal finish observed before cancellation")
                    text = choice.get("text", "")
                    if text:
                        rid = event.get("id")
                        q.require(isinstance(rid, str) and 0 < len(rid) <= 1024,
                                  "missing actual streamed response ID")
                        observed.append({"id": rid, "text": text})
            if observed:
                # Shut down the actual connected transport, not an unrelated
                # cancellation endpoint. Do not drain the response to completion.
                transport.shutdown(socket.SHUT_RDWR)
                receipt.update(passed=True, observed=observed,
                               disposition="client closed after real text; no terminal SSE observed")
                return receipt
        q.require(not cancel, "EOF before cancellable real text")
        data = q.strict_json(raw)
        message = data.get("error", {}).get("message", "")
        expected = f"Prompt too long: {len(body['prompt'])} tokens exceeds max_seq_len {client.args.context_limit}"
        q.require(message == expected, "400 did not establish exact configured context boundary")
        q.require(not data.get("choices"), "invalid prompt unexpectedly generated choices")
        receipt.update(passed=True, response=data)
        return receipt
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        if response is not None:
            response.close()
        conn.close()
        receipt["seconds"] = round(time.monotonic() - started, 6)
        receipt["raw_response"] = raw.decode("utf-8", errors="replace")
        client.save(label + ".json", receipt)


def cancellation_wave(client, count):
    prepared = []
    for index in range(count):
        tokens = client.tokens(
            f"Write a long detailed numbered checklist for exploring garden {q.NAMES[index]}. "
            "Include at least sixty distinct steps and explain each one. Begin immediately.\nChecklist:\n1.")
        q.require(len(tokens) + 256 <= client.args.context_limit, "cancel request budget")
        prepared.append({"model": client.args.model, "prompt": tokens, "max_tokens": 256,
                         "temperature": 0, "seed": 1, "stream": True,
                         "stream_options": {"include_usage": True}})
    barrier = threading.Barrier(count)
    def issue(index):
        barrier.wait(timeout=min(30, client.remaining()))
        return exchange(client, f"cancel-c{count}-r{index}", prepared[index], 200, True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        receipts = list(pool.map(issue, range(count)))
    client.request("post-cancel-health", "/health")
    return [{"label": r["label"], "observed": r["observed"], "seconds": r["seconds"]}
            for r in receipts]


def answers(client, count):
    prepared = []
    for index in range(count):
        body = q.chat_body(client, [{"role": "user", "content":
                            f"Calculate 17 * 9 + {index}. Reply only with the integer."}], [], q.CAP)
        probe, reference = client.request("reuse-answer-calibration", "/v1/chat/completions",
                                          {**body, "max_tokens": 1})
        counted = q.usage(probe, None, 1, client.args.context_limit)["prompt_tokens"]
        q.require(counted + q.CAP <= client.args.context_limit, "reuse answer context budget")
        prepared.append({"kind": "reuse-answer", "index": index,
                         "expected": str(153 + index), "input_tokens": counted, "body": body,
                         "calibrations": [{"receipt": reference, "prompt_tokens": counted}]})
    return q.wave(client, prepared)


def tools(client, count):
    prepared = []
    for index in range(count):
        name, key = "lookup_reuse_" + str(index), "REUSE-" + q.NAMES[index]
        schema = [{"type": "function", "function": {"name": name,
                   "description": "Return the requested case result.",
                   "parameters": {"type": "object", "additionalProperties": False,
                                  "properties": {"case_id": {"type": "string"}},
                                  "required": ["case_id"]}}}]
        body = q.chat_body(client, [{"role": "user", "content":
                             f"Call {name} once with case_id {key}. Wait for the result; do not guess."}], schema, 192)
        probe, reference = client.request("reuse-tool-calibration", "/v1/chat/completions",
                                          {**body, "max_tokens": 1})
        counted = q.usage(probe, None, 1, client.args.context_limit)["prompt_tokens"]
        q.require(counted + 192 <= client.args.context_limit, "reuse tool context budget")
        prepared.append({"kind": "auto-tool", "index": index, "body": body,
                         "input_tokens": counted, "calibration_receipt": reference,
                         "tool_name": name, "tool_args": {"case_id": key},
                         "expected": q.RESULTS[index]})
    first = q.wave(client, prepared)
    # The shared helper retains the ACTUAL assistant message, call ID and the
    # literal result; tools are empty and choice none for the final answer.
    return q.wave(client, [q.prepare_tool(client, i, first[i]) for i in range(count)])


def main():
    global q
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-limit", type=int, choices=(4096, 8192, 16384), required=True)
    parser.add_argument("--quality-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--deadline", type=int, default=600)
    args = parser.parse_args()
    q = load_quality(args.quality_sha256)
    q.require(len(args.concurrency) == len(set(args.concurrency)), "duplicate concurrency")
    q.require(1 <= args.timeout <= args.deadline <= 1800, "bounded positive deadlines required")
    client = q.Client(args)
    report = {"passed": False, "context_limit": args.context_limit,
              "quality_sha256": args.quality_sha256, "concurrency": args.concurrency,
              "cancel_output_cap": 256, "waves": [],
              "scope": "actual over-context400; observed real stream then client close; subsequent answer/tool reuse. Health is not proof of GPU release or cancellation completion."}
    try:
        models, _ = client.request("identity", "/v1/models")
        q.require(any(m.get("id") == args.model for m in models.get("data", [])), "model identity mismatch")
        client.request("initial-health", "/health")
        filler = client.tokens("The archive records routine garden observations. ")
        for count in (args.context_limit, args.context_limit + 1):
            prompt = (filler * ((count + len(filler) - 1) // len(filler)))[:count]
            body = {"model": args.model, "prompt": prompt, "max_tokens": 1,
                    "temperature": 0, "stream": False}
            exchange(client, "invalid-context-" + str(count), body, 400)
            client.request("post-invalid-health", "/health")
        for count in args.concurrency:
            result = {"concurrency": count, "cancelled_streams": cancellation_wave(client, count)}
            report["waves"].append(result)
            result["answers"] = answers(client, count)
            result["tool_followups"] = tools(client, count)
            result["passed"] = True
        client.request("final-health", "/health")
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        try:
            client.request("failure-health", "/health")
        except Exception as health_error:
            report["failure_health_error"] = str(health_error)
    finally:
        client.save("summary.json", report)
        print(json.dumps({"passed": report["passed"], "context_limit": args.context_limit,
                          "concurrency": args.concurrency, "error": report.get("error"),
                          "output_directory": str(client.directory)}), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
