# SPDX-License-Identifier: AGPL-3.0-only
"""Streaming performance baseline harness for Atlas /v1/chat/completions.

Issues one streaming request per workload-repeat against an OpenAI-compatible
endpoint and records per request: wall start, time-to-first-content-token
(TTFT), every streamed chunk arrival offset, inter-token gaps, TPOT,
end-to-end latency, server-reported usage (requests carry
``stream_options.include_usage``) and ``finish_reason``.

Measurement discipline (.claude/skills/measurement-discipline/SKILL.md):

* No wall-clock tok/s is ever computed. ``decode_tok_per_s`` is derived from
  TPOT only (decode span / (completion_tokens - 1)), so prefill never leaks
  into a decode rate. TTFT covers queue + network + prefill and is reported
  separately, never divided into a rate.
* TTFT is the first ``delta.content`` token. ``first_output_token_s`` is the
  first output-bearing chunk of any kind (content, reasoning content or tool
  calls) so reasoning workloads do not mislabel thinking tokens as content.
* Default workloads use non-counting prompts; counting/enumeration prompts
  inflate decode statistics and must not headline a baseline.
* Every raw record and summary carries a provenance status. A complete
  fingerprint is labelled ``fingerprinted``; this is not model qualification.
* Initial server/cache state is explicit. ``cold_start`` is null when the
  initial state is unknown, and ``first_request`` records request order alone.
"""

import argparse
import hashlib
import json
import math
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path

HARNESS_VERSION = "1.1"

# Caller-supplied fingerprint keys required for a fingerprinted summary. Names
# follow the provenance convention in tests/exl3_validation/README.md.
REQUIRED_FINGERPRINT_KEYS = (
    "image_id",
    "source_commit",
    "serve_flags",
    "environment",
    "hosts",
    "checkpoint_revision",
)

PROMPT_CLASSES = ("prose", "code", "tool-call", "reasoning-on", "reasoning-off")

DEFAULT_WORKLOADS = Path(__file__).with_name("perf_workloads.json")


def estimate_tokens(text):
    """Approximate token count (~4 chars/token); usage.prompt_tokens is truth."""
    return max(1, round(len(text) / 4))


def round6(value):
    return None if value is None else round(value, 6)


def percentile(values, p):
    """Nearest-rank percentile; None for empty input."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, math.ceil(p / 100.0 * len(vals)) - 1)
    return vals[k]


def percentiles(values):
    vals = [v for v in values if v is not None]
    return {
        "n": len(vals),
        "p50": round6(percentile(vals, 50)),
        "p95": round6(percentile(vals, 95)),
        "p99": round6(percentile(vals, 99)),
    }


def missing_fingerprint_keys(fingerprint, model_id=None):
    """Required keys that are absent/empty in the caller fingerprint, plus the
    harness-side model id resolved from /v1/models."""
    missing = []
    for key in REQUIRED_FINGERPRINT_KEYS:
        value = (fingerprint or {}).get(key)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(key)
    if not model_id:
        missing.append("model_id")
    return missing


def harness_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_workloads(path):
    """Load and validate the workload matrix; fail fast on any gap (PCND)."""
    doc = json.loads(Path(path).read_text())
    classes = doc.get("prompt_classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"{path}: prompt_classes must be a nonempty object")
    for name, cls in list(classes.items()):
        if "paragraphs" not in cls and "paragraphs_ref" in cls:
            ref = cls["paragraphs_ref"]
            if ref not in classes or "paragraphs" not in classes[ref]:
                raise ValueError(f"prompt class {name!r}: bad paragraphs_ref {ref!r}")
            cls = dict(cls)
            cls["paragraphs"] = classes[ref]["paragraphs"]
            classes[name] = cls
        if not cls.get("task") or not cls.get("paragraphs"):
            raise ValueError(f"prompt class {name!r}: needs task and paragraphs")
    workloads = doc.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError(f"{path}: workloads must be a nonempty list")
    seen = set()
    for wl in workloads:
        name = wl.get("name")
        for key in ("name", "prompt_class", "approx_input_tokens", "max_tokens",
                    "temperature", "seed", "repeats"):
            if key not in wl:
                raise ValueError(f"workload {name!r}: missing {key}")
        if name in seen:
            raise ValueError(f"workload {name!r}: duplicate name")
        seen.add(name)
        if wl["prompt_class"] not in PROMPT_CLASSES:
            raise ValueError(f"workload {name!r}: unknown prompt_class "
                             f"{wl['prompt_class']!r}")
        if wl["prompt_class"] not in classes:
            raise ValueError(f"workload {name!r}: prompt_class "
                             f"{wl['prompt_class']!r} has no prompt_classes entry")
        if not isinstance(wl["repeats"], int) or wl["repeats"] < 3:
            raise ValueError(f"workload {name!r}: repeats must be >= 3 "
                             "(measurement discipline: n>=3)")
        if not isinstance(wl["approx_input_tokens"], int) or wl["approx_input_tokens"] < 1:
            raise ValueError(f"workload {name!r}: approx_input_tokens must be >= 1")
        if not isinstance(wl["max_tokens"], int) or wl["max_tokens"] < 1:
            raise ValueError(f"workload {name!r}: max_tokens must be >= 1")
    return doc


def build_prompt(class_doc, approx_tokens):
    """Expand the class task with its paragraphs to the approximate length.

    The last paragraph is truncated at a word boundary so short targets do
    not overshoot by a whole paragraph.
    """
    text = class_doc["task"]
    paragraphs = class_doc["paragraphs"]
    i = 0
    while estimate_tokens(text) < approx_tokens:
        para = paragraphs[i % len(paragraphs)]
        i += 1
        room_chars = (approx_tokens - estimate_tokens(text)) * 4
        if len(para) > room_chars:
            para = (para[:room_chars].rsplit(" ", 1)[0] or para[:room_chars])
        text = text + "\n\n" + para
    return text


def build_body(model, workload, class_doc, prompt):
    body = {}
    body.update(class_doc.get("request_extra") or {})
    body.update(workload.get("request_extra") or {})
    body.update({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": workload["max_tokens"],
        "temperature": workload["temperature"],
        "seed": workload["seed"],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    return body


def fetch_models(endpoint, timeout):
    """Return (advertised model ids, error string or None)."""
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/models")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.loads(resp.read())
        ids = [m.get("id") for m in doc.get("data", []) if m.get("id")]
        return ids, None
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal here
        return [], f"{type(exc).__name__}: {exc}"


def stream_chat(endpoint, body, timeout):
    """POST one streaming chat request and collect arrival timings.

    Returns a dict with raw timing fields plus ``error`` (None on success).
    All offsets are seconds since request start (monotonic), with the absolute
    wall-clock start preserved separately.
    """
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    wall_start_epoch = time.time()
    t0 = time.perf_counter()
    chunk_offsets = []
    token_offsets = []
    output_text = []
    ttft = None
    first_output = None
    finish_reason = None
    usage = None
    error = None
    done = False
    event_name = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "strict").rstrip("\r\n")
                if not line:
                    event_name = None
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                    continue
                if not line.startswith("data:"):
                    raise ValueError(f"invalid SSE field: {line[:80]!r}")
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    done = True
                    break
                off = time.perf_counter() - t0
                if not payload:
                    raise ValueError("empty SSE data payload")
                if event_name == "error":
                    raise ValueError(f"server SSE error: {payload[:200]}")
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid SSE JSON: {exc.msg}") from exc
                if not isinstance(chunk, dict):
                    raise ValueError("SSE JSON payload must be an object")
                if chunk.get("error") is not None:
                    raise ValueError(f"server SSE error: {chunk['error']}")
                chunk_offsets.append(off)
                choices = chunk.get("choices") or []
                if not isinstance(choices, list):
                    raise ValueError("SSE choices must be an array")
                if choices and not isinstance(choices[0], dict):
                    raise ValueError("SSE choice must be an object")
                delta = (choices[0].get("delta") or {}) if choices else {}
                if not isinstance(delta, dict):
                    raise ValueError("SSE delta must be an object")
                has_output = bool(delta.get("content")
                                  or delta.get("reasoning_content")
                                  or delta.get("tool_calls"))
                for key in ("reasoning_content", "content"):
                    if isinstance(delta.get(key), str):
                        output_text.append(key[0] + delta[key])
                if has_output:
                    if first_output is None:
                        first_output = off
                    token_offsets.append(off)
                if delta.get("content") and ttft is None:
                    ttft = off
                if choices and choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                if chunk.get("usage") is not None:
                    if not isinstance(chunk["usage"], dict):
                        raise ValueError("SSE usage must be an object")
                    usage = chunk["usage"]
                event_name = None
        if not done:
            raise ValueError("SSE stream ended before [DONE]")
    except Exception as exc:  # noqa: BLE001 - recorded per request
        error = f"{type(exc).__name__}: {exc}"
    e2e = time.perf_counter() - t0

    # Partial streams are diagnostic failures, never samples. Preserve the
    # failure and elapsed wall time while withholding every derived statistic.
    if error is not None:
        chunk_offsets = []
        token_offsets = []
        ttft = None
        first_output = None
        finish_reason = None
        usage = None

    decode_span = (token_offsets[-1] - token_offsets[0]) if len(token_offsets) >= 2 else None
    completion = (usage or {}).get("completion_tokens")
    tpot = None
    if decode_span is not None and isinstance(completion, int) and completion >= 2:
        tpot = decode_span / (completion - 1)
    gaps = [token_offsets[i + 1] - token_offsets[i]
            for i in range(len(token_offsets) - 1)]
    return {
        "error": error,
        "output_sha256": hashlib.sha256("\x00".join(output_text).encode()).hexdigest() if error is None else None,
        "output_head": "".join(t[1:] for t in output_text)[:400] if error is None else None,
        "wall_start_epoch": wall_start_epoch,
        "ttft_s": round6(ttft),
        "first_output_token_s": round6(first_output),
        "chunk_offsets_s": [round6(o) for o in chunk_offsets],
        "token_chunk_offsets_s": [round6(o) for o in token_offsets],
        "inter_token_gaps_s": [round6(g) for g in gaps],
        "decode_span_s": round6(decode_span),
        # Decode-only rate: 1/TPOT. No wall tok/s exists anywhere in this
        # harness (prefill must never mix into a decode rate).
        "tpot_s": round6(tpot),
        "decode_tok_per_s": round6(None if not tpot else 1.0 / tpot),
        "e2e_s": round6(e2e),
        "usage": usage,
        "finish_reason": finish_reason,
    }


def run_one(endpoint, model, workload, class_doc, prompt, timeout,
            client_id, cold_start, repeat_index, first_request, provenance_status):
    res = stream_chat(endpoint,
                      build_body(model, workload, class_doc, prompt),
                      timeout)
    return {
        "provenance_status": provenance_status,
        "workload": workload["name"],
        "prompt_class": workload["prompt_class"],
        "repeat_index": repeat_index,
        "client_id": client_id,
        "cold_start": cold_start,
        "first_request": first_request,
        "first_in_workload": repeat_index == 0,
        "wall_start_epoch": res["wall_start_epoch"],
        "wall_start_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(res["wall_start_epoch"])),
        "request": {
            "max_tokens": workload["max_tokens"],
            "temperature": workload["temperature"],
            "seed": workload["seed"],
            "approx_input_tokens": workload["approx_input_tokens"],
            "prompt_chars": len(prompt),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            # Effective reasoning/tool policy, so a raw record alone shows the
            # deliberate thinking mode (measurement-discipline rule 1).
            "request_extra": {
                **(class_doc.get("request_extra") or {}),
                **(workload.get("request_extra") or {}),
            },
        },
        "ttft_s": res["ttft_s"],
        "first_output_token_s": res["first_output_token_s"],
        "chunk_offsets_s": res["chunk_offsets_s"],
        "token_chunk_offsets_s": res["token_chunk_offsets_s"],
        "inter_token_gaps_s": res["inter_token_gaps_s"],
        "decode_span_s": res["decode_span_s"],
        "tpot_s": res["tpot_s"],
        "decode_tok_per_s": res["decode_tok_per_s"],
        "e2e_s": res["e2e_s"],
        "usage": res["usage"],
        "finish_reason": res["finish_reason"],
        "output_sha256": res.get("output_sha256"),
        "output_head": res.get("output_head"),
        "error": res["error"],
    }


def summarize(records):
    """Per-workload p50/p95/p99 over successful requests, in workload order."""
    order = []
    grouped = {}
    for rec in records:
        if rec["workload"] not in grouped:
            grouped[rec["workload"]] = []
            order.append(rec["workload"])
        grouped[rec["workload"]].append(rec)
    out = {}
    for name in order:
        recs = grouped[name]
        ok = [r for r in recs if r["error"] is None]
        finishes = {}
        for rec in recs:
            if rec["finish_reason"]:
                finishes[rec["finish_reason"]] = finishes.get(rec["finish_reason"], 0) + 1
        out[name] = {
            "prompt_class": recs[0]["prompt_class"],
            "request": recs[0]["request"],
            "n": len(recs),
            "n_ok": len(ok),
            "n_failed": len(recs) - len(ok),
            "cold_starts": sum(1 for r in recs if r["cold_start"] is True),
            "unknown_cold_start_state": sum(1 for r in recs if r["cold_start"] is None),
            "finish_reasons": finishes,
            "prompt_tokens": percentiles([(r["usage"] or {}).get("prompt_tokens") for r in ok]),
            "completion_tokens": percentiles([(r["usage"] or {}).get("completion_tokens") for r in ok]),
            "ttft_s": percentiles([r["ttft_s"] for r in ok]),
            "tpot_s": percentiles([r["tpot_s"] for r in ok]),
            "e2e_s": percentiles([r["e2e_s"] for r in ok]),
            "decode_tok_per_s": percentiles([r["decode_tok_per_s"] for r in ok]),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", required=True, help="server base URL")
    ap.add_argument("--model", required=True, help="model id for the request body")
    ap.add_argument("--workloads", default=str(DEFAULT_WORKLOADS))
    ap.add_argument("--raw-out", required=True, help="per-request JSONL output")
    ap.add_argument("--summary-out", required=True, help="summary JSON output")
    ap.add_argument("--initial-state", required=True,
                    choices=("cold", "warm", "unknown"),
                    help="observed server/cache state before the first request")
    ap.add_argument("--fingerprint", help="caller fingerprint JSON (merged verbatim)")
    ap.add_argument("--unqualified", action="store_true",
                    help="allow a summary with incomplete fingerprint provenance")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="number of concurrent clients (default 1)")
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--workload", action="append",
                    help="run only this workload name (repeatable)")
    args = ap.parse_args(argv)
    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    raw_path = Path(args.raw_out)
    summary_path = Path(args.summary_out)
    if raw_path.resolve() == summary_path.resolve():
        ap.error("--raw-out and --summary-out must be different paths")
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite an existing output path")

    doc = load_workloads(args.workloads)
    classes = doc["prompt_classes"]
    workloads = [w for w in doc["workloads"]
                 if not args.workload or w["name"] in args.workload]
    if not workloads:
        raise ValueError("--workload matched no workloads")

    caller_fp = {}
    if args.fingerprint:
        caller_fp = json.loads(Path(args.fingerprint).read_text())
    advertised, models_error = fetch_models(args.endpoint, args.timeout)
    # PCND: no implicit fallback to advertised[0] — a summary may only claim
    # the model id the server advertises for what was actually requested.
    model_id = args.model if args.model in advertised else None

    missing = missing_fingerprint_keys(caller_fp, model_id)
    summary_allowed = (not missing) or args.unqualified
    provenance_status = "unqualified" if missing else "fingerprinted"

    prompts = {w["name"]: build_prompt(classes[w["prompt_class"]],
                                       w["approx_input_tokens"])
               for w in workloads}
    tasks = queue.Queue()
    for wl in workloads:
        for rep in range(wl["repeats"]):
            tasks.put((wl, rep))

    records = []
    lock = threading.Lock()
    first_request_claimed = False
    with raw_path.open("x") as raw_f:
        def worker(client_id):
            nonlocal first_request_claimed
            while True:
                try:
                    wl, rep = tasks.get_nowait()
                except queue.Empty:
                    return
                with lock:
                    first_request = not first_request_claimed
                    first_request_claimed = True
                cold = (True if args.initial_state == "cold" and first_request else
                        False if args.initial_state == "warm" or not first_request else
                        None)
                try:
                    rec = run_one(args.endpoint, args.model, wl, classes[wl["prompt_class"]],
                                  prompts[wl["name"]], args.timeout,
                                  client_id, cold, rep, first_request,
                                  provenance_status)
                except Exception as exc:  # noqa: BLE001 - never drop queued work
                    rec = {"provenance_status": provenance_status,
                           "workload": wl["name"],
                           "prompt_class": wl["prompt_class"],
                           "repeat_index": rep, "client_id": client_id,
                           "cold_start": cold, "first_request": first_request,
                           "first_in_workload": rep == 0,
                           "wall_start_epoch": None, "wall_start_utc": None,
                           "request": {}, "usage": None, "finish_reason": None,
                           "ttft_s": None, "first_output_token_s": None,
                           "chunk_offsets_s": [], "token_chunk_offsets_s": [],
                           "inter_token_gaps_s": [], "decode_span_s": None,
                           "tpot_s": None, "decode_tok_per_s": None,
                           "e2e_s": None,
                           "error": f"{type(exc).__name__}: {exc}"}
                with lock:
                    records.append(rec)
                    raw_f.write(json.dumps(rec) + "\n")
                    raw_f.flush()

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(args.concurrency)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    summary = {
        "provenance_status": provenance_status,
        "initial_state": args.initial_state,
        "harness": {
            "file": Path(__file__).name,
            "version": HARNESS_VERSION,
            "sha256": harness_sha256(),
        },
        "endpoint": args.endpoint,
        "requested_model": args.model,
        "model_id": model_id,
        "advertised_models": advertised,
        "models_fetch_error": models_error,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cli_args": vars(args),
        "workloads_file": {
            "path": str(args.workloads),
            "sha256": hashlib.sha256(Path(args.workloads).read_bytes()).hexdigest(),
        },
        # Caller-supplied fingerprint merged in verbatim.
        "fingerprint": caller_fp,
        "missing_fingerprint_keys": missing,
        "measurement_notes": (
            "tpot_s = decode_span_s / (completion_tokens - 1); "
            "decode_tok_per_s is decode-only (1/tpot_s); no wall-clock tok/s "
            "is computed, so prefill never mixes into a decode rate. TTFT is "
            "the first delta.content token; first_output_token_s also counts "
            "reasoning/tool-call chunks."
        ),
        "totals": {
            "requests": len(records),
            "ok": sum(1 for r in records if r["error"] is None),
            "failed": sum(1 for r in records if r["error"] is not None),
            "cold_starts": sum(1 for r in records if r["cold_start"] is True),
            "unknown_cold_start_state": sum(1 for r in records if r["cold_start"] is None),
        },
        "workloads": summarize(records),
    }
    if summary_allowed:
        with summary_path.open("x") as summary_f:
            summary_f.write(json.dumps(summary, indent=2) + "\n")
    else:
        print(f"refusing to write {args.summary_out}: missing required "
              f"fingerprint keys {missing}; supply them via --fingerprint or "
              "pass --unqualified", file=sys.stderr)
    if not summary_allowed:
        return 2
    return 3 if summary["totals"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
