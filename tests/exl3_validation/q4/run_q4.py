#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Q4 task-quality runner and comparator.

Run every case of a frozen corpus N times against an OpenAI-compatible chat
endpoint, score each trial with the case's objective checker, keep every raw
request/response, and report per-category pass rates with Wilson 95%
intervals against the provisional thresholds in thresholds.json:

  python3 tests/exl3_validation/q4/run_q4.py --endpoint http://HOST:8888 \
      --model MODEL --profile serial-qualified --trials 5 --out runs/q4/serial-1

Compare two runs of the same corpus (exit 1 when a category regresses, i.e.
B's interval lies entirely below A's):

  python3 tests/exl3_validation/q4/run_q4.py --compare A/summary.json B/summary.json

Scoring is checker-only. Nothing is compared with a reference engine's
output, and a failed request (HTTP error, malformed body) is a failed trial,
never skipped, so neither a shared reference error nor an outage can count
as a correct answer.
"""
import argparse
import hashlib
import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checkers import CHECKERS, check_code  # noqa: E402

DEFAULT_CASES = os.path.join(HERE, "corpus.jsonl")
THRESHOLDS = os.path.join(HERE, "thresholds.json")


def wilson_interval(successes, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def chat_url(endpoint):
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("endpoint must be an http(s) URL")
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        return endpoint
    if parsed.path not in ("", "/"):
        raise ValueError("endpoint must be a server origin or a .../chat/completions URL")
    return endpoint.rstrip("/") + "/v1/chat/completions"


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    if not cases:
        raise ValueError("no cases in " + path)
    ids = [case.get("id") for case in cases]
    bad = [cid for cid in ids if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", cid)]
    if bad:
        raise ValueError(f"case ids must be [A-Za-z0-9_-]+ (used as file names): {bad}")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in " + path)
    for case in cases:
        if case.get("checker") not in CHECKERS:
            raise ValueError(f"{case.get('id')}: unknown checker {case.get('checker')!r}")
    return cases


def build_payload(case, model):
    payload = {"model": model, "messages": case["messages"], "max_tokens": case["max_tokens"],
               "stream": False}
    payload.update(case["sampling"])
    for key in ("tools", "tool_choice", "response_format"):
        if key in case:
            payload[key] = case[key]
    if "thinking" in case:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(case["thinking"])}
    return payload


def post_json(url, payload, timeout):
    """Return (http_status, body or None, error or None); never raises for
    HTTP/transport/JSON failures, which are recorded as failed trials."""
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        return error.code, None, f"HTTP {error.code}: {error.read()[:2000].decode('utf-8', 'replace')}"
    except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
        return None, None, f"transport: {type(error).__name__}: {error}"
    try:
        return status, json.loads(raw), None
    except ValueError as error:
        return status, None, f"body is not JSON: {error}"


def response_view(body):
    """Normalise one chat completion for the checkers; raises ValueError on a
    body that is not a well-formed single-choice completion."""
    if not isinstance(body, dict) or "error" in body:
        raise ValueError("error body: " + json.dumps(body)[:500])
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("choice has no message")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("message.content must be a string or null")
    return {"content": content or "", "tool_calls": message.get("tool_calls") or None,
            "finish_reason": choices[0].get("finish_reason")}


def score(case, view, tmp_root):
    checker = case["checker"]
    if checker == "code":
        return check_code(view, case["checker_params"], tmp_root=tmp_root)
    return CHECKERS[checker](view, case["checker_params"])


def run_case(case, url, model, trials, timeout, tmp_root):
    payload = build_payload(case, model)
    rows = []
    for trial in range(1, trials + 1):
        started = time.time()
        status, body, error = post_json(url, payload, timeout)
        row = {"trial": trial, "http_status": status, "elapsed_s": round(time.time() - started, 3),
               "raw_response": body, "error": error, "passed": False}
        if error is None:
            try:
                view = response_view(body)
            except ValueError as bad:
                row["error"] = "malformed response: " + str(bad)
            else:
                row["view"] = view
                row["usage"] = body.get("usage")
                try:
                    result = score(case, view, tmp_root)
                except Exception as bad:  # a checker crash is a failed trial, not a lost run
                    result = {"passed": False, "reason": f"checker error: {type(bad).__name__}: {bad}"}
                row["passed"], row["reason"] = bool(result["passed"]), result["reason"]
        if row["error"] is not None:
            row["reason"] = "request failed: " + row["error"]
        rows.append(row)
    return {"id": case["id"], "category": case["category"], "request": payload, "trials": trials,
            "passes": sum(row["passed"] for row in rows), "outputs": rows}


def summarize(case_results, thresholds, trials):
    categories = {}
    for result in case_results:
        bucket = categories.setdefault(result["category"],
                                       {"passes": 0, "n": 0, "cases": 0, "cases_majority_passed": 0})
        bucket["passes"] += result["passes"]
        bucket["n"] += result["trials"]
        bucket["cases"] += 1
        bucket["cases_majority_passed"] += 2 * result["passes"] > result["trials"]
    verdicts = {}
    for name, bucket in sorted(categories.items()):
        low, high = wilson_interval(bucket["passes"], bucket["n"])
        bucket["pass_rate"] = bucket["passes"] / bucket["n"]
        bucket["wilson95"] = [low, high]
        # Trials of one temperature-0 case are strongly correlated, so the
        # pooled interval above is optimistic about precision. The case-level
        # interval (a case counts when most of its trials passed) treats the
        # independent unit as the case; read both before calling a change.
        bucket["case_wilson95"] = list(wilson_interval(bucket["cases_majority_passed"], bucket["cases"]))
        rule = thresholds.get(name)
        if not isinstance(rule, dict):
            verdict = "NO_THRESHOLD"
        elif trials < rule["min_trials"]:
            verdict = "INSUFFICIENT_TRIALS"
        else:
            verdict = "PASS" if bucket["pass_rate"] >= rule["pass_rate"] else "FAIL"
        bucket["threshold"] = rule if isinstance(rule, dict) else None
        bucket["verdict"] = verdicts[name] = verdict
    return categories, verdicts


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def run(args):
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    url = chat_url(args.endpoint)
    tmp_root = args.tmp_root or os.environ.get("TMPDIR")
    if not tmp_root or not os.path.isdir(tmp_root):
        raise SystemExit("set TMPDIR (or --tmp-root) to an existing directory for the code checker")
    cases = load_cases(args.cases)
    with open(THRESHOLDS, encoding="utf-8") as handle:
        thresholds = json.load(handle)
    # Refuse to overwrite any earlier run.
    os.makedirs(args.out, exist_ok=False)
    case_dir = os.path.join(args.out, "cases")
    os.mkdir(case_dir)
    results = []
    for case in cases:
        result = run_case(case, url, args.model, args.trials, args.timeout, tmp_root)
        with open(os.path.join(case_dir, case["id"] + ".json"), "x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, ensure_ascii=False)
        results.append(result)
        print(f"{case['id']:6s} {result['passes']}/{result['trials']}", flush=True)
    categories, verdicts = summarize(results, thresholds, args.trials)
    summary = {"profile": args.profile, "model": args.model, "endpoint": url, "trials": args.trials,
               "cases_file": os.path.abspath(args.cases), "cases_sha256": sha256_file(args.cases),
               "thresholds_sha256": sha256_file(THRESHOLDS),
               "started_utc": args.started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "categories": categories,
               "cases": {r["id"]: {"category": r["category"], "passes": r["passes"], "trials": r["trials"]}
                         for r in results}}
    with open(os.path.join(args.out, "summary.json"), "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    for name, bucket in categories.items():
        low, high = bucket["wilson95"]
        clow, chigh = bucket["case_wilson95"]
        print(f"{name:13s} {bucket['passes']:4d}/{bucket['n']:<4d} rate={bucket['pass_rate']:.3f} "
              f"wilson95=[{low:.3f},{high:.3f}] cases={bucket['cases_majority_passed']}/{bucket['cases']} "
              f"case_wilson95=[{clow:.3f},{chigh:.3f}] {bucket['verdict']}")
    return 0 if all(v == "PASS" for v in verdicts.values()) else 1


def compare(path_a, path_b):
    with open(path_a, encoding="utf-8") as handle:
        a = json.load(handle)
    with open(path_b, encoding="utf-8") as handle:
        b = json.load(handle)
    if a.get("cases_sha256") != b.get("cases_sha256"):
        raise SystemExit("refusing to compare runs of different case files")
    regressions = []
    print(f"A={a.get('profile')} ({path_a})\nB={b.get('profile')} ({path_b})")
    for name in sorted(set(a["categories"]) | set(b["categories"])):
        ca, cb = a["categories"].get(name), b["categories"].get(name)
        if ca is None or cb is None:
            print(f"{name:13s} present in only one run")
            regressions.append(name)
            continue
        (al, ah), (bl, bh) = ca["wilson95"], cb["wilson95"]
        diff = cb["pass_rate"] - ca["pass_rate"]
        if bh < al:
            flag = "REGRESSION"
            regressions.append(name)
        elif bl > ah:
            flag = "IMPROVED"
        else:
            flag = "overlap"
        if "case_wilson95" in ca and "case_wilson95" in cb and cb["case_wilson95"][1] < ca["case_wilson95"][0]:
            flag += " (also at case level)"
        print(f"{name:13s} A={ca['pass_rate']:.3f}[{al:.3f},{ah:.3f}] "
              f"B={cb['pass_rate']:.3f}[{bl:.3f},{bh:.3f}] diff={diff:+.3f} {flag}")
    flips = sorted(cid for cid, row in a.get("cases", {}).items()
                   if cid in b.get("cases", {}) and row["trials"] and b["cases"][cid]["trials"]
                   and row["passes"] / row["trials"] > b["cases"][cid]["passes"] / b["cases"][cid]["trials"])
    if flips:
        print("cases with a lower pass fraction in B: " + ", ".join(flips))
    return 1 if regressions else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs=2, metavar=("A_SUMMARY", "B_SUMMARY"))
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--out")
    parser.add_argument("--profile")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--tmp-root")
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    missing = [flag for flag in ("endpoint", "model", "out", "profile") if not getattr(args, flag)]
    if missing:
        parser.error("required: " + ", ".join("--" + flag for flag in missing))
    args.started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
