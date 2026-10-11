# SPDX-License-Identifier: AGPL-3.0-only
"""F2 concurrency qualification harness: interleaved == isolated.

Runs a declared set of greedy requests (temperature 0, fixed max_tokens,
``return_token_ids`` and top-k logprobs when the server accepts them):

1. ISOLATED: one at a time, ``--isolated-repeats`` times. Repeats that differ
   mean the server is not deterministic even alone, which is reported as a
   separate failure so it is not blamed on batching.
2. INTERLEAVED: for each C in ``--concurrency`` (default 2 and 4) every case
   is started ``--stagger`` seconds after the previous one, at most C in
   flight, so new prefills land while earlier requests decode. The phase
   must show real overlap: some request must receive its first token while
   another is still decoding. Equality under a server that silently runs
   requests one at a time proves nothing about batching, so such a phase
   fails ("server appears to serialize requests"); that is the expected
   result on a launcher that admits a single sequence. A phase whose
   requests were not even sent concurrently (stagger too long) also fails.
3. CANCEL: the same at the largest C, with one victim stream closed after
   ``--cancel-after`` output chunks while the others continue. The victim's
   prefix and every survivor must match isolated output, and an isolated
   repeat of the victim afterwards must still match.

Every interleaved output is compared with its isolated output token for
token (``token_ids``, else logprob tokens, else text; the basis is recorded)
and the first divergent position plus the logprob gap there is reported.
Each request has an overall deadline (``--timeout``), so a hang is a recorded
failure, not a stuck run. ``/health`` must return 200 after every phase.

Measurement labels: per-request TTFT/TPOT follow perf_baseline (TPOT is the
decode span / (completion_tokens - 1)). The aggregate rate is completion
tokens over the phase wall window; it mixes prefill, queueing and decode and
is named ``aggregate_completion_tok_per_s_wall_unqualified``, which is a load
indicator, never a decode rate or a qualified performance claim.

The receipt (``--out``, never overwritten) carries endpoint, model, C list,
cases hash, harness git sha/dirty flag and file hashes, UTC time and the
caller fingerprint. Exit status: 0 pass, 1 any failure.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import harness_client as hc

HARNESS_VERSION = "1.0"

# Short, varied prompts whose greedy continuations are long enough to keep
# several requests decoding at once. Prompt lengths differ so prefill and
# decode of different sizes overlap.
DEFAULT_CASES = [
    {"name": "prose-lighthouse",
     "messages": [{"role": "user", "content":
                   "Describe, in plain prose, a lighthouse keeper's evening routine."}]},
    {"name": "code-rust-sort",
     "messages": [{"role": "user", "content":
                   "Write a Rust function that sorts a Vec<u32> with insertion sort, "
                   "then explain its complexity."}]},
    {"name": "list-rivers",
     "messages": [{"role": "user", "content":
                   "Name several major rivers of Europe and one fact about each."}]},
    {"name": "long-context-summary",
     "messages": [{"role": "user", "content": (
         "Summarise the following notes in three sentences.\n\n" + " ".join(
             f"Note {i}: the depot received shipment {i * 37 % 101} of copper "
             f"wire and logged it in bay {i % 7}." for i in range(60)))}]},
    {"name": "json-shape",
     "messages": [{"role": "user", "content":
                   "Produce a JSON object describing a fictional library with "
                   "three books, each with title, author and year."}]},
    {"name": "math-explain",
     "messages": [{"role": "user", "content":
                   "Explain step by step why the sum of the first n odd numbers is n squared."}]},
]


def load_cases(path):
    """Cases file: {"cases": [{"name", "messages", optional "request_extra",
    optional "max_tokens"}]}. Fails fast on gaps or duplicates."""
    if path is None:
        return [dict(c) for c in DEFAULT_CASES]
    doc = json.loads(Path(path).read_text())
    cases = doc.get("cases") if isinstance(doc, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: cases must be a nonempty list")
    seen = set()
    for case in cases:
        name = case.get("name")
        if not name or name in seen:
            raise ValueError(f"{path}: case names must be unique and nonempty ({name!r})")
        seen.add(name)
        if not isinstance(case.get("messages"), list) or not case["messages"]:
            raise ValueError(f"case {name!r}: messages must be a nonempty list")
    return cases


def build_body(model, case, params):
    body = {}
    body.update(params.get("request_extra") or {})
    body.update(case.get("request_extra") or {})
    body.update({
        "model": model,
        "messages": case["messages"],
        "max_tokens": case.get("max_tokens", params["max_tokens"]),
        "temperature": 0,
        "seed": params["seed"],
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    })
    if params.get("top_logprobs"):
        body["logprobs"] = True
        body["top_logprobs"] = params["top_logprobs"]
    return body


def run_isolated(base_url, model, cases, params, timeout):
    """Run every case alone, ``isolated_repeats`` times. If the very first
    request is rejected with a 4xx while logprobs were requested, logprobs
    are disabled for the whole run (recorded in params)."""
    runs = []
    for rep in range(params["isolated_repeats"]):
        records = {}
        for case in cases:
            res = hc.stream_chat(base_url, build_body(model, case, params), timeout)
            if (not runs and not records and params.get("top_logprobs")
                    and res.status is not None and 400 <= res.status < 500):
                k = params["top_logprobs"]
                params["top_logprobs"] = 0
                retry = hc.stream_chat(base_url, build_body(model, case, params), timeout)
                if retry.ok:
                    # Logprobs, not the case, were rejected: run without them.
                    params["logprobs_rejected"] = res.error
                    res = retry
                else:
                    params["top_logprobs"] = k
            records[case["name"]] = res.to_record()
            records[case["name"]]["repeat"] = rep
        runs.append(records)
    return runs


def overlap_stats(records):
    """Evidence that requests were actually processed together.

    Counts ordered pairs (a, b) of requests where, inside a's decode window
    (a's first output chunk to a's completion):

    * ``sent_during_other_decode``: b was sent. This is client-side only:
      it proves the harness offered concurrent load.
    * ``first_token_during_other_decode``: b's first output chunk arrived.
      This is server-side evidence: the server produced tokens for two
      streams in the same window. A server that queues requests and runs
      them one at a time scores zero here, however the client sends them.

    ``max_in_flight`` is the peak number of requests between send and
    completion (client view).
    """
    spans = [r for r in records if r.get("t_start_rel") is not None
             and r.get("t_end_rel") is not None]
    sent = first = 0
    for a in spans:
        if a.get("t_first_rel") is None:
            continue
        for b in spans:
            if b is a:
                continue
            if a["t_first_rel"] < b["t_start_rel"] < a["t_end_rel"]:
                sent += 1
            if (b.get("t_first_rel") is not None
                    and a["t_first_rel"] < b["t_first_rel"] < a["t_end_rel"]):
                first += 1
    events = sorted([(r["t_start_rel"], 1) for r in spans]
                    + [(r["t_end_rel"], -1) for r in spans], key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return {"sent_during_other_decode": sent, "first_token_during_other_decode": first,
            "max_in_flight": peak}


def run_interleaved(base_url, model, cases, params, timeout, concurrency, stagger,
                    victim=None, cancel_after=None):
    """Start case i at t0 + i*stagger with at most ``concurrency`` in flight.

    When ``victim`` names a case, that stream is closed after
    ``cancel_after`` output chunks.
    """
    slots = threading.BoundedSemaphore(concurrency)
    t0 = time.perf_counter()
    out = {}
    lock = threading.Lock()

    def worker(i, case):
        delay = t0 + i * stagger - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        with slots:
            res = hc.stream_chat(base_url, build_body(model, case, params), timeout,
                                 cancel_after_chunks=(cancel_after if case["name"] == victim
                                                      else None))
        rec = res.to_record()
        rec["t_start_rel"] = hc.round6(res.t_start - t0)
        rec["t_first_rel"] = hc.round6(None if res.t_first is None else res.t_first - t0)
        rec["t_end_rel"] = hc.round6(res.t_end - t0)
        with lock:
            out[case["name"]] = rec

    threads = [threading.Thread(target=worker, args=(i, c), daemon=True)
               for i, c in enumerate(cases)]
    for th in threads:
        th.start()
    for th in threads:
        # stream_chat enforces its own deadline; the join bound only guards
        # against a harness bug and still reports the stuck case.
        th.join(timeout + stagger * len(cases) + 30)
    for case in cases:
        out.setdefault(case["name"], {"error": "harness: worker did not finish",
                                      "cancelled": False})
    return out


def phase_summary(records, isolated, victim=None):
    """Compare a phase against isolated output; return (summary, failures)."""
    failures = []
    comparisons = {}
    ok = [r for r in records.values() if r.get("error") is None and not r.get("cancelled")]
    for name, rec in records.items():
        if name == victim:
            if rec.get("error") is not None:
                failures.append(f"{name}: victim request failed: {rec['error']}")
                continue
            if not rec.get("cancelled"):
                failures.append(f"{name}: cancellation not exercised (stream finished first)")
            cmp = hc.compare_sequences(isolated[name], rec, prefix_only=True)
            comparisons[name] = cmp
            if not cmp["equal"]:
                failures.append(f"{name}: cancelled prefix diverges at "
                                f"{cmp['first_divergence']} ({cmp['basis']})")
            continue
        if rec.get("error") is not None:
            failures.append(f"{name}: {rec['error']}")
            continue
        cmp = hc.compare_sequences(isolated[name], rec)
        comparisons[name] = cmp
        if not cmp["equal"]:
            failures.append(
                f"{name}: diverges from isolated at {cmp['first_divergence']} "
                f"({cmp['basis']}: {cmp['reference_token']!r} vs {cmp['candidate_token']!r}, "
                f"logprob gap {cmp['logprob_gap']}, finish {cmp['finish_reason_mismatch']})")
    starts = [r["t_start_rel"] for r in ok if r.get("t_start_rel") is not None]
    ends = [r["t_end_rel"] for r in ok if r.get("t_end_rel") is not None]
    completion = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in ok)
    window = (max(ends) - min(starts)) if starts and ends else None
    summary = {
        "comparisons": comparisons,
        "overlap": overlap_stats(list(records.values())),
        "ttft_s": hc.percentiles([r.get("ttft_s") for r in ok]),
        "tpot_s": hc.percentiles([r.get("tpot_s") for r in ok]),
        "completion_tokens_total": completion,
        "phase_wall_window_s": hc.round6(window),
        "aggregate_completion_tok_per_s_wall_unqualified":
            hc.round6(completion / window if window else None),
    }
    if summary["overlap"]["sent_during_other_decode"] == 0:
        failures.append("phase not interleaved: no request was sent while another "
                        "was decoding (lower --stagger or raise max_tokens)")
    elif summary["overlap"]["first_token_during_other_decode"] == 0:
        failures.append("no overlapping streams observed: no request produced its first "
                        "token while another was decoding; the server appears to "
                        "serialize requests")
    return summary, failures


def run_f2(base_url, model, cases, *, max_tokens, top_logprobs, seed, concurrency,
           stagger, timeout, isolated_repeats, cancel_after, victim=None,
           request_extra=None):
    """Run every phase and return (verdict dict, list of failure strings)."""
    params = {"max_tokens": max_tokens, "top_logprobs": top_logprobs, "seed": seed,
              "isolated_repeats": isolated_repeats,
              "request_extra": request_extra if request_extra is not None
              else {"chat_template_kwargs": {"enable_thinking": False}}}
    failures = []
    health = {}

    def check(label):
        healthy, detail = hc.check_health(base_url)
        health[label] = {"healthy": healthy, "detail": detail}
        if not healthy:
            failures.append(f"server unhealthy after {label}: {detail}")

    iso_runs = run_isolated(base_url, model, cases, params, timeout)
    isolated = iso_runs[0]
    for name, rec in isolated.items():
        if rec.get("error") is not None:
            failures.append(f"isolated {name}: {rec['error']}")
    isolated_repeat_cmp = {}
    for rep, run in enumerate(iso_runs[1:], start=1):
        for name, rec in run.items():
            if rec.get("error") is not None:
                failures.append(f"isolated repeat {rep} {name}: {rec['error']}")
                continue
            cmp = hc.compare_sequences(isolated[name], rec)
            isolated_repeat_cmp[f"{name}#{rep}"] = cmp
            if not cmp["equal"]:
                failures.append(f"isolated nondeterminism: {name} repeat {rep} diverges "
                                f"at {cmp['first_divergence']} ({cmp['basis']})")
    check("isolated")

    phases = {}
    if not any(f.startswith("isolated ") for f in failures):
        for c in concurrency:
            recs = run_interleaved(base_url, model, cases, params, timeout, c, stagger)
            summary, fails = phase_summary(recs, isolated)
            phases[f"interleaved_c{c}"] = {"concurrency": c, "records": recs, **summary}
            failures.extend(f"interleaved C={c}: {f}" for f in fails)
            check(f"interleaved_c{c}")
        c = max(concurrency)
        victim = victim or cases[0]["name"]
        recs = run_interleaved(base_url, model, cases, params, timeout, c, stagger,
                               victim=victim, cancel_after=cancel_after)
        summary, fails = phase_summary(recs, isolated, victim=victim)
        phases[f"cancel_c{c}"] = {"concurrency": c, "victim": victim,
                                  "cancel_after_chunks": cancel_after,
                                  "records": recs, **summary}
        failures.extend(f"cancel C={c}: {f}" for f in fails)
        check(f"cancel_c{c}")
        # The cancelled request's slot/state must be reusable: an isolated
        # repeat of the victim must reproduce its original output.
        victim_case = next(cs for cs in cases if cs["name"] == victim)
        rep = hc.stream_chat(base_url, build_body(model, victim_case, params),
                             timeout).to_record()
        cmp = (hc.compare_sequences(isolated[victim], rep)
               if rep.get("error") is None else None)
        phases["post_cancel_repeat"] = {"victim": victim, "record": rep, "comparison": cmp}
        if rep.get("error") is not None:
            failures.append(f"post-cancel repeat of {victim}: {rep['error']}")
        elif not cmp["equal"]:
            failures.append(f"post-cancel repeat of {victim} diverges at "
                            f"{cmp['first_divergence']} ({cmp['basis']})")
        check("post_cancel_repeat")
    else:
        failures.append("interleaved phases skipped: isolated reference incomplete")

    bases = sorted({cmp["basis"] for ph in phases.values()
                    for cmp in (ph.get("comparisons") or {}).values()})
    result = {
        "request_params": params,
        "isolated": iso_runs,
        "isolated_repeat_comparisons": isolated_repeat_cmp,
        "phases": phases,
        "health": health,
        "comparison_bases": bases,
        "token_level": bool(bases) and "text" not in bases,
    }
    if bases and not result["token_level"]:
        failures.append("comparison fell back to text: no token_ids or logprob tokens "
                        "from the server, so equality is not token-level evidence")
    return result, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", required=True, help="server base URL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="receipt JSON (never overwritten)")
    ap.add_argument("--cases", help="cases JSON file (default: built-in cases)")
    ap.add_argument("--fingerprint", help="caller fingerprint JSON (merged verbatim)")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--top-logprobs", type=int, default=5, help="0 disables logprobs")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--stagger", type=float, default=0.25,
                    help="seconds between interleaved request starts")
    ap.add_argument("--cancel-after", type=int, default=4,
                    help="output chunks before the victim stream is closed")
    ap.add_argument("--victim", help="case name to cancel (default: first case)")
    ap.add_argument("--isolated-repeats", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="overall per-request deadline in seconds")
    ap.add_argument("--thinking", action="store_true",
                    help="leave thinking enabled (default sends enable_thinking=false)")
    args = ap.parse_args(argv)
    args.concurrency = sorted(set(args.concurrency))
    if any(c < 2 for c in args.concurrency):
        ap.error("--concurrency values must be >= 2")
    if args.isolated_repeats < 1 or args.max_tokens < 1 or args.cancel_after < 1:
        ap.error("--isolated-repeats, --max-tokens and --cancel-after must be >= 1")
    if args.cancel_after >= args.max_tokens:
        ap.error("--cancel-after must be below --max-tokens to exercise cancellation")
    hc.refuse_existing(args.out)

    cases = load_cases(args.cases)
    if args.victim and args.victim not in {c["name"] for c in cases}:
        ap.error(f"--victim {args.victim!r} is not a case name")
    caller_fp = json.loads(Path(args.fingerprint).read_text()) if args.fingerprint else {}
    models, models_error = hc.fetch_models(args.endpoint)
    advertised = [m.get("id") for m in models]
    model_id = args.model if args.model in advertised else None
    git_sha, git_dirty = hc.git_state()
    started = hc.utc_now()
    result, failures = run_f2(
        args.endpoint, args.model, cases, max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs, seed=args.seed, concurrency=args.concurrency,
        stagger=args.stagger, timeout=args.timeout,
        isolated_repeats=args.isolated_repeats, cancel_after=args.cancel_after,
        victim=args.victim,
        request_extra={} if args.thinking else None)
    missing = hc.missing_fingerprint_keys(caller_fp, model_id)
    here = Path(__file__).resolve().parent
    receipt = {
        "harness": {"file": Path(__file__).name, "version": HARNESS_VERSION,
                    "sha256": hc.file_sha256(Path(__file__)),
                    "client_sha256": hc.file_sha256(here / "harness_client.py"),
                    "git_sha": git_sha, "git_dirty": git_dirty},
        "fingerprint": {
            "endpoint": args.endpoint,
            "model": args.model,
            "model_id": model_id,
            "concurrency": args.concurrency,
            "cases_sha256": hc.canonical_sha256(cases),
            "harness_git_sha": git_sha,
            "utc_start": started,
            "utc_end": hc.utc_now(),
        },
        "caller_fingerprint": caller_fp,
        "missing_fingerprint_keys": missing,
        "provenance_status": "unqualified" if missing else "fingerprinted",
        "advertised_models": models,
        "models_fetch_error": models_error,
        "cli_args": vars(args),
        "cases": cases,
        "measurement_notes": (
            "ttft_s is the first output-bearing chunk; tpot_s = decode span / "
            "(completion_tokens - 1). aggregate_completion_tok_per_s_wall_unqualified "
            "divides completion tokens by the phase wall window and mixes queueing, "
            "prefill and decode: a load indicator, not a performance claim."),
        **result,
        "failures": failures,
        "verdict": "fail" if failures else "pass",
    }
    hc.write_json_exclusive(args.out, receipt)
    print(f"F2 {receipt['verdict']}: {len(failures)} failure(s); receipt {args.out}")
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
