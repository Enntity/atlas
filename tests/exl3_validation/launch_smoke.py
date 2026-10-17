#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Launch smoke test for Atlas GLM-5.3-Flash EXL3 TP2 serving.

Greedy short completion, long generation, multi-turn, and sampled requests
against an OpenAI-compatible /v1/chat/completions endpoint. Exit nonzero if
any request fails or returns empty text; coherence is judged by the operator
from the saved transcript, not by heuristics.
"""
import argparse
import json
import sys
import time
import urllib.request

def chat(endpoint, model, messages, temperature=None, max_tokens=256, seed=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature if temperature is not None else 0.0}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    choice = out["choices"][0]
    text = choice["message"]["content"] or ""
    usage = out.get("usage", {})
    return {"text": text, "finish_reason": choice.get("finish_reason"),
            "usage": usage, "wall_s": round(dt, 2)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    failures = []

    def run(name, messages, **kw):
        print(f"--- {name} ...", flush=True)
        try:
            r = chat(args.endpoint, args.model, messages, **kw)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: request failed: {e}")
            print(f"    FAILED: {e}", flush=True)
            return None
        print(f"    {r['wall_s']}s finish={r['finish_reason']} "
              f"out_tokens={r['usage'].get('completion_tokens')}", flush=True)
        print("    " + r["text"][:400].replace("\n", "\n    "), flush=True)
        if not r["text"].strip():
            failures.append(f"{name}: empty text")
        results.append({"name": name, **r})
        return r

    # 1. short greedy
    run("short-greedy",
        [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        max_tokens=64)

    # 2. long generation for coherence (a few hundred tokens)
    run("long-generation",
        [{"role": "user", "content": "Write a detailed 400-word explanation of how "
          "TCP congestion control works, aimed at a systems engineer."}],
        max_tokens=700)

    # 3. multi-turn
    run("multi-turn-first",
        [{"role": "user", "content": "My name is Dana and I keep bees. Remember that."}],
        max_tokens=64)
    first = results[-1]["text"] if results else ""
    run("multi-turn-followup",
        [{"role": "user", "content": "My name is Dana and I keep bees. Remember that."},
         {"role": "assistant", "content": first},
         {"role": "user", "content": "What is my name and what do I do? Answer in one sentence."}],
        max_tokens=64)

    # 4. sampled request (temperature > 0)
    run("sampled",
        [{"role": "user", "content": "Give me one short lighthearted joke."}],
        temperature=0.8, max_tokens=128, seed=42)

    report = {"endpoint": args.endpoint, "model": args.model,
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "results": results, "failures": failures}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{len(results)} ok, {len(failures)} failed; wrote {args.out}")
    if failures:
        print("FAILURES:", *failures, sep="\n  ")
        sys.exit(1)

if __name__ == "__main__":
    main()
