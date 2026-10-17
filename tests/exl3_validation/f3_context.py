# SPDX-License-Identifier: AGPL-3.0-only
"""F3 context and prefix-reuse qualification harness.

(a) Context ladder. For each declared prompt length (templated prompt tokens,
calibrated exactly with Atlas ``POST /tokenize``) a deterministic filler
document hides a passphrase (the needle) and a code line embedding a tool-call
JSON string. The model must return the passphrase and reproduce the code line
byte for byte. TTFT and ``usage.prompt_tokens`` are recorded and
``usage.prompt_tokens`` must equal the calibrated length. Rungs whose prompt
plus ``max_tokens`` would exceed the advertised ``max_model_len`` are not run:
Atlas rejects only ``prompt >= max_model_len`` (api/chat/mod.rs) while vLLM
also counts the requested output (vllm/renderers/params.py:442-450), so that
gap is server-specific and not part of this gate. Beyond the limit (prompt =
max_model_len and max_model_len + 256) the server must answer with a clean
4xx JSON error (not a dropped connection or 5xx), stay healthy, and then
reproduce a small sanity request exactly.

(b) Prefix reuse. A shared long prefix (headed by a per-run nonce so the
first request is a genuine miss) is followed by different suffixes:
cold then warm for suffix 0, then repeated requests for the other suffixes.
Every repeat must equal the first output token for token. Reported
``usage.prompt_tokens_details.cached_tokens`` and the warm/cold TTFT ratio
are recorded. vLLM never counts the last prompt token as a cache hit and hits
are block aligned (vllm/v1/core/kv_cache_manager.py:255-263), so the cached
count is evidence, not an equality gate; ``--require-cache-hit`` makes a
missing or zero warm count a failure. Finally, before each suffix is
repeated, ``--evict-prompts`` unrelated prompts are sent to push the prefix
out of the cache, so each repeat is a miss compared against the earlier
(hit) output. Whether eviction happened is reported (``eviction_observed``
true/false/null, from cached_tokens) but cannot be forced.

The receipt (``--out``, never overwritten) has the same fingerprint fields as
f2_concurrency.py. Exit status: 0 pass, 1 any failure.
"""

import argparse
import http.client
import json
import random
import secrets
import sys
from pathlib import Path

import harness_client as hc

HARNESS_VERSION = "1.0"

DEFAULT_LADDER = [1024, 2047, 2048, 2049, 4000]

FILLER_WORDS = (
    "the river carried silt past quiet farms while clerks counted grain in "
    "stone houses and the harbor lamps burned through long winter nights as "
    "ships returned with copper timber salt wool and letters from distant "
    "towns where markets opened at dawn and closed when the bells rang over "
    "narrow streets full of carts horses merchants children and tired "
    "travelers looking for bread water shelter maps news work and rest"
).split()

CODE_LINE_TEMPLATE = (
    'let k{n} = fnv1a(&[0x2f, 0x9c, 0x{h:02x}]) ^ 0x5bd1e995; '
    '// tool:{{"name":"lookup_v2","arguments":{{"id":"Q-{n}","flags":[1,2]}}}}'
)

QUESTION = (
    "\n\nThe document above contains one passphrase sentence and one line that "
    "starts with CODE:. Reply with exactly two lines and nothing else. Line 1: "
    "the passphrase only. Line 2: the text after 'CODE: ' copied exactly, "
    "character for character."
)


def filler_words(n, seed):
    """First n words of a deterministic, prefix-stable filler stream."""
    rng = random.Random(seed)
    words = []
    for i in range(n):
        word = rng.choice(FILLER_WORDS)
        if i % 13 == 12:
            word += "."
        words.append(word)
    return words


def needle_for(seed):
    rng = random.Random(f"needle-{seed}")
    return f"{rng.choice(['violet', 'amber', 'cobalt', 'saffron'])}-{rng.randrange(1000, 9999)}-" \
           f"{rng.choice(['harbor', 'lantern', 'quarry', 'orchard'])}"


def code_line_for(seed):
    rng = random.Random(f"code-{seed}")
    return CODE_LINE_TEMPLATE.format(n=rng.randrange(1000, 9999), h=rng.randrange(256))


def build_document(n_words, seed, pad=""):
    """Needle at 1/3 depth, code line at 2/3 depth, then ``pad`` at the end."""
    words = filler_words(n_words, seed)
    a, b = n_words // 3, (2 * n_words) // 3
    needle = f"The passphrase is {needle_for(seed)}."
    code = f"\nCODE: {code_line_for(seed)}\n"
    text = " ".join(words[:a]) + f" {needle} " + " ".join(words[a:b]) + code \
        + " ".join(words[b:])
    return "Document:\n" + text + pad + QUESTION


def ladder_messages(n_words, seed, pad=""):
    return [{"role": "user", "content": build_document(n_words, seed, pad)}]


PAD_UNITS = (" .", " ,", " a", " ;", " x", " -")


def calibrate(count_fn, target, seed, max_words=1 << 20):
    """Find (n_words, pad, count) with count_fn(messages) == target.

    Binary-searches the filler length for the largest count <= target, then
    greedily appends short pad units that each add a token. Returns the best
    found and ``exact`` False when ``target`` cannot be hit, which the caller
    treats as a failure: boundary rungs (2047/2048/2049) are meaningless when
    off by one.
    """
    def count(n, pad=""):
        return count_fn(ladder_messages(n, seed, pad))

    base = count(0)
    if base > target:
        return {"n_words": 0, "pad": "", "count": base, "exact": False}
    lo, hi = 0, 1
    while count(hi) <= target:
        lo, hi = hi, hi * 2
        if hi > max_words:
            raise hc.HarnessError(f"cannot reach {target} tokens within {max_words} words")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(mid) <= target:
            lo = mid
        else:
            hi = mid
    pad = ""
    current = count(lo)
    while current < target:
        best = None
        for unit in PAD_UNITS:
            c = count(lo, pad + unit)
            if current < c <= target and (best is None or c > best[1]):
                best = (unit, c)
        if best is None:
            break
        pad += best[0]
        current = best[1]
    return {"n_words": lo, "pad": pad, "count": current, "exact": current == target}


def chat_body(model, messages, max_tokens, seed, extra):
    body = dict(extra)
    body.update({"model": model, "messages": messages, "max_tokens": max_tokens,
                 "temperature": 0, "seed": seed, "n": 1,
                 "stream_options": {"include_usage": True}, "return_token_ids": True})
    return body


SANITY_MESSAGES = [{"role": "user", "content": "Name the three primary colors of light."}]


def run_ladder(base_url, model, *, ladder, max_model_len, max_tokens, seed, extra,
               timeout, allow_inexact):
    failures = []
    count_fn = lambda msgs: hc.tokenize_messages(base_url, model, msgs, extra)  # noqa: E731
    sanity_ref = hc.stream_chat(base_url, chat_body(model, SANITY_MESSAGES, 32, seed, extra),
                                timeout).to_record()
    if sanity_ref["error"] is not None:
        failures.append(f"sanity reference request failed: {sanity_ref['error']}")
    rungs = []
    limit = max_model_len - max_tokens
    for target in ladder:
        if target > limit:
            # A declared rung that cannot run is a failure, never a silent
            # pass; the over-limit probes below cover the rejection path.
            rungs.append({"target_tokens": target, "skipped":
                          f"prompt + max_tokens exceeds max_model_len {max_model_len}"})
            failures.append(f"rung {target}: exceeds max_model_len {max_model_len} - "
                            f"max_tokens {max_tokens}; not run")
            continue
        rung_seed = seed * 100003 + target
        cal = calibrate(count_fn, target, rung_seed)
        rung = {"target_tokens": target, "actual_tokens": cal["count"],
                "exact": cal["exact"], "n_words": cal["n_words"], "pad": cal["pad"]}
        rungs.append(rung)
        if not cal["exact"] and not allow_inexact:
            failures.append(f"rung {target}: calibration reached {cal['count']} tokens, not exact")
            continue
        msgs = ladder_messages(cal["n_words"], rung_seed, cal["pad"])
        rec = hc.stream_chat(base_url, chat_body(model, msgs, max_tokens, seed, extra),
                             timeout).to_record()
        needle, code = needle_for(rung_seed), code_line_for(rung_seed)
        rung.update({
            "record": rec,
            "needle": needle,
            "code_line": code,
            "needle_found": needle in rec["content"],
            "code_exact": code in rec["content"],
            "usage_prompt_tokens": (rec["usage"] or {}).get("prompt_tokens"),
        })
        if rec["error"] is not None:
            failures.append(f"rung {target}: {rec['error']}")
            continue
        if rung["usage_prompt_tokens"] != cal["count"]:
            failures.append(f"rung {target}: usage.prompt_tokens {rung['usage_prompt_tokens']} "
                            f"!= /tokenize count {cal['count']}")
        if not rung["needle_found"]:
            failures.append(f"rung {target}: needle {needle!r} not reproduced")
        if not rung["code_exact"]:
            failures.append(f"rung {target}: code/tool line not reproduced exactly")

    over = []
    for target in (max_model_len, max_model_len + 256):
        rung_seed = seed * 100003 + target
        cal = calibrate(count_fn, target, rung_seed)
        msgs = ladder_messages(cal["n_words"], rung_seed, cal["pad"])
        res = hc.stream_chat(base_url, chat_body(model, msgs, max_tokens, seed, extra), timeout)
        err = res.error_json.get("error") if isinstance(res.error_json, dict) else None
        healthy, detail = hc.check_health(base_url)
        entry = {"target_tokens": target, "actual_tokens": cal["count"],
                 "status": res.status, "error": res.error, "error_body": res.error_body,
                 "clean_4xx": (res.status is not None and 400 <= res.status < 500
                               and isinstance(err, dict)),
                 "healthy_after": healthy, "health_detail": detail}
        over.append(entry)
        if cal["count"] < max_model_len:
            failures.append(f"over-limit {target}: calibration only reached {cal['count']} "
                            "tokens, below max_model_len")
        if not entry["clean_4xx"]:
            failures.append(f"over-limit {target}: expected a 4xx JSON error, got status "
                            f"{res.status} ({res.error})")
        if not healthy:
            failures.append(f"over-limit {target}: server unhealthy afterwards: {detail}")
    sanity_after = hc.stream_chat(base_url, chat_body(model, SANITY_MESSAGES, 32, seed, extra),
                                  timeout).to_record()
    sanity_cmp = None
    if sanity_after["error"] is not None:
        failures.append(f"sanity request after over-limit rungs failed: {sanity_after['error']}")
    elif sanity_ref["error"] is None:
        sanity_cmp = hc.compare_sequences(sanity_ref, sanity_after)
        if not sanity_cmp["equal"]:
            failures.append("sanity request after over-limit rungs diverges at "
                            f"{sanity_cmp['first_divergence']} ({sanity_cmp['basis']})")
    return {"rungs": rungs, "over_limit": over, "sanity_reference": sanity_ref,
            "sanity_after": sanity_after, "sanity_comparison": sanity_cmp}, failures


PREFIX_SUFFIXES = (
    "Question: which goods did the ships return with? Answer in one sentence.",
    "Question: when did the markets open and close? Answer in one sentence.",
    "Question: list three things the travelers looked for. Answer briefly.",
)


def prefix_messages(nonce, prefix_words, suffix, seed):
    head = f"Session {nonce}. Read the record below.\n\n"
    body = " ".join(filler_words(prefix_words, seed))
    return [{"role": "user", "content": head + body + "\n\n" + suffix}]


def run_prefix(base_url, model, *, prefix_words, evict_prompts, evict_words, max_tokens,
               seed, extra, timeout, nonce, require_cache_hit):
    failures = []
    events = []

    def send(label, messages):
        rec = hc.stream_chat(base_url, chat_body(model, messages, max_tokens, seed, extra),
                             timeout).to_record()
        events.append({"label": label, "record": rec})
        if rec["error"] is not None:
            failures.append(f"{label}: {rec['error']}")
        return rec

    def same(label, ref, rec):
        if ref["error"] is not None or rec["error"] is not None:
            return None
        cmp = hc.compare_sequences(ref, rec)
        if cmp["basis"] == "text":
            failures.append(f"{label}: compared as text only; the server returned no "
                            "token_ids, so equality is not token-level evidence")
        if not cmp["equal"]:
            failures.append(f"{label}: diverges at {cmp['first_divergence']} ({cmp['basis']}: "
                            f"{cmp['reference_token']!r} vs {cmp['candidate_token']!r}, "
                            f"logprob gap {cmp['logprob_gap']})")
        return cmp

    msgs = [prefix_messages(nonce, prefix_words, s, seed) for s in PREFIX_SUFFIXES]
    cold = send("cold s0", msgs[0])
    warm = send("warm s0", msgs[0])
    comparisons = {"warm s0 vs cold s0": same("warm s0 vs cold s0", cold, warm)}
    first = [cold]
    for i in range(1, len(msgs)):
        a = send(f"prefix-hit s{i}", msgs[i])
        b = send(f"repeat s{i}", msgs[i])
        comparisons[f"repeat s{i} vs prefix-hit s{i}"] = same(f"repeat s{i} vs prefix-hit s{i}", a, b)
        first.append(a)
    healthy, detail = hc.check_health(base_url)
    if not healthy:
        failures.append(f"server unhealthy after prefix reuse: {detail}")
    # Each post-eviction repeat re-warms the shared prefix, so the cache is
    # refilled before every suffix; otherwise only the first repeat after the
    # fill could be a miss and s1/s2 would compare hit against hit.
    after = []
    hit_cached = [warm["cached_tokens"]] + [
        e["record"]["cached_tokens"] for e in events if e["label"].startswith("repeat s")]
    for i in range(len(msgs)):
        for k in range(evict_prompts):
            send(f"evict s{i} {k}", prefix_messages(
                f"{nonce}-evict-{i}-{k}", evict_words, "Question: what burned at night?",
                seed + 1 + k))
        rec = send(f"post-evict s{i}", msgs[i])
        comparisons[f"post-evict s{i} vs first s{i}"] = same(
            f"post-evict s{i} vs first s{i}", first[i], rec)
        after.append(rec)
    healthy2, detail2 = hc.check_health(base_url)
    if not healthy2:
        failures.append(f"server unhealthy after eviction: {detail2}")

    observed = []
    for before, rec in zip(hit_cached, after):
        if before is None or rec["cached_tokens"] is None:
            observed.append(None)
        else:
            observed.append(rec["cached_tokens"] < before)
    if None in observed:
        eviction_observed = None
    else:
        eviction_observed = all(observed)
    warm_cached = warm["cached_tokens"]
    if require_cache_hit and not warm_cached:
        failures.append(f"warm s0 reported cached_tokens={warm_cached}; a prefix hit was required")
    ratio = (warm["ttft_s"] / cold["ttft_s"]
             if warm["ttft_s"] is not None and cold["ttft_s"] else None)
    return {
        "nonce": nonce,
        "events": events,
        "comparisons": comparisons,
        "cached_tokens": {e["label"]: e["record"]["cached_tokens"] for e in events},
        "warm_over_cold_ttft_ratio": hc.round6(ratio),
        "eviction_observed": eviction_observed,
        "eviction_observed_per_suffix": observed,
        "healthy_after_reuse": healthy,
        "healthy_after_evict": healthy2,
    }, failures


def resolve_max_model_len(models, model, override):
    if override is not None:
        return override, "cli"
    for m in models:
        if m.get("id") == model and isinstance(m.get("max_model_len"), int):
            return m["max_model_len"], "/v1/models"
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="receipt JSON (never overwritten)")
    ap.add_argument("--fingerprint", help="caller fingerprint JSON (merged verbatim)")
    ap.add_argument("--ladder", type=int, nargs="+", default=DEFAULT_LADDER,
                    help="templated prompt lengths; max_model_len - max_tokens is "
                         "always added as the top rung")
    ap.add_argument("--max-model-len", type=int,
                    help="override the limit advertised by /v1/models")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--allow-inexact", action="store_true",
                    help="run rungs whose calibrated length misses the target")
    ap.add_argument("--prefix-words", type=int, default=1500)
    ap.add_argument("--evict-prompts", type=int, default=8)
    ap.add_argument("--evict-words", type=int, default=2500)
    ap.add_argument("--nonce", help="prefix nonce (default: random per run)")
    ap.add_argument("--require-cache-hit", action="store_true")
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--skip-prefix", action="store_true")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="overall per-request deadline in seconds")
    ap.add_argument("--thinking", action="store_true",
                    help="leave thinking enabled (default sends enable_thinking=false)")
    args = ap.parse_args(argv)
    if args.skip_ladder and args.skip_prefix:
        ap.error("--skip-ladder and --skip-prefix together would test nothing")
    hc.refuse_existing(args.out)
    extra = {} if args.thinking else {"chat_template_kwargs": {"enable_thinking": False}}

    caller_fp = json.loads(Path(args.fingerprint).read_text()) if args.fingerprint else {}
    models, models_error = hc.fetch_models(args.endpoint)
    advertised = [m.get("id") for m in models]
    model_id = args.model if args.model in advertised else None
    max_model_len, limit_source = resolve_max_model_len(models, args.model, args.max_model_len)
    git_sha, git_dirty = hc.git_state()
    started = hc.utc_now()
    failures = []
    ladder = sorted(set(args.ladder))
    result = {}
    if not args.skip_ladder:
        if max_model_len is None:
            failures.append("max_model_len unknown: not advertised and no --max-model-len")
        else:
            ladder = sorted(set(ladder) | {max_model_len - args.max_tokens})
            try:
                result["ladder"], fails = run_ladder(
                    args.endpoint, args.model, ladder=ladder, max_model_len=max_model_len,
                    max_tokens=args.max_tokens, seed=args.seed, extra=extra,
                    timeout=args.timeout, allow_inexact=args.allow_inexact)
                failures.extend(f"ladder: {f}" for f in fails)
            except (hc.HarnessError, OSError, http.client.HTTPException) as exc:
                failures.append(f"ladder aborted: {exc}")
    nonce = args.nonce or secrets.token_hex(8)
    if not args.skip_prefix:
        result["prefix_reuse"], fails = run_prefix(
            args.endpoint, args.model, prefix_words=args.prefix_words,
            evict_prompts=args.evict_prompts, evict_words=args.evict_words,
            max_tokens=args.max_tokens, seed=args.seed, extra=extra, timeout=args.timeout,
            nonce=nonce, require_cache_hit=args.require_cache_hit)
        failures.extend(f"prefix: {f}" for f in fails)
    missing = hc.missing_fingerprint_keys(caller_fp, model_id)
    here = Path(__file__).resolve().parent
    cases = {"ladder": ladder, "max_model_len": max_model_len, "max_tokens": args.max_tokens,
             "seed": args.seed, "prefix_words": args.prefix_words,
             "evict_prompts": args.evict_prompts, "evict_words": args.evict_words,
             "filler_words": FILLER_WORDS, "code_template": CODE_LINE_TEMPLATE,
             "question": QUESTION, "suffixes": PREFIX_SUFFIXES, "request_extra": extra}
    receipt = {
        "harness": {"file": Path(__file__).name, "version": HARNESS_VERSION,
                    "sha256": hc.file_sha256(Path(__file__)),
                    "client_sha256": hc.file_sha256(here / "harness_client.py"),
                    "git_sha": git_sha, "git_dirty": git_dirty},
        "fingerprint": {
            "endpoint": args.endpoint,
            "model": args.model,
            "model_id": model_id,
            "max_model_len": max_model_len,
            "max_model_len_source": limit_source,
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
        **result,
        "failures": failures,
        "verdict": "fail" if failures else "pass",
    }
    hc.write_json_exclusive(args.out, receipt)
    print(f"F3 {receipt['verdict']}: {len(failures)} failure(s); receipt {args.out}")
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
