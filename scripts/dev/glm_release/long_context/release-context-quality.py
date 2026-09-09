#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded non-spec release quality, one independently launched context profile.

Example: python3 release-context-quality.py --base-url http://127.0.0.1:8888
  --model EXPECTED_ID --context-limit 8192 --output-dir NEW_RECEIPT_DIRECTORY

Run separately at 4096/8192/16384 with the actual server context cap. This client
does not launch/reconfigure servers or prove decode occupancy: correlate server
traces. Near-full single-user requests use the REAL /v1/chat/completions route,
explicit thinking=true/budget16, and at most three one-token sizing probes.
The base /tokenize(messages) template is NOT assumed to match the chat variant.
Actual prompt usage, measured on the identical request except max_tokens, is
required to match final generation usage exactly. Rendered input
lies in [context-224,context-128], with128 total output tokens; retrieval/facts are
checked. Tool chats target context-512 input tokens, with their distinct lookup
keys embedded early/middle/late. Their exact rendered inputs are measured by
at most three one-token preparation requests,
because /tokenize(messages) omits tool schemas and tool-call metadata. All input
preparation/calibration finishes BEFORE each concurrent completion barrier.
Literal tool results are fixtures, not execution of an external tool.
"""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.parse

FILLER = ("The archive records routine observations about maps, gardens, weather, "
          "workshops, rivers, libraries, and scientific instruments. Each entry is "
          "independent and contains no secret project identifiers. ")
NAMES = ("AURORA", "NEBULA", "ORBIT", "QUASAR")
RESULTS = ("RESULT-A6819", "RESULT-B2473", "RESULT-C9536", "RESULT-D4082")
CITIES = ("Oslo", "Kyoto", "Lima", "Accra")
KINDS = ("niah-early", "niah-middle", "niah-late", "linked-facts")
CAP = 128


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def strict_json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


class Client:
    def __init__(self, args):
        parsed = urllib.parse.urlsplit(args.base_url)
        require(parsed.scheme == "http" and parsed.hostname and not parsed.username
                and not parsed.password and parsed.path in ("", "/")
                and not parsed.query and not parsed.fragment, "require HTTP origin")
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.args = args
        self.end = time.monotonic() + args.deadline
        self.lock = threading.Lock()
        self.serial = self.bytes_written = 0
        self.directory = Path(args.output_dir)
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)

    def remaining(self):
        seconds = self.end - time.monotonic()
        require(seconds > 0, "whole-driver deadline exceeded")
        return seconds

    def save(self, name, data):
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode()
        with self.lock:
            self.bytes_written += len(encoded)
            require(self.bytes_written <= 64 * 1024 * 1024, "receipt byte budget")
        fd = os.open(self.directory / name,
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)

    def request(self, label, path, body=None):
        with self.lock:
            self.serial += 1
            serial = self.serial
        require(serial <= 512, "HTTP request count budget")
        receipt = {"label": label, "path": path, "request": body}
        started = time.monotonic()
        end = min(self.end, started + self.args.timeout)
        raw = bytearray()
        conn = http.client.HTTPConnection(self.host, self.port,
                                         timeout=min(self.args.timeout, self.remaining()))
        try:
            payload = None if body is None else json.dumps(body).encode()
            conn.request("GET" if body is None else "POST", path, payload,
                         {"Content-Type": "application/json"})
            require(time.monotonic() < end, "HTTP deadline before response")
            transport = conn.sock
            transport.settimeout(end - time.monotonic())
            response = conn.getresponse()
            receipt["status"] = response.status
            while True:
                require(time.monotonic() < end, "HTTP deadline before read")
                # Keep the original socket reference even if HTTPConnection
                # detaches it for Connection: close; the response owns its
                # makefile until EOF. Long prefill may legitimately take >30s.
                if transport.fileno() >= 0:
                    transport.settimeout(max(0.001, end - time.monotonic()))
                part = response.read1(16384)
                require(time.monotonic() < end, "HTTP deadline after read")
                if not part:
                    break
                raw.extend(part)
                require(len(raw) <= 2 * 1024 * 1024, "HTTP response byte bound")
            receipt["raw_response"] = raw.decode("utf-8")
            require(response.status == 200, "HTTP status " + str(response.status))
            data = strict_json(raw)
            require(isinstance(data, dict) and "error" not in data, "API error response")
            receipt["response"] = data
            return data, f"http-{serial:04d}.json"
        except Exception as error:
            receipt["error"] = str(error)
            receipt["raw_response"] = raw.decode("utf-8", errors="replace")
            raise
        finally:
            conn.close()
            receipt["seconds"] = round(time.monotonic() - started, 6)
            self.save(f"http-{serial:04d}.json", receipt)

    def tokens(self, text):
        data, _ = self.request("prepare-tokenize", "/tokenize", {"prompt": text})
        tokens = data.get("tokens")
        require(isinstance(tokens, list) and tokens
                and all(type(t) is int and 0 <= t <= 0xffffffff for t in tokens),
                "invalid tokenization")
        require(data.get("count") == len(tokens), "tokenize count mismatch")
        return tokens


def usage(data, expected, cap, context):
    value = data.get("usage", {})
    p, c, total = (value.get(key) for key in
                   ("prompt_tokens", "completion_tokens", "total_tokens"))
    require(all(type(n) is int for n in (p, c, total)), "missing exact integer usage")
    require(p > 0 and 0 < c <= cap and p + c == total <= context,
            "invalid usage or context overrun")
    require(p + cap <= context, "requested output envelope exceeds context")
    require(expected is None or p == expected, "rendered input changed or truncated")
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": total,
            "output_cap": cap, "input_plus_output_cap": p + cap}


def choice(data):
    require(isinstance(data.get("choices"), list) and len(data["choices"]) == 1,
            "expected exactly one choice")
    return data["choices"][0]


def long_prompt(client, kind, index):
    target = client.args.context_limit - CAP
    needle = NAMES[index] + "-" + str(6819 + index * 37)
    prefix = "Read the archive carefully and answer only its final question.\n"
    if kind == "linked-facts":
        projects = ("Amber", "Birch", "Cedar", "Dahlia")
        leaders = ("Iris", "Noor", "Mika", "Vera")
        stock, removed = 31 + index * 4, 6 + index
        texts = [f"\nFACT: Project {projects[index]} is led by {leaders[index]}.\n",
                 f"\nFACT: {leaders[index]}'s project is based in {CITIES[index]} and held {stock} crates.\n",
                 f"\nFACT: Exactly {removed} crates were removed from Project {projects[index]}; no other stock changes occurred.\n"]
        answer = {"leader": leaders[index], "city": CITIES[index], "crates": stock - removed}
        question = f"\nQuestion: For Project {projects[index]}, give its leader, city and remaining crates. Return only JSON with keys leader, city, crates."
        positions = (.05, .50, .95)
    else:
        texts = [f"\nIMPORTANT FACT: The secret project codename is {needle}.\n"]
        answer = needle
        question = "\nQuestion: What is the secret project codename? Reply with only the codename."
        positions = ({"niah-early": .05, "niah-middle": .50, "niah-late": .95}[kind],)
    calibrations = []
    def calibrate(repeats):
        content, used, character_offsets = prefix, 0, []
        for position, fact in zip(positions, texts):
            before = round(repeats * position)
            content += FILLER * (before - used)
            used = before
            character_offsets.append(len(content))
            content += fact
        content += FILLER * (repeats - used) + question
        body = {"model": client.args.model, "messages": [{"role": "user", "content": content}],
                "tools": [], "tool_choice": "none", "max_tokens": CAP,
                "temperature": 0, "seed": 1, "stream": False,
                "chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 16}
        data, reference = client.request("prepare-real-chat-sizing", "/v1/chat/completions",
                                         {**body, "max_tokens": 1})
        count = usage(data, None, 1, client.args.context_limit)["prompt_tokens"]
        # Length1 is expected here; it is sizing evidence, not a quality pass.
        calibrations.append({"receipt": reference, "prompt_tokens": count,
                             "finish_reason": choice(data).get("finish_reason")})
        return count, character_offsets, body
    # The same actual chat handler/template resolves both sizing and generation.
    # Never fabricate or splice prompt tokens, or assume thinking=false overrides
    # GLM's open-think generation suffix (the API reconciles it to thinking).
    base, _, _ = calibrate(0)
    filler_width = len(client.tokens(FILLER))
    repeats = (target - base) // filler_width
    require(repeats > 0, "insufficient context for templated facts")
    count, character_offsets, body = calibrate(repeats)
    if not target - 96 <= count <= target:
        actual_width = (count - base) / repeats
        require(actual_width > 0, "nonpositive rendered filler growth")
        repeats = max(0, math.floor((target - base) / actual_width))
        count, character_offsets, body = calibrate(repeats)
    require(target - 96 <= count <= target, "three sizing probes missed near-full input window")
    return {"kind": kind, "index": index, "expected": answer,
            "fact_offsets": None, "fact_character_offsets": character_offsets,
            "calibrations": calibrations, "requested_positions": positions,
            "input_tokens": count, "body": body}


def chat_body(client, messages, tools, cap):
    # Tool-present GLM may close thinking in its template; no-tools followups
    # retain this explicit small budget rather than false->modelmax reconciliation.
    return {"model": client.args.model, "messages": messages, "tools": tools,
            "tool_choice": "auto" if tools else "none", "temperature": 0,
            "seed": 1, "max_tokens": cap, "stream": False,
            "chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 16}


def prepare_tool(client, index, first=None):
    name, key = "lookup_release_" + str(index), "CASE-" + NAMES[index]
    if first is None:
        tools = [{"type": "function", "function": {"name": name,
                  "description": "Look up the exact case identifier and wait for its result.",
                  "parameters": {"type": "object", "additionalProperties": False,
                                 "properties": {"case_id": {"type": "string"}},
                                 "required": ["case_id"]}}}]
        position = (.05, .50, .95, .50)[index]
        fact = f"\nIMPORTANT: The lookup case_id for this request is {key}.\n"
        question = f"\nFind the lookup case_id in this archive and call {name} exactly once with that case_id. Do not invent a result; wait for the tool response."
        def messages_with(repeats):
            before = round(repeats * position)
            return [{"role": "user", "content": "Read this archive.\n" + FILLER * before
                     + fact + FILLER * (repeats - before) + question}]
        messages = messages_with(0)
        kind, cap = "auto-tool", 192
    else:
        assistant = first["assistant"]
        call = assistant["tool_calls"][0]
        messages = first["request"]["messages"] + [assistant,
            {"role": "tool", "tool_call_id": call["id"],
             "content": json.dumps({"result_code": RESULTS[index]})},
            {"role": "user", "content": "Reply with exactly the result_code from the tool response. No other text."}]
        tools, kind, cap = [], "tool-result", 128
    body = chat_body(client, messages, tools, cap)
    # This exact-template probe is an explicitly recorded one-token generation,
    # not /tokenize pretending to account for schemas, IDs, or injected prompts.
    calibrations = []
    def calibrate(request):
        probe, reference = client.request(kind + "-input-calibration", "/v1/chat/completions",
                                          {**request, "max_tokens": 1})
        counted = usage(probe, None, 1, client.args.context_limit)["prompt_tokens"]
        calibrations.append({"receipt": reference, "prompt_tokens": counted})
        return counted
    counted = calibrate(body)
    if first is None:
        # First measure the actual fixed tool-template overhead using a tiny
        # input. Repeated filler then targets a 512-token reserve, never the
        # absolute context edge; a third actual probe corrects BPE joins.
        target = client.args.context_limit - 512
        filler_width = len(client.tokens(FILLER))
        repeats = max(0, (target - counted) // filler_width)
        require(repeats > 0, "tool schema leaves no long-context input budget")
        body["messages"] = messages_with(repeats)
        counted = calibrate(body)
        if not target - 128 <= counted <= target:
            delta = math.ceil((counted - target) / filler_width)
            repeats = max(0, repeats - delta)
            body["messages"] = messages_with(repeats)
            counted = calibrate(body)
        require(target - 128 <= counted <= target,
                "three probes could not establish near-full tool input budget")
    require(counted + cap <= client.args.context_limit, "tool request envelope")
    return {"kind": kind, "index": index, "body": body, "input_tokens": counted,
            "calibrations": calibrations, "expected": RESULTS[index],
            "tool_name": name, "tool_args": {"case_id": key}}


def wave(client, prepared):
    barrier = threading.Barrier(len(prepared))
    def submit(item):
        body, kind, index = item["body"], item["kind"], item["index"]
        barrier.wait(timeout=min(30, client.remaining()))
        data, reference = client.request(kind, "/v1/chat/completions" if "messages" in body
                                         else "/v1/completions", body)
        counts = usage(data, item["input_tokens"], body["max_tokens"], client.args.context_limit)
        selected = choice(data)
        result = {"kind": kind, "index": index, "receipt": reference, **counts}
        if kind == "auto-tool":
            assistant = selected.get("message", {})
            calls = assistant.get("tool_calls", [])
            require(selected.get("finish_reason") == "tool_calls" and len(calls) == 1,
                    "expected one structured auto tool call")
            call = calls[0]
            require(assistant.get("role") == "assistant" and call.get("type") == "function"
                    and isinstance(call.get("id"), str) and 0 < len(call["id"]) <= 1024,
                    "invalid actual tool call identity")
            function = call.get("function", {})
            require(function.get("name") == item["tool_name"]
                    and strict_json(function.get("arguments", "")) == item["tool_args"],
                    "wrong tool or crossed arguments")
            result.update(assistant=assistant, request=body)
        else:
            require(selected.get("finish_reason") == "stop", "generation did not stop normally")
            message = selected.get("message", {})
            require(not message.get("tool_calls"), "unexpected additional tool call")
            text = message.get("content") if "messages" in body else selected.get("text")
            require(isinstance(text, str), "missing visible output")
            answer = strict_json(text.strip()) if kind == "linked-facts" else text.strip()
            require(answer == item["expected"], "wrong exact answer or coherence result")
            foreign = [p["expected"] for p in prepared if p["index"] != index
                       and isinstance(p["expected"], str) and p["expected"] in text]
            require(not foreign, "peer result leakage")
            result.update(output=text, fact_offsets=item.get("fact_offsets"),
                          fact_character_offsets=item.get("fact_character_offsets"),
                          calibrations=item.get("calibrations"),
                          requested_positions=item.get("requested_positions"))
        result["passed"] = True
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prepared)) as pool:
        rows = list(pool.map(submit, prepared))
    _, health = client.request("post-wave-health", "/health")
    summary = {"concurrency": len(rows), "kind": prepared[0]["kind"], "passed": True,
               "health_receipt": health, "rows": rows}
    client.save(f"wave-c{len(rows)}-{prepared[0]['kind']}.json", summary)
    print(json.dumps({"concurrency": len(rows), "kind": prepared[0]["kind"],
                      "passed": True, "prompt_tokens": [r["prompt_tokens"] for r in rows]}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-limit", type=int, choices=(4096, 8192, 16384), required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--deadline", type=float, default=3600)
    args = parser.parse_args()
    require(len(set(args.concurrency)) == len(args.concurrency), "duplicate concurrency")
    require(all(math.isfinite(t) and 0 < t <= 7200 for t in (args.timeout, args.deadline)),
            "timeouts must be finite, positive and <=7200 seconds")
    client = Client(args)
    result = {"passed": False, "context_limit": args.context_limit,
              "concurrency": args.concurrency,
              "near_full_input_window": [args.context_limit - CAP - 96, args.context_limit - CAP],
              "scope": "non-spec profile asserted by launcher; real chat long retrieval/linked-facts with explicit soft thinking budget16 and exact actual prompt usage, near-full calibrated tools and actual-ID result roundtrips; client barrier is not proof of device occupancy"}
    try:
        models, _ = client.request("identity", "/v1/models")
        require(any(row.get("id") == args.model for row in models.get("data", [])), "model identity mismatch")
        client.request("initial-health", "/health")
        for count in args.concurrency:
            for kind in KINDS:
                wave(client, [long_prompt(client, kind, i) for i in range(count)])
            tools = wave(client, [prepare_tool(client, i) for i in range(count)])
            wave(client, [prepare_tool(client, i, tools[i]) for i in range(count)])
        client.request("final-health", "/health")
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
        # The post-failure probe is also bounded by the whole-driver deadline.
        try:
            _, result["failure_health_receipt"] = client.request("failure-health", "/health")
        except Exception as health_error:
            result["failure_health_error"] = str(health_error)
    finally:
        result["http_requests"] = client.serial
        client.save("summary.json", result)
        print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
