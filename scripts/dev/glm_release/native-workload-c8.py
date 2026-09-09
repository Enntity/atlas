# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned capacity-eight stdin workload: python3 - BASE_URL EXPECTED_MODEL < this-file.

No imports/files beyond Python's standard library; no tool/code execution.
Coding provenance: scripts/benchmark_glm53_concurrency.py and the literal
docs/experiments/glm53-20260906/workloads/lru-cache-completion.txt, as used by
v29-c8-graphs-matrix.json. Same request fields and aggregate denominators;
stdlib read1 replaces requests.iter_lines (client TTFT buffering may differ).
Quality provenance: existing concurrent_answers/tools and concurrent_niah
clients. Tools deliberately use auto, NOT the unsupported named-forced route.
Launch must explicitly supply --disable-tool-grammar true for this profile.
"""

import ast
import concurrent.futures
import hashlib
import http.client
import json
import math
import statistics
import sys
import threading
import time
import urllib.parse

LRU = """Complete the following Python module. Implement an LRU cache without importing
OrderedDict or using a third-party package. Use a dictionary and a doubly linked
list with sentinel nodes. Include get, put, delete, clear, __len__, and an items
method returning items from most recently used to least recently used. Reject
nonpositive capacities. Updating an existing key must move it to the front.
Deleting a missing key returns False. Include a small unittest.TestCase with
distinct tests for eviction order, updating existing keys, deletion, clearing,
and invalid capacity. Output only the complete Python source, with short
docstrings explaining the invariants. Do not abbreviate methods or tests.

Python source:
import unittest


class _Node:
"""
LITERAL_SHA = "940f3a003a1ac66b5195b27a78e31b92f5622bd29d221a35949056bdd87477d1"
TOKEN_SHA = "8b104308b377752ad9803d298be34e01cedf9f8ac577cacfea653c3359e7aebf"
ANSWERS = [
    "Calculate 37 * 14 - 86. Reply with only the integer answer.",
    "Stable-sort these items by their number ascending: birch=3, ash=1, cedar=3, dogwood=2. Preserve original order for ties. Reply only with item names separated by commas, without spaces.",
    "Write a Python function named square with one argument n, returning n*n. Use exactly one return statement, no imports, annotations, docstring or other statements. Reply only with the function code.",
    "Return exactly one JSON object with these three keys and no others: city is the string Oslo, count is the integer 7, active is boolean true. Do not use Markdown or add explanations.",
    "Calculate 23 * 17 + 19. Reply with only the integer answer.",
    "Calculate 91 * 8 - 37. Reply with only the integer answer.",
    "Calculate 144 / 12 + 58. Reply with only the integer answer.",
    "Calculate 63 * 9 - 42. Reply with only the integer answer.",
]
TOOLS = [
    ("lookup_weather", {"city": "Oslo", "unit": "celsius"},
     "Find the current weather in Oslo, using celsius units. You must call the lookup_weather tool exactly once with city Oslo and unit celsius. Do not guess the weather or provide an answer before the tool result."),
    ("lookup_inventory", {"sku": "BLUE-17", "warehouse": "north"},
     "Check the inventory for SKU BLUE-17 in the north warehouse. You must call lookup_inventory exactly once with sku BLUE-17 and warehouse north. Do not invent the stock count or provide an answer before the tool result."),
    ("lookup_library", {"title": "River Atlas", "branch": "west"},
     "Find River Atlas at the west library branch. You must call lookup_library exactly once with title River Atlas and branch west; wait for the result."),
    ("lookup_train", {"route": "T42", "station": "Central"},
     "Check train T42 at Central. You must call lookup_train exactly once with route T42 and station Central; wait for the result."),
    ("lookup_parcel", {"tracking": "PK-581", "carrier": "swift"},
     "Find parcel PK-581 with carrier swift. You must call lookup_parcel exactly once with tracking PK-581 and carrier swift; wait for the result."),
    ("lookup_flight", {"flight": "AZ73", "airport": "Riverton"},
     "Check flight AZ73 at airport Riverton. You must call lookup_flight exactly once with flight AZ73 and airport Riverton; wait for the result."),
    ("lookup_hotel", {"booking": "HT-209", "surname": "Meyer"},
     "Find hotel booking HT-209 for surname Meyer. You must call lookup_hotel exactly once with booking HT-209 and surname Meyer; wait for the result."),
    ("lookup_event", {"event": "EV-936", "venue": "Oak Hall"},
     "Check event EV-936 at venue Oak Hall. You must call lookup_event exactly once with event EV-936 and venue Oak Hall; wait for the result."),
]
TOOL_RESULTS = ["WX-4821", "ST-7396", "BK-1568", "TR-9042",
                "PC-6273", "FL-3185", "HT-8604", "EV-2957"]
NEEDLES = ["AURORA-6193", "NEBULA-6193", "ORBIT-6193", "QUASAR-6193",
           "PULSAR-8427", "COMET-3506", "NOVA-9714", "ZENITH-2685"]
NIAH_LENGTHS = (768, 800, 832, 864, 896, 928, 960, 992)
NIAH_CAPS = (32, 16, 32, 16, 32, 16, 32, 16)
NIAH_POSITIONS = (.05, .50, .95, .50, .05, .50, .95, .50)
FILLER = ("The archive records routine observations about maps, gardens, weather, "
          "workshops, rivers, libraries, and scientific instruments. Each entry is "
          "independent and contains no secret project identifiers. ")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def loads(raw):
    def invalid_constant(value):
        raise ValueError("non-finite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def fingerprint(text):
    data = text.encode("utf-8")
    return {"completion_text": text, "completion_text_bytes": len(data),
            "completion_text_sha256": hashlib.sha256(data).hexdigest()}


class Client:
    def __init__(self, base, model):
        url = urllib.parse.urlsplit(base)
        require(url.scheme == "http" and url.hostname and not url.username
                and not url.password and url.path in ("", "/")
                and not url.query and not url.fragment, "require literal HTTP base URL")
        require(model and len(model) <= 4096, "require explicit expected model")
        self.host, self.port, self.model = url.hostname, url.port or 80, model

    def request(self, path, body, streaming=False):
        started = time.perf_counter()
        deadline = started + 180
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        chunks, pending, usage, first, finish = [], b"", None, None, None
        received, text_bytes, done = 0, 0, False
        try:
            conn.request("POST", path, json.dumps(body).encode(),
                         {"Content-Type": "application/json", "Accept": "text/event-stream" if streaming else "application/json"})
            response = conn.getresponse()
            require(response.status == 200, "HTTP status " + str(response.status))
            while not done:
                require(time.perf_counter() < deadline, "request total deadline")
                part = response.read1(4096)
                require(time.perf_counter() < deadline, "request total deadline")
                if not part:
                    break
                received += len(part)
                require(received <= (262144 if streaming else 32768), "response byte bound")
                pending += part
                if not streaming:
                    continue
                while b"\n" in pending:
                    raw, pending = pending.split(b"\n", 1)
                    raw = raw.rstrip(b"\r")
                    if not raw.startswith(b"data: "):
                        continue
                    if raw == b"data: [DONE]":
                        done = True
                        break
                    event = loads(raw[6:])
                    require("error" not in event, "SSE error: " + str(event.get("error"))[:1024])
                    for choice in event.get("choices", []):
                        text = choice.get("text", "")
                        if text:
                            if first is None:
                                first = time.perf_counter()
                            text_bytes += len(text.encode("utf-8"))
                            require(text_bytes <= 16384, "completion text bound")
                            chunks.append(text)
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
                    if event.get("usage"):
                        usage = event["usage"]
        finally:
            conn.close()
        ended = time.perf_counter()
        if not streaming:
            return {"response": loads(pending), "started": started, "ended": ended}
        require(done and first is not None and usage is not None, "incomplete SSE receipt")
        check_usage(usage, body["max_tokens"], len(body["prompt"]))
        require(started <= first < ended, "invalid timing order")
        return {"started": started, "first_text": first, "ended": ended,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "ttft_ms": finite(usage["time_to_first_token_ms"]),
                "decode_tps": finite(usage["response_token/s"]),
                "client_ttft_ms": (first - started) * 1000,
                "finish_reason": finish, "usage": usage, **fingerprint("".join(chunks))}

    def tokenize(self, text):
        tokens = self.request("/tokenize", {"prompt": text})["response"]["tokens"]
        require(isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens), "invalid tokens")
        return tokens

    def complete(self, prompt, cap):
        require(1 <= len(prompt) <= 1024 and len(prompt) + cap <= 2044, "completion context budget")
        return self.request("/v1/completions", {
            "model": self.model, "prompt": prompt, "max_tokens": cap,
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "seed": 1}, True)

    def chat(self, prompt, tool=None):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 256 if tool else 128, "stream": False,
                "thinking_token_budget": 32, "chat_template_kwargs": {"enable_thinking": True}}
        if tool:
            name, args, _ = tool
            body.update(tool_choice="auto", tools=[{"type": "function", "function": {
                "name": name, "description": "Look up the requested values.",
                "parameters": {"type": "object", "additionalProperties": False,
                               "properties": {key: {"type": "string"} for key in args},
                               "required": list(args)}}}])
        result = self.request("/v1/chat/completions", body)
        result["request"] = body
        return result

    def tool_followup(self, first, index):
        assistant = first["response"]["choices"][0]["message"]
        call = assistant["tool_calls"][0]
        require(assistant["role"] == "assistant" and isinstance(call.get("id"), str)
                and 0 < len(call["id"]) <= 1024, "missing actual assistant call identity")
        # Preserve the actual assistant message/call and its exact ID. This is a
        # literal result fixture, not execution of the named external tool.
        result = {"result_code": TOOL_RESULTS[index]}
        body = {"model": self.model,
                "messages": [first["request"]["messages"][0], assistant,
                             {"role": "tool", "tool_call_id": call["id"],
                              "content": json.dumps(result, separators=(",", ":"))},
                             {"role": "user", "content": "Reply with exactly the result_code from the tool response, with no other text."}],
                "tools": [], "tool_choice": "none",
                "temperature": 0, "max_tokens": 64, "stream": False,
                "thinking_token_budget": 32,
                "chat_template_kwargs": {"enable_thinking": True}}
        followup = self.request("/v1/chat/completions", body)
        followup["request"] = body
        followup["literal_tool_result"] = result
        return followup


def finite(value):
    value = float(value)
    require(math.isfinite(value) and value >= 0, "invalid server timing")
    return value


def check_usage(usage, cap, prompt=None):
    p, c = usage["prompt_tokens"], usage["completion_tokens"]
    require(type(p) is int and 1 <= p <= 1024 and p + cap <= 2044, "reported prompt/context budget")
    require(type(c) is int and 1 <= c <= cap, "reported completion budget")
    require(prompt is None or p == prompt, "prompt token count mismatch")


def wave(functions):
    barrier = threading.Barrier(len(functions))
    def invoke(fn):
        try:
            barrier.wait(timeout=30)
            return fn()
        except Exception as error:
            return {"error": str(error)[:2048]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(functions)) as pool:
        return list(pool.map(invoke, functions))


def validate_chat(row, index, tool=False):
    require("error" not in row, row.get("error", ""))
    response = row["response"]
    check_usage(response["usage"], 256 if tool else 128)
    require(len(response["choices"]) == 1, "expected one choice")
    choice = response["choices"][0]
    if tool:
        require(choice["finish_reason"] == "tool_calls", "expected structured tool_calls finish")
        calls = choice["message"]["tool_calls"]
        require(len(calls) == 1 and calls[0]["type"] == "function", "expected exactly one function call")
        function = calls[0]["function"]
        require(function["name"] == TOOLS[index][0] and isinstance(function["arguments"], str)
                and loads(function["arguments"]) == TOOLS[index][1], "wrong or crossed tool arguments")
        return
    require(choice["finish_reason"] == "stop", "answer did not finish normally")
    text = choice["message"]["content"].strip()
    require(0 < len(text) <= 16384, "empty or oversized answer")
    if index < 2:
        require(text == ["432", "ash,dogwood,birch,cedar"][index], "wrong exact answer")
    elif index == 2:
        if text.startswith("```python\n") and text.endswith("\n```"):
            text = text[10:-4]
        require(ast.dump(ast.parse(text)) == ast.dump(ast.parse("def square(n):\n    return n*n")), "wrong Python AST")
    elif index == 3:
        value = loads(text)
        require(set(value) == {"city", "count", "active"} and value["city"] == "Oslo"
                and type(value["count"]) is int and value["count"] == 7
                and value["active"] is True, "wrong JSON answer")
    else:
        require(text == ["410", "691", "70", "525"][index - 4], "wrong exact arithmetic answer")


def validate_tool_followup(row, index):
    require("error" not in row, row.get("error", ""))
    response = row["response"]
    check_usage(response["usage"], 64)
    require(len(response["choices"]) == 1, "expected one followup choice")
    choice = response["choices"][0]
    require(choice["finish_reason"] == "stop", "tool followup did not stop normally")
    message = choice["message"]
    require(not message.get("tool_calls"), "tools-disabled followup emitted another call")
    text = message["content"].strip()
    require(text == TOOL_RESULTS[index]
            and all(other not in text for i, other in enumerate(TOOL_RESULTS) if i != index),
            "wrong or crossed tool followup result")


def qualify(rows, validator):
    for index, row in enumerate(rows):
        try:
            validator(row, index)
            row["passed"] = True
        except Exception as error:
            row.update(passed=False, validation_error=str(error)[:2048])
    require(all(row["passed"] for row in rows), "quality wave failed; timing not issued")


def niah_prompt(client, target, needle, position):
    prefix = client.tokenize("Read the archive carefully. Remember any explicit important fact.\n\n")
    secret = client.tokenize("\nIMPORTANT FACT: The secret project codename is " + needle + ".\n")
    question = client.tokenize("\nQuestion: What is the secret project codename? Reply with only the codename:")
    filler = client.tokenize(FILLER)
    spare = target - len(prefix) - len(secret) - len(question)
    require(spare > 0, "NIAH framing exceeds target")
    before = min(max(round(target * position) - len(prefix), 0), spare)
    def repeat(n):
        return (filler * ((n + len(filler) - 1) // len(filler)))[:n]
    return prefix + repeat(before) + secret + repeat(spare - before) + question


def summarize(rows):
    require(all("error" not in row for row in rows), "benchmark request failed")
    start = min(row["started"] for row in rows)
    first = min(row["first_text"] for row in rows)
    last = max(row["ended"] for row in rows)
    total = sum(row["completion_tokens"] for row in rows)
    return {"concurrency": len(rows), "requests": rows,
            # Keep a bad terminal receipt in the report before the caller fails.
            "all_outputs_reached_cap": all(row["completion_tokens"] == 256
                                            and row["finish_reason"] == "length" for row in rows),
            "completion_tokens_total": total, "batch_wall_seconds": round(last - start, 6),
            "aggregate_e2e_tps": round(total / (last - start), 3),
            "aggregate_decode_window_tps": round(total / (last - first), 3),
            "aggregate_post_first_token_tps": round((total - len(rows)) / (last - first), 3),
            "median_client_ttft_ms": round(statistics.median(r["client_ttft_ms"] for r in rows), 3),
            "median_ttft_ms": round(statistics.median(r["ttft_ms"] for r in rows), 3),
            "median_session_decode_tps": round(statistics.median(r["decode_tps"] for r in rows), 3),
            "sum_session_decode_tps": round(sum(r["decode_tps"] for r in rows), 3)}


def run(report):
    require(len(sys.argv) == 3, "usage: python3 - BASE_URL EXPECTED_MODEL < native-workload-c8.py")
    client = Client(sys.argv[1], sys.argv[2])
    report.update(base_url=sys.argv[1], model=sys.argv[2])
    report["quality"] = {"answers": [], "auto_tools": [], "tool_followups": [], "niah": []}
    quality = report["quality"]
    quality["answers"] = wave([lambda i=i: client.chat(ANSWERS[i]) for i in range(8)])
    qualify(quality["answers"], validate_chat)
    quality["auto_tools"] = wave([lambda case=case: client.chat(case[2], case) for case in TOOLS])
    qualify(quality["auto_tools"], lambda row, i: validate_chat(row, i, True))
    quality["tool_followups"] = wave([
        lambda i=i: client.tool_followup(quality["auto_tools"][i], i) for i in range(8)])
    qualify(quality["tool_followups"], validate_tool_followup)
    prompts = [niah_prompt(client, NIAH_LENGTHS[i], NEEDLES[i], NIAH_POSITIONS[i]) for i in range(8)]
    quality["niah"] = wave([lambda i=i: client.complete(prompts[i], NIAH_CAPS[i]) for i in range(8)])
    def validate_needle(row, i):
        require("error" not in row, row.get("error", ""))
        peers = [needle for j, needle in enumerate(NEEDLES) if j != i]
        row.update(needle=NEEDLES[i], peer_needles=peers, position=NIAH_POSITIONS[i],
                   requested_prompt_tokens=NIAH_LENGTHS[i], requested_output_cap=NIAH_CAPS[i])
        text = row["completion_text"].casefold()
        require(NEEDLES[i].casefold() in text and all(peer.casefold() not in text for peer in peers), "own needle absent or foreign needle present")
    qualify(quality["niah"], validate_needle)
    require(hashlib.sha256(LRU.encode()).hexdigest() == LITERAL_SHA, "embedded literal differs")
    prompt = client.tokenize(LRU)
    digest = hashlib.sha256(json.dumps(prompt, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    report["coding_prompt"] = {"literal_sha256": LITERAL_SHA, "token_sha256": digest, "prompt_tokens": len(prompt)}
    require(len(prompt) == 148 and digest == TOKEN_SHA, "coding tokens differ from v29 receipt")
    report["coding"] = []
    for width in range(1, 9):
        item = {"concurrency": width, "warmup_runs": [], "runs": []}
        report["coding"].append(item)
        for iteration in range(4):
            rows = wave([lambda: client.complete(prompt, 256) for _ in range(width)])
            batch = summarize(rows) if all("error" not in row for row in rows) else {"requests": rows, "all_outputs_reached_cap": False}
            item["warmup_runs" if iteration == 0 else "runs"].append(batch)
            require(batch["all_outputs_reached_cap"], "coding output failed matched 256-token cap")
        item["medians"] = {key: round(statistics.median(row[key] for row in item["runs"]), 3)
                           for key in item["runs"][0] if key.endswith("_tps") or key.endswith("ttft_ms")}
    report["passed"] = True


if __name__ == "__main__":
    result = {"passed": False, "scope": "selected capacity-eight MTP qualification, not full API or reference-workload parity",
              "selected_active_capacity": 8, "unissued_concurrencies": "above C8",
              "known_unsupported": "named-forced tool_choice requires compiled grammar; not requested or passed",
              "required_launch_option": "--disable-tool-grammar true (auto tool route only)",
              "quality_scope": "one simultaneous wave each: eight distinct answers, eight strict structured auto tool calls, eight tools-disabled followups using actual call IDs and distinct literal result codes, eight distinct needles checked against all seven peers; no actual external tool execution or broad quality claim",
              "concurrency_scope": "barrier immediately before HTTP requests; server traces, not client width, establish active slot occupancy",
              "network_bounds": "30s blocking I/O timeout; 180s request deadline checked around each read (at most one 30s I/O overshoot); response and final output byte caps",
              "chat_timing_scope": "nonstreaming chat retains server usage/TTFT and full wall time; client first-text TTFT is measured only for streaming completions",
              "benchmark": {"source": "scripts/benchmark_glm53_concurrency.py; v29-c8-graphs-matrix.json",
                            "literal_source": "docs/experiments/glm53-20260906/workloads/lru-cache-completion.txt",
                            "prompt_tokens": 148, "output_cap": 256, "temperature": 0, "seed": 1,
                            "warmup_waves_each": 1, "measured_waves_each": 3, "forced_output_cap": False,
                            "watchdog_override": False, "identical_coding_prompts_are_not_cross_slot_quality_proof": True,
                            "denominators": "e2e=sum(tokens)/(last_end-first_start); decode_window=sum(tokens)/(last_end-first_text); post_first=(sum(tokens)-width)/(last_end-first_text); not GPU timers",
                            "client_difference": "stdlib read1 vs requests.iter_lines; buffering may affect client TTFT/decode-window",
                            "coding_correctness": "raw capped text retained; generated program not executed or certified"}}
    try:
        run(result)
    except Exception as failure:
        result["error"] = str(failure)[:2048]
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) + 1 > 1000000:
        encoded = b'{"passed":false,"error":"output exceeded 1000000 bytes; no success claim"}'
        result["passed"] = False
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()
    sys.exit(0 if result["passed"] else 1)
