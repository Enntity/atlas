#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Distinct forced-tool tasks at C1..8; no external tools are executed.

This checks structured output and argument ownership, not automatic tool choice.
Short replies need server batch traces to establish actual decode occupancy.
--self-test runs CPU client/validator tests with stubbed HTTP only.
"""
import argparse
import concurrent.futures
import contextlib
import io
import json
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.request

CASES = (
    ("lookup_weather", {"city": "Oslo", "unit": "celsius"},
     "Find the current weather in Oslo, using celsius units. You must call the "
     "lookup_weather tool exactly once with city Oslo and unit celsius. "
     "Do not guess the weather or provide an answer before the tool result."),
    ("lookup_inventory", {"sku": "BLUE-17", "warehouse": "north"},
     "Check the inventory for SKU BLUE-17 in the north warehouse. You must call "
     "lookup_inventory exactly once with sku BLUE-17 and warehouse north. "
     "Do not invent the stock count or provide an answer before the tool result."),
    ("lookup_library", {"title": "River Atlas", "branch": "west"},
     "Find River Atlas at the west library branch. You must call lookup_library "
     "exactly once with title River Atlas and branch west; wait for the result."),
    ("lookup_train", {"route": "T42", "station": "Central"},
     "Check train T42 at Central. You must call lookup_train exactly once "
     "with route T42 and station Central; wait for the result."),
    ("lookup_parcel", {"tracking": "PKG-503", "carrier": "Acorn"},
     "Track PKG-503 with Acorn. You must call lookup_parcel exactly once "
     "with tracking PKG-503 and carrier Acorn; wait for the result."),
    ("lookup_booking", {"reference": "BK-611", "surname": "Mora"},
     "Find booking BK-611 for Mora. You must call lookup_booking exactly once "
     "with reference BK-611 and surname Mora; wait for the result."),
    ("lookup_museum", {"museum": "Harbor", "day": "Tuesday"},
     "Check Harbor museum on Tuesday. You must call lookup_museum exactly once "
     "with museum Harbor and day Tuesday; wait for the result."),
    ("lookup_repair", {"ticket": "REP-829", "device": "printer"},
     "Check repair REP-829 for the printer. You must call lookup_repair exactly "
     "once with ticket REP-829 and device printer; wait for the result."),
)


def payload(index, model):
    name, arguments, prompt = CASES[index]
    return {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "function", "function": {
            "name": name, "description": "Look up the requested values.",
            "parameters": {"type": "object", "additionalProperties": False,
                           "properties": {key: {"type": "string"} for key in arguments},
                           "required": list(arguments)}}}],
        "tool_choice": {"type": "function", "function": {"name": name}},
        "temperature": 0, "max_tokens": 256, "stream": False,
        "thinking_token_budget": 32, "chat_template_kwargs": {"enable_thinking": True},
    }


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON keys")
    return result


def validate(index, body):
    try:
        choices = body["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] != "tool_calls":
            return False
        calls = choices[0]["message"]["tool_calls"]
        if len(calls) != 1 or calls[0]["type"] != "function":
            return False
        function = calls[0]["function"]
        expected_name, expected_args, _ = CASES[index]
        if function["name"] != expected_name or not isinstance(function["arguments"], str):
            return False
        arguments = json.loads(function["arguments"], object_pairs_hook=unique_object)
        return arguments == expected_args
    except (KeyError, TypeError, ValueError, IndexError, RecursionError):
        return False


class ValidatorTests(unittest.TestCase):
    @staticmethod
    def response(index):
        name, arguments, _ = CASES[index]
        return {"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "tool_calls": [{"type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}]}}]}

    def test_correct(self):
        for index in range(len(CASES)):
            self.assertTrue(validate(index, self.response(index)))
            self.assertIn("must call", payload(index, "test-model")["messages"][0]["content"])

    def test_wrong(self):
        for index in range(len(CASES)):
            for bad in ("{}", "not json", '{"city":"Oslo","city":"Oslo","unit":"celsius"}'):
                body = self.response(index)
                body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = bad
                self.assertFalse(validate(index, body))
            body = self.response(index)
            body["choices"][0]["finish_reason"] = "length"
            self.assertFalse(validate(index, body))
            body = self.response(index)
            body["choices"][0]["message"]["tool_calls"] *= 2
            self.assertFalse(validate(index, body))

    def test_crossed_args(self):
        for index in range(len(CASES)):
            for peer in range(len(CASES)):
                if peer == index:
                    continue
                self.assertFalse(validate(index, self.response(peer)))
                body = self.response(index)
                body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(CASES[peer][1])
                self.assertFalse(validate(index, body))

    def test_actual_client_full_wave(self):
        for width in range(1, 9):
            seen = []
            lock = threading.Lock()
            http_wave = threading.Barrier(width)

            def urlopen(req, timeout):
                task = json.loads(req.data)
                index = next(i for i, case in enumerate(CASES)
                             if case[2] == task["messages"][0]["content"])
                with lock:
                    seen.append(index)
                http_wave.wait(timeout=5)
                body = self.response(index)
                body["usage"] = {"prompt_tokens": 100}
                response = mock.MagicMock(status=200)
                response.read.return_value = json.dumps(body).encode()
                response.__enter__.return_value = response
                return response

            output = io.StringIO()
            with mock.patch("sys.argv", ["tools", "--concurrency", str(width)]), \
                    mock.patch.object(urllib.request, "urlopen", side_effect=urlopen), \
                    contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as done:
                main()
            self.assertEqual(done.exception.code, 0, output.getvalue())
            receipt = json.loads(output.getvalue())
            self.assertEqual(receipt["concurrency"], width)
            self.assertEqual(sorted(seen), list(range(max(2, width))))

    def test_actual_client_refuses_crossed_tool(self):
        body = self.response(1)
        body["usage"] = {"prompt_tokens": 100}
        response = mock.MagicMock(status=200)
        response.read.return_value = json.dumps(body).encode()
        response.__enter__.return_value = response
        args = argparse.Namespace(model="test-model", base_url="http://unused", context_limit=2048)
        with mock.patch.object(urllib.request, "urlopen", return_value=response):
            receipt = request(0, args, threading.Barrier(1))
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["response"], body)


def request(index, args, barrier):
    task = payload(index, args.model)
    receipt = {"index": index, "payload": task, "passed": False, "error": None}
    start = time.time()
    try:
        barrier.wait(timeout=10)
        start = time.time()
        req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(task).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as response:
            receipt["http_status"] = response.status
            receipt["raw_response"] = response.read().decode()
        body = json.loads(receipt["raw_response"])
        receipt["response"] = body
        usage = body.get("usage") or {}
        receipt["usage"] = usage
        prompt_tokens = usage.get("prompt_tokens")
        budget_ok = (type(prompt_tokens) is int and prompt_tokens > 0
                     and prompt_tokens + task["max_tokens"] <= args.context_limit)
        receipt["passed"] = budget_ok and validate(index, body)
        if not receipt["passed"]:
            receipt["error"] = "wrong/incomplete structured tool call or invalid reported context budget"
    except urllib.error.HTTPError as error:
        receipt["http_status"] = error.code
        receipt["raw_response"] = error.read().decode(errors="replace")
        receipt["error"] = str(error)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    receipt["started_at_unix"] = start
    receipt["elapsed_seconds"] = time.time() - start
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="/var/tmp/models/glm53-flash-nvfp4")
    parser.add_argument("--context-limit", type=int, choices=[2044, 2048, 16384], default=2048)
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValidatorTests)
        raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    barrier = threading.Barrier(args.concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(request, index, args, barrier) for index in range(max(2, args.concurrency))]
        receipts = [future.result() for future in futures]
    passed = all(row["passed"] for row in receipts)
    print(json.dumps({"passed": passed, "concurrency": args.concurrency,
                      "scope": "Forced named-tool calls, not automatic tool selection; no tools executed.",
                      "requests": receipts}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
