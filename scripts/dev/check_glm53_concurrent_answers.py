#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Four small answer checks, not a benchmark or full quality evaluation.

Short answers may never reach actual C4 decode: correlate server batch traces.
No generated code is executed. Run --self-test for CPU-only validator tests.
"""

import argparse
import ast
import concurrent.futures
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

PROMPTS = (
    "Calculate 37 * 14 - 86. Reply with only the integer answer.",
    "Stable-sort these items by their number ascending: birch=3, ash=1, "
    "cedar=3, dogwood=2. Preserve original order for ties. "
    "Reply only with item names separated by commas, without spaces.",
    "Write a Python function named square with one argument n, returning n*n. "
    "Use exactly one return statement, no imports, annotations, docstring or "
    "other statements. Reply only with the function code.",
    "Return exactly one JSON object with these three keys and no others: "
    "city is the string Oslo, count is the integer 7, active is boolean true. "
    "Do not use Markdown or add explanations.",
)
ANSWERS = ("432", "ash,dogwood,birch,cedar", "def square(n):\n    return n*n",
           '{"city":"Oslo","count":7,"active":true}')


class ValidatorTests(unittest.TestCase):
    def test_correct_answers(self):
        for index, answer in enumerate(ANSWERS):
            self.assertTrue(validate(index, answer))

    def test_junk_empty_repetition_and_cross_answers(self):
        for index, answer in enumerate(ANSWERS):
            for invalid in ("", "!" * 128, "I cannot answer", answer + "\n" + answer,
                            ANSWERS[(index + 1) % 4]):
                self.assertFalse(validate(index, invalid), (index, invalid))

    def test_wrong_answers(self):
        wrong = ("433", "ash,dogwood,cedar,birch", "def square(n):\n    return n+n",
                 '{"city":"Oslo","count":true,"active":true}')
        for index, answer in enumerate(wrong):
            self.assertFalse(validate(index, answer))

    def test_json_duplicates_extra_and_wrong_types(self):
        for answer in ('{"city":"Oslo","count":7,"active":true,"extra":0}',
                       '{"city":"Oslo","count":7,"active":true,"count":7}',
                       '{"city":"Oslo","count":7.0,"active":true}'):
            self.assertFalse(validate(3, answer))

    def test_python_not_executed_and_no_extra_statements(self):
        for answer in ("import os\ndef square(n):\n    return n*n",
                       "def square(n):\n    return __import__('os').system('false')",
                       ANSWERS[2] + "\nsquare(3)"):
            self.assertFalse(validate(2, answer))


def validate(index, answer):
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 16384:
        return False
    answer = answer.strip()
    if index in (0, 1):
        return answer == ANSWERS[index]
    try:
        if index == 2:
            if answer.startswith("```python\n") and answer.endswith("\n```"):
                answer = answer[10:-4]
            # Parse only. Never compile, eval, exec, import or invoke generated code.
            return ast.dump(ast.parse(answer)) == ast.dump(ast.parse(ANSWERS[2]))
        if index == 3:
            def unique(pairs):
                result = dict(pairs)
                if len(result) != len(pairs):
                    raise ValueError("duplicate JSON keys")
                return result
            obj = json.loads(answer, object_pairs_hook=unique)
            return (isinstance(obj, dict) and set(obj) == {"city", "count", "active"}
                    and obj["city"] == "Oslo" and type(obj["count"]) is int
                    and obj["count"] == 7 and obj["active"] is True)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return False
    return False


def request(index, args, barrier):
    payload = {"model": args.model, "messages": [{"role": "user", "content": PROMPTS[index]}],
               "temperature": 0, "max_tokens": 128, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    receipt = {"index": index, "prompt": PROMPTS[index], "passed": False,
               "output": None, "usage": None, "error": None}
    started = time.time()
    try:
        barrier.wait(timeout=10)
        started = time.time()
        req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as response:
            receipt["http_status"] = response.status
            receipt["raw_response"] = response.read().decode()
        body = json.loads(receipt["raw_response"])
        receipt["response"] = body
        choice = body["choices"][0]
        receipt["output"] = choice["message"]["content"]
        receipt["usage"] = body.get("usage")
        receipt["finish_reason"] = choice.get("finish_reason")
        receipt["passed"] = validate(index, receipt["output"]) and choice.get("finish_reason") == "stop"
        usage = receipt["usage"] or {}
        if usage.get("prompt_tokens", 0) + 128 > args.context_limit:
            receipt["passed"] = False
            receipt["error"] = "reported prompt plus output budget exceeds context limit"
        elif not receipt["passed"]:
            receipt["error"] = "invalid answer or incomplete completion"
    except urllib.error.HTTPError as error:
        receipt["http_status"] = error.code
        receipt["raw_response"] = error.read().decode(errors="replace")
        receipt["error"] = str(error)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    receipt["started_at_unix"] = started
    receipt["elapsed_seconds"] = time.time() - started
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="/var/tmp/models/glm53-flash-nvfp4")
    parser.add_argument("--context-limit", type=int, choices=[2048], default=2048,
                        help="must match the restarted server's bounded context cap")
    parser.add_argument("--self-test", action="store_true", help="CPU only; no HTTP requests")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValidatorTests)
        raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    barrier = threading.Barrier(4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda index: request(index, args, barrier), range(4)))
    passed = all(receipt["passed"] for receipt in receipts)
    print(json.dumps({"passed": passed, "context_limit": args.context_limit,
                      "max_tokens": 128, "timeout_seconds": 90,
                      "scope": "Four answer checks only, not a benchmark or full quality evaluation. "
                               "Short outputs may not exercise C4: server batch traces are required.",
                      "requests": receipts}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
