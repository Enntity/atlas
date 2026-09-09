#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Small distinct answer checks, not a benchmark or full quality evaluation.

Short answers may never reach actual C4 decode: correlate server batch traces.
No generated code is executed. Run --self-test for CPU-only client/validator tests.
Initial v8-off run with thinking disabled passed arithmetic/JSON but failed
sort/code: empty visible answers, finish_reason=length at 128 tokens, with correct
answers in reasoning_content. GLM's template forced thinking despite the flag.
This revision explicitly enables thinking with a 32-token budget; answer checks
remain strict and never accept hidden reasoning in place of visible answers.
"""

import argparse
import ast
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
    "Calculate 91 - 38. Reply with only the integer answer.",
    "Calculate 12 * 13. Reply with only the integer answer.",
    "Calculate 144 / 12. Reply with only the integer answer.",
    "Calculate 8 * 9 + 5. Reply with only the integer answer.",
)
ANSWERS = ("432", "ash,dogwood,birch,cedar", "def square(n):\n    return n*n",
           '{"city":"Oslo","count":7,"active":true}', "53", "156", "12", "77")


class ValidatorTests(unittest.TestCase):
    def test_context_receipt_accepts_only_tested_profiles(self):
        self.assertEqual(parse_args([]).context_limit, 2048)
        self.assertEqual(parse_args(["--context-limit", "2044"]).context_limit, 2044)
        self.assertEqual(parse_args(["--context-limit", "16384"]).context_limit, 16384)
        for invalid in ["0", "4096", "32768"]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--context-limit", invalid])

    def test_concurrency_is_explicit_and_keeps_existing_default(self):
        self.assertEqual(parse_args([]).concurrency, 4)
        self.assertEqual(parse_args(["--concurrency", "1"]).concurrency, 1)
        for width in range(2, 9):
            self.assertEqual(parse_args(["--concurrency", str(width)]).concurrency, width)
        for invalid in ["0", "9"]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--concurrency", invalid])

    def test_payload_reserves_visible_answer_budget(self):
        for index in range(4):
            payload = request_payload(index, "test-model")
            self.assertEqual(payload["model"], "test-model")
            self.assertEqual(payload["messages"][0]["content"], PROMPTS[index])
            self.assertEqual(payload["max_tokens"], 128)
            self.assertEqual(payload["thinking_token_budget"], 32)
            self.assertIs(payload["chat_template_kwargs"]["enable_thinking"], True)
            self.assertEqual(payload["max_tokens"] - payload["thinking_token_budget"], 96)

    def test_correct_answers(self):
        for index, answer in enumerate(ANSWERS):
            self.assertTrue(validate(index, answer))

    def test_junk_empty_repetition_and_cross_answers(self):
        for index, answer in enumerate(ANSWERS):
            for invalid in ("", "!" * 128, "I cannot answer", answer + "\n" + answer,
                            ANSWERS[(index + 1) % len(ANSWERS)]):
                self.assertFalse(validate(index, invalid), (index, invalid))
            for peer, other in enumerate(ANSWERS):
                if peer != index:
                    self.assertFalse(validate(index, other), (index, peer))

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

    def test_actual_client_full_waves(self):
        for width in range(1, 9):
            seen = []
            lock = threading.Lock()
            http_wave = threading.Barrier(width)

            def urlopen(req, timeout):
                task = json.loads(req.data)
                index = PROMPTS.index(task["messages"][0]["content"])
                with lock:
                    seen.append(index)
                http_wave.wait(timeout=5)
                body = {"choices": [{"finish_reason": "stop", "message": {
                    "content": ANSWERS[index]}}], "usage": {"prompt_tokens": 100}}
                response = mock.MagicMock(status=200)
                response.read.return_value = json.dumps(body).encode()
                response.__enter__.return_value = response
                return response

            output = io.StringIO()
            with mock.patch("sys.argv", ["answers", "--concurrency", str(width)]), \
                    mock.patch.object(urllib.request, "urlopen", side_effect=urlopen), \
                    contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as done:
                main()
            self.assertEqual(done.exception.code, 0, output.getvalue())
            receipt = json.loads(output.getvalue())
            count = 4 if width in (1, 2, 4) else (6 if width == 3 else width)
            self.assertEqual(sorted(seen), list(range(count)))
            self.assertEqual(len(receipt["requests"]), count)

    def test_actual_client_refuses_crossed_answer(self):
        body = {"choices": [{"finish_reason": "stop", "message": {"content": ANSWERS[1]}}],
                "usage": {"prompt_tokens": 100}}
        response = mock.MagicMock(status=200)
        response.read.return_value = json.dumps(body).encode()
        response.__enter__.return_value = response
        with mock.patch.object(urllib.request, "urlopen", return_value=response):
            receipt = request(0, parse_args([]), threading.Barrier(1))
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["output"], ANSWERS[1])


def validate(index, answer):
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 16384:
        return False
    answer = answer.strip()
    if index in (0, 1, 4, 5, 6, 7):
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


def request_payload(index, model):
    return {"model": model, "messages": [{"role": "user", "content": PROMPTS[index]}],
            "temperature": 0, "max_tokens": 128, "stream": False,
            "thinking_token_budget": 32, "chat_template_kwargs": {"enable_thinking": True}}


def request(index, args, barrier):
    payload = request_payload(index, args.model)
    receipt = {"index": index, "prompt": PROMPTS[index], "passed": False,
               "payload": payload, "output": None, "usage": None, "error": None}
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="/var/tmp/models/glm53-flash-nvfp4")
    parser.add_argument("--context-limit", type=int, choices=[2044, 2048, 16384], default=2048,
                        help="must match the restarted server's bounded context cap")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=4,
                        help="C1/C2/C4 retain four checks; C3 runs six, C5..8 run one full wave")
    parser.add_argument("--self-test", action="store_true", help="CPU only; no HTTP requests")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValidatorTests)
        raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    barrier = threading.Barrier(args.concurrency)
    count = 4 if args.concurrency in (1, 2, 4) else (6 if args.concurrency == 3 else args.concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        receipts = list(pool.map(lambda index: request(index, args, barrier), range(count)))
    passed = all(receipt["passed"] for receipt in receipts)
    print(json.dumps({"passed": passed, "context_limit": args.context_limit,
                      "concurrency": args.concurrency,
                      "max_tokens": 128, "thinking_token_budget": 32, "timeout_seconds": 90,
                      "prior_failure": "Initial v8-off thinking-disabled run passed arithmetic/JSON "
                                       "but failed sort/code with empty visible answers and length at 128; "
                                       "correct answers appeared only in template-forced reasoning. "
                                       "Now uses an explicit 32-token thinking budget; validators unchanged.",
                      "scope": "Distinct answer checks only, not a benchmark or full quality evaluation. "
                               "Short outputs may not reach requested occupancy: server traces are required.",
                      "requests": receipts}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
