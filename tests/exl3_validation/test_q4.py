# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the Q4 task-quality corpus, checkers and runner.

Standard library only. Covers every checker on correct and incorrect
outputs, reasoning stripping, fenced-code extraction, code sandboxing,
repetition detection, the Wilson interval, corpus schema / reproducibility /
answer validity (each reference solution and expected value must pass its
own checker, and a near-miss must fail), and the runner plus --compare
against a local fake server.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

Q4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q4")
sys.path.insert(0, Q4)

import build_corpus  # noqa: E402
import checkers  # noqa: E402
import generate_long_context  # noqa: E402
import run_q4  # noqa: E402

SERVING_LIMIT = 4096


def scratch():
    """A temporary directory under $TMPDIR when set (never mutates env)."""
    return tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None)


def load(name):
    with open(os.path.join(Q4, name), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fenced(code):
    return "Here you go:\n```python\n" + code + "```\n"


def tool_response(name, arguments):
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"content": None, "finish_reason": "tool_calls",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": raw}}]}


class StripReasoningTest(unittest.TestCase):
    def test_complete_orphan_and_unterminated_blocks(self):
        self.assertEqual(checkers.strip_reasoning("<think>Answer: 1</think>Answer: 2"), "Answer: 2")
        self.assertEqual(checkers.strip_reasoning("prefilled reasoning 1</think>\nAnswer: 2"), "Answer: 2")
        self.assertEqual(checkers.strip_reasoning("Answer: 2<think>still thinking Answer: 1"), "Answer: 2")
        self.assertEqual(checkers.strip_reasoning("[THINK]x[/THINK]y"), "y")
        self.assertEqual(checkers.strip_reasoning(None), "")


class ArithmeticTest(unittest.TestCase):
    check = staticmethod(checkers.check_arithmetic)

    def test_final_answer_line(self):
        self.assertTrue(self.check("17*23 = 391\nAnswer: 378", {"expected": "378"})["passed"])
        self.assertTrue(self.check("**Answer:** $38.88", {"expected": "38.88"})["passed"])
        self.assertTrue(self.check("Answer: 4,410", {"expected": "4410"})["passed"])
        self.assertTrue(self.check("Answer: 9.90", {"expected": "9.9"})["passed"])
        self.assertTrue(self.check("Answer: −4", {"expected": "-4"})["passed"])

    def test_wrong_or_missing_answer_fails(self):
        # The right number appears in the working but the final answer is wrong.
        self.assertFalse(self.check("378 looks close.\nAnswer: 377", {"expected": "378"})["passed"])
        self.assertFalse(self.check("The result is 378.", {"expected": "378"})["passed"])
        self.assertFalse(self.check("Answer: 1,13", {"expected": "113"})["passed"])
        self.assertFalse(self.check("Answer: 4", {"expected": "-4"})["passed"])

    def test_answer_inside_reasoning_does_not_count(self):
        text = "<think>Answer: 42</think>Answer: 41"
        self.assertFalse(self.check(text, {"expected": "42"})["passed"])


class CodeTest(unittest.TestCase):
    PARAMS = {"entry_point": "add", "hidden_tests": "assert mod.add(2, 3) == 5\n", "timeout_sec": 5}

    def run_code(self, text, params=None):
        with scratch() as root:
            result = checkers.check_code(text, params or self.PARAMS, tmp_root=root)
            self.assertEqual(os.listdir(root), [], "sandbox directory not cleaned up")
            return result

    def test_correct_and_incorrect(self):
        self.assertTrue(self.run_code(fenced("def add(a, b):\n    return a + b\n"))["passed"])
        self.assertFalse(self.run_code(fenced("def add(a, b):\n    return a - b\n"))["passed"])
        self.assertFalse(self.run_code(fenced("def plus(a, b):\n    return a + b\n"))["passed"])
        self.assertFalse(self.run_code("I cannot write that.")["passed"])

    def test_extraction_prefers_defining_python_block_and_ignores_reasoning(self):
        text = ("<think>```python\ndef add(a, b):\n    return a + b\n```</think>"
                "```bash\necho hi\n```\n```python\ndef add(a, b):\n    return b + a\n```")
        self.assertTrue(self.run_code(text)["passed"])
        only_think = "<think>```python\ndef add(a, b):\n    return a + b\n```</think>Done."
        self.assertFalse(self.run_code(only_think)["passed"])
        self.assertTrue(self.run_code("def add(a, b):\n    return a + b\n")["passed"])

    def test_timeout_stdin_and_exit_are_failures(self):
        params = dict(self.PARAMS, timeout_sec=1.5)
        self.assertIn("timed out", self.run_code(fenced("def add(a, b):\n    while True:\n        pass\n"),
                                                 params)["reason"])
        self.assertFalse(self.run_code(fenced("x = input()\ndef add(a, b):\n    return a + b\n"))["passed"])
        cheat = "import sys\nprint('Q4_PASS')\nsys.exit(0)\ndef add(a, b):\n    return 0\n"
        self.assertFalse(self.run_code(fenced(cheat))["passed"])

    def test_sandbox_env_and_location(self):
        with scratch() as root:
            params = {"entry_point": "probe", "timeout_sec": 5,
                      "hidden_tests": "import os\nenv, cwd = mod.probe()\n"
                                      "assert 'PATH' not in env and 'HOME' not in env, env\n"
                                      f"assert cwd.startswith({os.path.realpath(root)!r}), cwd\n"}
            code = "import os\ndef probe():\n    return dict(os.environ), os.path.realpath(os.getcwd())\n"
            self.assertTrue(checkers.check_code(fenced(code), params, tmp_root=root)["passed"])

    def test_refuses_missing_tmp_root(self):
        with self.assertRaises(RuntimeError):
            checkers.check_code(fenced("def add(a, b):\n    return a + b\n"), self.PARAMS,
                                tmp_root="/nonexistent/q4")


class ToolArgsTest(unittest.TestCase):
    PARAMS = {"name": "create_event", "expected_args": {"title": "Q4 review", "date": "2026-11-03",
                                                        "duration_minutes": 90},
              "schema": build_corpus.TOOLBOX[3]["function"]["parameters"]}

    def check(self, response):
        return checkers.check_tool_args(response, self.PARAMS)

    def test_exact_call_passes(self):
        args = {"duration_minutes": 90, "date": "2026-11-03", "title": "Q4 review"}
        self.assertTrue(self.check(tool_response("create_event", args))["passed"])

    def test_wrong_calls_fail(self):
        good = self.PARAMS["expected_args"]
        self.assertFalse(self.check(tool_response("create_event", dict(good, duration_minutes=60)))["passed"])
        self.assertFalse(self.check(tool_response("create_event", dict(good, duration_minutes=90.0)))["passed"])
        self.assertFalse(self.check(tool_response("create_event", dict(good, notes="x")))["passed"])
        self.assertFalse(self.check(tool_response("web_search", good))["passed"])
        self.assertFalse(self.check(tool_response("create_event", "{not json"))["passed"])
        self.assertFalse(self.check({"tool_calls": ["create_event"]})["passed"])
        two = tool_response("create_event", good)
        two["tool_calls"] *= 2
        self.assertFalse(self.check(two)["passed"])
        # Arguments written as text instead of a tool call do not count.
        self.assertFalse(self.check({"content": json.dumps(good), "tool_calls": None})["passed"])

    def test_strict_equality(self):
        self.assertFalse(checkers.same_json(True, 1))
        self.assertTrue(checkers.same_json({"a": [1, 2.0]}, {"a": [1.0, 2]}))
        self.assertFalse(checkers.same_json({"a": 1}, {"a": 1, "b": None}))


class StructuredTest(unittest.TestCase):
    PARAMS = {"schema": {"type": "object", "additionalProperties": False, "required": ["a", "b"],
                         "properties": {"a": {"type": "integer"}, "b": {"type": ["string", "null"]}}},
              "expected": {"a": 3, "b": None}}

    def check(self, text):
        return checkers.check_structured(text, self.PARAMS)

    def test_valid_documents(self):
        self.assertTrue(self.check('{"a": 3, "b": null}')["passed"])
        self.assertTrue(self.check('```json\n{"b": null, "a": 3}\n```')["passed"])
        self.assertTrue(self.check('<think>{"a": 1}</think>{"a": 3, "b": null}')["passed"])

    def test_invalid_documents(self):
        self.assertFalse(self.check('Sure! {"a": 3, "b": null}')["passed"])
        self.assertFalse(self.check('{"a": 4, "b": null}')["passed"])
        self.assertFalse(self.check('{"a": 3}')["passed"])
        self.assertFalse(self.check('{"a": 3, "b": null, "c": 1}')["passed"])
        self.assertFalse(self.check('{"a": 3, "a": 3, "b": null}')["passed"])
        self.assertFalse(self.check('{"a": true, "b": null}')["passed"])


class MultilingualTest(unittest.TestCase):
    def check(self, text, language, groups):
        return checkers.check_multilingual(text, {"language": language, "all_of": groups})["passed"]

    def test_language_and_facts(self):
        self.assertTrue(self.check("La capitale de l'Australie est Canberra.", "fr", [["Canberra"]]))
        self.assertFalse(self.check("The capital of Australia is Canberra.", "fr", [["Canberra"]]))
        self.assertFalse(self.check("La capitale de l'Australie est Sydney.", "fr", [["Canberra"]]))
        self.assertTrue(self.check("Eine Spinne hat acht Beine.", "de", [["8", "acht"]]))
        self.assertFalse(self.check("Eine Spinne hat 18 Beine.", "de", [["8", "acht"]]))
        self.assertTrue(self.check("世界上最大的海洋是太平洋。", "zh", [["太平洋"]]))
        self.assertFalse(self.check("世界で一番大きい海は太平洋です。", "zh", [["太平洋"]]))
        self.assertTrue(self.check("一年は１２か月です。", "ja", [["12", "十二"]]))
        self.assertFalse(self.check("A year has 12 months.", "ja", [["12", "十二"]]))
        self.assertTrue(self.check("El planeta más grande es Júpiter.", "es", [["Júpiter", "Jupiter"]]))


class LongContextTest(unittest.TestCase):
    PARAMS = {"needle": "ABC-1234", "forbidden": ["XYZ-9876"]}

    def test_needle_and_distractors(self):
        self.assertTrue(checkers.check_long_context("ABC-1234", self.PARAMS)["passed"])
        self.assertFalse(checkers.check_long_context("XYZ-9876", self.PARAMS)["passed"])
        self.assertFalse(checkers.check_long_context("ABC-1234 or XYZ-9876", self.PARAMS)["passed"])
        self.assertFalse(checkers.check_long_context("<think>ABC-1234</think>unsure", self.PARAMS)["passed"])

    def test_generator_is_deterministic_and_places_needle(self):
        a = generate_long_context.generate(7, 5000, 0.9)
        self.assertEqual(a, generate_long_context.generate(7, 5000, 0.9))
        self.assertNotEqual(a[0], generate_long_context.generate(8, 5000, 0.9)[0])
        prompt, needle, others, locker = a
        self.assertEqual(prompt.count(needle), 1)
        self.assertIn(f"locker {locker} was assigned the access code {needle}.", prompt)
        self.assertGreater(prompt.index(needle) / len(prompt), 0.7)
        self.assertEqual(len(set(others + [needle])), len(others) + 1)


class SustainedTest(unittest.TestCase):
    def check(self, response, params):
        return checkers.check_sustained(response, params)["passed"]

    def test_sequences(self):
        ints = {"mode": "ints", "expected": list(range(1, 11))}
        self.assertTrue(self.check({"content": "\n".join(map(str, range(1, 11))), "finish_reason": "stop"}, ints))
        skipped = [n for n in range(1, 11) if n != 7]
        self.assertFalse(self.check({"content": " ".join(map(str, skipped)), "finish_reason": "stop"}, ints))
        self.assertFalse(self.check({"content": "\n".join(map(str, range(1, 11))), "finish_reason": "length"}, ints))
        lines = {"mode": "lines", "expected": ["Alabama", "Alaska"]}
        self.assertTrue(self.check("Alabama\n\n  Alaska \n", lines))
        self.assertFalse(self.check("1. Alabama\n2. Alaska", lines))
        primes = {"mode": "ints", "expected": [2, 3, 5]}
        self.assertTrue(self.check("Here are the first 3 primes:\n2, 3, 5", primes))
        self.assertFalse(self.check("1. 2\n2. 3\n3. 5", primes))
        doc = {"mode": "json", "expected": [{"id": 1}, {"id": 2}]}
        self.assertTrue(self.check('[{"id": 1}, {"id": 2}]', doc))
        self.assertFalse(self.check('[{"id": 1}]', doc))
        looped = checkers.check_sustained("1 2 3 4 4 4 4 4", {"mode": "ints", "expected": [1, 2, 3, 4, 5]})
        self.assertIn("repetition loop", looped["reason"])

    def test_prose_and_loops(self):
        prose = {"mode": "prose", "min_words": 20}
        text = " ".join(f"word{i}" for i in range(30))
        self.assertTrue(self.check({"content": text, "finish_reason": "stop"}, prose))
        self.assertFalse(self.check({"content": "too short", "finish_reason": "stop"}, prose))
        looped = text + " the sea is calm" * 6
        self.assertFalse(self.check({"content": looped, "finish_reason": "stop"}, prose))
        self.assertTrue(checkers.has_repetition_loop("a b c a b c a b c a b c"))
        self.assertFalse(checkers.has_repetition_loop("a b c a b c a b c d"))
        self.assertFalse(checkers.has_repetition_loop("\n".join(map(str, range(200)))))


class WilsonTest(unittest.TestCase):
    def test_known_values(self):
        low, high = run_q4.wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=4)
        self.assertAlmostEqual(high, 0.7634, places=4)
        low, high = run_q4.wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.2775, places=4)
        self.assertAlmostEqual(run_q4.wilson_interval(10, 10)[0], 0.7225, places=4)
        self.assertEqual(run_q4.wilson_interval(0, 0), (0.0, 1.0))
        with self.assertRaises(ValueError):
            run_q4.wilson_interval(11, 10)


class CorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.heldout, cls.tuning = load("corpus.jsonl"), load("tuning.jsonl")
        cls.all = cls.heldout + cls.tuning

    def test_committed_files_match_builder(self):
        heldout, tuning = build_corpus.build()
        for name, rows in (("corpus.jsonl", heldout), ("tuning.jsonl", tuning)):
            with open(os.path.join(Q4, name), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), build_corpus.render(rows), name + " is stale")

    def test_schema(self):
        self.assertGreaterEqual(len(self.heldout), 60)
        self.assertEqual(len(self.tuning), 10)
        ids = [case["id"] for case in self.all]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({case["split"] for case in self.heldout}, {"heldout"})
        self.assertEqual({case["split"] for case in self.tuning}, {"tuning"})
        counts = {}
        for case in self.all:
            self.assertEqual(case["category"], case["checker"], case["id"])
            self.assertIn(case["checker"], checkers.CHECKERS)
            self.assertLessEqual(checkers.REQUIRED_PARAMS[case["checker"]], set(case["checker_params"]))
            self.assertEqual(case["sampling"]["temperature"], 0.0)
            self.assertIsInstance(case["max_tokens"], int)
            self.assertTrue(case["messages"] and all(m["content"] for m in case["messages"]))
            counts[case["category"]] = counts.get(case["category"], 0) + (case["split"] == "heldout")
        self.assertEqual(set(counts), set(checkers.CHECKERS))
        self.assertGreaterEqual(min(counts.values()), 8, counts)

    def test_tuning_disjoint_from_heldout(self):
        prompts = lambda rows: {json.dumps(case["messages"], sort_keys=True) for case in rows}
        self.assertTrue(prompts(self.heldout).isdisjoint(prompts(self.tuning)))
        self.assertTrue({c["id"] for c in self.heldout}.isdisjoint({c["id"] for c in self.tuning}))
        self.assertEqual(len(prompts(self.heldout)), len(self.heldout), "duplicate held-out prompts")

    def test_long_context_fits_serving_limit(self):
        long_cases = [c for c in self.heldout if c["category"] == "long_context"]
        near = [c for c in long_cases if c["measured_content_tokens"] > 3500]
        self.assertGreaterEqual(len(near), 3)
        self.assertGreaterEqual(len(long_cases) - len(near), 6)
        for case in long_cases:
            # The chat template needs a few tokens; reasoning needs headroom.
            self.assertLessEqual(case["measured_content_tokens"] + case["max_tokens"], SERVING_LIMIT - 64)
            self.assertGreaterEqual(case["max_tokens"], 256)
            self.assertGreaterEqual(case["measured_content_tokens"], 2900)

    def test_expected_answers_pass_and_near_misses_fail(self):
        """Every frozen expectation is satisfiable by a correct answer and is
        not satisfied by an obviously wrong one."""
        for case in self.all:
            params, kind = case["checker_params"], case["checker"]
            with self.subTest(case=case["id"]):
                if kind == "arithmetic":
                    good = "Working...\nAnswer: " + params["expected"]
                    bad = "Answer: " + str(build_corpus.Fraction(params["expected"]) + 1)
                elif kind == "code":
                    continue  # covered by test_reference_solutions_pass
                elif kind == "tool_args":
                    names = [t["function"]["name"] for t in case["tools"]]
                    self.assertIn(params["name"], names)
                    self.assertEqual(case["tool_choice"], "auto")
                    good = tool_response(params["name"], params["expected_args"])
                    key = sorted(params["expected_args"])[0]
                    bad = tool_response(params["name"], dict(params["expected_args"], **{key: "zz"}))
                elif kind == "structured":
                    good = json.dumps(params["expected"], ensure_ascii=False)
                    bad = json.dumps({"wrong": True})
                    if "response_format" in case:
                        self.assertEqual(case["response_format"]["json_schema"]["schema"], params["schema"])
                elif kind == "multilingual":
                    continue  # facts depend on phrasing; covered by MultilingualTest
                elif kind == "long_context":
                    good, bad = params["needle"], params["forbidden"][0]
                    self.assertIn(params["needle"], case["messages"][0]["content"])
                elif kind == "sustained":
                    if params["mode"] == "prose":
                        continue
                    exp = params["expected"]
                    if params["mode"] == "json":
                        good, bad = json.dumps(exp), json.dumps(exp[:-1])
                    else:
                        good = "\n".join(map(str, exp))
                        bad = "\n".join(map(str, exp[:-1]))
                    good = {"content": good, "finish_reason": "stop"}
                    bad = {"content": bad, "finish_reason": "stop"}
                self.assertTrue(checkers.CHECKERS[kind](good, params)["passed"])
                self.assertFalse(checkers.CHECKERS[kind](bad, params)["passed"])

    def test_reference_solutions_pass(self):
        code_cases = [c for c in self.all if c["checker"] == "code"]
        self.assertEqual({c["id"] for c in code_cases}, set(build_corpus.REFERENCE_SOLUTIONS))
        with scratch() as root:
            for case in code_cases:
                with self.subTest(case=case["id"]):
                    ref = build_corpus.REFERENCE_SOLUTIONS[case["id"]]
                    result = checkers.check_code(fenced(ref), case["checker_params"], tmp_root=root)
                    self.assertTrue(result["passed"], result["reason"])
                    broken = ref.replace("return", "return None and", 1)
                    self.assertFalse(checkers.check_code(fenced(broken), case["checker_params"],
                                                         tmp_root=root)["passed"])

    def test_multilingual_answers_satisfiable(self):
        answers = {"ml01": "La capitale de l'Australie est Canberra.", "ml02": "Eine Spinne hat acht Beine.",
                   "ml03": "世界上面积最大的海洋是太平洋。", "ml04": "日本で一番高い山は富士山です。",
                   "ml05": "Un hexagone a six côtés.", "ml06": "Das Element mit dem Symbol O heißt Sauerstoff.",
                   "ml07": "一个星期有七天。", "ml08": "一年は12か月あります。",
                   "ml09": "El planeta más grande del sistema solar es Júpiter.",
                   "ml10": "Die Katze schläft auf dem Sofa.", "ml11": "Je voudrais un verre d'eau, s'il vous plaît.",
                   "ml12": "今天是星期一。", "tu07": "Der Tag hat 24 Stunden.", "tu08": "Le ciel est bleu."}
        cases = {c["id"]: c for c in self.all if c["checker"] == "multilingual"}
        self.assertEqual(set(cases), set(answers))
        for cid, text in answers.items():
            with self.subTest(case=cid):
                self.assertTrue(checkers.check_multilingual(text, cases[cid]["checker_params"])["passed"])


class FakeChat(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeChat.requests.append(payload)
        prompt = payload["messages"][0]["content"]
        if "server error" in prompt:
            self.reply(500, {"error": {"message": "Answer: 42"}})
        elif "error body" in prompt:
            self.reply(200, {"error": {"message": "Answer: 42"}})
        elif payload.get("tools"):
            self.reply(200, {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "get_stock_price", "arguments": "{\"ticker\": \"NVDA\"}"}}]}}]})
        else:
            self.reply(200, {"choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "<think>Answer: 41</think>6*7\nAnswer: 42"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 5}})

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class RunnerTest(unittest.TestCase):
    def setUp(self):
        FakeChat.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def write_cases(self, root):
        tool = next(c for c in load("corpus.jsonl") if c["id"] == "ta08")
        rows = [build_corpus.case("x01", "arithmetic", "What is 6*7?", "arithmetic", {"expected": "42"}, 32),
                build_corpus.case("x02", "arithmetic", "server error", "arithmetic", {"expected": "42"}, 32),
                build_corpus.case("x03", "arithmetic", "error body", "arithmetic", {"expected": "42"}, 32),
                tool]
        path = os.path.join(root, "cases.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(build_corpus.render(rows))
        return path

    def test_run_scores_with_checkers_and_keeps_raw_outputs(self):
        with scratch() as root:
            cases, out = self.write_cases(root), os.path.join(root, "run")
            argv = ["--endpoint", self.endpoint, "--model", "m", "--out", out, "--trials", "2",
                    "--cases", cases, "--profile", "fake", "--tmp-root", root]
            self.assertEqual(run_q4.main(argv), 1)  # 2 trials < min_trials, and failures
            with open(os.path.join(out, "summary.json"), encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["cases"]["x01"]["passes"], 2)
            # HTTP 500 and a 200 error body fail even though they contain "Answer: 42".
            self.assertEqual(summary["cases"]["x02"]["passes"], 0)
            self.assertEqual(summary["cases"]["x03"]["passes"], 0)
            self.assertEqual(summary["cases"]["ta08"]["passes"], 2)
            arithmetic = summary["categories"]["arithmetic"]
            self.assertEqual((arithmetic["passes"], arithmetic["n"]), (2, 6))
            self.assertEqual(arithmetic["wilson95"], list(run_q4.wilson_interval(2, 6)))
            self.assertEqual(arithmetic["verdict"], "INSUFFICIENT_TRIALS")
            self.assertEqual((arithmetic["cases_majority_passed"], arithmetic["cases"]), (1, 3))
            self.assertEqual(arithmetic["case_wilson95"], list(run_q4.wilson_interval(1, 3)))
            with open(os.path.join(out, "cases", "x02.json"), encoding="utf-8") as handle:
                failed = json.load(handle)
            self.assertEqual(failed["outputs"][0]["http_status"], 500)
            self.assertIn("request failed", failed["outputs"][0]["reason"])
            with open(os.path.join(out, "cases", "x01.json"), encoding="utf-8") as handle:
                ok = json.load(handle)
            self.assertEqual(ok["outputs"][1]["usage"]["prompt_tokens"], 9)
            self.assertIn("raw_response", ok["outputs"][0])
            sent = FakeChat.requests[0]
            self.assertEqual((sent["temperature"], sent["seed"]), (0.0, build_corpus.SEED))
            self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
            tool_request = next(r for r in FakeChat.requests if r.get("tools"))
            self.assertEqual(tool_request["tool_choice"], "auto")
            # Never overwrite an earlier run.
            with self.assertRaises(FileExistsError):
                run_q4.main(argv)

    def test_case_ids_must_be_safe_file_names(self):
        with scratch() as root:
            path = os.path.join(root, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(build_corpus.render([build_corpus.case(
                    "../escape", "arithmetic", "q", "arithmetic", {"expected": "1"}, 8)]))
            with self.assertRaises(ValueError):
                run_q4.load_cases(path)

    def test_compare_flags_only_non_overlapping_drops(self):
        def summary(path, passes, sha="s"):
            low, high = run_q4.wilson_interval(passes, 50)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"profile": path, "cases_sha256": sha, "cases": {}, "categories": {
                    "code": {"passes": passes, "n": 50, "pass_rate": passes / 50, "wilson95": [low, high]}}},
                    handle)
            return path

        with scratch() as root:
            a = summary(os.path.join(root, "a.json"), 48)
            self.assertEqual(run_q4.compare(a, summary(os.path.join(root, "b.json"), 20)), 1)
            self.assertEqual(run_q4.compare(a, summary(os.path.join(root, "c.json"), 46)), 0)
            self.assertEqual(run_q4.compare(summary(os.path.join(root, "d.json"), 20), a), 0)
            with self.assertRaises(SystemExit):
                run_q4.compare(a, summary(os.path.join(root, "e.json"), 48, sha="other"))


if __name__ == "__main__":
    unittest.main()
