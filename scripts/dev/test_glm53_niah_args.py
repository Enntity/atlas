# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only admission bounds for the independent GLM needle harness."""

import contextlib
import io
import json
import re
import sys
import threading
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_glm53_concurrent_niah import main, needles_for, parse_args


class NeedleArgsTests(unittest.TestCase):
    def four_rows(self):
        return ["--prompt-tokens", "768", "800", "832", "896",
                "--output-tokens", "32", "64", "96", "128"]

    def rejects(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parse_args(argv)
        self.assertEqual(error.exception.code, 2)

    def test_default_three_rows_unchanged(self):
        args = parse_args([])
        self.assertEqual(args.prompt_tokens, [2047, 2048, 2049])
        self.assertEqual(args.output_tokens, [32, 64, 96])
        self.assertIsNone(args.context_limit)

    def test_four_rows_need_explicit_sufficient_context(self):
        self.rejects(self.four_rows())
        self.rejects(self.four_rows() + ["--context-limit", "1023"])
        args = parse_args(self.four_rows() + ["--context-limit", "1024"])
        self.assertEqual(len(args.prompt_tokens), 4)
        self.assertEqual(args.context_limit, 1024)

    def test_context_bound_also_applies_to_smaller_batches(self):
        self.rejects(["--prompt-tokens", "1000", "--output-tokens", "25",
                      "--context-limit", "1024"])
        args = parse_args(["--prompt-tokens", "1000", "--output-tokens", "24",
                           "--context-limit", "1024"])
        self.assertEqual(args.context_limit, 1024)

    def test_invalid_lengths_counts_and_timeouts_fail_before_http(self):
        for argv in [
            ["--prompt-tokens", *["100"] * 9,
             "--output-tokens", *["32"] * 9, "--context-limit", "2048"],
            ["--prompt-tokens", "0", "--output-tokens", "1"],
            ["--prompt-tokens", "100"],
            ["--repetitions", "0"], ["--position", "nan"],
            ["--timeout", "nan"], ["--timeout", "inf"], ["--timeout", "0"],
        ]:
            self.rejects(argv)

    def test_needles_are_distinct_across_rows_and_repetitions(self):
        first, second = needles_for(8, 0), needles_for(8, 1)
        self.assertEqual(len(set(first)), 8)
        self.assertTrue(first[3].startswith("QUASAR-"))
        self.assertFalse(set(first) & set(second))
        self.assertEqual(needles_for(3, 0), first[:3])
        for count, repetition in [(0, 0), (9, 0), (4, -1)]:
            with self.assertRaises(ValueError):
                needles_for(count, repetition)

    def test_all_larger_widths_require_context(self):
        for count in range(4, 9):
            argv = ["--prompt-tokens", *["768"] * count,
                    "--output-tokens", *["32"] * count]
            self.rejects(argv)
            self.assertEqual(len(parse_args(argv + ["--context-limit", "2048"]).prompt_tokens), count)

    def test_actual_client_dispatch_and_cross_needle_refusal(self):
        # Run the actual tokenization + streaming client with only requests.post
        # stubbed. Character-token encoding is a CPU boundary, not model numerics.
        import benchmark_glm53_niah as client
        for count, crossed in [(n, False) for n in range(1, 9)] + [(8, True)]:
            needles = needles_for(count, 0)
            seen = []
            lock = threading.Lock()
            http_wave = threading.Barrier(count)
            encode_event = json.dumps

            def post(url, json, timeout, stream=False):
                response = mock.MagicMock()
                response.__enter__.return_value = response
                if url.endswith("/tokenize"):
                    response.json.return_value = {"tokens": list(json["prompt"].encode())}
                    return response
                self.assertTrue(url.endswith("/v1/completions"))
                prompt = bytes(json["prompt"]).decode()
                needle = re.search(r"codename is ([A-Z]+-\d+)\.", prompt)[1]
                with lock:
                    seen.append((needle, len(json["prompt"]), json["max_tokens"]))
                http_wave.wait(timeout=5)
                text = needle + (" " + needles[1] if crossed and needle == needles[0] else "")
                event = {"choices": [{"text": text}], "usage": {
                    "prompt_tokens": len(json["prompt"]), "completion_tokens": 8,
                    "time_to_first_token_ms": 10, "response_token/s": 1}}
                response.iter_lines.return_value = ["data: " + encode_event(event),
                                                     "data: [DONE]"]
                return response

            prompts = [str(768 + i * 8) for i in range(count)]
            outputs = [str(32 + i * 4) for i in range(count)]
            argv = ["niah", "--prompt-tokens", *prompts, "--output-tokens", *outputs,
                    "--context-limit", "2048"]
            output = io.StringIO()
            with mock.patch("sys.argv", argv), mock.patch.object(client.requests, "post", side_effect=post), \
                    contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as done:
                main()
            self.assertEqual(done.exception.code, 1 if crossed else 0)
            receipt = json.loads(output.getvalue())
            self.assertEqual(sorted(seen), sorted(zip(needles, map(int, prompts), map(int, outputs))))
            self.assertEqual(len(receipt["requests"]), count)
            self.assertEqual(receipt["requests"][0]["foreign_needles"], [needles[1]] if crossed else [])


if __name__ == "__main__":
    unittest.main()
