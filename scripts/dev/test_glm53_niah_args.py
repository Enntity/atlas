# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only admission bounds for the independent GLM needle harness."""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_glm53_concurrent_niah import needles_for, parse_args


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
            ["--prompt-tokens", "1", "2", "3", "4", "5",
             "--output-tokens", "1", "2", "3", "4", "5"],
            ["--prompt-tokens", "0", "--output-tokens", "1"],
            ["--prompt-tokens", "100"],
            ["--repetitions", "0"], ["--position", "nan"],
            ["--timeout", "nan"], ["--timeout", "inf"], ["--timeout", "0"],
        ]:
            self.rejects(argv)

    def test_needles_are_distinct_across_rows_and_repetitions(self):
        first, second = needles_for(4, 0), needles_for(4, 1)
        self.assertEqual(len(set(first)), 4)
        self.assertTrue(first[3].startswith("QUASAR-"))
        self.assertFalse(set(first) & set(second))
        self.assertEqual(needles_for(3, 0), first[:3])
        for count, repetition in [(0, 0), (5, 0), (4, -1)]:
            with self.assertRaises(ValueError):
                needles_for(count, repetition)


if __name__ == "__main__":
    unittest.main()
