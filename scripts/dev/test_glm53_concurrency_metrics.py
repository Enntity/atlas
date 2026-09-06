# SPDX-License-Identifier: AGPL-3.0-only
"""GPU-free metric checks for the GLM concurrency harness."""

import argparse
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_glm53_concurrency import summarize_batch, summarize_runs, workload_metadata


class MetricsTests(unittest.TestCase):
    def rows(self):
        return [
            dict(started=1.0, first_text=2.0, ended=5.0, prompt_tokens=1000,
                 completion_tokens=96, ttft_ms=800.0, decode_tps=20.0),
            dict(started=1.1, first_text=3.0, ended=6.0, prompt_tokens=1000,
                 completion_tokens=96, ttft_ms=900.0, decode_tps=30.0),
        ]

    def test_e2e_and_post_first_token_windows_are_distinct(self):
        result = summarize_batch(self.rows(), 96)
        self.assertEqual(result["aggregate_e2e_tps"], 38.4)
        self.assertEqual(result["aggregate_decode_window_tps"], 48.0)
        self.assertEqual(result["aggregate_post_first_token_tps"], 47.5)
        self.assertEqual(result["sum_session_decode_tps"], 50.0)
        self.assertEqual(result["batch_wall_seconds"], 5.0)
        self.assertEqual(result["median_client_ttft_ms"], 1450.0)
        self.assertEqual(result["median_ttft_ms"], 850.0)
        self.assertTrue(result["all_outputs_reached_cap"])

    def test_early_stop_is_not_reported_as_fixed_output(self):
        rows = self.rows()
        rows[1]["completion_tokens"] = 49
        result = summarize_batch(rows, 96)
        self.assertFalse(result["all_outputs_reached_cap"])
        self.assertEqual(result["completion_tokens_total"], 145)
        self.assertEqual(result["output_tokens_requested_total"], 192)
        self.assertEqual(result["aggregate_post_first_token_tps"], 35.75)

    def test_single_token_streams_have_no_post_first_tokens(self):
        rows = self.rows()
        for row in rows:
            row["completion_tokens"] = 1
        result = summarize_batch(rows, 1)
        self.assertEqual(result["aggregate_decode_window_tps"], 0.5)
        self.assertEqual(result["aggregate_post_first_token_tps"], 0.0)

    def test_repetition_permission_does_not_force_output_cap(self):
        rows = self.rows()
        first = summarize_batch(rows, 96)
        complete = summarize_runs(2, True, [first])
        self.assertTrue(complete["all_outputs_reached_cap"])
        self.assertFalse(complete["forced_output_cap"])
        rows[1]["completion_tokens"] = 49
        second = summarize_batch(rows, 96)
        result = summarize_runs(2, True, [first, second])
        self.assertFalse(result["all_outputs_reached_cap"])
        self.assertFalse(result["forced_output_cap"])
        self.assertEqual(result["median_aggregate_post_first_token_tps"], 41.625)
        self.assertEqual(result["median_aggregate_decode_window_tps"], 42.125)

    def test_workload_fingerprint_has_documented_stable_encoding(self):
        args = argparse.Namespace(model="test-model", prompt_tokens=3, output_tokens=96,
                                  repetitions=2, allow_repetition=True)
        result = workload_metadata(args, [1, 23, 4])
        self.assertEqual(result["prompt_token_sha256"],
                         hashlib.sha256(b"[1,23,4]").hexdigest())
        self.assertNotEqual(result["prompt_token_sha256"],
                            workload_metadata(args, [12, 3, 4])["prompt_token_sha256"])
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["prompt_source"], "generated")
        self.assertFalse(result["forced_output_cap"])
        self.assertNotIn("literal_prompt_sha256", result)
        literal = workload_metadata(args, [1, 23, 4], "Hello\r\n🌌")
        self.assertEqual(literal["prompt_source"], "literal")
        self.assertEqual(literal["literal_prompt_sha256"],
                         hashlib.sha256("Hello\r\n🌌".encode("utf-8")).hexdigest())

    def test_invalid_completion_counts_and_mixed_prompt_lengths_fail(self):
        for count in [0, -1, 97]:
            rows = self.rows()
            rows[0]["completion_tokens"] = count
            with self.assertRaises(ValueError):
                summarize_batch(rows, 96)
        rows = self.rows()
        rows[0]["prompt_tokens"] = 999
        with self.assertRaises(ValueError):
            summarize_batch(rows, 96)

    def test_invalid_time_order_or_empty_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_batch([], 96)
        rows = self.rows()
        rows[0]["ended"] = 1.5
        with self.assertRaises(ValueError):
            summarize_batch(rows, 96)
        for timing in [float("nan"), float("inf")]:
            rows = self.rows()
            rows[0]["ended"] = timing
            with self.assertRaises(ValueError):
                summarize_batch(rows, 96)

    def test_empty_or_mismatched_measured_batches_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_runs(2, False, [])
        with self.assertRaises(ValueError):
            summarize_runs(3, False, [summarize_batch(self.rows(), 96)])


if __name__ == "__main__":
    unittest.main()
