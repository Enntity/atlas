# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the real CLI loop without issuing any network requests."""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_glm53_concurrency as benchmark


class WarmupReceiptTests(unittest.TestCase):
    def run_cli(self, short_warmup=False, short_measured=False):
        calls = []

        def fake_batch(args, concurrency, prompt):
            ordinal = len(calls) % 4
            calls.append((concurrency, ordinal, prompt))
            count = 4 if (short_warmup and ordinal == 0) or (
                short_measured and ordinal == 2
            ) else 8
            # Warmup is deliberately much slower than every measured wave.
            elapsed = 100.0 if ordinal == 0 else float(ordinal + 1)
            return benchmark.summarize_batch([
                dict(started=0.0, first_text=0.5, ended=elapsed,
                     prompt_tokens=2, completion_tokens=count,
                     ttft_ms=500.0, decode_tps=count / (elapsed - 0.5),
                     finish_reason="stop" if count < 8 else "length",
                     completion_text_sha256=f"owner-{concurrency}-{slot}-{ordinal}",
                     completion_text_bytes=count)
                for slot in range(concurrency)
            ], args.output_tokens)

        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["benchmark_glm53_concurrency.py", "--min-concurrency", "1",
                "--max-concurrency", "4", "--repetitions", "3",
                "--output-tokens", "8"]
        with patch.object(sys, "argv", argv), \
             patch("benchmark_glm53.make_prompt", return_value=[1, 2]), \
             patch.object(benchmark, "batch", side_effect=fake_batch), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            benchmark.main()
        return json.loads(stdout.getvalue()), [
            json.loads(line) for line in stderr.getvalue().splitlines()
        ], calls

    def test_c1_through_c4_preserve_warmup_without_measuring_it(self):
        results, progress, calls = self.run_cli()
        self.assertEqual(results, progress)
        self.assertEqual(calls, [(c, wave, [1, 2])
                                 for c in range(1, 5) for wave in range(4)])
        for concurrency, result in enumerate(results, 1):
            self.assertEqual(len(result["warmup_runs"]), 1)
            self.assertEqual(len(result["runs"]), 3)
            warmup = result["warmup_runs"][0]
            self.assertEqual(warmup["batch_wall_seconds"], 100.0)
            self.assertTrue(result["warmup_all_outputs_reached_cap"])
            self.assertTrue(result["all_outputs_reached_cap"])
            self.assertFalse(result["forced_output_cap"])
            self.assertEqual(result["median_aggregate_e2e_tps"],
                             round(8 * concurrency / 3, 3))
            for slot, request in enumerate(warmup["requests"]):
                self.assertEqual(request["completion_text_sha256"],
                                 f"owner-{concurrency}-{slot}-0")
                self.assertEqual(request["completion_tokens"], 8)
                self.assertEqual(request["end_offset_ms"], 100000.0)

    def test_warmup_early_stop_does_not_change_measured_cap_flag(self):
        results, _, _ = self.run_cli(short_warmup=True)
        for result in results:
            self.assertFalse(result["warmup_all_outputs_reached_cap"])
            self.assertTrue(result["all_outputs_reached_cap"])
            self.assertEqual(result["warmup_runs"][0]["requests"][0]["finish_reason"],
                             "stop")

    def test_measured_early_stop_is_not_hidden_by_complete_warmup(self):
        results, _, _ = self.run_cli(short_measured=True)
        for result in results:
            self.assertTrue(result["warmup_all_outputs_reached_cap"])
            self.assertFalse(result["all_outputs_reached_cap"])
            self.assertEqual(result["runs"][1]["requests"][0]["completion_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
