# SPDX-License-Identifier: AGPL-3.0-only
"""CPU gates for the bounded cross-engine chat comparison client."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_glm53_reference_chat import make_payload, read_stream, summarize_wave


def event(**value):
    return "data: " + json.dumps(value)


class ReferenceChatTests(unittest.TestCase):
    def lines(self, field="reasoning_content", count=256):
        return [
            event(choices=[{"delta": {"role": "assistant"}}]),
            event(choices=[{"delta": {field: "first"}}]),
            event(choices=[{"delta": {"content": "answer"}, "finish_reason": "length"}]),
            event(choices=[], usage={"prompt_tokens": 32, "completion_tokens": count}),
            "data: [DONE]",
        ]

    def test_payload_keeps_peer_prompt_and_explicit_eos_difference(self):
        payload = make_payload("model")
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["temperature"], 0)
        self.assertNotIn("ignore_eos", payload)
        self.assertNotIn("thinking_token_budget", payload)
        self.assertNotIn("repetition_detection", payload)
        self.assertEqual(payload["messages"][0]["content"],
                         "Write a compact Python function that validates a topological ordering. "
                         "Return code only.")

    def test_both_reasoning_names_and_content_start_ttft(self):
        for field in ("reasoning", "reasoning_content", "content", "tool_calls"):
            times = iter([2.0, 5.0])
            row = read_stream(self.lines(field), 1.0, lambda: next(times))
            self.assertEqual(row["first_text"], 2.0)
            self.assertEqual(row["ended"], 5.0)
            self.assertEqual(row["completion_tokens"], 256)
            self.assertEqual(row["decode_tps"], 85.0)

    def test_incomplete_and_error_streams_are_rejected(self):
        bad = [self.lines()[:-1], self.lines()[:3] + ["data: [DONE]"],
               [event(error={"message": "failed"})], ["data: [DONE]"]]
        for lines in bad:
            with self.assertRaises(ValueError):
                read_stream(lines, 1.0, lambda: 2.0)

    def test_usage_counts_require_positive_integers(self):
        for count in (0, -1, True, 1.5, "256", 257):
            times = iter([2.0, 5.0])
            with self.assertRaises(ValueError):
                read_stream(self.lines(count=count), 1.0, lambda: next(times))

    def test_wave_wall_includes_pool_start_and_end_like_reference(self):
        row = dict(started=2.0, first_text=3.0, ended=6.0, prompt_tokens=32,
                   completion_tokens=256, ttft_ms=1000, decode_tps=85.0,
                   finish_reason="length")
        result = summarize_wave([row], 1.0, 7.0)
        self.assertAlmostEqual(result["reference_aggregate_e2e_tps"], 256 / 6, places=3)
        self.assertEqual(result["aggregate_e2e_tps"], 64.0)
        self.assertTrue(result["all_outputs_reached_cap"])
        self.assertFalse(result["forced_output_cap"])

    def test_early_stop_is_retained_not_a_fixed_output_comparison(self):
        row = dict(started=2.0, first_text=3.0, ended=6.0, prompt_tokens=32,
                   completion_tokens=40, ttft_ms=1000, decode_tps=13.0,
                   finish_reason="stop")
        result = summarize_wave([row], 1.0, 7.0)
        self.assertFalse(result["all_outputs_reached_cap"])
        with self.assertRaises(ValueError):
            summarize_wave([row], 3.0, 7.0)


if __name__ == "__main__":
    unittest.main()
