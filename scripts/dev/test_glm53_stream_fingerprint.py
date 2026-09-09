# SPDX-License-Identifier: AGPL-3.0-only
"""Mocked SSE checks for retained completion text, identity and timing."""

import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_glm53_concurrency import one_request, summarize_batch


class FingerprintTests(unittest.TestCase):
    def request(self, chunks):
        events = ["data: " + json.dumps({"choices": [{"text": chunk}]})
                  for chunk in chunks]
        events += ["data: " + json.dumps({
            "choices": [{"text": "", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 4,
                      "time_to_first_token_ms": 1000, "response_token/s": 2},
        }), "data: [DONE]"]
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = iter(events)
        with patch("requests.post", return_value=response) as post, patch(
            "benchmark_glm53_concurrency.time.perf_counter", side_effect=[1, 2, 5]
        ):
            result = one_request("http://unused", "test", [1, 2], 4, 10)
        self.assertEqual(post.call_args.kwargs["json"]["seed"], 1)
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], 0)
        self.assertNotIn("repetition_detection", post.call_args.kwargs["json"])
        return result

    def test_unicode_digest_is_independent_of_sse_chunk_boundaries(self):
        first = self.request(["", "hé", "llo 🌌\n"])
        second = self.request(["héllo 🌌\n"])
        expected = "héllo 🌌\n".encode("utf-8")
        self.assertEqual(first["completion_text_sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(first["completion_text_bytes"], len(expected))
        self.assertEqual(first, second)
        self.assertEqual(first["completion_text"], expected.decode("utf-8"))

    def test_changed_output_has_distinct_digest_and_same_timing_metrics(self):
        first, second = self.request(["same"]), self.request(["else"])
        self.assertNotEqual(first["completion_text_sha256"], second["completion_text_sha256"])
        summary = summarize_batch([first], 4)
        self.assertEqual(summary["aggregate_e2e_tps"], 1)
        self.assertEqual(summary["requests"][0]["completion_text_sha256"],
                         first["completion_text_sha256"])
        self.assertEqual(summary["requests"][0]["completion_text_bytes"], 4)
        self.assertEqual(summary["requests"][0]["completion_text"], "same")


if __name__ == "__main__":
    unittest.main()
