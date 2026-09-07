#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
import copy
import unittest

from summarize_glm53_c1_mtp import summarize


def benchmark():
    run = {
        "concurrency": 1, "prompt_tokens_each": 148,
        "completion_tokens_total": 256, "output_tokens_requested_total": 256,
        "all_outputs_reached_cap": True, "aggregate_e2e_tps": 27.123,
        "requests": [{"completion_tokens": 256, "finish_reason": "length"}],
    }
    return [{
        "concurrency": 1, "all_outputs_reached_cap": True,
        "median_aggregate_e2e_tps": 27.123,
        "median_aggregate_post_first_token_tps": 28.456,
        "workload": {"prompt_tokens": 148, "output_tokens_requested_per_stream": 256,
                     "warmup_batches_per_concurrency": 1,
                     "measured_batches_per_concurrency": 2},
        "runs": [copy.deepcopy(run), copy.deepcopy(run)],
    }]


def window(prompt=148, cap=256, tokens=256, timings=3):
    start = f"Chunked prefill start: {prompt} prompt tokens, chunk_size=148, max_tokens={cap}\n"
    for i in range(timings):
        start += (f"MTP verify timing [25 steps, seq_len={200+i*30}]: "
                  f"fwd={100+i}.25ms(x1.0) propose={10+i}.50ms(x1.0) TOTAL=125ms(x1.0)\n")
    return start + (f"Done: {tokens} tokens (length) 29.1 tok/s, TTFT=424.0ms, "
                    "serial=0.00 mtp=1.00 p1=0.882 mean_na=2.395 tok_step=3.395 "
                    "regime_reprobes=0 depth=k5 depth_switches=0\n")


class SummaryTests(unittest.TestCase):
    def test_preserves_metrics_orders_requests_and_discards_first_timing(self):
        doc = benchmark()
        original = copy.deepcopy(doc)
        text = window(29, 128, 69) + "\x1b[32m" + window() * 3 + "\x1b[0m"
        report = summarize(doc, text)
        self.assertEqual(report["benchmark_metrics"]["median_aggregate_e2e_tps"], 27.123)
        self.assertEqual(report["benchmark_metrics"]["median_aggregate_post_first_token_tps"], 28.456)
        self.assertEqual(report["unmatched_complete_requests"], 1)
        self.assertEqual([r["warmup"] for r in report["requests"]], [True, False, False])
        for row in report["requests"]:
            self.assertEqual(row["mean_na"], 2.395)
            self.assertEqual(row["depth"], "k5")
            self.assertEqual(row["discarded_timing_windows"], 1)
            self.assertEqual([w["fwd_ms"] for w in row["clean_timing_windows"]], [101.25, 102.25])
        self.assertEqual(doc, original)

    def test_no_clean_window_is_empty_not_an_estimate(self):
        report = summarize(benchmark(), window(timings=0) + window(timings=1) * 2)
        self.assertEqual([r["clean_timing_windows"] for r in report["requests"]], [[], [], []])
        self.assertEqual([r["discarded_timing_windows"] for r in report["requests"]], [0, 1, 1])

    def test_rejects_wrong_matching_count_never_last_n(self):
        for count in [0, 2, 4]:
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "matching request count"):
                summarize(benchmark(), window() * count)

    def test_rejects_partial_interleaved_and_orphan_records(self):
        for text in [window() * 3 + "Chunked prefill start: 29 prompt tokens, chunk_size=29, max_tokens=5\n",
                     window().split("Done:")[0] + window() * 3,
                     "Done: 5 tokens (stop)\n" + window() * 3]:
            with self.subTest(text=text[:50]), self.assertRaisesRegex(ValueError, "incomplete|interleaved|orphan"):
                summarize(benchmark(), text)

    def test_rejects_c4_and_missing_or_bad_metadata(self):
        cases = []
        c4 = benchmark(); c4[0]["concurrency"] = 4; cases.append(c4)
        mixed = benchmark() * 2; cases.append(mixed)
        for key in ["prompt_tokens", "output_tokens_requested_per_stream",
                    "warmup_batches_per_concurrency", "measured_batches_per_concurrency"]:
            doc = benchmark(); del doc[0]["workload"][key]; cases.append(doc)
            doc = benchmark(); doc[0]["workload"][key] = -1; cases.append(doc)
        for doc in cases:
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                summarize(doc, window() * 3)

    def test_rejects_cap_and_measured_completion_mismatch(self):
        doc = benchmark(); doc[0]["runs"][0]["all_outputs_reached_cap"] = False
        with self.assertRaisesRegex(ValueError, "cap integrity"):
            summarize(doc, window() * 3)
        with self.assertRaisesRegex(ValueError, "completion count"):
            summarize(benchmark(), window() + window(tokens=255) + window())

    def test_short_outputs_report_not_relabelled_as_cap(self):
        doc = benchmark(); doc[0]["all_outputs_reached_cap"] = False
        run = doc[0]["runs"][0]; run["all_outputs_reached_cap"] = False
        run["completion_tokens_total"] = 100; run["requests"][0]["completion_tokens"] = 100
        report = summarize(doc, window() + window(tokens=100) + window())
        self.assertFalse(report["benchmark_metrics"]["all_outputs_reached_cap"])
        self.assertEqual(report["requests"][1]["completion_tokens"], 100)

    def test_rejects_malformed_matching_diagnostics(self):
        for changed in [window().replace("mean_na=2.395", "mean_na=nan"),
                        window().replace("propose=10.50ms", "propose=nanms"),
                        window().replace("[25 steps", "[24 steps")]:
            with self.assertRaisesRegex(ValueError, "diagnostic|timing"):
                summarize(benchmark(), changed + window() * 2)

    def test_rejects_inconsistent_runs_and_nonfinite_original_metrics(self):
        changes = [
            lambda d: d[0]["runs"].pop(),
            lambda d: d[0]["runs"][0].update(concurrency=4),
            lambda d: d[0]["runs"][0].update(prompt_tokens_each=149),
            lambda d: d[0]["runs"][0].update(output_tokens_requested_total=255),
            lambda d: d[0]["runs"][0]["requests"].append({"completion_tokens": 256}),
            lambda d: d[0].update(median_aggregate_e2e_tps=float("nan")),
            lambda d: d[0]["workload"].update(measured_batches_per_concurrency=True),
            lambda d: d[0]["workload"].update(warmup_batches_per_concurrency=2**63-1),
        ]
        for change in changes:
            doc = benchmark(); change(doc)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize(doc, window() * 3)

    def test_zero_warmups_and_unrelated_trailing_complete_request(self):
        doc = benchmark(); doc[0]["workload"]["warmup_batches_per_concurrency"] = 0
        report = summarize(doc, window() * 2 + window(30, 128, 10))
        self.assertEqual([r["warmup"] for r in report["requests"]], [False, False])
        self.assertEqual(report["clean_measured_timing_window_count"], 4)
        self.assertEqual(report["diagnostic_median_fwd_ms"], 101.75)

    def test_malformed_start_and_missing_mtp_fields_fail_closed(self):
        for text in [window().replace("max_tokens=256", "max_tokens=unknown") + window() * 2,
                     window().replace(" depth=k5", "") + window() * 2,
                     window().replace("serial=0.00", "serial=1.01") + window() * 2]:
            with self.assertRaises(ValueError):
                summarize(benchmark(), text)


if __name__ == "__main__":
    unittest.main()
