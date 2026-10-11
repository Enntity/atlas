# SPDX-License-Identifier: AGPL-3.0-only
"""CPU unit tests for perf_baseline.py using a local fake SSE server.

Covers timing extraction, usage parsing, percentile math and fingerprint
enforcement. No network beyond localhost, no external dependencies.
"""

import json
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import perf_baseline


def content_chunk(text, finish=None):
    return {"id": "c", "choices": [
        {"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


def reasoning_chunk(text):
    return {"id": "c", "choices": [
        {"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}]}


def usage_chunk(usage):
    return {"id": "c", "choices": [], "usage": usage}


FAKE_USAGE = {
    "prompt_tokens": 111,
    "completion_tokens": 6,
    "total_tokens": 117,
    "completion_tokens_details": {"reasoning_tokens": 0},
}


def default_script(delay=0.02, usage=None):
    usage = FAKE_USAGE if usage is None else usage
    script = [{"delay": delay, "chunk": content_chunk(f"tok{i} ")} for i in range(5)]
    script.append({"delay": delay, "chunk": content_chunk("end", finish="stop")})
    script.append({"delay": 0.0, "chunk": usage_chunk(usage)})
    return script


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.endswith("/v1/models"):
            body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(payload)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in self.server.script:
            time.sleep(event["delay"])
            payload = event["chunk"] if isinstance(event["chunk"], str) \
                else json.dumps(event["chunk"])
            try:
                if event.get("event"):
                    self.wfile.write(b"event: " + event["event"].encode() + b"\n")
                self.wfile.write(b"data: " + payload.encode() + b"\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                return
        if self.server.terminal:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
            except BrokenPipeError:
                pass


class FakeServer:
    def __init__(self, script=None, terminal=True):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.script = default_script() if script is None else script
        self.httpd.terminal = terminal
        self.httpd.requests = []
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


WORKLOADS_DOC = {
    "schema_version": 1,
    "prompt_classes": {
        "prose": {"task": "Explain the notes.", "paragraphs": ["Background paragraph."]}
    },
    "workloads": [
        {"name": "prose_short", "prompt_class": "prose", "approx_input_tokens": 128,
         "max_tokens": 32, "temperature": 0.0, "seed": 7, "repeats": 3}
    ],
}


def full_fingerprint():
    return {
        "image_id": "sha256:abc",
        "source_commit": "11671adf",
        "serve_flags": ["--tp", "2"],
        "environment": {"ATLAS_KDA_FLA_REFERENCE": "0"},
        "hosts": ["spark0", "spark1"],
        "checkpoint_revision": "dda99c7e",
    }


def run_harness(out_dir, server, fingerprint="complete", unqualified=False,
                concurrency=1, workloads_doc=None, model="fake-model",
                initial_state="unknown"):
    out_dir = Path(out_dir)
    run_dir = out_dir / f"run-{len(list(out_dir.glob('run-*')))}"
    run_dir.mkdir()
    wl_path = run_dir / "workloads.json"
    wl_path.write_text(json.dumps(workloads_doc or WORKLOADS_DOC))
    argv = ["--endpoint", server.url, "--model", model,
            "--workloads", str(wl_path),
            "--raw-out", str(run_dir / "raw.jsonl"),
            "--summary-out", str(run_dir / "summary.json"),
            "--initial-state", initial_state,
            "--concurrency", str(concurrency)]
    if fingerprint == "complete":
        fp = full_fingerprint()
    elif fingerprint == "none":
        fp = None
    else:
        fp = fingerprint
    if fp is not None:
        fp_path = run_dir / "fp.json"
        fp_path.write_text(json.dumps(fp))
        argv += ["--fingerprint", str(fp_path)]
    if unqualified:
        argv += ["--unqualified"]
    code = perf_baseline.main(argv)
    raw_path = run_dir / "raw.jsonl"
    records = [json.loads(line) for line in raw_path.read_text().splitlines()] \
        if raw_path.exists() else []
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return code, records, summary


class HarnessRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def start_server(self, script=None, terminal=True):
        server = FakeServer(script, terminal=terminal)
        self.addCleanup(server.close)
        return server

    def test_timing_extraction(self):
        server = self.start_server()
        code, records, summary = run_harness(self.dir, server)
        self.assertEqual(code, 0)
        self.assertEqual(len(records), 3)
        for rec in records:
            self.assertIsNone(rec["error"])
            self.assertGreater(rec["wall_start_epoch"], 0)
            self.assertTrue(rec["wall_start_utc"].endswith("Z"))
            # Six content chunks plus one usage chunk arrive over the stream.
            self.assertEqual(len(rec["chunk_offsets_s"]), 7)
            self.assertEqual(len(rec["token_chunk_offsets_s"]), 6)
            self.assertEqual(len(rec["inter_token_gaps_s"]), 5)
            offsets = rec["token_chunk_offsets_s"]
            self.assertEqual(sorted(offsets), offsets)
            self.assertGreater(rec["ttft_s"], 0.0)
            self.assertEqual(rec["ttft_s"], offsets[0])
            self.assertEqual(rec["first_output_token_s"], offsets[0])
            self.assertGreaterEqual(rec["e2e_s"], rec["ttft_s"])
            self.assertAlmostEqual(
                sum(rec["inter_token_gaps_s"]),
                offsets[-1] - offsets[0], delta=1e-3)
            self.assertAlmostEqual(
                rec["decode_span_s"], offsets[-1] - offsets[0], delta=1e-3)
            # TPOT = decode span / (completion_tokens - 1).
            self.assertAlmostEqual(
                rec["tpot_s"], rec["decode_span_s"] / 5, delta=1e-3)
            self.assertAlmostEqual(
                rec["decode_tok_per_s"], 1.0 / rec["tpot_s"], delta=0.1)
        # Unknown initial cache state stays unknown on the first request.
        self.assertEqual([r["cold_start"] for r in records],
                         [None, False, False])
        self.assertEqual([r["first_request"] for r in records],
                         [True, False, False])
        self.assertEqual([r["first_in_workload"] for r in records],
                         [True, False, False])
        self.assertEqual(summary["totals"]["cold_starts"], 0)
        self.assertEqual(summary["totals"]["unknown_cold_start_state"], 1)
        self.assertEqual(summary["provenance_status"], "fingerprinted")
        self.assertEqual(summary["initial_state"], "unknown")

    def test_usage_parsing_and_request_contract(self):
        server = self.start_server()
        code, records, summary = run_harness(self.dir, server)
        self.assertEqual(code, 0)
        body = server.httpd.requests[0]
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["seed"], 7)
        for rec in records:
            # Usage is recorded verbatim, including detail blocks.
            self.assertEqual(rec["usage"], FAKE_USAGE)
            self.assertEqual(rec["finish_reason"], "stop")
            self.assertIn("request_extra", rec["request"])
        wl = summary["workloads"]["prose_short"]
        self.assertEqual(wl["prompt_tokens"]["p50"], 111)
        self.assertEqual(wl["completion_tokens"]["p50"], 6)
        self.assertEqual(wl["finish_reasons"], {"stop": 3})
        self.assertEqual(wl["n"], 3)
        self.assertEqual(wl["n_ok"], 3)

    def test_reasoning_ttft_is_first_content_token(self):
        usage = dict(FAKE_USAGE, completion_tokens=6)
        script = ([{"delay": 0.01, "chunk": reasoning_chunk(f"think{i} ")}
                   for i in range(3)]
                  + [{"delay": 0.01, "chunk": content_chunk(f"tok{i} ")}
                     for i in range(2)]
                  + [{"delay": 0.01, "chunk": content_chunk("end", finish="stop")},
                     {"delay": 0.0, "chunk": usage_chunk(usage)}])
        server = self.start_server(script)
        code, records, _ = run_harness(self.dir, server)
        self.assertEqual(code, 0)
        rec = records[0]
        offsets = rec["token_chunk_offsets_s"]
        self.assertEqual(len(offsets), 6)
        # TTFT is the first content token; thinking tokens only set
        # first_output_token_s.
        self.assertEqual(rec["first_output_token_s"], offsets[0])
        self.assertEqual(rec["ttft_s"], offsets[3])
        self.assertGreater(rec["ttft_s"], rec["first_output_token_s"])
        self.assertAlmostEqual(rec["tpot_s"], rec["decode_span_s"] / 5, delta=1e-3)

    def test_tpot_none_for_single_token_completion(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
        script = [{"delay": 0.01, "chunk": content_chunk("only", finish="stop")},
                  {"delay": 0.0, "chunk": usage_chunk(usage)}]
        server = self.start_server(script)
        code, records, _ = run_harness(self.dir, server)
        self.assertEqual(code, 0)
        rec = records[0]
        self.assertIsNone(rec["tpot_s"])
        self.assertIsNone(rec["decode_tok_per_s"])
        self.assertIsNone(rec["decode_span_s"])
        self.assertEqual(rec["inter_token_gaps_s"], [])

    def test_malformed_sse_json_fails_closed_without_timing_statistics(self):
        script = ([{"delay": 0.0, "chunk": content_chunk("partial")}]
                  + [{"delay": 0.0, "chunk": "{not json"}]
                  + default_script(delay=0.002))
        server = self.start_server(script)
        code, records, _ = run_harness(self.dir, server)
        self.assertEqual(code, 3)
        rec = records[0]
        self.assertIn("invalid SSE JSON", rec["error"])
        self.assertIsNone(rec["usage"])
        self.assertIsNone(rec["finish_reason"])
        self.assertIsNone(rec["ttft_s"])
        self.assertIsNone(rec["tpot_s"])
        self.assertEqual(rec["token_chunk_offsets_s"], [])

    def test_truncated_sse_without_done_fails_closed(self):
        server = self.start_server(terminal=False)
        code, records, summary = run_harness(self.dir, server)
        self.assertEqual(code, 3)
        self.assertIn("before [DONE]", records[0]["error"])
        self.assertIsNone(records[0]["ttft_s"])
        self.assertEqual(records[0]["chunk_offsets_s"], [])
        self.assertEqual(summary["workloads"]["prose_short"]["n_ok"], 0)

    def test_unknown_initial_state_is_not_called_cold(self):
        server = self.start_server()
        code, records, summary = run_harness(self.dir, server,
                                              initial_state="unknown")
        self.assertEqual(code, 0)
        self.assertEqual(records[0]["cold_start"], None)
        self.assertTrue(records[0]["first_request"])
        self.assertEqual(summary["totals"]["unknown_cold_start_state"], 1)

    def test_existing_output_is_never_overwritten(self):
        server = self.start_server()
        run_dir = self.dir / "existing"
        run_dir.mkdir()
        wl_path = run_dir / "workloads.json"
        wl_path.write_text(json.dumps(WORKLOADS_DOC))
        raw_path = run_dir / "raw.jsonl"
        raw_path.write_text("previous evidence\n")
        with self.assertRaises(FileExistsError):
            perf_baseline.main([
                "--endpoint", server.url, "--model", "fake-model",
                "--workloads", str(wl_path), "--raw-out", str(raw_path),
                "--summary-out", str(run_dir / "summary.json"),
                "--initial-state", "unknown", "--unqualified",
            ])
        self.assertEqual(raw_path.read_text(), "previous evidence\n")
        self.assertEqual(server.httpd.requests, [])

    def test_sse_error_event_fails_closed(self):
        script = ([{"delay": 0.0, "chunk": content_chunk("partial")}]
                  + [{"delay": 0.0, "event": "error",
                      "chunk": {"error": {"message": "backend failed"}}}])
        server = self.start_server(script)
        code, records, _ = run_harness(self.dir, server)
        self.assertEqual(code, 3)
        self.assertIn("server SSE error", records[0]["error"])
        self.assertIsNone(records[0]["ttft_s"])

    def test_concurrent_clients_flag_own_cold_starts(self):
        server = self.start_server(default_script(delay=0.002))
        code, records, summary = run_harness(
            self.dir, server, concurrency=2, initial_state="cold")
        self.assertEqual(code, 0)
        self.assertEqual(len(records), 3)
        # A cold start is the first request of the run, not each worker's
        # first request. Repeats remain indexed within their workload.
        self.assertEqual(summary["totals"]["cold_starts"], 1)
        self.assertEqual(sum(r["cold_start"] for r in records), 1)
        self.assertEqual(sum(r["first_request"] for r in records), 1)
        self.assertEqual(sorted(r["repeat_index"] for r in records), [0, 1, 2])
        self.assertEqual(sum(r["first_in_workload"] for r in records), 1)

    def test_failed_request_keeps_repeat_cold_and_nullable_timing_metadata(self):
        server = self.start_server()
        with patch.object(perf_baseline, "run_one",
                          side_effect=RuntimeError("request setup failed")):
            code, records, summary = run_harness(
                self.dir, server, initial_state="cold")
        self.assertEqual(code, 3)
        self.assertEqual([r["repeat_index"] for r in records], [0, 1, 2])
        self.assertEqual([r["cold_start"] for r in records], [True, False, False])
        self.assertEqual([r["first_in_workload"] for r in records],
                         [True, False, False])
        for rec in records:
            self.assertIsNone(rec["ttft_s"])
            self.assertIsNone(rec["tpot_s"])
            self.assertIsNone(rec["e2e_s"])
            self.assertEqual(rec["chunk_offsets_s"], [])
            self.assertEqual(rec["error"], "RuntimeError: request setup failed")
        self.assertEqual(summary["totals"]["cold_starts"], 1)


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self):
        values = list(range(1, 11))
        self.assertEqual(perf_baseline.percentile(values, 50), 5)
        self.assertEqual(perf_baseline.percentile(values, 95), 10)
        self.assertEqual(perf_baseline.percentile(values, 99), 10)
        self.assertEqual(perf_baseline.percentile([7], 50), 7)

    def test_empty_and_none(self):
        self.assertIsNone(perf_baseline.percentile([], 50))
        stats = perf_baseline.percentiles([None, None])
        self.assertEqual(stats, {"n": 0, "p50": None, "p95": None, "p99": None})
        stats = perf_baseline.percentiles([1.0, None, 2.0, 3.0])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["p50"], 2.0)


class WorkloadValidationTests(unittest.TestCase):
    def check_rejected(self, mutate, pattern):
        doc = json.loads(json.dumps(WORKLOADS_DOC))
        mutate(doc)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.json"
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, pattern):
                perf_baseline.load_workloads(path)

    def test_repeats_below_three_rejected(self):
        self.check_rejected(lambda d: d["workloads"][0].update(repeats=2),
                            "repeats")

    def test_unknown_prompt_class_rejected(self):
        self.check_rejected(
            lambda d: d["workloads"][0].update(prompt_class="counting"),
            "prompt_class")

    def test_missing_field_rejected(self):
        self.check_rejected(lambda d: d["workloads"][0].pop("seed"), "missing")

    def test_default_matrix_builds_prompts_near_target(self):
        doc = perf_baseline.load_workloads(perf_baseline.DEFAULT_WORKLOADS)
        classes = doc["prompt_classes"]
        self.assertEqual(len(doc["workloads"]), 10)
        for wl in doc["workloads"]:
            prompt = perf_baseline.build_prompt(
                classes[wl["prompt_class"]], wl["approx_input_tokens"])
            est = perf_baseline.estimate_tokens(prompt)
            self.assertLess(abs(est - wl["approx_input_tokens"]), 10, wl["name"])
            # Non-counting prompts: no enumeration/counting tasks in the
            # instruction line (background prose may describe counters etc.).
            low = classes[wl["prompt_class"]]["task"].lower()
            for banned in ("how many", "count how", "list all", "enumerate"):
                self.assertNotIn(banned, low, wl["name"])
        # The reasoning pair must differ only in reasoning_effort.
        on = classes["reasoning-on"]
        off = classes["reasoning-off"]
        self.assertEqual(on["task"], off["task"])
        self.assertEqual(on["paragraphs"], off["paragraphs"])
        self.assertNotEqual(on["request_extra"], off["request_extra"])
        # ... and the workload entries must be matched one-variable pairs:
        # identical except name and prompt_class (measurement discipline).
        by_name = {w["name"]: w for w in doc["workloads"]}
        for on_name, off_name in (("reasoning_on_short", "reasoning_off_short"),
                                  ("reasoning_on_near_limit",
                                   "reasoning_off_near_limit")):
            left = {k: v for k, v in by_name[on_name].items()
                    if k not in ("name", "prompt_class")}
            right = {k: v for k, v in by_name[off_name].items()
                     if k not in ("name", "prompt_class")}
            self.assertEqual(left, right, (on_name, off_name))


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.server = FakeServer(default_script(delay=0.001))
        self.addCleanup(self.server.close)

    def test_unadvertised_model_refuses_summary(self):
        # A model the server does not advertise must never enter a fingerprint.
        code, records, summary = run_harness(
            self.dir, self.server, fingerprint=full_fingerprint(),
            model="typo-model")
        self.assertEqual(code, 2)
        self.assertIsNone(summary)
        self.assertEqual({r["provenance_status"] for r in records}, {"unqualified"})
        code, records, summary = run_harness(
            self.dir, self.server, fingerprint=full_fingerprint(),
            model="typo-model", unqualified=True)
        self.assertEqual(code, 0)
        self.assertEqual(summary["provenance_status"], "unqualified")
        self.assertIn("model_id", summary["missing_fingerprint_keys"])
        self.assertIsNone(summary["model_id"])
        self.assertEqual(summary["requested_model"], "typo-model")

    def test_missing_keys_detected(self):
        self.assertEqual(
            perf_baseline.missing_fingerprint_keys({}, "m"),
            list(perf_baseline.REQUIRED_FINGERPRINT_KEYS))
        self.assertEqual(perf_baseline.missing_fingerprint_keys(
            full_fingerprint(), None), ["model_id"])
        self.assertEqual(perf_baseline.missing_fingerprint_keys(
            full_fingerprint(), "m"), [])

    def test_summary_refused_without_keys_or_flag(self):
        code, records, summary = run_harness(
            self.dir, self.server, fingerprint={"image_id": "only-one"})
        self.assertEqual(code, 2)
        self.assertIsNone(summary)
        self.assertEqual(len(records), 3)
        # Raw records are still produced and identify incomplete provenance.
        for rec in records:
            self.assertEqual(rec["provenance_status"], "unqualified")

    def test_unqualified_flag_writes_labelled_outputs(self):
        code, records, summary = run_harness(
            self.dir, self.server, fingerprint={"image_id": "only-one"},
            unqualified=True)
        self.assertEqual(code, 0)
        self.assertEqual(summary["provenance_status"], "unqualified")
        self.assertNotIn("image_id", summary["missing_fingerprint_keys"])
        self.assertIn("source_commit", summary["missing_fingerprint_keys"])
        for rec in records:
            self.assertEqual(rec["provenance_status"], "unqualified")

    def test_fingerprinted_summary_does_not_claim_model_qualification(self):
        fp = full_fingerprint()
        code, records, summary = run_harness(self.dir, self.server, fingerprint=fp)
        self.assertEqual(code, 0)
        self.assertEqual(summary["provenance_status"], "fingerprinted")
        self.assertEqual(summary["missing_fingerprint_keys"], [])
        # Verbatim merge, including nested structure.
        self.assertEqual(summary["fingerprint"], fp)
        self.assertEqual(summary["fingerprint"], full_fingerprint())
        for rec in records:
            self.assertEqual(rec["provenance_status"], "fingerprinted")
        harness = summary["harness"]
        self.assertEqual(harness["version"], perf_baseline.HARNESS_VERSION)
        self.assertEqual(harness["sha256"], perf_baseline.harness_sha256())
        self.assertEqual(summary["endpoint"], self.server.url)
        self.assertEqual(summary["model_id"], "fake-model")
        self.assertEqual(summary["advertised_models"], ["fake-model"])
        self.assertTrue(summary["date_utc"].endswith("Z"))
        self.assertEqual(summary["cli_args"]["model"], "fake-model")
        self.assertEqual(summary["cli_args"]["concurrency"], 1)
        self.assertEqual(summary["cli_args"]["initial_state"], "unknown")
        self.assertIn("sha256", summary["workloads_file"])


if __name__ == "__main__":
    unittest.main()
