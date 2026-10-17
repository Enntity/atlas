# SPDX-License-Identifier: AGPL-3.0-only
import copy
import json
import tempfile
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from collect_logits import collect
from compare_logits import compare, negative_controls
from parity_protocol import load_artifact, read_raw, token_sha, validate_cases, validate_provenance


def provenance(engine):
    digest = "0" * 64
    return {"engine": engine, "engine_build": {"commit": "test", "binary_sha256": digest,
            "source_sha256": digest, "image": "test-image"}, "checkpoint_sha256": {"weights": digest},
            "checkpoint_revision": "immutable-test", "tokenizer_sha256": {"tokenizer": digest},
            "config_sha256": {"config": digest}, "execution": {"tp": 2, "ep": 1, "kv_dtype": "bf16",
            "activation_dtype": "bf16", "speculative": False, "eager": True, "prefix_caching": False},
            "serve_flags": ["serve", "test"], "environment": {}, "hardware": {"gpu_uuids": ["a", "b"],
            "driver": "test", "cuda": "test", "nccl": "test"}, "hostname": "test", "utc": "test"}


def thresholds():
    return {"abs": 0, "rel": 0, "kl": 0, "cross_entropy": 0, "top_k": 2, "argmax": "exact"}


def manifest(engine):
    cases = [{"id": "case", "prompt_ids": [1, 2], "continuation_ids": [3]}]
    return {"provenance": provenance(engine), "cases": cases, "tokens_sha256": token_sha(cases),
            "vocab": 4, "generated_text": {"case": ["x", "x"]}, "thresholds": thresholds(), "request_policy": {"temperature": 0, "seed": 42, "max_tokens": 1}}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference, self.candidate = manifest("vllm"), manifest("atlas")
        self.arrays = {"case": np.array([[0., 1., 2., 3.], [3., 2., 1., 0.]], dtype=np.float32)}

    def compare(self, arrays):
        return compare(self.reference, self.candidate, self.arrays, arrays, 0, 0, 0, 0, 2)

    def test_identical_rows_pass_and_negative_controls_are_detected(self):
        self.assertTrue(self.compare(self.arrays)["passed"])
        self.assertTrue(negative_controls(self.reference, self.candidate, self.arrays, 0, 0, 0, 0, 2)["passed"])
        self.assertEqual(self.compare(self.arrays)["cases"][0]["max_cross_entropy_error"], 0)

    def test_changed_logits_and_argmax_fail(self):
        changed = {"case": self.arrays["case"].copy()}
        changed["case"][0, 0] = 10
        result = self.compare(changed)
        self.assertFalse(result["passed"])
        self.assertGreater(result["cases"][0]["max_kl_reference_to_candidate"], 0)
        self.assertEqual(result["cases"][0]["argmax_matches"], 1)

    def test_actual_http_text_difference_fails_even_when_raw_logits_match(self):
        self.candidate["generated_text"]["case"][0] = "different"
        result = self.compare(self.arrays)
        self.assertFalse(result["passed"])
        self.assertFalse(result["cases"][0]["generated_text_matches"])
        self.assertTrue(result["cases"][0]["allclose"])

    def test_uniform_vocab_does_not_qualify_negative_control(self):
        uniform = {"case": np.zeros((2, 4), np.float32)}
        self.assertFalse(negative_controls(self.reference, self.candidate, uniform, 0, 0, 0, 0, 2)["passed"])

    def test_shape_finite_tokens_provenance_and_thresholds_fail_closed(self):
        for arrays in ({"case": np.zeros((1, 4))}, {"case": np.full((2, 4), np.nan)}):
            with self.assertRaises(ValueError):
                self.compare(arrays)
        for field in ("cases", "tokens_sha256", "vocab"):
            altered = copy.deepcopy(self.candidate)
            altered[field] = None
            with self.assertRaises(ValueError):
                compare(self.reference, altered, self.arrays, self.arrays, 0, 0, 0, 0, 2)
        altered = copy.deepcopy(self.candidate)
        altered["provenance"]["execution"]["kv_dtype"] = "fp8"
        with self.assertRaises(ValueError):
            compare(self.reference, altered, self.arrays, self.arrays, 0, 0, 0, 0, 2)
        with self.assertRaises(ValueError):
            compare(self.reference, self.candidate, self.arrays, self.arrays, -1, 0, 0, 0, 2)

    def test_case_and_provenance_validation(self):
        for cases in ([], [{"id": "../case", "prompt_ids": [1], "continuation_ids": []}],
                      [{"id": "case", "prompt_ids": [True], "continuation_ids": []}]):
            with self.assertRaises(ValueError):
                validate_cases(cases)
        bad = provenance("atlas")
        bad["execution"]["speculative"] = "false"
        with self.assertRaises(ValueError):
            validate_provenance(bad)


class CollectorTests(unittest.TestCase):
    def serve(self, directory, extra=0, missing=False, observed_headers=None):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if not (directory / "capture-enabled").is_file():
                    raise RuntimeError("collector did not enable capture")
                requests.append(request)
                if observed_headers is not None:
                    observed_headers.append(self.headers.get("x-request-id"))
                sequence = len(list(directory.glob("logits-*.json")))
                for offset in range(0 if missing else 1 + extra):
                    filename = f"logits-{sequence + offset:06d}"
                    data = np.array([[100, 100, 100, 100], [0, 1, 2, 3]], dtype="<f4")
                    (directory / (filename + ".f32")).write_bytes(data.tobytes())
                    (directory / (filename + ".json")).write_text(json.dumps({"dtype": "f32-le", "rows": 2, "vocab": 4, "file": filename + ".f32"}))
                response = json.dumps({"choices": [{"text": "x"}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(response)
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}", requests

    def test_serial_exact_prefixes_last_row_and_artifact_fingerprints(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            capture = root / "capture"
            capture.mkdir()
            endpoint, requests = self.serve(capture)
            cases = manifest("atlas")["cases"]
            collect(endpoint, "model", cases, capture, root / "out", provenance("atlas"), 1, thresholds())
            saved, arrays = load_artifact(root / "out")
            self.assertEqual([r["prompt"] for r in requests], [[1, 2], [1, 2, 3]])
            self.assertTrue(all(r["max_tokens"] == 1 and r["temperature"] == 0 and r["seed"] == 42 for r in requests))
            np.testing.assert_array_equal(arrays["case"], [[0, 1, 2, 3], [0, 1, 2, 3]])
            self.assertEqual(saved["tokens_sha256"], token_sha(cases))
            self.assertFalse((capture / "capture-enabled").exists())
            self.assertTrue((root / "out" / "pre_registration.json").is_file())
            damaged = copy.deepcopy(saved)
            damaged["captures"][0]["row_sha256"] = "bad"
            (root / "out" / "manifest.json").write_text(json.dumps(damaged))
            with self.assertRaises(ValueError):
                load_artifact(root / "out")

    def test_explicit_request_ids_reach_http_and_preserved_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            capture.mkdir()
            headers = []
            endpoint, _ = self.serve(capture, observed_headers=headers)
            request_ids = ["c2-arithmetic-r0", "c2-arithmetic-r1"]
            result = collect(endpoint, "model", manifest("atlas")["cases"], capture,
                             root / "out", provenance("atlas"), 1, thresholds(),
                             request_ids=request_ids)
            self.assertEqual(headers, request_ids)
            self.assertEqual(result["request_ids"], request_ids)
            self.assertEqual([record["request_id"] for record in result["captures"]], request_ids)
            registered = json.loads((root / "out" / "pre_registration.json").read_text())
            self.assertEqual(registered["request_ids"], request_ids)
            loaded, _ = load_artifact(root / "out")
            self.assertEqual(loaded["request_ids"], request_ids)

    def test_default_collection_does_not_add_request_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            capture.mkdir()
            headers = []
            endpoint, _ = self.serve(capture, observed_headers=headers)
            result = collect(endpoint, "model", manifest("atlas")["cases"], capture,
                             root / "out", provenance("atlas"), 1, thresholds())
            self.assertEqual(headers, [None, None])
            self.assertNotIn("request_ids", result)
            self.assertTrue(all("request_id" not in item for item in result["captures"]))

    def test_request_ids_refuse_invalid_schedule_before_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid = [[], ["one"], ["one", "one"], "ab", ["one", 2],
                       ["one", ""], ["one", "bad\r\nheader"], ["one", "white space"],
                       ["one", "caf\u00e9"]]
            for request_ids in invalid:
                with self.subTest(request_ids=request_ids), self.assertRaisesRegex(ValueError, "request IDs"):
                    collect("http://unused", "model", manifest("atlas")["cases"],
                            root / "missing-capture", root / "out", provenance("atlas"),
                            1, thresholds(), request_ids=request_ids)
                self.assertEqual(list(root.iterdir()), [])

    def test_comparator_cli_preserves_report_and_runs_negative_controls(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for engine in ("atlas", "vllm"):
                capture = root / (engine + "-capture")
                capture.mkdir()
                endpoint, _ = self.serve(capture)
                collect(endpoint, "model", manifest(engine)["cases"], capture, root / engine, provenance(engine), 1, thresholds())
            report = root / "report.json"
            command = [sys.executable, str(Path(__file__).parent / "compare_logits.py"),
                       "--reference", str(root / "vllm"), "--candidate", str(root / "atlas"),
                       "--out", str(report), "--abs", "0", "--rel", "0", "--kl", "0", "--ce-max-error", "0", "--top-k", "2"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads(report.read_text())
            self.assertTrue(saved["passed"])
            self.assertTrue(saved["negative_controls"]["shuffled_vocabulary_detected"])
            drift = command.copy()
            drift[drift.index("--abs") + 1] = "0.1"
            drift[drift.index("--out") + 1] = str(root / "drift-report.json")
            self.assertNotEqual(subprocess.run(drift, capture_output=True, text=True).returncode, 0)
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)

    def test_missing_and_ambiguous_captures_fail(self):
        for extra, missing in ((1, False), (0, True)):
            with tempfile.TemporaryDirectory() as root:
                root = Path(root)
                capture = root / "capture"
                capture.mkdir()
                endpoint, _ = self.serve(capture, extra, missing)
                with self.assertRaises(ValueError):
                    collect(endpoint, "model", manifest("atlas")["cases"], capture, root / "out", provenance("atlas"), 0.1, thresholds())

                self.assertFalse((capture / "capture-enabled").exists())

    def test_existing_capture_marker_is_never_taken_over(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            capture = root / "capture"
            capture.mkdir()
            marker = capture / "capture-enabled"
            marker.write_text("another-run")
            with self.assertRaises(ValueError):
                collect("http://unused", "model", manifest("atlas")["cases"], capture, root / "out", provenance("atlas"), 1, thresholds())
            self.assertEqual(marker.read_text(), "another-run")
            self.assertFalse((root / "out").exists())

    def test_raw_dtype_shape_and_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            sidecar = root / "logits-000000.json"
            for metadata in ({"dtype": "bf16"}, {"dtype": "f32-le", "rows": 1, "vocab": 4, "file": "../x.f32"}):
                sidecar.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    read_raw(sidecar)


if __name__ == "__main__":
    unittest.main()
