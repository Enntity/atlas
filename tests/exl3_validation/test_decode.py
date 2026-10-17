# SPDX-License-Identifier: AGPL-3.0-only
import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from collect_decode import collect_decode, completion, load_decode
from compare_decode import compare_decode
from test_parity import provenance, manifest


def response(prompt, ids=None):
    choice = {"text": "x" * 16, "finish_reason": "length"}
    if ids is not None:
        choice["token_ids"] = ids
    return {"choices": [choice], "usage": {"prompt_tokens": len(prompt), "completion_tokens": 16}}


class DecodeTests(unittest.TestCase):
    def serve(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(request)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(response(request["prompt"])).encode())
        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}", requests

    def artifact(self, root, engine):
        endpoint, requests = self.serve()
        capture = root / (engine + "-capture")
        capture.mkdir()
        meta = provenance(engine)
        meta["environment"]["ATLAS_EXL3_LOGITS_DIR"] = str(capture)
        collect_decode(endpoint, "model", manifest(engine)["cases"], root / engine, meta, 1)
        self.assertFalse((capture / "capture-enabled").exists())
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["prompt"], [1, 2])
        self.assertEqual(requests[0]["max_tokens"], 16)
        self.assertEqual(requests[0]["temperature"], 0)
        self.assertEqual(requests[0]["seed"], 42)
        return load_decode(root / engine)

    def test_actual_http_files_and_cli_compare_preserve_generation_policy(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            reference = self.artifact(root, "vllm")
            candidate = self.artifact(root, "atlas")
            result = compare_decode(reference, candidate)
            self.assertTrue(result["passed"])
            self.assertIsNone(result["cases"][0]["token_ids_match"])
            report = root / "report.json"
            command = [sys.executable, str(Path(__file__).parent / "compare_decode.py"),
                       "--reference", str(root / "vllm"), "--candidate", str(root / "atlas"), "--out", str(report)]
            executed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            self.assertTrue(json.loads(report.read_text())["passed"])
            self.assertNotEqual(subprocess.run(command, capture_output=True, text=True).returncode, 0)

    def test_each_surface_difference_and_exposed_token_ids_fail(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            reference, original = self.artifact(root, "vllm"), self.artifact(root, "atlas")
            for field, value in (("text", " " + "x" * 16), ("finish_reason", "stop"), ("completion_tokens", 15)):
                altered = copy.deepcopy(original)
                altered["responses"][0]["completion"][field] = value
                self.assertFalse(compare_decode(reference, altered)["passed"])
            reference["responses"][0]["completion"]["token_ids"] = [1] * 16
            original["responses"][0]["completion"]["token_ids"] = [2] * 16
            self.assertFalse(compare_decode(reference, original)["passed"])

    def test_response_token_count_and_case_coverage_fail_closed(self):
        with self.assertRaises(ValueError):
            completion(response([1, 2], [1]), [1, 2])
        data = response([1, 2])
        data["usage"]["prompt_tokens"] = 3
        with self.assertRaises(ValueError):
            completion(data, [1, 2])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            reference, candidate = self.artifact(root, "vllm"), self.artifact(root, "atlas")
            candidate["responses"] = []
            with self.assertRaises(ValueError):
                compare_decode(reference, candidate)

    def test_enabled_capture_and_damaged_response_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            capture = root / "capture"
            capture.mkdir()
            marker = capture / "capture-enabled"
            marker.write_text("owned")
            meta = provenance("atlas")
            meta["environment"]["ATLAS_EXL3_LOGITS_DIR"] = str(capture)
            with self.assertRaises(ValueError):
                collect_decode("http://unused", "model", manifest("atlas")["cases"], root / "blocked", meta, 1)
            self.assertEqual(marker.read_text(), "owned")
            self.artifact(root, "atlas")
            raw = root / "atlas" / "responses" / "case.json"
            raw.write_text("{}")
            with self.assertRaises(ValueError):
                load_decode(root / "atlas")


if __name__ == "__main__":
    unittest.main()
