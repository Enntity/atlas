# SPDX-License-Identifier: AGPL-3.0-only
"""CPU unit tests for harness_client.py, f2_concurrency.py and f3_context.py.

A fake OpenAI-compatible server (http.server in a thread) generates a
deterministic token stream from the prompt and can be told to misbehave:
diverge under concurrency, hang, drip keepalives, drop streams, crash on long
prompts, or change its output on a prefix-cache hit. Every misbehaviour must
be detected by the harness. Localhost only; no external dependencies.
"""

import hashlib
import json
import re
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import f2_concurrency as f2
import f3_context as f3
import harness_client as hc

MODEL = "fake-model"
TEMPLATE_OVERHEAD = 3
BLOCK = 16


def prompt_text(messages):
    return "\n".join(m.get("content") or "" for m in messages)


def prompt_words(messages):
    return prompt_text(messages).split()


def answer_pieces(text, max_tokens):
    """Deterministic answer for a prompt, split into token pieces."""
    needle = re.search(r"The passphrase is (\S+)\.", text)
    code = re.search(r"\nCODE: (.*)\n", text)
    if needle and code:
        answer = needle.group(1) + "\n" + code.group(1)
        pieces = [answer[i:i + 8] for i in range(0, len(answer), 8)]
    else:
        digest = hashlib.sha256(text.encode()).hexdigest()
        pieces = [f" w{digest[i % 60:i % 60 + 4]}{i}" for i in range(max_tokens + 4)]
    return pieces


def token_id(piece, pos):
    return int(hashlib.sha256(f"{pos}:{piece}".encode()).hexdigest()[:6], 16) % 50000


class FakeServer:
    def __init__(self, **cfg):
        self.cfg = {"max_model_len": 400, "delay": 0.005, "diverge_concurrent": False,
                    "hang_substring": None, "drip_substring": None, "truncate": False,
                    "crash_over": None, "warm_diverge": False, "token_ids": True,
                    "logprobs": True, "reject_logprobs": False, "serialize": False,
                    "mangle": None,
                    "cache_capacity_words": 10 ** 9}
        self.cfg.update(cfg)
        self.lock = threading.Lock()
        self.serial = threading.Lock()   # held per stream in serialize mode
        self.active = 0
        self.max_active = 0
        self.disconnects = 0
        self.crashed = False
        self.cache = []          # LRU list of prompt word lists
        self.requests = []
        self.stop = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_json(self, status, obj):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.send_json(200, {"data": [{"id": MODEL, "max_model_len":
                                                   server.cfg["max_model_len"]}]})
                elif self.path == "/health":
                    self.send_json(503 if server.crashed else 200, {})
                else:
                    self.send_json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    self.send_json(200, {"count": len(prompt_words(body["messages"]))
                                         + TEMPLATE_OVERHEAD, "tokens": []})
                    return
                server.requests.append(body)
                server.chat(self, body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    def cached_prefix(self, words):
        """Longest word prefix shared with a cached prompt; updates the LRU."""
        with self.lock:
            best = 0
            for prev in self.cache:
                n = 0
                for a, b in zip(prev, words):
                    if a != b:
                        break
                    n += 1
                best = max(best, n)
            self.cache.append(words)
            while sum(len(p) for p in self.cache) > self.cfg["cache_capacity_words"]:
                self.cache.pop(0)
        # Like vLLM, the last prompt token is never a hit and hits are whole
        # blocks (vllm/v1/core/kv_cache_manager.py:255-263).
        return min(best, len(words) - 1) // BLOCK * BLOCK

    def chat(self, h, body):
        cfg = self.cfg
        text = prompt_text(body["messages"])
        words = prompt_words(body["messages"])
        n_prompt = len(words) + TEMPLATE_OVERHEAD
        if self.crashed:
            h.close_connection = True
            return
        if cfg["crash_over"] is not None and n_prompt > cfg["crash_over"]:
            self.crashed = True
            h.close_connection = True
            return   # drop the connection with no response
        if cfg["reject_logprobs"] and body.get("logprobs"):
            h.send_json(400, {"error": {"message": "logprobs unsupported"}})
            return
        if n_prompt >= cfg["max_model_len"]:
            h.send_json(400, {"error": {"message": "Prompt too long", "type": "invalid_request_error"}})
            return
        cached = self.cached_prefix(words)
        max_tokens = body["max_tokens"]
        if cfg["mangle"] == "code":
            text = text.replace("0x5bd1e995", "0x5bd1e996")
        elif cfg["mangle"] == "needle":
            text = re.sub(r"The passphrase is (\S+)\.", "The passphrase is unknown.", text)
        pieces = answer_pieces(text, max_tokens)
        finish = "stop" if len(pieces) <= max_tokens else "length"
        pieces = pieces[:max_tokens]
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        if cfg["serialize"]:
            self.serial.acquire()
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            def emit(obj):
                data = obj if isinstance(obj, str) else json.dumps(obj)
                h.wfile.write(b"data: " + data.encode() + b"\n\n")
                h.wfile.flush()

            emit({"choices": [{"index": 0, "delta": {"role": "assistant"},
                               "finish_reason": None}]})
            for pos, piece in enumerate(pieces):
                time.sleep(cfg["delay"])
                if cfg["hang_substring"] and cfg["hang_substring"] in text and pos == 1:
                    self.stop.wait()
                    return
                if cfg["drip_substring"] and cfg["drip_substring"] in text and pos == 1:
                    while not self.stop.wait(0.05):
                        h.wfile.write(b": keepalive\n\n")
                        h.wfile.flush()
                    return
                if cfg["truncate"] and pos == 2:
                    return
                tid, lp = token_id(piece, pos), -0.01 * (pos % 7)
                if pos == 3 and cfg["diverge_concurrent"] and self.active > 1:
                    piece, tid, lp = " XX", 99999, -5.0
                if pos == 1 and cfg["warm_diverge"] and cached > 0:
                    piece, tid, lp = " YY", 88888, -4.0
                choice = {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                if cfg["token_ids"]:
                    choice["token_ids"] = [tid]
                if cfg["logprobs"] and body.get("logprobs"):
                    choice["logprobs"] = {"content": [{
                        "token": piece, "logprob": lp,
                        "top_logprobs": [{"token": piece, "logprob": lp}]}]}
                emit({"choices": [choice]})
            emit({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            emit({"choices": [], "usage": {
                "prompt_tokens": n_prompt, "completion_tokens": len(pieces),
                "total_tokens": n_prompt + len(pieces),
                "prompt_tokens_details": {"cached_tokens": cached}}})
            emit("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            with self.lock:
                self.disconnects += 1
        finally:
            with self.lock:
                self.active -= 1
            if cfg["serialize"]:
                self.serial.release()


SMALL_CASES = [
    {"name": f"case{i}", "messages": [{"role": "user", "content": f"prompt number {i} " * (i + 1)}]}
    for i in range(4)
]


def f2_run(url, **overrides):
    kwargs = dict(max_tokens=12, top_logprobs=2, seed=1, concurrency=[2, 4],
                  stagger=0.01, timeout=5.0, isolated_repeats=2, cancel_after=3)
    kwargs.update(overrides)
    return f2.run_f2(url, MODEL, SMALL_CASES, **kwargs)


def body_for(content, max_tokens=8, **extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 1}
    body.update(extra)
    return body


class ClientTests(unittest.TestCase):
    def test_stream_collects_tokens_usage_and_done(self):
        with FakeServer() as srv:
            res = hc.stream_chat(srv.url, body_for("hello there"), timeout=5)
        self.assertIsNone(res.error)
        self.assertTrue(res.done)
        self.assertEqual(len(res.token_ids), 8)
        self.assertEqual(res.lp_tokens, [p for p in answer_pieces("hello there", 8)[:8]])
        self.assertEqual(res.usage["completion_tokens"], 8)
        self.assertEqual(res.finish_reason, "length")
        self.assertEqual(res.cached_tokens(), 0)
        self.assertIsNotNone(res.ttft_s)

    def test_deadline_trips_on_silent_hang(self):
        with FakeServer(hang_substring="HANG") as srv:
            t0 = time.perf_counter()
            res = hc.stream_chat(srv.url, body_for("HANG now"), timeout=0.5)
            elapsed = time.perf_counter() - t0
        self.assertTrue(res.timed_out)
        self.assertIn("timeout", res.error)
        self.assertLess(elapsed, 2.0)

    def test_deadline_trips_on_dripping_server(self):
        # Keepalives every 50 ms never trip a per-read socket timeout of
        # 0.5 s; only an overall deadline ends this request.
        with FakeServer(drip_substring="DRIP") as srv:
            t0 = time.perf_counter()
            res = hc.stream_chat(srv.url, body_for("DRIP now"), timeout=0.5)
            elapsed = time.perf_counter() - t0
        self.assertTrue(res.timed_out)
        self.assertLess(elapsed, 2.0)

    def test_stream_without_done_is_error(self):
        with FakeServer(truncate=True) as srv:
            res = hc.stream_chat(srv.url, body_for("short"), timeout=5)
        self.assertIn("ended before [DONE]", res.error)

    def test_cancel_closes_socket_and_server_sees_it(self):
        with FakeServer(delay=0.02) as srv:
            res = hc.stream_chat(srv.url, body_for("cancel me", max_tokens=40), timeout=5,
                                 cancel_after_chunks=3)
            deadline = time.time() + 3
            while srv.disconnects == 0 and time.time() < deadline:
                time.sleep(0.02)
            self.assertTrue(res.cancelled)
            self.assertIsNone(res.error)
            self.assertEqual(len(res.token_ids), 3)  # the role chunk is not output
            self.assertEqual(srv.disconnects, 1)

    def test_cancel_event_from_other_thread(self):
        with FakeServer(hang_substring="HANG") as srv:
            ev = threading.Event()
            threading.Timer(0.2, ev.set).start()
            t0 = time.perf_counter()
            res = hc.stream_chat(srv.url, body_for("HANG"), timeout=10, cancel_event=ev)
        self.assertTrue(res.cancelled)
        self.assertFalse(res.timed_out)
        self.assertLess(time.perf_counter() - t0, 2.0)

    def test_malformed_logprobs_are_rejected(self):
        res = hc.StreamResult()
        res.t_start = 0.0
        with self.assertRaises(hc.HarnessError):
            hc._consume_chunk(res, {"choices": [{"delta": {"content": "a"},
                                                 "logprobs": "bogus"}]}, 1.0)
        hc._consume_chunk(res, {"choices": [{"delta": {"content": "a"},
                                             "logprobs": {"content": None}}]}, 1.0)
        self.assertEqual(res.content, "a")

    def test_http_error_is_captured(self):
        with FakeServer(max_model_len=5) as srv:
            res = hc.stream_chat(srv.url, body_for("far too many words here"), timeout=5)
        self.assertEqual(res.status, 400)
        self.assertEqual(res.error_json["error"]["message"], "Prompt too long")

    def test_write_json_exclusive_refuses_overwrite(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.json"
            hc.write_json_exclusive(out, {"a": 1})
            with self.assertRaises(FileExistsError):
                hc.write_json_exclusive(out, {"a": 2})
            with self.assertRaises(FileExistsError):
                hc.refuse_existing(out)
            self.assertEqual(json.loads(out.read_text()), {"a": 1})

    def test_compare_reports_divergence_and_logprob_gap(self):
        ref = {"token_ids": [1, 2, 3, 4], "logprobs": [-0.1, -0.2, -0.3, -0.4],
               "finish_reason": "length"}
        cand = {"token_ids": [1, 2, 9, 4], "logprobs": [-0.1, -0.25, -2.3, -0.4],
                "finish_reason": "length"}
        cmp = hc.compare_sequences(ref, cand)
        self.assertFalse(cmp["equal"])
        self.assertEqual(cmp["first_divergence"], 2)
        self.assertEqual((cmp["reference_token"], cmp["candidate_token"]), (3, 9))
        self.assertAlmostEqual(cmp["logprob_gap"], 2.0)
        self.assertAlmostEqual(cmp["max_abs_logprob_diff_before_divergence"], 0.05)

    def test_compare_prefix_length_and_finish(self):
        ref = {"token_ids": [1, 2, 3], "finish_reason": "stop"}
        self.assertTrue(hc.compare_sequences(ref, {"token_ids": [1, 2]},
                                             prefix_only=True)["equal"])
        short = hc.compare_sequences(ref, {"token_ids": [1, 2], "finish_reason": "stop"})
        self.assertEqual(short["first_divergence"], 2)
        fin = hc.compare_sequences(ref, {"token_ids": [1, 2, 3], "finish_reason": "length"})
        self.assertFalse(fin["equal"])
        self.assertEqual(fin["finish_reason_mismatch"], ["stop", "length"])

    def test_compare_falls_back_to_shared_basis(self):
        ref = {"token_ids": [1, 2], "logprob_tokens": ["a", "b"], "content": "ab"}
        cand = {"logprob_tokens": ["a", "c"], "content": "ac"}
        cmp = hc.compare_sequences(ref, cand)
        self.assertEqual(cmp["basis"], "logprob_tokens")
        self.assertEqual(cmp["first_divergence"], 1)
        text = hc.compare_sequences({"content": "abc"}, {"content": "abd"})
        self.assertEqual((text["basis"], text["first_divergence"]), ("text", 2))


class F2Tests(unittest.TestCase):
    def test_behaving_server_passes_with_real_overlap(self):
        with FakeServer() as srv:
            result, failures = f2_run(srv.url)
            self.assertEqual(failures, [])
            self.assertGreaterEqual(srv.max_active, 3)
            self.assertEqual(srv.disconnects, 1)  # the cancelled victim
        for name in ("interleaved_c2", "interleaved_c4", "cancel_c4"):
            overlap = result["phases"][name]["overlap"]
            self.assertGreater(overlap["sent_during_other_decode"], 0)
            self.assertGreater(overlap["first_token_during_other_decode"], 0)
        self.assertEqual(result["comparison_bases"], ["token_ids"])
        self.assertTrue(result["token_level"])
        victim = result["phases"]["cancel_c4"]["records"]["case0"]
        self.assertTrue(victim["cancelled"])
        self.assertTrue(result["phases"]["post_cancel_repeat"]["comparison"]["equal"])
        self.assertIsNotNone(result["phases"]["interleaved_c4"][
            "aggregate_completion_tok_per_s_wall_unqualified"])

    def assert_divergence_detected(self, srv_cfg, basis):
        with FakeServer(diverge_concurrent=True, **srv_cfg) as srv:
            result, failures = f2_run(srv.url)
        self.assertTrue(any("diverges from isolated at 3" in f for f in failures), failures)
        cmps = result["phases"]["interleaved_c4"]["comparisons"]
        bad = [c for c in cmps.values() if not c["equal"]]
        self.assertTrue(bad)
        self.assertTrue(all(c["first_divergence"] == 3 and c["basis"] == basis for c in bad))
        return bad, failures

    def test_divergence_under_concurrency_detected_on_token_ids(self):
        bad, _ = self.assert_divergence_detected({}, "token_ids")
        self.assertEqual(bad[0]["candidate_token"], 99999)
        self.assertIsNotNone(bad[0]["logprob_gap"])

    def test_divergence_detected_on_logprob_tokens(self):
        bad, _ = self.assert_divergence_detected({"token_ids": False}, "logprob_tokens")
        self.assertEqual(bad[0]["candidate_token"], " XX")

    def test_divergence_detected_on_text_and_flagged_not_token_level(self):
        # Text basis compares characters; the swapped piece is still found,
        # and the run is flagged as lacking token-level evidence.
        with FakeServer(diverge_concurrent=True, token_ids=False, logprobs=False) as srv:
            result, failures = f2_run(srv.url)
        self.assertFalse(result["token_level"])
        self.assertTrue(any("fell back to text" in f for f in failures))
        self.assertTrue(any("diverges from isolated" in f for f in failures))

    def test_rejected_logprobs_fall_back_and_are_recorded(self):
        with FakeServer(reject_logprobs=True) as srv:
            result, failures = f2_run(srv.url)
        self.assertEqual(failures, [])
        self.assertIn("logprobs unsupported", result["request_params"]["logprobs_rejected"])
        self.assertEqual(result["request_params"]["top_logprobs"], 0)

    def test_case_rejection_is_not_blamed_on_logprobs(self):
        cases = [{"name": "huge", "messages": [{"role": "user", "content": "w " * 500}]}]
        with FakeServer() as srv:
            result, failures = f2.run_f2(srv.url, MODEL, cases, max_tokens=12,
                                         top_logprobs=2, seed=1, concurrency=[2],
                                         stagger=0.01, timeout=5, isolated_repeats=1,
                                         cancel_after=3)
        self.assertNotIn("logprobs_rejected", result["request_params"])
        self.assertEqual(result["request_params"]["top_logprobs"], 2)
        self.assertTrue(any(f.startswith("isolated huge: HarnessError: HTTP 400") for f in failures))

    def test_hang_is_reported_not_stuck(self):
        cases = SMALL_CASES[:3] + [{"name": "hang", "messages": [
            {"role": "user", "content": "HANG please"}]}]
        with FakeServer(hang_substring="HANG") as srv:
            t0 = time.perf_counter()
            result, failures = f2.run_f2(srv.url, MODEL, cases, max_tokens=12,
                                         top_logprobs=0, seed=1, concurrency=[2],
                                         stagger=0.01, timeout=0.5, isolated_repeats=1,
                                         cancel_after=3)
            elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 10)
        self.assertTrue(any(f.startswith("isolated hang: timeout") for f in failures), failures)

    def test_server_that_serializes_requests_is_detected(self):
        # Client-side concurrency is real here, but the server streams one
        # request at a time, so equality would prove nothing about batching.
        with FakeServer(serialize=True) as srv:
            result, failures = f2_run(srv.url, concurrency=[2])
            self.assertEqual(srv.max_active, 1)
        overlap = result["phases"]["interleaved_c2"]["overlap"]
        self.assertGreater(overlap["sent_during_other_decode"], 0)
        self.assertEqual(overlap["first_token_during_other_decode"], 0)
        self.assertTrue(any("appears to serialize requests" in f for f in failures), failures)

    def test_serial_execution_is_not_accepted_as_interleaved(self):
        with FakeServer() as srv:
            _, failures = f2_run(srv.url, stagger=0.5, concurrency=[2])
        self.assertTrue(any("not interleaved" in f for f in failures), failures)

    def test_cancellation_that_never_fires_is_reported(self):
        with FakeServer() as srv:
            _, failures = f2_run(srv.url, max_tokens=4, cancel_after=10)
        self.assertTrue(any("cancellation not exercised" in f for f in failures), failures)

    def test_unhealthy_server_is_reported(self):
        with FakeServer() as srv:
            srv.crashed = True
            _, failures = f2_run(srv.url)
        self.assertTrue(any("unhealthy" in f for f in failures))

    def test_main_writes_receipt_and_refuses_overwrite(self):
        with FakeServer(max_model_len=4096) as srv, TemporaryDirectory() as tmp:
            out = Path(tmp) / "f2.json"
            argv = ["--endpoint", srv.url, "--model", MODEL, "--out", str(out),
                    "--max-tokens", "12", "--cancel-after", "3", "--stagger", "0.01",
                    "--timeout", "5", "--concurrency", "2"]
            self.assertEqual(f2.main(argv), 0)
            receipt = json.loads(out.read_text())
            fp = receipt["fingerprint"]
            self.assertEqual(fp["endpoint"], srv.url)
            self.assertEqual(fp["model_id"], MODEL)
            self.assertEqual(fp["concurrency"], [2])
            self.assertEqual(fp["cases_sha256"], hc.canonical_sha256(f2.DEFAULT_CASES))
            self.assertIn("harness_git_sha", fp)
            self.assertRegex(fp["utc_start"], r"^\d{4}-\d\d-\d\dT")
            self.assertEqual(receipt["provenance_status"], "unqualified")
            self.assertEqual(receipt["verdict"], "pass")
            with self.assertRaises(FileExistsError):
                f2.main(argv)


def ladder_run(url, **overrides):
    kwargs = dict(ladder=[96, 127, 128, 129], max_model_len=400, max_tokens=48, seed=7,
                  extra={}, timeout=5.0, allow_inexact=False)
    kwargs.update(overrides)
    return f3.run_ladder(url, MODEL, **kwargs)


def prefix_run(url, **overrides):
    kwargs = dict(prefix_words=60, evict_prompts=3, evict_words=80, max_tokens=10, seed=3,
                  extra={}, timeout=5.0, nonce="n0", require_cache_hit=True)
    kwargs.update(overrides)
    return f3.run_prefix(url, MODEL, **kwargs)


class F3Tests(unittest.TestCase):
    def test_calibration_hits_boundary_lengths_exactly(self):
        with FakeServer() as srv:
            count = lambda m: hc.tokenize_messages(srv.url, MODEL, m)  # noqa: E731
            for target in (127, 128, 129):
                cal = f3.calibrate(count, target, seed=11)
                self.assertTrue(cal["exact"], cal)
                self.assertEqual(count(f3.ladder_messages(cal["n_words"], 11, cal["pad"])),
                                 target)

    def test_ladder_passes_on_behaving_server(self):
        with FakeServer() as srv:
            result, failures = ladder_run(srv.url)
        self.assertEqual(failures, [])
        rungs = result["rungs"]
        self.assertEqual([r["actual_tokens"] for r in rungs], [96, 127, 128, 129])
        self.assertTrue(all(r["needle_found"] and r["code_exact"] for r in rungs))
        self.assertTrue(all(o["clean_4xx"] and o["healthy_after"] for o in result["over_limit"]))
        self.assertTrue(result["sanity_comparison"]["equal"])

    def test_rung_beyond_output_budget_is_not_run_and_fails(self):
        with FakeServer() as srv:
            result, failures = ladder_run(srv.url, ladder=[96, 360])
        self.assertIn("skipped", result["rungs"][1])
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("rung 360: exceeds max_model_len"))

    def test_crash_on_long_prompt_is_detected(self):
        with FakeServer(crash_over=300) as srv:
            _, failures = ladder_run(srv.url)
        self.assertTrue(any("expected a 4xx JSON error" in f for f in failures), failures)
        self.assertTrue(any("unhealthy afterwards" in f for f in failures), failures)
        self.assertTrue(any("sanity request after over-limit" in f for f in failures), failures)

    def test_wrong_code_line_is_detected(self):
        with FakeServer(mangle="code") as srv:
            result, failures = ladder_run(srv.url, ladder=[96])
        self.assertTrue(any(f == "rung 96: code/tool line not reproduced exactly"
                            for f in failures), failures)
        self.assertTrue(result["rungs"][0]["needle_found"])
        self.assertFalse(any("needle" in f for f in failures))

    def test_lost_needle_is_detected(self):
        with FakeServer(mangle="needle") as srv:
            result, failures = ladder_run(srv.url, ladder=[96])
        self.assertTrue(any(f.startswith("rung 96: needle ") and f.endswith("not reproduced")
                            for f in failures), failures)
        self.assertTrue(result["rungs"][0]["code_exact"])

    def test_crash_between_rungs_is_detected(self):
        # A server that crashes past 120 tokens fails the 127+ rungs.
        with FakeServer(crash_over=120) as srv:
            _, failures = ladder_run(srv.url)
        self.assertTrue(any(f.startswith("rung 127:") for f in failures), failures)

    def test_prefix_reuse_passes_and_observes_eviction(self):
        with FakeServer(cache_capacity_words=200) as srv:
            result, failures = prefix_run(srv.url)
        self.assertEqual(failures, [])
        cached = result["cached_tokens"]
        self.assertEqual(cached["cold s0"], 0)
        self.assertGreater(cached["warm s0"], 60)
        self.assertGreater(cached["prefix-hit s1"], 60)
        self.assertTrue(result["eviction_observed"])
        self.assertEqual(cached["post-evict s0"], 0)
        self.assertIsNotNone(result["warm_over_cold_ttft_ratio"])

    def test_without_eviction_reports_not_observed(self):
        with FakeServer() as srv:
            result, failures = prefix_run(srv.url)
        self.assertEqual(failures, [])
        self.assertFalse(result["eviction_observed"])

    def test_warm_output_change_is_detected(self):
        with FakeServer(warm_diverge=True, cache_capacity_words=200) as srv:
            result, failures = prefix_run(srv.url)
        self.assertTrue(any(f.startswith("warm s0 vs cold s0: diverges at 1") for f in failures),
                        failures)
        # After eviction s0 is a miss again and matches cold; s1 was first a
        # hit, so its post-eviction miss differs from that first output.
        self.assertTrue(result["comparisons"]["post-evict s0 vs first s0"]["equal"])
        self.assertFalse(result["comparisons"]["post-evict s1 vs first s1"]["equal"])

    def test_text_only_comparison_is_not_accepted(self):
        with FakeServer(token_ids=False) as srv:
            _, failures = prefix_run(srv.url)
        self.assertTrue(any("compared as text only" in f for f in failures), failures)

    def test_required_cache_hit_missing_is_failure(self):
        with FakeServer(cache_capacity_words=0) as srv:
            _, failures = prefix_run(srv.url)
        self.assertTrue(any("cached_tokens=0" in f for f in failures), failures)

    def test_main_rejects_skipping_everything(self):
        with TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            f3.main(["--endpoint", "http://127.0.0.1:9", "--model", MODEL,
                     "--out", str(Path(tmp) / "x.json"), "--skip-ladder", "--skip-prefix"])

    def test_main_records_unreachable_tokenizer_as_failure(self):
        with FakeServer() as srv, TemporaryDirectory() as tmp:
            out = Path(tmp) / "f3.json"
            srv.httpd.RequestHandlerClass.do_POST = lambda h: h.send_json(404, {})
            rc = f3.main(["--endpoint", srv.url, "--model", MODEL, "--out", str(out),
                          "--skip-prefix", "--timeout", "5"])
            receipt = json.loads(out.read_text())
        self.assertEqual(rc, 1)
        self.assertTrue(any(f.startswith("ladder aborted: /tokenize failed")
                            for f in receipt["failures"]), receipt["failures"])

    def test_main_receipt_and_overwrite_refusal(self):
        with FakeServer(cache_capacity_words=200) as srv, TemporaryDirectory() as tmp:
            out = Path(tmp) / "f3.json"
            argv = ["--endpoint", srv.url, "--model", MODEL, "--out", str(out),
                    "--ladder", "96", "128", "--max-tokens", "48", "--prefix-words", "60",
                    "--evict-prompts", "3", "--evict-words", "80", "--timeout", "5",
                    "--nonce", "fixed"]
            self.assertEqual(f3.main(argv), 0)
            receipt = json.loads(out.read_text())
            self.assertEqual(receipt["fingerprint"]["max_model_len"], 400)
            self.assertEqual(receipt["fingerprint"]["max_model_len_source"], "/v1/models")
            self.assertEqual(receipt["cases"]["ladder"], [96, 128, 352])
            self.assertEqual(receipt["prefix_reuse"]["nonce"], "fixed")
            self.assertEqual(receipt["verdict"], "pass")
            with self.assertRaises(FileExistsError):
                f3.main(argv)


if __name__ == "__main__":
    unittest.main()
