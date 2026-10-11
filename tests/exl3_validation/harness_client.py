# SPDX-License-Identifier: AGPL-3.0-only
"""Standard-library HTTP client shared by the F2/F3 qualification harnesses.

Used by ``f2_concurrency.py`` and ``f3_context.py``. Provides:

* ``stream_chat``: one streaming ``/v1/chat/completions`` request with strict
  SSE parsing (same discipline as ``perf_baseline.stream_chat``), an overall
  wall-clock deadline and cooperative cancellation. The deadline and the
  cancellation are both enforced by a watchdog thread that shuts the socket
  down, so a server that stalls mid-read or drips bytes cannot hang a run and
  a cancelled stream is visibly closed on the server side.
* per-token capture (``token_ids`` from ``return_token_ids``, logprob tokens,
  logprobs and top-k alternatives) and ``compare_sequences``, which reports
  the first divergent position and the logprob gap there;
* small JSON helpers (``/v1/models``, ``/health``, ``/tokenize``) and
  ``write_json_exclusive``, which refuses to overwrite an output file.
"""

import hashlib
import http.client
import json
import math
import socket
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path

# Required caller fingerprint keys; same convention as perf_baseline.py and
# the provenance section of tests/exl3_validation/README.md.
REQUIRED_FINGERPRINT_KEYS = (
    "image_id",
    "source_commit",
    "serve_flags",
    "environment",
    "hosts",
    "checkpoint_revision",
)

BASES = ("token_ids", "logprob_tokens", "text")


class HarnessError(Exception):
    """Protocol or transport failure for one request."""


# ---------------------------------------------------------------- utilities

def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_sha256(obj):
    """sha256 of the canonical (sorted, compact) JSON encoding of ``obj``."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state(path=None):
    """(HEAD sha, dirty flag) for the repository holding ``path``.

    Dirty means the harness directory has uncommitted changes, so the sha
    alone does not identify the code that ran. (None, None) without git.
    """
    cwd = str(Path(path or __file__).resolve().parent)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=cwd,
                                check=True, capture_output=True, text=True,
                                timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return sha or None, bool(status.strip())


def missing_fingerprint_keys(fingerprint, model_id):
    missing = []
    for key in REQUIRED_FINGERPRINT_KEYS:
        value = (fingerprint or {}).get(key)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(key)
    if not model_id:
        missing.append("model_id")
    return missing


def refuse_existing(*paths):
    """Raise FileExistsError if any output path already exists."""
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")


def write_json_exclusive(path, obj):
    """Write JSON with mode 'x' so an existing receipt is never clobbered."""
    with Path(path).open("x") as fh:
        fh.write(json.dumps(obj, indent=2) + "\n")


def percentile(values, p):
    """Nearest-rank percentile over non-None values; None when empty."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, math.ceil(p / 100.0 * len(vals)) - 1)
    return vals[k]


def percentiles(values):
    vals = [v for v in values if v is not None]
    return {"n": len(vals), "p50": round6(percentile(vals, 50)),
            "p95": round6(percentile(vals, 95)), "p99": round6(percentile(vals, 99))}


def round6(value):
    return None if value is None else round(value, 6)


# ------------------------------------------------------------- plain JSON IO

def _connect(base_url, timeout):
    url = urllib.parse.urlparse(base_url)
    if url.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme in {base_url!r}")
    port = url.port or (443 if url.scheme == "https" else 80)
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    return cls(url.hostname or "localhost", port, timeout=timeout), url.path.rstrip("/")


def request_json(base_url, method, path, body=None, timeout=60.0):
    """Blocking JSON request. Returns (status, parsed body or None, raw text).

    Transport failures raise; HTTP error statuses are returned to the caller.
    """
    conn, prefix = _connect(base_url, timeout)
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    try:
        conn.request(method, prefix + path, body=data, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
        status = resp.status
    finally:
        conn.close()
    try:
        parsed = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        parsed = None
    return status, parsed, raw


def fetch_models(base_url, timeout=30.0):
    """Return (list of model objects, error string or None)."""
    try:
        status, doc, raw = request_json(base_url, "GET", "/v1/models", timeout=timeout)
    except (OSError, http.client.HTTPException) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if status != 200 or not isinstance(doc, dict):
        return [], f"HTTP {status}: {raw[:200]}"
    return [m for m in doc.get("data") or [] if isinstance(m, dict)], None


def check_health(base_url, timeout=30.0):
    """(healthy, detail). Healthy means GET /health returned 200."""
    try:
        status, _, raw = request_json(base_url, "GET", "/health", timeout=timeout)
    except (OSError, http.client.HTTPException) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return status == 200, f"HTTP {status}: {raw[:200]}"


def tokenize_messages(base_url, model, messages, extra=None, timeout=60.0):
    """Templated prompt token count from Atlas ``POST /tokenize``.

    ``extra`` must carry the same template inputs (chat_template_kwargs,
    tools) as the chat request: /tokenize and /v1/chat/completions share one
    prompt builder, so the counts agree only when the inputs agree.
    """
    body = {"model": model, "messages": messages}
    body.update(extra or {})
    status, doc, raw = request_json(base_url, "POST", "/tokenize", body, timeout=timeout)
    if status != 200 or not isinstance(doc, dict) or not isinstance(doc.get("count"), int):
        raise HarnessError(f"/tokenize failed: HTTP {status}: {raw[:200]}")
    return doc["count"]


# ----------------------------------------------------------------- streaming

class StreamResult:
    """Everything observed for one streaming request."""

    def __init__(self):
        self.status = None          # HTTP status; None if no response line
        self.error = None           # str when the request failed
        self.error_body = None      # raw body of a non-2xx response
        self.error_json = None      # parsed body of a non-2xx response
        self.cancelled = False      # client cancelled on purpose
        self.timed_out = False      # overall deadline expired
        self.done = False           # saw data: [DONE]
        self.token_ids = []
        self.lp_tokens = []         # logprobs.content[].token
        self.logprobs = []          # logprobs.content[].logprob
        self.top_logprobs = []      # logprobs.content[].top_logprobs
        self.content = ""
        self.reasoning = ""
        self.finish_reason = None
        self.usage = None
        self.n_output_chunks = 0
        self.t_start = None         # perf_counter at request start
        self.t_first = None         # perf_counter at first output chunk
        self.t_last = None          # perf_counter at last output chunk
        self.t_end = None
        self.ttft_s = None          # first output-bearing chunk of any kind

    @property
    def ok(self):
        return self.error is None and not self.cancelled

    def cached_tokens(self):
        details = (self.usage or {}).get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
            return details["cached_tokens"]
        return None

    def tpot_s(self):
        """Decode span / (completion_tokens - 1), as in perf_baseline."""
        completion = (self.usage or {}).get("completion_tokens")
        if (self.t_first is None or self.t_last is None
                or not isinstance(completion, int) or completion < 2):
            return None
        return (self.t_last - self.t_first) / (completion - 1)

    def to_record(self):
        tpot = self.tpot_s()
        return {
            "status": self.status,
            "error": self.error,
            "error_body": self.error_body,
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "done": self.done,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "cached_tokens": self.cached_tokens(),
            "token_ids": self.token_ids,
            "logprob_tokens": self.lp_tokens,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "content": self.content,
            "reasoning": self.reasoning,
            "n_output_chunks": self.n_output_chunks,
            "ttft_s": round6(self.ttft_s),
            "tpot_s": round6(tpot),
            "decode_tok_per_s": round6(1.0 / tpot if tpot else None),
            "e2e_s": round6(None if self.t_end is None else self.t_end - self.t_start),
        }


def _consume_chunk(res, chunk, now):
    if not isinstance(chunk, dict):
        raise HarnessError("SSE JSON payload must be an object")
    if chunk.get("error") is not None:
        raise HarnessError(f"server SSE error: {json.dumps(chunk['error'])[:300]}")
    if chunk.get("usage") is not None:
        if not isinstance(chunk["usage"], dict):
            raise HarnessError("SSE usage must be an object")
        res.usage = chunk["usage"]
    choices = chunk.get("choices") or []
    if not isinstance(choices, list):
        raise HarnessError("SSE choices must be an array")
    if len(choices) > 1:
        raise HarnessError("harness requests n=1 but got several choices")
    if not choices:
        return
    choice = choices[0]
    if not isinstance(choice, dict):
        raise HarnessError("SSE choice must be an object")
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        raise HarnessError("SSE delta must be an object")
    ids = choice.get("token_ids") or []
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        raise HarnessError("SSE token_ids must be an integer array")
    res.token_ids.extend(ids)
    lp = choice.get("logprobs")
    if lp is not None:
        entries = lp.get("content") if isinstance(lp, dict) else lp
        if entries is None:
            entries = []
        if not isinstance(lp, dict) or not isinstance(entries, list):
            raise HarnessError("SSE logprobs must be {content: [...]}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise HarnessError("SSE logprobs.content entries must be objects")
            res.lp_tokens.append(entry.get("token"))
            res.logprobs.append(entry.get("logprob"))
            res.top_logprobs.append([[t.get("token"), t.get("logprob")]
                                     for t in entry.get("top_logprobs") or []])
    text = delta.get("content")
    reasoning = delta.get("reasoning_content")
    if text:
        res.content += text
    if reasoning:
        res.reasoning += reasoning
    if text or reasoning or delta.get("tool_calls") or ids:
        res.n_output_chunks += 1
        if res.t_first is None:
            res.t_first = now
            res.ttft_s = now - res.t_start
        res.t_last = now
    if choice.get("finish_reason"):
        res.finish_reason = choice["finish_reason"]


def stream_chat(base_url, body, timeout, cancel_event=None, cancel_after_chunks=None):
    """Run one streaming chat request and return a StreamResult.

    ``timeout`` is an overall deadline in seconds for the whole request
    (connect, queueing, prefill and every chunk), not a per-read timeout.
    ``cancel_event`` (threading.Event) aborts the request when another thread
    sets it; ``cancel_after_chunks`` cancels after that many output-bearing
    chunks. Cancellation closes the socket so the server sees the client go
    away. Per-request failures never raise: they land in ``result.error``.
    """
    body = dict(body)
    body["stream"] = True
    res = StreamResult()
    cancel_event = cancel_event or threading.Event()
    finished = threading.Event()
    reason = {}
    conn, prefix = _connect(base_url, timeout)
    res.t_start = time.perf_counter()
    deadline = res.t_start + timeout

    # http.client drops conn.sock once a will-close response is read, so the
    # watchdog keeps its own reference to the connected socket.
    held = {}

    def watchdog():
        while not finished.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                reason.setdefault("why", "timeout")
                break
            if cancel_event.wait(min(remaining, 0.05)):
                reason.setdefault("why", "cancel")
                break
        if finished.is_set():
            return
        sock = held.get("sock")
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    dog = threading.Thread(target=watchdog, daemon=True)
    dog.start()
    try:
        conn.connect()
        held["sock"] = conn.sock
        if "why" in reason:
            raise HarnessError("deadline or cancellation before connect completed")
        conn.request("POST", prefix + "/v1/chat/completions",
                     body=json.dumps(body).encode("utf-8"),
                     headers={"Content-Type": "application/json",
                              "Accept": "text/event-stream"})
        resp = conn.getresponse()
        res.status = resp.status
        if resp.status >= 300:
            res.error_body = resp.read().decode("utf-8", "replace")
            try:
                res.error_json = json.loads(res.error_body)
            except json.JSONDecodeError:
                res.error_json = None
            raise HarnessError(f"HTTP {resp.status}: {res.error_body[:300]}")
        event_name = None
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "strict").rstrip("\r\n")
            if not line:
                event_name = None
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
                continue
            if not line.startswith("data:"):
                raise HarnessError(f"invalid SSE field: {line[:80]!r}")
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                res.done = True
                break
            if event_name == "error":
                raise HarnessError(f"server SSE error: {payload[:300]}")
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise HarnessError(f"invalid SSE JSON: {exc.msg}") from exc
            before = res.n_output_chunks
            _consume_chunk(res, chunk, time.perf_counter())
            if (cancel_after_chunks is not None and res.n_output_chunks != before
                    and res.n_output_chunks >= cancel_after_chunks):
                reason.setdefault("why", "cancel")
                break
            if cancel_event.is_set():
                reason.setdefault("why", "cancel")
                break
        if not res.done and "why" not in reason:
            raise HarnessError("SSE stream ended before [DONE]")
    except (OSError, http.client.HTTPException, HarnessError, UnicodeDecodeError) as exc:
        if "why" not in reason:
            res.error = f"{type(exc).__name__}: {exc}"
    finally:
        finished.set()
        res.t_end = time.perf_counter()
        # Closing (rather than draining) the socket is the cancellation
        # signal the server's disconnect detection keys off.
        conn.close()
    why = reason.get("why")
    if why == "timeout":
        res.timed_out = True
        res.error = f"timeout: request exceeded the {timeout}s deadline"
    elif why == "cancel" and not res.done:
        res.cancelled = True
    return res


# ---------------------------------------------------------------- comparison

def _sequence(rec, basis):
    if basis == "token_ids":
        return list(rec.get("token_ids") or [])
    if basis == "logprob_tokens":
        return list(rec.get("logprob_tokens") or [])
    return list((rec.get("reasoning") or "") + (rec.get("content") or ""))


def best_basis(rec):
    for basis in BASES[:2]:
        if _sequence(rec, basis):
            return basis
    return "text"


def compare_sequences(reference, candidate, prefix_only=False):
    """Compare two request records (StreamResult.to_record) token for token.

    Uses the finest basis both records carry: token_ids, then logprob token
    strings, then characters of reasoning+content (labelled "text", which is
    not token-level evidence). ``prefix_only`` accepts a candidate that is a
    strict prefix of the reference (a cancelled stream). Reports the first
    divergent index, the tokens there and, when both sides carry logprobs
    aligned with the compared sequence, the logprob gap at that index.
    """
    basis = max(best_basis(reference), best_basis(candidate), key=BASES.index)
    ref_seq = _sequence(reference, basis)
    cand_seq = _sequence(candidate, basis)
    first = None
    for i in range(min(len(ref_seq), len(cand_seq))):
        if ref_seq[i] != cand_seq[i]:
            first = i
            break
    if first is None and len(ref_seq) != len(cand_seq):
        if not (prefix_only and len(cand_seq) < len(ref_seq)):
            first = min(len(ref_seq), len(cand_seq))
    out = {
        "basis": basis,
        "equal": first is None,
        "reference_len": len(ref_seq),
        "candidate_len": len(cand_seq),
        "first_divergence": first,
        "reference_token": None,
        "candidate_token": None,
        "logprob_gap": None,
        "max_abs_logprob_diff_before_divergence": None,
        "finish_reason_mismatch": None,
    }
    if (not prefix_only and first is None
            and reference.get("finish_reason") != candidate.get("finish_reason")):
        out["equal"] = False
        out["finish_reason_mismatch"] = [reference.get("finish_reason"),
                                         candidate.get("finish_reason")]
    ref_lp = reference.get("logprobs") or []
    cand_lp = candidate.get("logprobs") or []
    aligned = (basis != "text" and len(ref_lp) == len(ref_seq)
               and len(cand_lp) == len(cand_seq))
    if first is not None:
        out["reference_token"] = ref_seq[first] if first < len(ref_seq) else None
        out["candidate_token"] = cand_seq[first] if first < len(cand_seq) else None
        if aligned and first < len(ref_lp) and first < len(cand_lp):
            a, b = ref_lp[first], cand_lp[first]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                out["logprob_gap"] = round6(abs(a - b))
    if aligned:
        stop = min(len(ref_seq) if first is None else first, len(ref_lp), len(cand_lp))
        diffs = [abs(ref_lp[i] - cand_lp[i]) for i in range(stop)
                 if isinstance(ref_lp[i], (int, float))
                 and isinstance(cand_lp[i], (int, float))]
        out["max_abs_logprob_diff_before_divergence"] = round6(max(diffs)) if diffs else None
    return out
