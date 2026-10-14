# SPDX-License-Identifier: AGPL-3.0-only
"""Collect one full-vocabulary row for each prescribed continuation prefix."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from parity_protocol import canonical, read_raw, token_sha, validate_cases, validate_provenance, validate_thresholds


def collect(endpoint, model, cases, capture_dir, destination, provenance, timeout, thresholds,
            request_ids=None):
    """Collect serial prefixes; optional request IDs follow case/step order.

    An explicit schedule binds C2 fixture invocations to x-request-id without
    changing the registered token cases or the default collection protocol.
    """
    cases = validate_cases(cases)
    if request_ids is not None:
        count = sum(len(case["continuation_ids"]) + 1 for case in cases)
        if (not isinstance(request_ids, list) or len(request_ids) != count
                or any(not isinstance(value, str) or not value
                       or any(ord(char) < 33 or ord(char) > 126 for char in value)
                       for value in request_ids)):
            raise ValueError("request IDs must be explicit visible ASCII strings, one per prescribed prefix")
        if len(set(request_ids)) != count:
            raise ValueError("request IDs must be unique across the collection schedule")
        request_ids = list(request_ids)
    provenance = validate_provenance(provenance)
    thresholds = validate_thresholds(thresholds)
    capture_dir, destination = Path(capture_dir), Path(destination)
    if not capture_dir.is_dir() or timeout <= 0:
        raise ValueError("capture directory and positive timeout are required")
    marker = capture_dir / "capture-enabled"
    if marker.exists():
        raise ValueError("capture-enabled marker must be absent before collection")
    if list(capture_dir.glob("logits-*")):
        raise ValueError("capture directory must contain no earlier logits")
    if destination.exists():
        raise ValueError("destination must be new to preserve earlier evidence")
    destination.mkdir(parents=True)
    responses = destination / "responses"
    responses.mkdir()
    registration = {"thresholds": thresholds, "registered_utc": datetime.now(timezone.utc).isoformat(),
                    "tokens_sha256": token_sha(cases), "provenance": provenance}
    if request_ids is not None:
        registration["request_ids"] = request_ids
    registration_file = destination / "pre_registration.json"
    registration_file.write_text(json.dumps(registration, indent=2))
    arrays, records, vocab = {}, [], None
    with marker.open("x") as enabled:
        enabled.write(hashlib.sha256(registration_file.read_bytes()).hexdigest())
    try:
        for case in cases:
            rows = []
            for step in range(len(case["continuation_ids"]) + 1):
                prefix = case["prompt_ids"] + case["continuation_ids"][:step]
                request = {"model": model, "prompt": prefix, "max_tokens": 1,
                           "temperature": 0, "seed": 42, "stream": False}
                before = set(capture_dir.glob("logits-*.json"))
                headers = {"Content-Type": "application/json"}
                if request_ids is not None:
                    headers["x-request-id"] = request_ids[len(records)]
                req = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions",
                                             data=canonical(request), headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as result:
                    response = json.loads(result.read())
                if "error" in response or not response.get("choices"):
                    raise ValueError("engine did not complete the prescribed request")
                filename = f"{case['id']}-{step:06d}.json"
                (responses / filename).write_text(json.dumps({"request": request, "response": response}, indent=2))
                deadline = time.monotonic() + timeout
                while True:
                    added = set(capture_dir.glob("logits-*.json")) - before
                    if len(added) > 1:
                        raise ValueError("ambiguous capture: more than one logits sidecar per request")
                    if added:
                        sidecar = next(iter(added))
                        if sidecar.name != f"logits-{len(records):06d}.json":
                            raise ValueError("capture files must be sequential from zero")
                        row, metadata = read_raw(sidecar)
                        break
                    if time.monotonic() >= deadline:
                        raise ValueError("missing logits capture for completed request")
                    time.sleep(0.02)
                if vocab is not None and row.size != vocab:
                    raise ValueError("vocabulary changed during collection")
                vocab = row.size
                if max(prefix) >= vocab:
                    raise ValueError("request token outside captured vocabulary")
                rows.append(row)
                records.append({"case": case["id"], "step": step, "prefix_sha256": hashlib.sha256(canonical(prefix)).hexdigest(),
                                "capture": str(sidecar.resolve()), "capture_metadata": metadata,
                                "raw_sha256": hashlib.sha256((sidecar.parent / metadata["file"]).read_bytes()).hexdigest(),
                                "row_sha256": hashlib.sha256(row.astype("<f4").tobytes()).hexdigest(), "response": filename})
                if request_ids is not None:
                    records[-1]["request_id"] = request_ids[len(records) - 1]
            arrays[case["id"]] = np.stack(rows)
    finally:
        marker.unlink()
    if thresholds["top_k"] > vocab:
        raise ValueError("preregistered top-k exceeds captured vocabulary")
    np.savez(destination / "logits.npz", **arrays)
    manifest = {"schema": "atlas-exl3-teacher-forced-v1", "provenance": provenance,
                "cases": cases, "tokens_sha256": token_sha(cases), "vocab": vocab,
                "thresholds": thresholds, "pre_registration_sha256": hashlib.sha256(registration_file.read_bytes()).hexdigest(),
                "endpoint": endpoint, "model": model, "timeout_seconds": timeout, "captures": records,
                "collected_utc": datetime.now(timezone.utc).isoformat(),
                "harness_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ("collect_logits.py", "parity_protocol.py")},
                "request_policy": {"temperature": 0, "seed": 42, "max_tokens": 1},
                "logits_sha256": hashlib.sha256((destination / "logits.npz").read_bytes()).hexdigest(),
                "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if request_ids is not None:
        manifest["request_ids"] = request_ids
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("endpoint", "model", "cases", "capture-dir", "out", "provenance", "thresholds"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--request-ids", help="JSON list of request IDs in case/step order")
    args = parser.parse_args()
    collect(args.endpoint, args.model, json.loads(Path(args.cases).read_text()),
            args.capture_dir, args.out, json.loads(Path(args.provenance).read_text()), args.timeout, json.loads(Path(args.thresholds).read_text()),
            request_ids=json.loads(Path(args.request_ids).read_text()) if args.request_ids else None)


if __name__ == "__main__":
    main()
