# SPDX-License-Identifier: AGPL-3.0-only
"""Collect persistent autoregressive completions on frozen prompt token IDs."""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from parity_protocol import canonical, token_sha, validate_cases, validate_provenance

POLICY = {"max_tokens": 16, "temperature": 0, "seed": 42, "stream": False}


def completion(response, prompt):
    if not isinstance(response, dict) or "error" in response:
        raise ValueError("engine returned an error")
    choices, usage = response.get("choices"), response.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("completion requires exactly one choice")
    choice = choices[0]
    if not isinstance(choice.get("text"), str) or not isinstance(choice.get("finish_reason"), str) or not choice["finish_reason"]:
        raise ValueError("completion requires text and terminal finish_reason")
    if not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] != len(prompt):
        raise ValueError("engine prompt-token count differs from prescribed prefix")
    count = usage.get("completion_tokens")
    if type(count) is not int or not 0 <= count <= POLICY["max_tokens"]:
        raise ValueError("completion_tokens must fit the prescribed request")
    ids = choice.get("token_ids")
    if ids is not None and (not isinstance(ids, list) or len(ids) != count or any(type(t) is not int or t < 0 for t in ids)):
        raise ValueError("exposed token_ids must match completion count and contain nonnegative integers")
    return {"text": choice["text"], "finish_reason": choice["finish_reason"], "completion_tokens": count, "token_ids": ids}


def collect_decode(endpoint, model, cases, destination, provenance, timeout):
    cases, provenance = validate_cases(cases), validate_provenance(provenance)
    if timeout <= 0:
        raise ValueError("positive timeout required")
    capture_dir = provenance["environment"].get("ATLAS_EXL3_LOGITS_DIR")
    if capture_dir and (Path(capture_dir) / "capture-enabled").exists():
        raise ValueError("disable the logits capture marker before persistent decode")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("decode destination must be new")
    destination.mkdir(parents=True)
    responses = destination / "responses"
    responses.mkdir()
    records = []
    for case in cases:
        request = {"model": model, "prompt": case["prompt_ids"], **POLICY}
        req = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions", data=canonical(request), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as result:
            response = json.loads(result.read())
        filename = case["id"] + ".json"
        raw = responses / filename
        raw.write_text(json.dumps({"request": request, "response": response}, indent=2))
        records.append({"case": case["id"], "response": filename,
                        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                        "completion": completion(response, case["prompt_ids"])})
    manifest = {"schema": "atlas-exl3-persistent-decode-v1", "cases": cases,
                "tokens_sha256": token_sha(cases), "provenance": provenance, "request_policy": POLICY,
                "endpoint": endpoint, "model": model, "timeout_seconds": timeout, "responses": records,
                "collected_utc": datetime.now(timezone.utc).isoformat(),
                "harness_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in ("collect_decode.py", "parity_protocol.py")}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_decode(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "atlas-exl3-persistent-decode-v1" or manifest.get("request_policy") != POLICY:
        raise ValueError("persistent decode schema/request policy mismatch")
    cases = validate_cases(manifest["cases"])
    validate_provenance(manifest["provenance"])
    if manifest.get("tokens_sha256") != token_sha(cases) or len(manifest["responses"]) != len(cases):
        raise ValueError("persistent decode case/token fingerprint mismatch")
    for case, record in zip(cases, manifest["responses"]):
        filename = case["id"] + ".json"
        if record.get("case") != case["id"] or record.get("response") != filename:
            raise ValueError("persistent decode response mapping mismatch")
        raw = root / "responses" / filename
        if record.get("sha256") != hashlib.sha256(raw.read_bytes()).hexdigest():
            raise ValueError("persistent decode response fingerprint mismatch")
        http = json.loads(raw.read_text())
        if http.get("request") != {"model": manifest["model"], "prompt": case["prompt_ids"], **POLICY}:
            raise ValueError("persistent decode HTTP request mismatch")
        if record.get("completion") != completion(http.get("response"), case["prompt_ids"]):
            raise ValueError("persistent decode completion summary mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("endpoint", "model", "cases", "out", "provenance"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    collect_decode(args.endpoint, args.model, json.loads(Path(args.cases).read_text()), args.out,
                   json.loads(Path(args.provenance).read_text()), args.timeout)


if __name__ == "__main__":
    main()
