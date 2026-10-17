# SPDX-License-Identifier: AGPL-3.0-only
"""Compare actual persistent decode outputs without text normalization."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from collect_decode import load_decode
from parity_protocol import validate_run_pair


def compare_decode(reference, candidate):
    validate_run_pair(reference, candidate, ("cases", "tokens_sha256", "request_policy"))
    if len(reference["responses"]) != len(reference["cases"]) or len(candidate["responses"]) != len(candidate["cases"]):
        raise ValueError("persistent decode responses must cover every case")
    reports = []
    for expected, actual in zip(reference["responses"], candidate["responses"]):
        if expected["case"] != actual["case"]:
            raise ValueError("persistent decode response case mismatch")
        expected, actual, name = expected["completion"], actual["completion"], expected["case"]
        matches = {field: expected[field] == actual[field] for field in ("text", "finish_reason", "completion_tokens")}
        ids_available = expected["token_ids"] is not None and actual["token_ids"] is not None
        ids_match = expected["token_ids"] == actual["token_ids"] if ids_available else None
        reports.append({"case": name, "passed": all(matches.values()) and (ids_match is not False),
                        "matches": matches, "token_ids_available_both": ids_available,
                        "token_ids_match": ids_match, "reference": expected, "candidate": actual})
    return {"passed": all(r["passed"] for r in reports), "cases": reports,
            "scope": "persistent greedy HTTP text/finish/count; token IDs checked only when both engines expose them"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference", "candidate", "out"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise ValueError("decode comparison report must be new")
    reference, candidate = load_decode(args.reference), load_decode(args.candidate)
    report = compare_decode(reference, candidate)
    report["compared_utc"] = datetime.now(timezone.utc).isoformat()
    report["manifests"] = {name: hashlib.sha256((Path(root) / "manifest.json").read_bytes()).hexdigest()
                           for name, root in (("reference", args.reference), ("candidate", args.candidate))}
    report["harness_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                               for name in ("compare_decode.py", "collect_decode.py", "parity_protocol.py")}
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "report": str(destination)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
