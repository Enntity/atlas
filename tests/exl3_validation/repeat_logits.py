# SPDX-License-Identifier: AGPL-3.0-only
"""Check same-engine reproducibility using unchanged preregistered gates."""
import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from compare_logits import log_softmax
from parity_protocol import load_artifact, validate_provenance, validate_thresholds


def compare_repeats(runs):
    if len(runs) < 2:
        raise ValueError("at least two independent captures are required")
    first = runs[0][0]
    limits = validate_thresholds(first["thresholds"])
    for manifest, arrays in runs:
        validate_provenance(manifest["provenance"])
        for field in ("cases", "tokens_sha256", "vocab", "request_policy", "thresholds"):
            if manifest[field] != first[field]:
                raise ValueError(f"repeat inputs differ: {field}")
        fingerprint = lambda value: {key: item for key, item in value.items() if key != "utc"}
        if fingerprint(manifest["provenance"]) != fingerprint(first["provenance"]):
            raise ValueError("repeat fingerprint differs")
        if set(arrays) != {case["id"] for case in first["cases"]}:
            raise ValueError("repeat case coverage mismatch")
        for case in first["cases"]:
            array = arrays[case["id"]]
            shape = (len(case["continuation_ids"]) + 1, first["vocab"])
            if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
                raise ValueError("repeat requires finite complete F32 vocabulary rows")
            if len(manifest["generated_text"][case["id"]]) != shape[0]:
                raise ValueError("repeat HTTP text coverage mismatch")
    pairs = []
    for i, j in itertools.combinations(range(len(runs)), 2):
        cases = []
        for case in first["cases"]:
            name = case["id"]
            left, right = runs[i][1][name], runs[j][1][name]
            error = np.abs(left.astype(np.float64) - right.astype(np.float64))
            lp, rp = log_softmax(left), log_softmax(right)
            kl = (np.exp(lp) * (lp - rp)).sum(axis=1)
            ce = max((abs(lp[row, token] - rp[row, token])
                      for row, token in enumerate(case["continuation_ids"])), default=0.)
            close = bool(np.all(error <= limits["abs"] + limits["rel"] * np.abs(left.astype(np.float64))))
            argmax = bool(np.array_equal(left.argmax(axis=1), right.argmax(axis=1)))
            text = runs[i][0]["generated_text"][name] == runs[j][0]["generated_text"][name]
            passed = close and argmax and text and bool(np.all(kl <= limits["kl"])) and ce <= limits["cross_entropy"]
            cases.append({"case": name, "passed": passed,
                          "bitwise_equal": left.tobytes() == right.tobytes(),
                          "allclose": close, "argmax_matches": argmax, "text_matches": text,
                          "max_abs_error": float(error.max()), "max_kl": float(kl.max()),
                          "max_cross_entropy_error": float(ce)})
        pairs.append({"left": i, "right": j, "passed": all(c["passed"] for c in cases), "cases": cases})
    return {"schema": "atlas-exl3-repeat-v1", "passed": all(p["passed"] for p in pairs),
            "thresholds": limits, "runs": len(runs), "pairs": pairs,
            "scope": "same-engine reproducibility; does not establish Atlas/vLLM parity"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise ValueError("report destination must be new")
    report = compare_repeats([load_artifact(root) for root in args.run])
    report["artifacts"] = [{"path": str(Path(root).resolve()),
                            "manifest_sha256": hashlib.sha256((Path(root) / "manifest.json").read_bytes()).hexdigest()}
                           for root in args.run]
    report["harness_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("repeat_logits.py", "compare_logits.py", "parity_protocol.py")}
    report["utc"] = datetime.now(timezone.utc).isoformat()
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"passed": report["passed"], "report": str(destination)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
