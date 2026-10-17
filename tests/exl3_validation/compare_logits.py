# SPDX-License-Identifier: AGPL-3.0-only
"""Compare full-vocabulary teacher-forced rows with explicit tolerances."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from parity_protocol import load_artifact, token_sha, validate_run_pair


def log_softmax(array):
    shifted = array.astype(np.float64) - array.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def validate_pair(reference, candidate):
    validate_run_pair(reference, candidate, ("cases", "tokens_sha256", "vocab", "request_policy", "thresholds"))


def compare(reference, candidate, ref_arrays, candidate_arrays, atol, rtol, kl_limit, ce_limit, top_k):
    validate_pair(reference, candidate)
    configured = {"abs": atol, "rel": rtol, "kl": kl_limit, "cross_entropy": ce_limit, "top_k": top_k, "argmax": "exact"}
    if reference["thresholds"] != configured:
        raise ValueError("comparison tolerances differ from preregistered thresholds")
    thresholds = (atol, rtol, kl_limit, ce_limit)
    if any(not np.isfinite(v) or v < 0 for v in thresholds):
        raise ValueError("tolerances must be explicitly finite and nonnegative")
    if type(top_k) is not int or not 1 <= top_k <= reference["vocab"]:
        raise ValueError("top-k must fit the complete vocabulary")
    reports = []
    for case in reference["cases"]:
        ref, actual = ref_arrays[case["id"]], candidate_arrays[case["id"]]
        expected = (len(case["continuation_ids"]) + 1, reference["vocab"])
        if ref.shape != expected or actual.shape != expected:
            raise ValueError("comparison logits full shape mismatch")
        if not np.isfinite(ref).all() or not np.isfinite(actual).all():
            raise ValueError("comparison logits contain nonfinite values")
        ref64, actual64 = ref.astype(np.float64), actual.astype(np.float64)
        error = np.abs(ref64 - actual64)
        ref_prob, actual_prob = log_softmax(ref64), log_softmax(actual64)
        kl = (np.exp(ref_prob) * (ref_prob - actual_prob)).sum(axis=1)
        # Stable vocabulary ID tie-breaking makes top-k reports deterministic.
        ref_top = np.argsort(-ref64, axis=1, kind="stable")[:, :top_k]
        actual_top = np.argsort(-actual64, axis=1, kind="stable")[:, :top_k]
        top2 = np.sort(ref64, axis=1)[:, -2:]
        ce_ref, ce_actual = [], []
        for row, token in enumerate(case["continuation_ids"]):
            ce_ref.append(float(-ref_prob[row, token]))
            ce_actual.append(float(-actual_prob[row, token]))
        ce_error = max((abs(a - b) for a, b in zip(ce_ref, ce_actual)), default=0.0)
        close = bool(np.all(error <= atol + rtol * np.abs(ref64)))
        same_argmax = bool(np.array_equal(ref_top[:, 0], actual_top[:, 0]))
        ref_text, actual_text = reference["generated_text"][case["id"]], candidate["generated_text"][case["id"]]
        if len(ref_text) != expected[0] or len(actual_text) != expected[0]:
            raise ValueError("HTTP choice rows do not cover every prefix")
        text_matches = ref_text == actual_text
        passed = text_matches and close and same_argmax and bool(np.all(kl <= kl_limit)) and ce_error <= ce_limit
        reports.append({"case": case["id"], "passed": passed, "allclose": close,
                        "generated_text_matches": text_matches, "reference_generated_text": ref_text,
                        "candidate_generated_text": actual_text,
                        "max_abs_error": float(error.max()),
                        "max_rel_error": float((error / np.maximum(np.abs(ref64), np.finfo(np.float32).tiny)).max()),
                        "max_kl_reference_to_candidate": float(kl.max()),
                        "argmax_matches": int((ref_top[:, 0] == actual_top[:, 0]).sum()), "rows": ref.shape[0],
                        "reference_argmax_margins": (top2[:, 1] - top2[:, 0]).tolist(),
                        "reference_topk": ref_top.tolist(), "candidate_topk": actual_top.tolist(),
                        "topk_set_overlap": [len(set(a) & set(b)) / top_k for a, b in zip(ref_top, actual_top)],
                        "reference_cross_entropy": ce_ref, "candidate_cross_entropy": ce_actual,
                        "max_cross_entropy_error": ce_error})
    return {"passed": all(report["passed"] for report in reports), "cases": reports,
            "thresholds": {"abs": atol, "rel": rtol, "kl": kl_limit, "cross_entropy": ce_limit, "top_k": top_k}}


def negative_controls(reference, candidate, arrays, atol, rtol, kl_limit, ce_limit, top_k):
    altered = copy.deepcopy(candidate)
    swapped = False
    for case in altered["cases"]:
        for field in ("prompt_ids", "continuation_ids"):
            tokens = case[field]
            pair = next(((i, j) for i in range(len(tokens)) for j in range(i + 1, len(tokens)) if tokens[i] != tokens[j]), None)
            if pair:
                i, j = pair
                tokens[i], tokens[j] = tokens[j], tokens[i]
                swapped = True
                break
        if swapped:
            break
    if not swapped:
        raise ValueError("negative token-swap control requires two distinct tokens in one prefix/continuation")
    altered["tokens_sha256"] = token_sha(altered["cases"])
    try:
        validate_pair(reference, altered)
    except ValueError:
        tokens_detected = True
    else:
        tokens_detected = False
    shifted = {name: np.roll(array, 1, axis=1) for name, array in arrays.items()}
    shifted_report = compare(reference, candidate, arrays, shifted, atol, rtol, kl_limit, ce_limit, top_k)
    return {"passed": tokens_detected and not shifted_report["passed"],
            "swapped_tokens_detected": tokens_detected, "swapped_tokens_sha256": altered["tokens_sha256"],
            "shuffled_vocabulary_detected": not shifted_report["passed"],
            "vocabulary_transform": "cyclic permutation: column j receives reference column (j-1) modulo vocab"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference", "candidate", "out"):
        parser.add_argument("--" + flag, required=True)
    for flag in ("abs", "rel", "kl", "ce-max-error"):
        parser.add_argument("--" + flag, type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise ValueError("report destination must be new")
    reference, ref_arrays = load_artifact(args.reference)
    candidate, candidate_arrays = load_artifact(args.candidate)
    report = compare(reference, candidate, ref_arrays, candidate_arrays, args.abs, args.rel, args.kl, args.ce_max_error, args.top_k)
    report["negative_controls"] = negative_controls(reference, candidate, ref_arrays, args.abs, args.rel, args.kl, args.ce_max_error, args.top_k)
    report["passed"] = report["passed"] and report["negative_controls"]["passed"]
    report["manifests"] = {label: hashlib.sha256((Path(root) / "manifest.json").read_bytes()).hexdigest()
                           for label, root in (("reference", args.reference), ("candidate", args.candidate))}
    report["compared_utc"] = datetime.now(timezone.utc).isoformat()
    report["harness_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ("compare_logits.py", "parity_protocol.py")}
    report["comparator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"passed": report["passed"], "report": str(destination)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
