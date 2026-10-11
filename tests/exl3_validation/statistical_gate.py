# SPDX-License-Identifier: AGPL-3.0-only
"""Pre-registered statistical logit gate (docs/design/exl3-statistical-gate.md).

H0: the candidate's teacher-forced logits are exchangeable with one more fresh
reference run. Every run is a list of collect_logits.py artifact directories
(one per suite, e.g. short16 then long-prefix), joined with commas.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from compare_logits import compare, log_softmax
from parity_protocol import load_artifact, token_sha

TOP_K = 10
CATASTROPHE_FACTOR = 3.0
FLOORS = {"sym_kl": 0.005, "ce_error": 0.15}  # frozen thresholds kept as never-catastrophic floor
METRICS = ("sym_kl", "ce_error", "topk_miss")
IDENTITY = ("cases", "tokens_sha256", "vocab", "request_policy")
PROVENANCE = ("checkpoint_sha256", "checkpoint_revision", "tokenizer_sha256", "config_sha256", "execution")


# ---------------------------------------------------------------- loading
def merge_run(artifacts):
    """[(manifest, arrays)] of one run -> run dict keyed by (suite, case)."""
    run = {"manifests": [m for m, _ in artifacts], "rows": {}, "text": {}, "cases": {}}
    for suite, (manifest, arrays) in enumerate(artifacts):
        for case in manifest["cases"]:
            key = (suite, case["id"])
            run["rows"][key] = arrays[case["id"]].astype(np.float64)
            run["text"][key] = manifest["generated_text"][case["id"]]
            run["cases"][key] = case
    return run


def validate_runs(references, candidates):
    if len(references) < 3:
        raise ValueError("gate requires at least 3 reference runs (leave-one-out needs 2)")
    if not candidates:
        raise ValueError("gate requires at least one candidate run")
    first = references[0]["manifests"]
    for run in references + candidates:
        if len(run["manifests"]) != len(first):
            raise ValueError("every run must supply the same suites")
        for base, manifest in zip(first, run["manifests"]):
            for field in IDENTITY:
                if manifest[field] != base[field]:
                    raise ValueError(f"runs differ in {field}")
            for field in PROVENANCE:
                if manifest["provenance"][field] != base["provenance"][field]:
                    raise ValueError(f"runs differ in provenance {field}")
    for group, label in ((references, "reference"), (candidates, "candidate")):
        profile = group[0]["manifests"][0]["provenance"]
        for run in group:
            for manifest in run["manifests"]:
                provenance = manifest["provenance"]
                if provenance["environment"] != profile["environment"]:
                    raise ValueError(f"{label} runs mix runtime environments")
                if provenance["serve_flags"] != profile["serve_flags"]:
                    raise ValueError(f"{label} runs mix serve configurations")
                if provenance["engine_build"] != profile["engine_build"]:
                    raise ValueError(f"{label} runs mix engine builds")
                if provenance.get("capture_hooks_sha256") != profile.get("capture_hooks_sha256"):
                    raise ValueError(f"{label} runs mix capture hook hashes")
    for run in references + candidates:
        if any(not np.isfinite(a).all() for a in run["rows"].values()):
            raise ValueError("nonfinite logits")
    base = candidates[0]
    for run in candidates[1:]:
        if any(not np.array_equal(run["rows"][k], base["rows"][k]) for k in base["rows"]) or run["text"] != base["text"]:
            raise ValueError("candidate runs are not byte-identical")


# ---------------------------------------------------------------- metrics
def row_metrics(x, y, case):
    """Per row: symmetric KL, CE error at the continuation token, 1 - top-k overlap."""
    lx, ly = log_softmax(x), log_softmax(y)
    px, py = np.exp(lx), np.exp(ly)
    sym = (px * (lx - ly)).sum(1) + (py * (ly - lx)).sum(1)
    tx = np.argsort(-x, axis=1, kind="stable")[:, :TOP_K]
    ty = np.argsort(-y, axis=1, kind="stable")[:, :TOP_K]
    miss = np.array([1 - len(set(a) & set(b)) / TOP_K for a, b in zip(tx, ty)])
    ce = np.full(x.shape[0], np.nan)  # last row has no continuation token
    for row, token in enumerate(case["continuation_ids"]):
        ce[row] = abs(lx[row, token] - ly[row, token])
    return {"sym_kl": sym, "ce_error": ce, "topk_miss": miss}


def items(run):
    """Ordered (key, row, metric) items that are defined (CE skips the last row)."""
    out = []
    for key, case in sorted(run["cases"].items()):
        rows = len(case["continuation_ids"]) + 1
        for row in range(rows):
            for metric in METRICS:
                if metric == "ce_error" and row >= len(case["continuation_ids"]):
                    continue
                out.append((key, row, metric))
    return out


def distance_table(runs):
    """d[i][j][(key,row,metric)] for all ordered pairs i<j (symmetric)."""
    n, table = len(runs), {}
    for i in range(n):
        for j in range(i + 1, n):
            entry = {}
            for key, case in runs[0]["cases"].items():
                metrics = row_metrics(runs[i]["rows"][key], runs[j]["rows"][key], case)
                for metric, values in metrics.items():
                    for row, value in enumerate(values):
                        if np.isfinite(value):
                            entry[(key, row, metric)] = float(value)
            table[(i, j)] = table[(j, i)] = entry
    return table


def average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def rank_scores(runs, table, keys=None):
    """T(X) for every member of S = runs (candidate last)."""
    n = len(runs)
    selected = [it for it in items(runs[0]) if keys is None or it[0] in keys]
    totals = np.zeros(n)
    for it in selected:
        s = [np.mean([table[(x, y)][it] for y in range(n) if y != x]) for x in range(n)]
        totals += average_ranks(s)
    return totals / max(len(selected), 1)


# ---------------------------------------------------------------- rules
def catastrophic(cand_index, ref_indices, table):
    """min_k d(A,R_k) <= max(c * max_{j<k} d(R_j,R_k), floor) for sym_kl and ce_error."""
    failures = []
    for it in table[(ref_indices[0], ref_indices[1])]:
        metric = it[2]
        if metric not in FLOORS:
            continue
        spread = max(table[(j, k)][it] for j in ref_indices for k in ref_indices if j < k)
        nearest = min(table[(cand_index, k)][it] for k in ref_indices)
        limit = max(CATASTROPHE_FACTOR * spread, FLOORS[metric])
        if nearest > limit:
            failures.append({"case": list(it[0]), "row": it[1], "metric": metric,
                             "nearest": nearest, "limit": limit})
    return failures


def discrete_rules(references, candidate):
    """Argmax and HTTP text: unanimous references must be matched, else in the observed set."""
    failures = []
    for key, case in sorted(candidate["cases"].items()):
        for row in range(len(case["continuation_ids"]) + 1):
            observed = {int(np.argmax(r["rows"][key][row])) for r in references}
            got = int(np.argmax(candidate["rows"][key][row]))
            if got not in observed:
                failures.append({"case": list(key), "row": row, "rule": "argmax",
                                 "candidate": got, "references": sorted(observed)})
            texts = {r["text"][key][row] for r in references}
            if candidate["text"][key][row] not in texts:
                failures.append({"case": list(key), "row": row, "rule": "text",
                                 "candidate": candidate["text"][key][row], "references": sorted(texts)})
    return failures


def evaluate(references, candidate):
    """Apply the registered rule to one candidate run against N reference runs."""
    runs = references + [candidate]
    n = len(references)
    table = distance_table(runs)
    ref_idx = list(range(n))
    scores = rank_scores(runs, table)
    t_a, t_refs = float(scores[-1]), [float(v) for v in scores[:-1]]
    p = (1 + sum(t >= t_a for t in t_refs)) / (n + 1)
    loo = []
    for k in ref_idx:
        rest = [j for j in ref_idx if j != k]
        loo.append({"reference": k, "failures": catastrophic(k, rest, table)})
    verifiable = all(not row["failures"] for row in loo)
    catastrophes = catastrophic(n, ref_idx, table)
    discrete = discrete_rules(references, candidate)
    suites = {}
    for suite in sorted({key[0] for key in candidate["cases"]}):
        keys = {key for key in candidate["cases"] if key[0] == suite}
        suite_scores = rank_scores(runs, table, keys)
        suites[str(suite)] = {"candidate_T": float(suite_scores[-1]),
                              "max_reference_T": float(max(suite_scores[:-1]))}
    reasons = []
    if t_a > max(t_refs):
        reasons.append("exchangeability rank: candidate is farther from the reference set than every reference")
    if not verifiable:
        reasons.append("catastrophic-row rule unverifiable: a leave-one-out reference fails it")
    if catastrophes:
        reasons.append(f"{len(catastrophes)} catastrophic row(s)")
    if discrete:
        reasons.append(f"{len(discrete)} argmax/text rule failure(s)")
    return {"passed": not reasons, "reasons": reasons, "references": n,
            "T_candidate": t_a, "T_references": t_refs, "p_value": p,
            "false_fail_bound": 1 / (n + 1), "suites_diagnostic": suites,
            "catastrophic_failures": catastrophes, "catastrophic_rule_verifiable": verifiable,
            "leave_one_out": loo, "discrete_failures": discrete,
            "distances_to_references": [
                {"reference": k, "metrics": {f"{it[0][0]}/{it[0][1]}/{it[1]}/{it[2]}": v
                                             for it, v in table[(n, k)].items()}}
                for k in ref_idx]}


def negative_controls(references, candidate):
    """Token-swap must be refused; a cyclic vocabulary shift must fail the gate."""
    manifests = [json.loads(json.dumps(m)) for m in candidate["manifests"]]
    target = next((m, c) for m in manifests for c in m["cases"] if len(set(c["prompt_ids"])) > 1)
    target[1]["prompt_ids"] = target[1]["prompt_ids"][::-1]
    target[0]["tokens_sha256"] = token_sha(target[0]["cases"])
    try:
        validate_runs(references, [dict(candidate, manifests=manifests)])
        tokens_detected = False
    except ValueError:
        tokens_detected = True
    shifted = dict(candidate, rows={k: np.roll(v, 1, axis=1) for k, v in candidate["rows"].items()})
    shifted_result = evaluate(references, shifted)
    return {"passed": tokens_detected and not shifted_result["passed"],
            "swapped_tokens_detected": tokens_detected,
            "shifted_vocabulary_detected": not shifted_result["passed"],
            "shifted_vocabulary_reasons": shifted_result["reasons"]}


def ulp_perturbed(run):
    """Must-pass control: every logit moved one f32 ulp toward +inf."""
    rows = {k: np.nextafter(v.astype(np.float32), np.float32(np.inf)).astype(np.float64)
            for k, v in run["rows"].items()}
    return dict(run, rows=rows)


# ---------------------------------------------------------------- CLI
def load_run(spec):
    return merge_run([load_artifact(path) for path in spec.split(",")])


def secondary_frozen(references, candidate_artifacts):
    """Frozen-threshold compare_logits report per reference (record only)."""
    out = []
    for r_spec in references:
        for r_path, (cand, cand_arrays) in zip(r_spec.split(","), candidate_artifacts):
            ref, ref_arrays = load_artifact(r_path)
            t = ref["thresholds"]
            try:
                report = compare(ref, cand, ref_arrays, cand_arrays, t["abs"], t["rel"], t["kl"],
                                 t["cross_entropy"], t["top_k"])
                out.append({"reference": r_path, "passed": report["passed"],
                            "cases": [{k: c[k] for k in ("case", "passed", "max_abs_error",
                                                          "max_kl_reference_to_candidate",
                                                          "max_cross_entropy_error")}
                                      for c in report["cases"]]})
            except ValueError as error:
                out.append({"reference": r_path, "skipped": str(error)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True,
                        help="one reference run: comma-joined artifact dirs, one per suite")
    parser.add_argument("--candidate", action="append",
                        help="one candidate run (repeat; all runs must be byte-identical)")
    parser.add_argument("--held-out-reference",
                        help="control: evaluate this reference run as the candidate (must pass)")
    parser.add_argument("--ulp-control", action="store_true",
                        help="control: also evaluate the candidate moved one ulp (must pass)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.candidate) == bool(args.held_out_reference):
        parser.error("give --candidate runs or --held-out-reference, not both")
    destination = Path(args.out)
    if destination.exists():
        raise ValueError("report destination must be new")
    references = [load_run(spec) for spec in args.reference]
    specs = args.candidate or [args.held_out_reference]
    candidates = [load_run(spec) for spec in specs]
    validate_runs(references, candidates)
    report = {"mode": "held-out-reference" if args.held_out_reference else "candidate",
              "gate": evaluate(references, candidates[0]),
              "negative_controls": negative_controls(references, candidates[0])}
    report["passed"] = report["gate"]["passed"] and report["negative_controls"]["passed"]
    if args.ulp_control:
        report["ulp_control"] = evaluate(references, ulp_perturbed(candidates[0]))
        report["passed"] = report["passed"] and report["ulp_control"]["passed"]
    if args.candidate:
        report["secondary_frozen_thresholds"] = secondary_frozen(
            args.reference, [load_artifact(p) for p in args.candidate[0].split(",")])
    report["inputs"] = {"references": args.reference, "candidates": specs}
    report["manifest_sha256"] = {path: hashlib.sha256((Path(path) / "manifest.json").read_bytes()).hexdigest()
                                 for spec in args.reference + specs for path in spec.split(",")}
    report["harness_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("statistical_gate.py", "compare_logits.py", "parity_protocol.py")}
    report["evaluated_utc"] = datetime.now(timezone.utc).isoformat()
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"passed": report["passed"], "reasons": report["gate"]["reasons"],
                      "p_value": report["gate"]["p_value"], "report": str(destination)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
