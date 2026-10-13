# SPDX-License-Identifier: AGPL-3.0-only
"""Fixed-design EXL3 logit gate v2. Collection is deliberately out of scope."""
import argparse
import hashlib
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import statistical_gate as v1
from compare_logits import log_softmax
from parity_protocol import load_artifact

PLAN_SCHEMA = "atlas-exl3-gate-v2-plan-v1"
CAMPAIGN_SCHEMA = "atlas-exl3-gate-v2-campaign-v1"
REFERENCE_ATTEMPTS = tuple(f"v2ref{i:02d}" for i in range(1, 16))
HELDOUT_ATTEMPTS = tuple(f"v2refho{i:02d}" for i in range(1, 4))
CANDIDATE_ATTEMPTS = ("v2cand01", "v2cand02")
PRIMARY_ATTEMPTS = REFERENCE_ATTEMPTS + HELDOUT_ATTEMPTS + CANDIDATE_ATTEMPTS
PERTURBATION_ATTEMPTS = ("v2pert-mhc_post_1p9", "v2pert-router_scale_2p4",
                          "v2pert-topk7_layer", "v2pert-drop_shared_layer",
                          "v2pert-bf16_routed_accum")
POST_ATTEMPTS = PERTURBATION_ATTEMPTS + ("v2mde",)
CACHE_CATEGORIES = ("vllm", "triton", "tilelang", "cuda_compute", "exl3_tune")
N_REFERENCES, N_HELDOUTS = 15, 3
G_MIN, K_MIN, C_RHO = 4.0, 20.0, 2.0


def validate_plan(plan):
    """Validate a frozen plan. This function never starts, schedules or retries a run."""
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported plan schema")
    if not isinstance(plan.get("campaign_id"), str) or not plan["campaign_id"]:
        raise ValueError("campaign_id is required")
    seed = plan.get("seed")
    if type(seed) is not int:
        raise ValueError("integer permutation seed is required")
    order = plan.get("attempt_order")
    if (not isinstance(order, list) or len(order) != len(PRIMARY_ATTEMPTS)
            or not all(isinstance(item, str) for item in order) or set(order) != set(PRIMARY_ATTEMPTS)):
        raise ValueError("primary attempt set must contain exactly the registered 20 attempts")
    expected = list(PRIMARY_ATTEMPTS)
    random.Random(seed).shuffle(expected)
    if order != expected:
        raise ValueError("attempt order is not the registered seeded permutation")
    if plan.get("post_attempt_order") != list(POST_ATTEMPTS):
        raise ValueError("post attempts must follow the registered perturbation and MDE order")
    layers = plan.get("perturbation_layers")
    if not isinstance(layers, dict) or set(layers) != {"topk7_layer", "drop_shared_layer"}:
        raise ValueError("both exact semantic perturbation layers must be frozen")
    if any(type(layer) is not int or not 3 <= layer <= 44 for layer in layers.values()):
        raise ValueError("perturbation layers must be routed layer indices 3..44")
    identities = plan.get("engine_identities")
    if not isinstance(identities, dict) or set(identities) != {"reference", "candidate"}:
        raise ValueError("reference and candidate engine identities must be frozen")
    for label, identity in identities.items():
        if (not isinstance(identity, dict) or set(identity) != {"engine", "build"}
                or identity["engine"] != ("vllm" if label == "reference" else "atlas")
                or not isinstance(identity["build"], dict)
                or set(identity["build"]) != {"commit", "binary_sha256", "source_sha256", "image"}
                or any(not isinstance(value, str) or not value for value in identity["build"].values())):
            raise ValueError(f"{label} engine identity must pin engine and complete build identity")
    if identities["reference"] == identities["candidate"]:
        raise ValueError("reference and candidate engine identities must be distinct")
    return {"valid": True, "attempt_count": len(PRIMARY_ATTEMPTS), "attempt_order": order,
            "post_attempt_order": list(POST_ATTEMPTS), "collection_executed": False}


def _mount_entry(mount):
    if not isinstance(mount, dict) or set(mount) != {"category", "host_path", "container_path", "shared_across_attempts", "state", "files"}:
        raise ValueError("cache mount requires category, paths, sharing, state and files")
    if mount["state"] not in ("present", "empty", "missing"):
        raise ValueError("cache mount state must explicitly be present, empty or missing")
    if not isinstance(mount["host_path"], str) or not isinstance(mount["container_path"], str):
        raise ValueError("cache mount host/container paths must be explicit strings")
    if type(mount["shared_across_attempts"]) is not bool or not isinstance(mount["files"], list):
        raise ValueError("cache mount sharing and file manifest are required")
    if mount["state"] == "present" and not mount["files"]:
        raise ValueError("present cache mount requires a nonempty sorted file manifest")
    if mount["state"] != "present" and mount["files"]:
        raise ValueError("empty/missing cache mount must have no files")
    previous = None
    for item in mount["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ValueError("cache file requires relative path, size and SHA256")
        path = item["path"]
        if (not isinstance(path, str) or not path or "\\" in path or path.startswith("/") or ".." in Path(path).parts
                or previous is not None and path <= previous):
            raise ValueError("cache paths must be safe, relative and sorted")
        if type(item["size"]) is not int or item["size"] < 0:
            raise ValueError("cache file size must be nonnegative")
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("cache file SHA256 is required")
        previous = path


def validate_cache_receipt(receipt):
    if not isinstance(receipt, dict) or set(receipt) != {"attempt", "captured_utc", "mounts"}:
        raise ValueError("cache receipt requires attempt, capture time and mounts")
    if not isinstance(receipt["attempt"], str) or not receipt["attempt"]:
        raise ValueError("cache receipt attempt name is required")
    try:
        captured = datetime.fromisoformat(receipt["captured_utc"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("cache receipt capture time must be ISO-8601 UTC") from error
    if captured.tzinfo is None or captured.utcoffset().total_seconds() != 0:
        raise ValueError("cache receipt capture time must be ISO-8601 UTC")
    mounts = receipt["mounts"]
    if not isinstance(mounts, list) or [m.get("category") for m in mounts if isinstance(m, dict)] != list(CACHE_CATEGORIES):
        raise ValueError("cache categories must cover vLLM, Triton, TileLang, CUDA compute and EXL3 tune")
    for mount in mounts:
        _mount_entry(mount)
    return {"valid": True, "attempt": receipt["attempt"], "captured_utc": receipt["captured_utc"],
            "cache_categories": list(CACHE_CATEGORIES)}


def empty_cache_receipt(attempt, captured_utc="2026-10-12T00:00:00+00:00"):
    """Fixture/helper for explicit all-absent accounting; not evidence of a real attempt."""
    return {"attempt": attempt, "captured_utc": captured_utc, "mounts": [
        {"category": category, "host_path": "", "container_path": "", "shared_across_attempts": False,
         "state": "missing", "files": []} for category in CACHE_CATEGORIES]}


def cache_receipt_pair_valid(pair, attempt):
    if not isinstance(pair, dict) or set(pair) != {"pre_start", "post_shutdown"}:
        raise ValueError(f"{attempt}: pre-start and post-shutdown cache receipts required")
    times = []
    for stage in pair.values():
        validate_cache_receipt(stage)
        if stage["attempt"] != attempt:
            raise ValueError(f"{attempt}: cache receipt attempt mismatch")
        times.append(datetime.fromisoformat(stage["captured_utc"].replace("Z", "+00:00")))
    if times[0] >= times[1]:
        raise ValueError(f"{attempt}: pre-start cache receipt must precede post-shutdown receipt")
    return True


def validate_observed_attempt_order(plan, cache_receipts):
    ordered = plan["attempt_order"] + plan["post_attempt_order"]
    for earlier, later in zip(ordered, ordered[1:]):
        finish = datetime.fromisoformat(cache_receipts[earlier]["post_shutdown"]["captured_utc"].replace("Z", "+00:00"))
        start = datetime.fromisoformat(cache_receipts[later]["pre_start"]["captured_utc"].replace("Z", "+00:00"))
        if finish >= start:
            raise ValueError(f"observed attempt sequence differs from frozen order: {earlier} before {later}")
    return {"valid": True, "attempt_order": ordered}


def row_bytes(row):
    return np.asarray(row, dtype="<f4").tobytes()


def row_hash(row):
    return hashlib.sha256(row_bytes(row)).hexdigest()


def _sym_distances(runs, key, row):
    values = []
    for left in range(len(runs)):
        for right in range(left + 1, len(runs)):
            values.append((_sym_kl_vectors(runs[left]["rows"][key][row],
                                           runs[right]["rows"][key][row]), left, right))
    return values


def _sym_kl_vectors(left, right):
    left = np.asarray(left, dtype=np.float64)[None, :]
    right = np.asarray(right, dtype=np.float64)[None, :]
    lx, ly = log_softmax(left), log_softmax(right)
    px, py = np.exp(lx), np.exp(ly)
    return float((px * (lx - ly)).sum() + (py * (ly - lx)).sum())


def separation_ratio(cross, within):
    if within == 0:
        return float("inf") if cross > 0 else None
    if cross == 0:
        return 0.0
    return cross / within


def select_gap(positive_distances):
    values = np.sort(np.asarray(positive_distances, dtype=np.float64), kind="stable")
    if len(values) < 2:
        return None
    ratios = values[1:] / values[:-1]
    index = int(np.argmax(ratios))  # np.argmax selects the first tied maximum.
    return {"gap_index": index, "d_lo": float(values[index]), "d_hi": float(values[index + 1]),
            "gap_ratio": float(ratios[index]), "threshold": float(np.sqrt(values[index] * values[index + 1]))}


def _components(count, pair_distances, threshold):
    parents = list(range(count))

    def find(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    for distance, left, right in pair_distances:
        if distance <= threshold:
            a, b = find(left), find(right)
            if a != b:
                parents[b] = a
    groups = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def analyze_modes(runs, key, row, pair_distances=None):
    """H4 demonstration and canonical byte-class mode construction for one row."""
    raw = [row_bytes(run["rows"][key][row]) for run in runs]
    hashes = [hashlib.sha256(value).hexdigest() for value in raw]
    unique = sorted(set(hashes))
    pairs = _sym_distances(runs, key, row) if pair_distances is None else pair_distances
    class_distances = {}
    for distance, left, right in pairs:
        class_distances[(hashes[left], hashes[right])] = distance
    classes = {digest: [i for i, item in enumerate(hashes) if item == digest] for digest in unique}
    if len(classes) == 1:
        return {"demonstrated": False, "reason": "one byte class", "byte_classes": hashes}
    if len(classes) == 2:
        a, b = unique
        cross = class_distances.get((a, b), class_distances.get((b, a), 0.0))
        within = max((distance for distance, left, right in pairs if hashes[left] == hashes[right]), default=0.0)
        if cross > 0 and within == 0:
            groups = [classes[a], classes[b]]
            return _mode_result(runs, key, row, groups, hashes, float("inf"), 0.0,
                                {"special_case": "two byte classes, zero within spread"})
    gap = select_gap([distance for distance, _, _ in pairs if distance > 0])
    if gap is None:
        return {"demonstrated": False, "reason": "fewer than two positive pair distances", "byte_classes": hashes}
    components = _components(len(runs), pairs, gap["threshold"])
    if gap["gap_ratio"] < G_MIN or len(components) != 2:
        return {"demonstrated": False, "reason": "gap ratio or component count", "gap": gap,
                "component_count": len(components), "byte_classes": hashes}
    within = max((distance for distance, a, b in pairs
                  if any(a in group and b in group for group in components)), default=0.0)
    cross = min((distance for distance, a, b in pairs
                 if not any(a in group and b in group for group in components)), default=0.0)
    sep = separation_ratio(cross, within)
    if sep is None or sep < K_MIN:
        return {"demonstrated": False, "reason": "separation below K_MIN", "gap": gap,
                "separation": sep, "byte_classes": hashes}
    return _mode_result(runs, key, row, components, hashes, sep, None, {"gap": gap})


def _mode_result(runs, key, row, groups, hashes, separation, _unused, detail):
    modes = []
    rho = 0.0
    for group in groups:
        counts = {}
        for index in group:
            counts[hashes[index]] = counts.get(hashes[index], 0) + 1
        center_hash = min(counts, key=lambda digest: (-counts[digest], digest))
        center_index = next(index for index in group if hashes[index] == center_hash)
        within_radius = max((_sym_kl_vectors(runs[index]["rows"][key][row],
                                              runs[center_index]["rows"][key][row])
                             for index in group), default=0.0)
        rho = max(rho, within_radius)
        modes.append({"members": group, "size": len(group), "center_hash": center_hash,
                      "center": runs[center_index]["rows"][key][row].copy(),
                      "within_radius": within_radius})
    sizes = [m["size"] for m in modes]
    return {"demonstrated": True, "modes": modes, "mode_sizes": sizes,
            "mode_frequencies": [{"count": size, "n": len(runs), "frequency": size / len(runs),
                                  "cp95": clopper_pearson(size, len(runs))} for size in sizes],
            "separation": separation, "rho": rho, "byte_classes": hashes, **detail}


def mode_membership(mode_result, row_value):
    if not mode_result.get("demonstrated"):
        return {"passed": True, "applicable": False}
    value = np.asarray(row_value, dtype=np.float32)
    digest = row_hash(value)
    distances = [_sym_kl_vectors(value, mode["center"])
                 for mode in mode_result["modes"]]
    if mode_result["rho"] == 0:
        passed = any(digest == mode["center_hash"] for mode in mode_result["modes"])
        return {"passed": passed, "applicable": True, "distances": distances,
                "byte_class_hash": digest, "ulp_control_failure": not passed and
                any(np.array_equal(np.nextafter(mode["center"], np.float32(np.inf), dtype=np.float32), value)
                    for mode in mode_result["modes"])}
    limit = C_RHO * mode_result["rho"]
    return {"passed": min(distances) <= limit, "applicable": True,
            "distances": distances, "limit": limit, "byte_class_hash": digest}


def mode_leave_one_out(references, key, row, pair_distances=None):
    output = []
    all_pairs = _sym_distances(references, key, row) if pair_distances is None else pair_distances
    for excluded in range(len(references)):
        fit_indices = [index for index in range(len(references)) if index != excluded]
        fit = [references[index] for index in fit_indices]
        local_index = {source: target for target, source in enumerate(fit_indices)}
        pairs = [(distance, local_index[left], local_index[right]) for distance, left, right in all_pairs
                 if left in local_index and right in local_index]
        result = analyze_modes(fit, key, row, pairs)
        check = mode_membership(result, references[excluded]["rows"][key][row])
        output.append({"excluded_reference": excluded, "fit_reference_count": len(fit),
                       "demonstrated": result.get("demonstrated", False), "passed": check["passed"],
                       "membership": check})
    return output


def evaluate_h3(references, candidate, table=None):
    if len(references) != N_REFERENCES:
        raise ValueError("H3 requires the fixed 15-reference ensemble")
    runs = references + [candidate]
    table = v1.distance_table(runs) if table is None else table
    n = len(references)
    items = v1.items(references[0])
    gated_item_count = sum(item[2] in v1.FLOORS for item in items)
    failures, loo, checks = [], [], []
    for item in items:
        metric = item[2]
        if metric not in v1.FLOORS:
            continue
        spread = max(table[(j, k)][item] for j in range(n) for k in range(j + 1, n))
        nearest = min(table[(n, k)][item] for k in range(n))
        limit = max(v1.CATASTROPHE_FACTOR * spread, v1.FLOORS[metric])
        checks.append({"item": _item_name(item), "spread": spread, "limit": limit,
                       "nearest_candidate": nearest, "passed": nearest <= limit})
        if nearest > limit:
            failures.append({"item": _item_name(item), "nearest": nearest, "spread": spread, "limit": limit})
    for excluded in range(n):
        fit = [index for index in range(n) if index != excluded]
        row_failures = []
        for item in items:
            metric = item[2]
            if metric not in v1.FLOORS:
                continue
            spread = max(table[(j, k)][item] for p, j in enumerate(fit) for k in fit[p + 1:])
            nearest = min(table[(excluded, other)][item] for other in fit)
            limit = max(v1.CATASTROPHE_FACTOR * spread, v1.FLOORS[metric])
            if nearest > limit:
                row_failures.append({"item": _item_name(item), "nearest": nearest, "spread": spread, "limit": limit})
        loo.append({"excluded_reference": excluded, "target_index": excluded,
                    "fit_reference_count": len(fit),
                    "fit_indices": fit, "items_checked": gated_item_count,
                    "failures": row_failures, "passed": not row_failures})
    return {"passed": not failures and all(entry["passed"] for entry in loo),
            "candidate_failures": failures, "candidate_items": checks, "leave_one_out": loo,
            "verifiable": all(entry["passed"] for entry in loo)}


def _item_name(item):
    return f"{item[0][0]}/{item[0][1]}/r{item[1]}/{item[2]}"


def evaluate_h4(references, candidates, table=None):
    if len(references) != N_REFERENCES:
        raise ValueError("H4 requires the fixed 15-reference ensemble")
    failures, loo, rows = [], [], []
    pair_table = None
    if table is not None:
        pair_table = {}
        for key, case in sorted(references[0]["cases"].items()):
            for row in range(len(case["continuation_ids"]) + 1):
                item = (key, row, "sym_kl")
                pair_table[(key, row)] = [
                    (table[(left, right)][item], left, right)
                    for left in range(len(references)) for right in range(left + 1, len(references))]
    for key, case in sorted(references[0]["cases"].items()):
        for row in range(len(case["continuation_ids"]) + 1):
            pairs = None if pair_table is None else pair_table[(key, row)]
            modes = analyze_modes(references, key, row, pairs)
            item_result = {"case": list(key), "row": row, "mode_result": modes}
            if modes.get("demonstrated"):
                candidate_counts = [0] * len(modes["modes"])
                outside_count = 0
                for index, candidate in enumerate(candidates):
                    membership = mode_membership(modes, candidate["rows"][key][row])
                    item_result.setdefault("candidate_membership", []).append(membership)
                    if membership["passed"]:
                        distances = membership.get("distances", [])
                        if distances:
                            nearest = min(range(len(distances)), key=distances.__getitem__)
                            candidate_counts[nearest] += 1
                        else:
                            digest = row_hash(candidate["rows"][key][row])
                            for mode_index, mode in enumerate(modes["modes"]):
                                if digest == mode["center_hash"]:
                                    candidate_counts[mode_index] += 1
                                    break
                    if not membership["passed"]:
                        outside_count += 1
                        failures.append({"case": list(key), "row": row, "candidate": index,
                                         "membership": membership})
                item_result["candidate_mode_frequency"] = [
                    {"center_hash": mode["center_hash"], "count": count, "n": len(candidates),
                     "frequency": count / len(candidates) if candidates else 0.0}
                    for mode, count in zip(modes["modes"], candidate_counts)]
                item_result["candidate_outside_mode_count"] = outside_count
            item_result["byte_ulp_evidence"] = [byte_ulp_evidence(
                references, candidate["rows"][key][row], key, row) for candidate in candidates]
            rows.append(item_result)
            for excluded, entry in enumerate(mode_leave_one_out(references, key, row, pairs)):
                if entry["demonstrated"] and not entry["passed"]:
                    loo.append({"case": list(key), "row": row, **entry})
    return {"passed": not failures and not loo, "candidate_failures": failures,
            "leave_one_out_failures": loo, "rows": rows,
            "verifiable": not loo}


def byte_ulp_evidence(references, value, key, row):
    value = np.asarray(value, dtype=np.float32)
    classes = {}
    for reference in references:
        raw = reference["rows"][key][row].astype(np.float32)
        classes.setdefault(row_hash(raw), raw)
    exact = row_hash(value)
    candidates = [(float(np.max(np.abs(value.astype(np.float64) - raw.astype(np.float64)))), digest, raw)
                  for digest, raw in classes.items()]
    max_abs, nearest_hash, nearest = min(candidates, key=lambda item: item[0])
    ulps = np.abs(_ordered_f32(value).astype(np.int64) - _ordered_f32(nearest).astype(np.int64))
    return {"byte_identical_to_reference": exact in classes,
            "byte_class_sha256": exact if exact in classes else None,
            "nearest_class_sha256": nearest_hash, "max_abs_logit_delta": max_abs,
            "max_f32_ulp_distance": int(np.max(ulps)), "reference_class_count": len(classes)}


def _ordered_f32(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return np.where(bits & np.uint32(0x80000000), ~bits, bits | np.uint32(0x80000000))


def clopper_pearson(successes, trials, confidence=0.95):
    """Exact two-sided binomial interval, solved from binomial tail probabilities."""
    alpha = 1.0 - confidence

    def cdf(k, p):
        return sum(math.comb(trials, j) * p ** j * (1 - p) ** (trials - j)
                   for j in range(k + 1))

    def bisect(fn, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            below = fn(mid) < target
            if below == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if successes == 0 else bisect(lambda p: 1 - cdf(successes - 1, p), alpha / 2, True)
    upper = 1.0 if successes == trials else bisect(lambda p: cdf(successes, p), alpha / 2, False)
    return [lower, upper]


def reference_verifiability(references):
    table = v1.distance_table(references + [references[0]])
    h3 = evaluate_h3(references, references[0], table)
    h4 = evaluate_h4(references, [references[0]], table)
    return {"verifiable": h3["verifiable"] and h4["verifiable"], "H3": h3, "H4": h4}


def evaluate_h5(references, candidate, table=None):
    """Unchanged v1 rank statistic; callable at historical N for continuity checks."""
    runs = references + [candidate]
    table = v1.distance_table(runs) if table is None else table
    scores = v1.rank_scores(runs, table)
    t_candidate = float(scores[-1])
    t_references = [float(value) for value in scores[:-1]]
    p_value = (1 + sum(value >= t_candidate for value in t_references)) / (len(references) + 1)
    suite_scores = {}
    for suite in sorted({key[0] for key in candidate["cases"]}):
        keys = {key for key in candidate["cases"] if key[0] == suite}
        scores_for_suite = v1.rank_scores(runs, table, keys)
        suite_scores[str(suite)] = {"candidate_T": float(scores_for_suite[-1]),
                                    "max_reference_T": float(max(scores_for_suite[:-1]))}
    return {"T_candidate": t_candidate, "T_references": t_references, "p_value": p_value,
            "false_fail_bound": 1 / (len(references) + 1), "suites_diagnostic": suite_scores,
            "passed_rank": t_candidate <= max(t_references),
            "T_implementation": "statistical_gate.py::rank_scores (unchanged v1)"}


def evaluate_one(references, candidate, candidate_runs=None):
    """Full H2-H5 outcome for one candidate, preserving v1's H5 implementation."""
    if len(references) != N_REFERENCES:
        raise ValueError("v2 evaluation requires exactly 15 references")
    runs = references + [candidate]
    table = v1.distance_table(runs)
    candidate_runs = [candidate] if candidate_runs is None else candidate_runs
    h3, h4 = evaluate_h3(references, candidate, table), evaluate_h4(references, candidate_runs, table)
    discrete = v1.discrete_rules(references, candidate)
    h5 = evaluate_h5(references, candidate, table)
    # v1's catastrophic rule is replaced by H3; its discrete rules and T are retained.
    failures = []
    if not h3["passed"]:
        failures.append("H3 envelope failure or unverifiable reference")
    if not h4["passed"]:
        failures.append("H4 mode membership failure or unverifiable reference")
    if discrete:
        failures.append("H2 discrete failure")
    if not h5["passed_rank"]:
        failures.append("H5 exchangeability rank failure")
    return {"passed": not failures, "reasons": failures, "references": N_REFERENCES,
            **h5, "H3": h3, "H4": h4,
            "discrete_failures": discrete,
            "T_implementation": "statistical_gate.py::rank_scores (unchanged v1)"}


def evaluate_heldouts(references, heldouts):
    if len(references) != N_REFERENCES or len(heldouts) != N_HELDOUTS:
        raise ValueError("v2 requires exactly 15 references and 3 pre-designated held-outs")
    reports = []
    for index, heldout in enumerate(heldouts):
        if any(heldout is reference for reference in references):
            raise ValueError("held-out reference cannot be included in the reference fit")
        result = evaluate_one(references, heldout)
        reports.append({"heldout_index": index, "reference_count": len(references),
                        "heldout_in_fit": False, "passed": result["passed"], "gate": result})
    return {"passed": all(row["passed"] for row in reports), "heldouts": reports}


def evaluate_campaign(plan, attempts, cache_receipts):
    """Evaluate recorded campaign inputs only; missing evidence is INCOMPLETE."""
    validate_plan(plan)
    required = set(PRIMARY_ATTEMPTS + POST_ATTEMPTS)
    missing = sorted(required - set(attempts))
    missing_receipts = sorted(required - set(cache_receipts))
    extra = sorted(set(attempts) - required)
    extra_receipts = sorted(set(cache_receipts) - required)
    if missing or missing_receipts or extra or extra_receipts:
        return {"status": campaign_status(False), "passed": False, "missing_attempts": missing,
                "missing_cache_receipts": missing_receipts, "unregistered_attempts": extra,
                "unregistered_cache_receipts": extra_receipts, "collection_executed": False}
    try:
        for attempt in required:
            cache_receipt_pair_valid(cache_receipts[attempt], attempt)
        validate_observed_attempt_order(plan, cache_receipts)
    except ValueError as error:
        return {"status": campaign_status(False), "passed": False, "reason": str(error), "collection_executed": False}
    try:
        loaded = {name: load_run(attempts[name]) for name in required}
        references = [loaded[name] for name in REFERENCE_ATTEMPTS]
        heldouts = [loaded[name] for name in HELDOUT_ATTEMPTS]
        candidates = [loaded[name] for name in CANDIDATE_ATTEMPTS]
        if any(run["manifests"][0]["provenance"]["engine"] != "vllm" for run in references + heldouts):
            raise ValueError("reference and held-out attempts must use the pinned vLLM engine")
        if any(run["manifests"][0]["provenance"]["engine"] != "atlas" for run in candidates):
            raise ValueError("candidate attempts must use the registered Atlas engine")
        for run in references + heldouts + candidates:
            perturbation = run["manifests"][0]["provenance"]["environment"].get("ATLAS_EXL3_GATE_PERTURBATION", "")
            if perturbation:
                raise ValueError("reference, held-out and candidate runs must have no semantic perturbation")
        expected_reference = plan["engine_identities"]["reference"]["build"]
        expected_candidate = plan["engine_identities"]["candidate"]["build"]
        if any(run["manifests"][0]["provenance"]["engine_build"] != expected_reference
               for run in references + heldouts):
            raise ValueError("reference/held-out engine build differs from frozen plan")
        if any(run["manifests"][0]["provenance"]["engine_build"] != expected_candidate
               for run in candidates):
            raise ValueError("candidate engine build differs from frozen plan")
        v1.validate_runs(references, candidates)
        v1.validate_runs(references, heldouts)
        validate_campaign_provenance(references + heldouts, "reference engine")
        validate_campaign_provenance(candidates, "candidate engine")
        verifiability = reference_verifiability(references)
        if not verifiability["verifiable"]:
            return {"status": campaign_status(True, False), "passed": False,
                    "reason": "H3/H4 reference leave-one-out failed", "reference_verifiability": verifiability,
                    "collection_executed": False}
        heldout_report = evaluate_heldouts(references, heldouts)
        candidate_report = evaluate_one(references, candidates[0], candidates)
        negative = negative_controls_v2(references, candidates[0])
        ulp_report = evaluate_one(references, v1.ulp_perturbed(candidates[0]))
        perturbations = {}
        expected_perturbations = {
            "v2pert-mhc_post_1p9": "mhc_post_1p9",
            "v2pert-router_scale_2p4": "router_scale_2p4",
            "v2pert-topk7_layer": f"topk7_layer={plan['perturbation_layers']['topk7_layer']}",
            "v2pert-drop_shared_layer": f"drop_shared_layer={plan['perturbation_layers']['drop_shared_layer']}",
            "v2pert-bf16_routed_accum": "bf16_routed_accum",
        }
        for name in PERTURBATION_ATTEMPTS:
            control = loaded[name]
            if control["manifests"][0]["provenance"]["engine_build"] != expected_candidate:
                raise ValueError(f"{name}: perturbation engine build differs from frozen plan")
            actual = control["manifests"][0]["provenance"]["environment"].get("ATLAS_EXL3_GATE_PERTURBATION")
            if actual != expected_perturbations[name]:
                raise ValueError(f"{name}: recorded semantic perturbation differs from registered control")
            validate_control_provenance(references, candidates[0], control, name,
                                        "ATLAS_EXL3_GATE_PERTURBATION", expected_perturbations[name])
            outcome = evaluate_one(references, control)
            perturbations[name] = {"must_fail": True, "passed_gate": outcome["passed"], "gate": outcome}
        mde = loaded["v2mde"]
        if mde["manifests"][0]["provenance"]["engine_build"] != expected_candidate:
            raise ValueError("MDE engine build differs from frozen plan")
        if mde["manifests"][0]["provenance"]["environment"].get("ATLAS_KDA_FLA_REFERENCE") != "0":
            raise ValueError("MDE attempt must record ATLAS_KDA_FLA_REFERENCE=0")
        validate_control_provenance(references, candidates[0], mde, "v2mde", "ATLAS_KDA_FLA_REFERENCE", "0")
        mde_report = evaluate_one(references, mde)
        all_must_pass = heldout_report["passed"] and candidate_report["passed"] and negative["passed"] and ulp_report["passed"]
        all_must_fail = all(not entry["passed_gate"] for entry in perturbations.values())
        passed = all_must_pass and all_must_fail
        receipt_diffs = {
            name: {before["category"]: before != after
                   for before, after in zip(cache_receipts[name]["pre_start"]["mounts"],
                                            cache_receipts[name]["post_shutdown"]["mounts"])}
            for name in sorted(required)}
        report = {"schema": CAMPAIGN_SCHEMA, "status": campaign_status(True, True, passed),
                "campaign_id": plan["campaign_id"], "passed": passed,
                "reference_verifiability": verifiability, "heldouts": heldout_report,
                "candidate": candidate_report, "negative_controls": negative,
                "ulp_control": ulp_report, "perturbations": perturbations,
                "mde": {"measured_only": True, "gate_diagnostic": mde_report},
                "cache_state_changed": receipt_diffs,
                "cache_receipts": {name: cache_receipts[name] for name in sorted(required)},
                "collection_executed": False}
        report["inputs"] = {name: attempts[name] for name in sorted(required)}
        report["attempt_provenance"] = {
            name: [run_manifest["provenance"] for run_manifest in loaded[name]["manifests"]]
            for name in sorted(required)}
        report["manifest_sha256"] = {
            path: hashlib.sha256((Path(path) / "manifest.json").read_bytes()).hexdigest()
            for spec in attempts.values() for path in spec.split(",")}
        report["harness_sha256"] = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("statistical_gate_v2.py", "statistical_gate.py", "compare_logits.py", "parity_protocol.py")}
        report["evaluated_utc"] = datetime.now(timezone.utc).isoformat()
        return report
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        return {"status": campaign_status(False), "passed": False, "reason": f"artifact/provenance invalid: {error}",
                "collection_executed": False}


def load_run(spec):
    return v1.merge_run([load_artifact(path) for path in spec.split(",")])


def validate_campaign_provenance(runs, label):
    """Check all stable artifact/provenance fields within one engine profile."""
    first = runs[0]["manifests"]
    stable_manifest_fields = ("schema", "thresholds", "endpoint",
                              "model", "timeout_seconds", "harness_sha256", "collector_sha256")
    for run in runs:
        if len(run["manifests"]) != len(first):
            raise ValueError(f"{label}: every attempt must contain the same suite count")
        for base, manifest in zip(first, run["manifests"]):
            for field in stable_manifest_fields:
                if field not in base or manifest.get(field) != base[field]:
                    raise ValueError(f"{label}: attempt artifacts differ in {field}")
            provenance = manifest["provenance"]
            reference_provenance = base["provenance"]
            if not isinstance(provenance.get("hardware"), dict) or not provenance.get("hostname"):
                raise ValueError(f"{label}: hardware and hostname provenance are required")
            stable = {key: value for key, value in provenance.items() if key != "utc"}
            reference_stable = {key: value for key, value in reference_provenance.items() if key != "utc"}
            if stable != reference_stable:
                raise ValueError(f"{label}: attempts differ in stable provenance fields")
            if (not isinstance(manifest["harness_sha256"], dict) or not manifest["harness_sha256"]
                    or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                           for value in manifest["harness_sha256"].values())):
                raise ValueError(f"{label}: complete harness SHA256 map is required")
            if (not isinstance(manifest["collector_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", manifest["collector_sha256"])):
                raise ValueError(f"{label}: collector SHA256 is required")
    return {"valid": True, "attempts": len(runs), "stable_provenance_fields": True}


def validate_control_provenance(reference_runs, candidate_base, control, label, changed_env_key, changed_env_value):
    """Controls may vary only their registered environment switch, not stable provenance."""
    v1.validate_runs(reference_runs, [control])
    for base_manifest, control_manifest in zip(candidate_base["manifests"], control["manifests"]):
        base_prov = {key: value for key, value in base_manifest["provenance"].items()
                     if key not in ("utc", "environment")}
        control_prov = {key: value for key, value in control_manifest["provenance"].items()
                        if key not in ("utc", "environment")}
        if changed_env_key == "ATLAS_KDA_FLA_REFERENCE":
            base_policy = base_prov.get("native_numerical_policy", {})
            control_policy = control_prov.get("native_numerical_policy", {})
            if (base_manifest["provenance"]["environment"].get(changed_env_key) != "1"
                    or base_policy.get("kda_core") != f"{changed_env_key}=1"):
                raise ValueError(f"{label}: candidate baseline KDA numerical policy is not reference mode")
            if control_policy.get("kda_core") != f"{changed_env_key}={changed_env_value}":
                raise ValueError(f"{label}: numerical policy does not match the registered KDA control")
            base_prov["native_numerical_policy"] = {
                key: value for key, value in base_policy.items() if key != "kda_core"}
            control_prov["native_numerical_policy"] = {
                key: value for key, value in control_policy.items() if key != "kda_core"}
        if control_prov != base_prov:
            raise ValueError(f"{label}: stable provenance differs from candidate profile")
        expected_environment = dict(base_manifest["provenance"]["environment"])
        expected_environment[changed_env_key] = changed_env_value
        if control_manifest["provenance"]["environment"] != expected_environment:
            raise ValueError(f"{label}: runtime environment differs beyond its registered control switch")
        for field in ("schema", "thresholds", "endpoint", "model",
                      "timeout_seconds", "harness_sha256", "collector_sha256"):
            if control_manifest.get(field) != base_manifest.get(field):
                raise ValueError(f"{label}: artifact metadata differs in {field}")
    return {"valid": True, "allowed_environment_change": {changed_env_key: changed_env_value}}


def negative_controls_v2(references, candidate):
    manifests = [json.loads(json.dumps(manifest)) for manifest in candidate["manifests"]]
    target = next((manifest, case) for manifest in manifests for case in manifest["cases"]
                  if len(set(case["prompt_ids"])) > 1)
    target[1]["prompt_ids"] = target[1]["prompt_ids"][::-1]
    try:
        v1.validate_runs(references, [dict(candidate, manifests=manifests)])
        token_swap_refused = False
    except ValueError:
        token_swap_refused = True
    shifted = dict(candidate, rows={key: np.roll(value, 1, axis=1) for key, value in candidate["rows"].items()})
    shifted_gate = evaluate_one(references, shifted)
    return {"passed": token_swap_refused and not shifted_gate["passed"],
            "token_swap_refused": token_swap_refused, "vocabulary_shift_failed_gate": not shifted_gate["passed"],
            "vocabulary_shift_gate": shifted_gate}


def campaign_status(complete, reference_verifiable=True, checks_passed=False):
    if not complete:
        return "INCOMPLETE"
    if not reference_verifiable:
        return "UNVERIFIABLE"
    return "PASS" if checks_passed else "FAIL"


def _load_campaign(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported campaign schema")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-plan", help="validate a frozen plan JSON; does not collect")
    modes.add_argument("--campaign", help="evaluate a recorded campaign JSON")
    parser.add_argument("--out", help="write JSON report to a new path")
    args = parser.parse_args()
    if args.validate_plan:
        report = validate_plan(json.loads(Path(args.validate_plan).read_text()))
        report["status"] = "PLAN_VALID"
    else:
        campaign = _load_campaign(args.campaign)
        report = evaluate_campaign(campaign["plan"], campaign.get("attempts", {}),
                                   campaign.get("cache_receipts", {}))
    encoded = json.dumps(_json_safe(report), indent=2, allow_nan=False)
    if args.out:
        destination = Path(args.out)
        if destination.exists():
            raise ValueError("report destination must be new")
        destination.write_text(encoded + "\n")
    print(encoded)
    raise SystemExit(0 if report.get("status") in ("PLAN_VALID", "PASS") else 1)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return {"sha256": hashlib.sha256(value.astype("<f4").tobytes()).hexdigest(),
                "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and np.isinf(value):
        return "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
