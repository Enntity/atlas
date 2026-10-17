# SPDX-License-Identifier: AGPL-3.0-only
"""Reference-only retrospective diagnosis; never collect or score candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tests/exl3_validation"))
import statistical_gate_v2 as gate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path, hashes, root):
    hashes[str(path.relative_to(root))] = file_sha(path)
    return json.loads(path.read_text())


def diagnose(root):
    hashes = {}
    plan = read_json(root / "frozen-plan.json", hashes, root)
    gate.validate_plan(plan)
    runs, metadata = [], []
    targets = (("longer", 3, 13), ("code", 1, 14))
    for index, name in enumerate(gate.REFERENCE_ATTEMPTS):
        base = root / "attempts" / name
        artifact = base / "attempt-files/short16"
        run = gate.load_run(str(artifact))
        runs.append(run)
        hashes[str((artifact / "manifest.json").relative_to(root))] = file_sha(artifact / "manifest.json")
        hashes[str((artifact / "logits.npz").relative_to(root))] = file_sha(artifact / "logits.npz")
        # Also inspect long-suite provenance, but do not read its arrays.
        long_manifest = read_json(base / "attempt-files/long/manifest.json", hashes, root)
        attempt = read_json(base / "attempt.json", hashes, root)
        shutdown = read_json(base / "shutdown.json", hashes, root)
        pre = read_json(base / "cache-pre.json", hashes, root)
        post = read_json(base / "cache-post.json", hashes, root)
        caches = {}
        for before, after in zip(pre["mounts"], post["mounts"]):
            if before["category"] != after["category"]:
                raise ValueError("cache category ordering differs")
            old = {f["path"]: f for f in before["files"]}
            new = {f["path"]: f for f in after["files"]}
            caches[before["category"]] = {
                "pre_sha256": digest(before), "post_sha256": digest(after),
                "pre_files": len(old), "post_files": len(new),
                "added": len(new.keys() - old.keys()), "removed": len(old.keys() - new.keys()),
                "changed": sum(old[p] != new[p] for p in old.keys() & new.keys()),
            }
        manifest = run["manifests"][0]
        contexts = {}
        for case, row, _ in targets:
            capture = next(c for c in manifest["captures"] if c["case"] == case and c["step"] == row)
            capture_name = Path(capture["capture"]).stem
            context = read_json(base / "capture-files" / f"head-position-{capture_name.split('-')[-1]}.json", hashes, root)
            contexts[f"{case}/r{row}"] = {"capture": capture_name, "prefix_sha256": capture["prefix_sha256"],
                "row_sha256": capture["row_sha256"], "context": {
                    k: v for k, v in context["context"].items() if k not in ("request_id", "chunk_counter")}}
        metadata.append({"index": index, "attempt": name, "order_position": plan["attempt_order"].index(name),
            "start_utc": attempt["start_utc"], "shutdown": shutdown,
            "short_collected_utc": manifest["collected_utc"], "long_collected_utc": long_manifest["collected_utc"],
            "stable_short_provenance_sha256": digest({k: v for k, v in manifest["provenance"].items() if k != "utc"}),
            "stable_long_provenance_sha256": digest({k: v for k, v in long_manifest["provenance"].items() if k != "utc"}),
            "normalized_create_commands_sha256": digest(json.loads(json.dumps(attempt["commands"]).replace(name, "ATTEMPT"))),
            "caches": caches, "targets": contexts})
    gate.v1.validate_runs(runs, [runs[0]])
    gate.validate_campaign_provenance(runs, "reference short suite")
    table = gate.v1.distance_table(runs + [runs[0]])
    h3 = gate.evaluate_h3(runs, runs[0], table)
    h4 = gate.evaluate_h4(runs, [], table)
    rows = {}
    for case, row, excluded in targets:
        key = (0, case)
        pairs = [(table[(i, j)][(key, row, "sym_kl")], i, j) for i in range(15) for j in range(i + 1, 15)]
        fit_indices = [i for i in range(15) if i != excluded]
        fit = [runs[i] for i in fit_indices]
        modes = gate.analyze_modes(fit, key, row)
        membership = gate.mode_membership(modes, runs[excluded]["rows"][key][row])
        hashes_by_ref = [gate.row_hash(run["rows"][key][row]) for run in runs]
        classes = {h: [i for i, value in enumerate(hashes_by_ref) if h == value] for h in sorted(set(hashes_by_ref))}
        nearest = min((d, j if i == excluded else i) for d, i, j in pairs if excluded in (i, j))
        fit_spread = max(d for d, i, j in pairs if excluded not in (i, j))
        array = runs[excluded]["rows"][key][row]
        peer = runs[nearest[1]]["rows"][key][row]
        rows[f"short16/{case}/r{row}"] = {
            "excluded_reference": excluded, "excluded_attempt": gate.REFERENCE_ATTEMPTS[excluded],
            "pairs": [{"left": i, "right": j, "sym_kl": d} for d, i, j in pairs],
            "byte_classes": classes, "nearest_reference": nearest[1], "nearest_sym_kl": nearest[0],
            "fit_spread_sym_kl": fit_spread,
            "h3_limit_sym_kl": max(gate.v1.CATASTROPHE_FACTOR * fit_spread, gate.v1.FLOORS["sym_kl"]),
            "full_modes": gate.analyze_modes(runs, key, row, pairs),
            "fit_indices": fit_indices, "leave_one_out_modes": modes, "membership": membership,
            "argmax_by_reference": [int(np.argmax(run["rows"][key][row])) for run in runs],
            "nearest_max_abs_logit_delta": float(np.max(np.abs(array - peer))),
            "nearest_changed_float_count": int(np.count_nonzero(array != peer)),
            "excluded_byte_ulp_vs_fit": gate.byte_ulp_evidence(fit, array, key, row),
        }
    return gate._json_safe({"schema": "atlas-exl3-reference-diagnosis-v1", "campaign_id": plan["campaign_id"],
        "scope": "15 reference short-suite arrays; long-suite metadata only; no candidate or held-out arrays",
        "collection_executed": False, "thresholds_changed": False,
        "reference_build": plan["engine_identities"]["reference"]["build"],
        "reference_provenance": runs[0]["manifests"][0]["provenance"],
        "request_policy": runs[0]["manifests"][0]["request_policy"],
        "short_suite_h3_failures": [x for x in h3["leave_one_out"] if not x["passed"]],
        "short_suite_h4_failures": h4["leave_one_out_failures"],
        "rows": rows, "chronology": sorted(metadata, key=lambda m: m["start_utc"]),
        "input_sha256": hashes,
        "analysis_sha256": file_sha(Path(__file__)),
        "evaluator_sha256": {n: file_sha(REPO / "tests/exl3_validation" / n) for n in
            ("statistical_gate_v2.py", "statistical_gate.py", "parity_protocol.py", "compare_logits.py")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.root)
    with args.out.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
