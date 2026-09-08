# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic exact-schema fixtures, not native numerical evidence."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import analyze_glm_mtp_hidden_trace as analyzer
from analyze_glm_mtp_hidden_trace import analyze, compare, parse_line


def row(rank=0, generation=1, attempt=1, step=0, **changes):
    result = dict(rank=rank, slot=0, generation=generation, attempt=attempt,
                  position=148 + (attempt - 1) * 4, seed=3, hidden_row=0,
                  step=step, input_token=3 if step == 0 else 7,
                  draft=7, eh_nvfp4=True, head_nvfp4=step > 0,
                  input_sha256=f"{step:064x}", final_sha256=f"{step + 1:064x}",
                  argmax_pair_bytes=None)
    result["self.cache_before"] = 147 + (attempt - 1) * 4 + step
    result["self.cache_after"] = result["self.cache_before"] + 1
    result.update(changes)
    return result


def wire(record):
    def value(key, val):
        if key == "argmax_pair_bytes":
            return "None" if val is None else f"Some({json.dumps(val)})"
        if isinstance(val, bool):
            return str(val).lower()
        return str(val)
    return "2026-09-08T12:00:00Z INFO module: GLM MTP HIDDEN_TRACE " + " ".join(
        f"{key}={value(key, val)}" for key, val in record.items()) + "\n"


def rows(rank, generation=1, attempts=8):
    return [row(rank, generation, attempt, step)
            for attempt in range(1, attempts + 1) for step in range(4)]


class TraceAnalysisTests(unittest.TestCase):
    def test_version2_post_eh_boundary_and_legacy_availability(self):
        records = [rows(0, 1), rows(1, 2)]
        for items in records:
            for item in items:
                item.update(trace_version=2, post_eh_sha256="a" * 64 if item["step"] == 0 else None)
        report = self.run_records(records)
        steps = report["requests"]["A"]["cross_rank"]
        self.assertTrue(steps[0]["post_eh_equal"])
        self.assertIsNone(steps[1]["post_eh_equal"])
        left = parse_line(wire(records[0][0]))
        right = {**left, "post_eh_sha256": "b" * 64}
        self.assertEqual(compare("A", left, "B", right)["classification"], "post_eh_difference")
        legacy = parse_line(wire(row()))
        result = compare("A", left, "B", legacy)
        self.assertIsNone(result["post_eh_equal"])
        self.assertEqual(result["post_eh_availability"], [True, False])
        self.assertEqual(result["classification"], "post_eh_availability_difference")

    def test_version2_strict_schema_step_and_request_consistency(self):
        valid = row(trace_version=2, post_eh_sha256="a" * 64)
        for changes in ({"trace_version": 3}, {"trace_version": 1},
                        {"post_eh_sha256": None}, {"post_eh_sha256": "A" * 64},
                        {"step": 1}, {"post_eh_sha256": "0"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_line(wire({**valid, **changes}))
        for missing in ("trace_version", "post_eh_sha256"):
            broken = {**valid}
            del broken[missing]
            with self.assertRaises(ValueError):
                parse_line(wire(broken))
        for fault in ("within_rank", "across_ranks"):
            records = [rows(0, 1), rows(1, 2)]
            for item in records[0][:1 if fault == "within_rank" else 32]:
                item.update(trace_version=2, post_eh_sha256="a" * 64 if item["step"] == 0 else None)
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "schema"):
                self.run_records(records)

    def test_explicit_legacy_and_extended_requests_keep_missing_boundary_evidence(self):
        records = [[], []]
        requests = []
        for generation, label in ((1, "legacy"), (2, "extended")):
            requests.append(dict(id=label, expected_attempts=8, ranks=[
                dict(rank=rank, path=f"rank{rank}.log", slot=0, generation=generation)
                for rank in range(2)]))
            for rank in range(2):
                items = rows(rank, generation)
                if generation == 2:
                    for item in items:
                        item.update(trace_version=2, post_eh_sha256="a" * 64 if item["step"] == 0 else None)
                records[rank].extend(items)
        report = self.run_records(records, requests, [["legacy", "extended"]])
        steps = report["comparisons"][0]["ranks"]["0"]["steps"]
        self.assertEqual(steps[0]["classification"], "post_eh_availability_difference")
        self.assertEqual(steps[0]["trace_versions"], [1, 2])
        self.assertEqual(steps[1]["classification"], "observed_agreement")
        self.assertIsNone(steps[1]["post_eh_equal"])
        left = parse_line(wire(row()))
        right = parse_line(wire(row(trace_version=2, post_eh_sha256="a" * 64, final_sha256="b" * 64)))
        result = compare("legacy", left, "extended", right)
        self.assertEqual(result["classification"], "final_hidden_difference")
        self.assertEqual(result["post_eh_availability"], [False, True])
        self.assertIsNone(result["post_eh_equal"])

    def run_records(self, records, requests=None, comparisons=None):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for rank, items in enumerate(records):
                (base / f"rank{rank}.log").write_text(
                    "ordinary startup line\n" + "".join(wire(item) for item in items))
            if requests is None:
                requests = [dict(id="A", expected_attempts=8, ranks=[
                    dict(rank=rank, path=f"rank{rank}.log", slot=0, generation=rank + 1)
                    for rank in range(2)])]
            return analyze(dict(requests=requests, comparisons=comparisons or []), base)

    def test_complete_both_rank_actual_generation_records(self):
        report = self.run_records([rows(0, 1), rows(1, 2)])
        self.assertEqual(report["requests"]["A"]["record_count"], 64)
        self.assertEqual(report["files"]["rank0.log"]["trace_records"], 32)
        self.assertEqual(len(report["requests"]["A"]["cross_rank"]), 32)
        self.assertTrue(all(item["classification"] == "observed_agreement"
                            for item in report["requests"]["A"]["cross_rank"]))
        self.assertIn("equal private KV", report["limitations"])

    def test_missing_duplicate_foreign_and_order_refuse(self):
        original = [rows(0, 1), rows(1, 2)]
        for fault in ("missing", "duplicate", "foreign", "order"):
            records = copy.deepcopy(original)
            if fault == "missing":
                records[0].pop()
            elif fault == "duplicate":
                records[0].append(records[0][0])
            elif fault == "foreign":
                records[0][0]["generation"] = 3
            else:
                records[0][0], records[0][1] = records[0][1], records[0][0]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.run_records(records)

    def test_malformed_fields_and_chain_refuse(self):
        for changes in ({"input_sha256": "0"}, {"step": 4}, {"attempt": 9},
                        {"input_token": 2}, {"self.cache_after": 900},
                        {"argmax_pair_bytes": [0]}, {"unexpected": 1}):
            records = [rows(0, 1), rows(1, 2)]
            records[0][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_records(records)
        records = [rows(0, 1), rows(1, 2)]
        records[0][1]["input_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            self.run_records(records)

    def test_all_six_request_selectors_cover_same_complete_rank_files(self):
        records = [[], []]
        requests = []
        for generation in range(1, 7):
            requests.append(dict(id=f"request{generation}", expected_attempts=8, ranks=[
                dict(rank=rank, path=f"rank{rank}.log", slot=0, generation=generation + 10 * rank)
                for rank in range(2)]))
            for rank in range(2):
                records[rank].extend(rows(rank, generation + 10 * rank))
        report = self.run_records(records, requests)
        self.assertEqual(len(report["requests"]), 6)
        self.assertEqual(report["files"]["rank0.log"]["trace_records"], 192)
        self.assertEqual(report["files"]["rank1.log"]["trace_records"], 192)
        with self.assertRaisesRegex(ValueError, "undeclared trace owner"):
            self.run_records(records, requests[:-1])

    def test_strict_wire_fields_ansi_optional_pairs_and_duplicate_detection(self):
        pairs = list(struct.pack("<fIfI", 1.25, 7, -2.5, 8))
        line = wire(row(argmax_pair_bytes=pairs))
        self.assertEqual(parse_line("\x1b[32m" + line + "\x1b[0m")["argmax_pair_bytes"], pairs)
        self.assertEqual(parse_line(wire(row()))["self.cache_before"], 147)
        for malformed in (line + " rank=0", line.replace("self.cache_before", "cache_before"),
                          line.replace("eh_nvfp4=true", "eh_nvfp4=1"),
                          line.replace("rank=0", "rank=-1"),
                          line.replace("generation=1", "generation=18446744073709551616"),
                          line.replace("GLM MTP HIDDEN_TRACE", "broken HIDDEN_TRACE"),
                          line.replace("argmax_pair_bytes=Some", "argmax_pair_bytes=Bad"),
                          line.replace("seed=3", "seed=3rank=0")):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                parse_line(malformed)
        for pairs in ([True] * 16, [256] * 16,
                      list(struct.pack("<fIfI", float("nan"), 0, 1.0, 0))):
            with self.assertRaises(ValueError):
                parse_line(wire(row(argmax_pair_bytes=pairs)))

    def test_comparison_classifies_observed_boundaries_without_kv_assumptions(self):
        left = row(argmax_pair_bytes=list(struct.pack("<fIfI", 1.0, 7, 0.5, 3)))
        cases = [({"input_sha256": "f" * 64}, "input_difference"),
                 ({"input_token": 4}, "input_difference"),
                 ({"final_sha256": "f" * 64}, "final_hidden_difference"),
                 ({"argmax_pair_bytes": list(struct.pack("<fIfI", 2.0, 7, 0.5, 3))}, "argmax_pair_difference"),
                 ({"draft": 8}, "draft_difference"), ({}, "observed_agreement")]
        for changes, expected in cases:
            right = {**left, **changes}
            result = compare("A", left, "B", right)
            self.assertEqual(result["classification"], expected)
            self.assertEqual(result["left"]["generation"], 1)
            self.assertEqual(result["argmax_pair_evidence"][0][1]["value"], 0.5)
        changed_path = compare("A", left, "B", {**left, "head_nvfp4": True})
        self.assertIn("head_nvfp4", changed_path["metadata_differences"])

    def test_missing_pair_evidence_is_not_reported_as_full_observed_agreement(self):
        left = row(argmax_pair_bytes=list(struct.pack("<fIfI", 1.0, 7, 0.5, 3)))
        right = {**left, "argmax_pair_bytes": None}
        for first, second in ((left, right), (right, left)):
            result = compare("A", first, "B", second)
            self.assertEqual(result["classification"], "argmax_pair_availability_difference")

    def test_explicit_startup_env_receipts_are_not_malformed_events(self):
        for line in ("INFO startup: ATLAS_GLM_MTP_HIDDEN_TRACE=1\n",
                     "GLM_MTP_HIDDEN_TRACE=0\n"):
            self.assertIsNone(parse_line(line))
        for line in ("GLM MTP HIDDEN_TRACE broken=1\n", "broken HIDDEN_TRACE\n",
                     "ATLAS_GLM_MTP_HIDDEN_TRACE=true\n"):
            with self.assertRaises(ValueError):
                parse_line(line)

    def test_cross_request_match_uses_position_seed_not_attempt_number(self):
        requests = [dict(id=label, expected_attempts=8, ranks=[
            dict(rank=rank, path=f"rank{rank}.log", slot=0, generation=generation)
            for rank in range(2)]) for label, generation in (("A", 1), ("B", 2))]
        records = []
        for rank in range(2):
            shifted = rows(rank, 2)
            for item in shifted:
                item["position"] += 4
            records.append(rows(rank, 1) + shifted)
        report = self.run_records(records, requests, [["A", "B"]])
        result = report["comparisons"][0]["ranks"]["0"]
        self.assertEqual(len(result["steps"]), 28)
        self.assertEqual(result["steps"][0]["left"]["attempt"], 2)
        self.assertEqual(result["steps"][0]["right"]["attempt"], 1)
        self.assertEqual(len(result["unmatched_left"]), 1)
        self.assertEqual(len(result["unmatched_right"]), 1)
        for item in records[0][:8]:
            item["position"] = 148
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.run_records(records, requests, [["A", "B"]])

    def test_manifest_ownership_types_counts_and_duplicate_comparisons_reject(self):
        good = [dict(id="A", expected_attempts=8, ranks=[
            dict(rank=rank, path=f"rank{rank}.log", slot=0, generation=rank + 1)
            for rank in range(2)])]
        for fault in range(6):
            requests = copy.deepcopy(good)
            if fault == 0:
                requests[0]["expected_attempts"] = True
            elif fault == 1:
                requests[0]["ranks"][0]["generation"] = 0
            elif fault == 2:
                requests[0]["ranks"][1]["rank"] = 0
            elif fault == 3:
                requests.append(requests[0])
            elif fault == 4:
                requests[0]["expected_attempts"] = 9
            else:
                requests[0]["extra"] = "unsupported"
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.run_records([rows(0, 1), rows(1, 2)], requests)
        for comparisons in ([["A", "A"]], [["A", "missing"]], [["A"]]):
            with self.assertRaises(ValueError):
                self.run_records([rows(0, 1), rows(1, 2)], good, comparisons)

    def test_line_and_file_bounds_fail_before_unbounded_read(self):
        with patch.object(analyzer, "MAX_LINE_BYTES", 32):
            with self.assertRaisesRegex(ValueError, "line exceeds bound"):
                self.run_records([rows(0, 1), rows(1, 2)])
        with patch.object(analyzer, "MAX_FILE_BYTES", 2048):
            with self.assertRaisesRegex(ValueError, "file exceeds bound"):
                self.run_records([rows(0, 1), rows(1, 2)])

    def test_cli_rejects_duplicate_manifest_keys_without_reading_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"requests": [], "requests": [], "comparisons": []}')
            result = subprocess.run([sys.executable, str(Path(analyzer.__file__)),
                                     "--manifest", str(path)], capture_output=True,
                                    text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn("duplicate JSON field", json.loads(result.stderr)["error"])

    def test_cli_complete_success_and_truncated_request_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for rank in range(2):
                (base / f"rank{rank}.log").write_text("".join(wire(item) for item in rows(rank, rank + 1, 1)))
            manifest = dict(requests=[dict(id="native-window", expected_attempts=1, ranks=[
                dict(rank=rank, path=f"rank{rank}.log", slot=0, generation=rank + 1)
                for rank in range(2)])], comparisons=[])
            path = base / "manifest.json"
            path.write_text(json.dumps(manifest))
            command = [sys.executable, str(Path(analyzer.__file__)), "--manifest", str(path)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["requests"]["native-window"]["record_count"], 8)
            (base / "rank1.log").write_text("ordinary log with no successful snapshots\n")
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn("missing/extra trace records", json.loads(result.stderr)["error"])


if __name__ == "__main__":
    unittest.main()
