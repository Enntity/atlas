# SPDX-License-Identifier: AGPL-3.0-only
"""Production-path fixtures for the registered v2 EXL3 gate semantics."""
import copy
import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np

import statistical_gate_v2 as gate
from statistical_gate import merge_run
from test_statistical_gate import CASES, VOCAB, manifest


def make_run(seed=0):
    rng = np.random.default_rng(seed)
    arrays = {}
    for case in CASES:
        rows = rng.normal(0, 0.02, (len(case["continuation_ids"]) + 1, VOCAB)).astype(np.float32)
        rows[:, 0] += 20
        arrays[case["id"]] = rows
    return merge_run([(manifest("vllm"), arrays)])


def change_row(run, value, row=0, case_id="a"):
    changed = copy.deepcopy(run)
    arr = changed["rows"][(0, case_id)].copy()
    arr[row] = np.asarray(value, dtype=np.float32)
    changed["rows"][(0, case_id)] = arr
    return changed


def add_campaign_metadata(run):
    for item in run["manifests"]:
        item.update({"schema": "atlas-exl3-teacher-forced-v1", "thresholds": {"fixture": True},
                     "pre_registration_sha256": "a" * 64, "endpoint": "http://127.0.0.1:8000",
                     "model": "fixture-model", "timeout_seconds": 30,
                     "harness_sha256": {"collector.py": "b" * 64}, "collector_sha256": "c" * 64})
    return run


def campaign_plan(order=None):
    required = gate.PRIMARY_ATTEMPTS
    if order is None:
        order = list(required)
        random.Random(17).shuffle(order)
    return {"schema": gate.PLAN_SCHEMA, "campaign_id": "fixture-campaign", "seed": 17,
            "attempt_order": list(order), "post_attempt_order": list(gate.POST_ATTEMPTS),
            "perturbation_layers": {"topk7_layer": 3, "drop_shared_layer": 4},
            "engine_identities": {"reference": {"engine": "vllm", "build": manifest("vllm")["provenance"]["engine_build"]},
                                  "candidate": {"engine": "atlas", "build": {**manifest("atlas")["provenance"]["engine_build"],
                                                                             "image": "atlas-test-image"}}}}


class H4Fixtures(unittest.TestCase):
    def setUp(self):
        self.base = make_run(100)
        self.key = (0, "a")
        self.a = np.zeros(VOCAB, dtype=np.float32)
        self.a[0] = 3
        self.b = self.a.copy()
        self.b[1] = 0.25

    def test_exact_two_byte_classes_zero_radius_and_ulp_candidate_failure(self):
        refs = [change_row(self.base, self.a if i < 12 else self.b) for i in range(15)]
        result = gate.analyze_modes(refs, self.key, 0)
        self.assertTrue(result["demonstrated"])
        self.assertEqual(result["mode_sizes"], [12, 3])
        self.assertEqual(result["rho"], 0.0)
        self.assertEqual(result["separation"], float("inf"))
        self.assertTrue(gate.mode_membership(result, self.a)["passed"])
        ulp = np.nextafter(self.a, np.float32(np.inf), dtype=np.float32)
        self.assertFalse(gate.mode_membership(result, ulp)["passed"])
        self.assertTrue(gate.mode_membership(result, ulp)["ulp_control_failure"])
        candidates = [change_row(self.base, ulp)]
        evaluated = gate.evaluate_h4(refs, candidates)
        self.assertFalse(evaluated["passed"])
        self.assertEqual(evaluated["candidate_failures"][0]["row"], 0)
        table = gate.v1.distance_table(refs + candidates)
        table_path = gate.evaluate_h4(refs, candidates, table)
        self.assertEqual(table_path["candidate_failures"], evaluated["candidate_failures"])
        pair_report = gate.evaluate_h4(refs, [refs[0], refs[0]])
        frequency = pair_report["rows"][0]["candidate_mode_frequency"]
        self.assertEqual(sum(item["count"] for item in frequency), 2)

    def test_rare_singleton_component_is_reported_by_the_h4_path(self):
        refs = []
        for index in range(15):
            delta = np.float32(index * 1e-4 if index < 14 else 0.3)
            row = self.a.copy()
            row[1] += delta
            refs.append(change_row(self.base, row))
        result = gate.analyze_modes(refs, self.key, 0)
        self.assertTrue(result["demonstrated"], result)
        self.assertIn(1, result["mode_sizes"])
        self.assertTrue(any(mode["within_radius"] == 0 for mode in result["modes"]))

    def test_one_byte_class_and_zero_cross_distance_do_not_demonstrate(self):
        same = [change_row(self.base, self.a) for _ in range(15)]
        self.assertFalse(gate.analyze_modes(same, self.key, 0)["demonstrated"])
        other = self.a + np.float32(2.0)  # byte-distinct additive shift, exactly equal softmax
        self.assertEqual(gate._sym_kl_vectors(self.a, other), 0.0)
        refs = [change_row(self.base, self.a if i < 8 else other) for i in range(15)]
        self.assertFalse(gate.analyze_modes(refs, self.key, 0)["demonstrated"])

    def test_singleton_spread_and_tied_gap_choose_first_pair(self):
        # Distances form two equal largest ratios; the first sorted gap is the registered split.
        distances = np.asarray([1.0, 2.0, 4.0, 8.0])
        split = gate.select_gap(distances)
        self.assertEqual(split["gap_index"], 0)
        self.assertEqual(split["d_lo"], 1.0)
        # Two singleton components have zero within spread and positive separation => infinity.
        self.assertEqual(gate.separation_ratio(1.0, 0.0), float("inf"))
        self.assertIsNone(gate.separation_ratio(0.0, 0.0))

    def test_loo_detection_uses_each_excluded_reference_set(self):
        refs = [change_row(self.base, self.a if i < 12 else self.b) for i in range(15)]
        report = gate.mode_leave_one_out(refs, self.key, 0)
        self.assertEqual(len(report), 15)
        self.assertTrue(all(row["fit_reference_count"] == 14 for row in report))
        self.assertTrue(all(row["passed"] for row in report))


class H3AndCampaignFixtures(unittest.TestCase):
    def test_clopper_pearson_is_exact_two_sided(self):
        lower, upper = gate.clopper_pearson(3, 15)
        self.assertAlmostEqual(lower, 0.0433, places=3)
        self.assertAlmostEqual(upper, 0.4809, places=3)

    def test_h3_loo_recomputes_spread_after_exclusion_and_never_self_compares(self):
        refs = [make_run(i) for i in range(15)]
        result = gate.evaluate_h3(refs, refs[0])
        self.assertEqual(len(result["leave_one_out"]), 15)
        self.assertTrue(all(len(row["fit_indices"]) == 14 for row in result["leave_one_out"]))
        self.assertTrue(all(row["excluded_reference"] not in row["fit_indices"] for row in result["leave_one_out"]))
        self.assertTrue(all(row["target_index"] not in row["fit_indices"] for row in result["leave_one_out"]))

    def test_plan_enforces_fixed_attempt_set_order_record_and_no_topups(self):
        plan = campaign_plan()
        self.assertTrue(gate.validate_plan(plan)["valid"])
        bad = campaign_plan(order=list(reversed(gate.PRIMARY_ATTEMPTS)))
        with self.assertRaisesRegex(ValueError, "seeded permutation"):
            gate.validate_plan(bad)
        bad = campaign_plan(order=list(gate.PRIMARY_ATTEMPTS) + ["v2ref16"])
        with self.assertRaisesRegex(ValueError, "attempt set"):
            gate.validate_plan(bad)

    def test_blocked_and_incomplete_campaigns_never_pass(self):
        plan = campaign_plan()
        incomplete = gate.evaluate_campaign(plan, {}, cache_receipts={})
        self.assertEqual(incomplete["status"], "INCOMPLETE")
        self.assertEqual(gate.campaign_status(True, reference_verifiable=False), "UNVERIFIABLE")
        self.assertEqual(gate.campaign_status(True, reference_verifiable=True, checks_passed=False), "FAIL")
        self.assertFalse(gate.campaign_status(True, reference_verifiable=False, checks_passed=True) == "PASS")
        order = plan["attempt_order"] + plan["post_attempt_order"]
        ordered_receipts = {}
        for index, attempt in enumerate(order):
            minute = index * 2
            ordered_receipts[attempt] = {
                "pre_start": gate.empty_cache_receipt(attempt, f"2026-10-12T00:{minute:02d}:00Z"),
                "post_shutdown": gate.empty_cache_receipt(attempt, f"2026-10-12T00:{minute + 1:02d}:00Z")}
        self.assertTrue(gate.validate_observed_attempt_order(plan, ordered_receipts)["valid"])
        reverse = dict(ordered_receipts)
        reverse[order[1]] = {"pre_start": gate.empty_cache_receipt(order[1], "2026-10-12T00:00:30Z"),
                             "post_shutdown": gate.empty_cache_receipt(order[1], "2026-10-12T00:00:40Z")}
        with self.assertRaisesRegex(ValueError, "observed attempt sequence"):
            gate.validate_observed_attempt_order(plan, reverse)

    def test_heldouts_are_evaluated_against_fixed_references_only(self):
        refs = [make_run(i) for i in range(15)]
        heldouts = [make_run(40 + i) for i in range(3)]
        result = gate.evaluate_heldouts(refs, heldouts)
        self.assertEqual(len(result["heldouts"]), 3)
        self.assertTrue(all(row["reference_count"] == 15 for row in result["heldouts"]))
        self.assertTrue(all(row["heldout_in_fit"] is False for row in result["heldouts"]))

    def test_full_campaign_accepts_distinct_heldouts_and_keeps_candidate_repeat_check(self):
        plan = campaign_plan()
        builds = plan["engine_identities"]
        runs = {}

        def profile_run(engine, seed, environment=None, native_policy=None):
            run = make_run(seed)
            for item in run["manifests"]:
                provenance = item["provenance"]
                provenance["engine"] = engine
                provenance["engine_build"] = copy.deepcopy(builds["reference" if engine == "vllm" else "candidate"]["build"])
                provenance["environment"] = dict(environment or {})
                if native_policy is not None:
                    provenance["native_numerical_policy"] = dict(native_policy)
                item.update({"schema": "atlas-exl3-teacher-forced-v1", "thresholds": {"fixture": True},
                             "pre_registration_sha256": "a" * 64, "endpoint": "http://127.0.0.1:8000",
                             "model": "fixture-model", "timeout_seconds": 30,
                             "harness_sha256": {"collector.py": "b" * 64}, "collector_sha256": "c" * 64})
            return run

        for index, name in enumerate(gate.REFERENCE_ATTEMPTS):
            runs[name] = profile_run("vllm", index)
        for index, name in enumerate(gate.HELDOUT_ATTEMPTS):
            runs[name] = profile_run("vllm", 40 + index)
        candidate = profile_run("atlas", 80, {"ATLAS_KDA_FLA_REFERENCE": "1"},
                                 {"kda_core": "ATLAS_KDA_FLA_REFERENCE=1"})
        runs["v2cand01"] = candidate
        runs["v2cand02"] = copy.deepcopy(candidate)
        perturbations = {
            "v2pert-mhc_post_1p9": "mhc_post_1p9",
            "v2pert-router_scale_2p4": "router_scale_2p4",
            "v2pert-topk7_layer": "topk7_layer=3",
            "v2pert-drop_shared_layer": "drop_shared_layer=4",
            "v2pert-bf16_routed_accum": "bf16_routed_accum",
        }
        for index, (name, value) in enumerate(perturbations.items()):
            runs[name] = profile_run("atlas", 90 + index,
                                     {"ATLAS_KDA_FLA_REFERENCE": "1", "ATLAS_EXL3_GATE_PERTURBATION": value},
                                     {"kda_core": "ATLAS_KDA_FLA_REFERENCE=1"})
        runs["v2mde"] = profile_run("atlas", 100, {"ATLAS_KDA_FLA_REFERENCE": "0"},
                                    {"kda_core": "ATLAS_KDA_FLA_REFERENCE=0"})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = {}
            for name in gate.PRIMARY_ATTEMPTS + gate.POST_ATTEMPTS:
                manifest_path = root / name / "manifest.json"
                manifest_path.parent.mkdir()
                manifest_path.write_text("{}")
                attempts[name] = str(manifest_path.parent)
            order = plan["attempt_order"] + plan["post_attempt_order"]
            receipts = {}
            for index, name in enumerate(order):
                start = datetime(2026, 10, 12, tzinfo=timezone.utc) + timedelta(minutes=2 * index)
                before = start.isoformat()
                after = (start + timedelta(minutes=1)).isoformat()
                receipts[name] = {"pre_start": gate.empty_cache_receipt(name, before),
                                  "post_shutdown": gate.empty_cache_receipt(name, after)}
            def fixture_gate_result(_references, candidate, _candidate_runs=None):
                environment = candidate["manifests"][0]["provenance"]["environment"]
                return {"passed": not environment.get("ATLAS_EXL3_GATE_PERTURBATION")}

            with mock.patch.object(gate, "load_run", side_effect=lambda spec: runs[Path(spec).name]), \
                    mock.patch.object(gate, "reference_verifiability", return_value={"verifiable": True}), \
                    mock.patch.object(gate, "evaluate_heldouts", return_value={"passed": True, "heldouts": []}), \
                    mock.patch.object(gate, "evaluate_one", side_effect=fixture_gate_result), \
                    mock.patch.object(gate, "negative_controls_v2", return_value={"passed": True}):
                report = gate.evaluate_campaign(plan, attempts, receipts)
                different_candidate = copy.deepcopy(candidate)
                different_candidate["rows"][(0, "a")][0, 0] += 1.0
                runs["v2cand02"] = different_candidate
                mismatched_repeat = gate.evaluate_campaign(plan, attempts, receipts)

        self.assertTrue(any(not np.array_equal(runs["v2refho01"]["rows"][key],
                                               runs["v2refho02"]["rows"][key])
                            for key in runs["v2refho01"]["rows"]))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(mismatched_repeat["status"], "INCOMPLETE")
        self.assertIn("candidate runs are not byte-identical", mismatched_repeat["reason"])

    def test_candidate_rank_and_negative_control_use_unchanged_v1_T(self):
        refs = [make_run(i) for i in range(15)]
        result = gate.evaluate_one(refs, refs[0])
        self.assertEqual(result["T_candidate"], gate.v1.evaluate(refs, refs[0])["T_candidate"])
        self.assertEqual(result["T_references"], gate.v1.evaluate(refs, refs[0])["T_references"])
        self.assertTrue(gate.negative_controls_v2(refs, refs[0])["passed"])

    def test_campaign_provenance_checks_hardware_and_only_allows_control_env_change(self):
        first = [add_campaign_metadata(make_run(seed)) for seed in (71, 72, 74)]
        second = add_campaign_metadata(make_run(73))
        self.assertTrue(gate.validate_campaign_provenance(first + [second], "reference")["valid"])
        changed = copy.deepcopy(second)
        changed["manifests"][0]["provenance"]["hardware"]["driver"] = "other-driver"
        with self.assertRaisesRegex(ValueError, "stable provenance"):
            gate.validate_campaign_provenance(first + [changed], "reference")

        candidate = add_campaign_metadata(make_run(73))
        candidate["manifests"][0]["provenance"]["engine"] = "atlas"
        candidate["manifests"][0]["provenance"]["engine_build"]["image"] = "atlas-image"
        control = copy.deepcopy(candidate)
        control["manifests"][0]["provenance"]["environment"]["ATLAS_EXL3_GATE_PERTURBATION"] = "mhc_post_1p9"
        self.assertTrue(gate.validate_control_provenance(
            first, candidate, control, "perturbation", "ATLAS_EXL3_GATE_PERTURBATION", "mhc_post_1p9"))
        control["manifests"][0]["provenance"]["environment"]["UNREGISTERED"] = "1"
        with self.assertRaisesRegex(ValueError, "runtime environment"):
            gate.validate_control_provenance(
                first, candidate, control, "perturbation", "ATLAS_EXL3_GATE_PERTURBATION", "mhc_post_1p9")

        mde = copy.deepcopy(candidate)
        mde_provenance = mde["manifests"][0]["provenance"]
        mde_provenance["environment"]["ATLAS_KDA_FLA_REFERENCE"] = "0"
        mde_provenance["native_numerical_policy"] = {"kda_core": "ATLAS_KDA_FLA_REFERENCE=0"}
        candidate["manifests"][0]["provenance"]["environment"]["ATLAS_KDA_FLA_REFERENCE"] = "1"
        candidate["manifests"][0]["provenance"]["native_numerical_policy"] = {
            "kda_core": "ATLAS_KDA_FLA_REFERENCE=1"}
        self.assertTrue(gate.validate_control_provenance(
            first, candidate, mde, "MDE", "ATLAS_KDA_FLA_REFERENCE", "0")["valid"])
        mde["manifests"][0]["provenance"]["native_numerical_policy"]["kda_core"] = \
            "ATLAS_KDA_FLA_REFERENCE=1"
        with self.assertRaisesRegex(ValueError, "numerical policy"):
            gate.validate_control_provenance(
                first, candidate, mde, "MDE", "ATLAS_KDA_FLA_REFERENCE", "0")

    def test_cache_receipt_requires_pre_post_mount_manifests_and_explicit_absence(self):
        receipt = gate.empty_cache_receipt("v2ref01")
        self.assertTrue(gate.validate_cache_receipt(receipt)["valid"])
        pair = {"pre_start": receipt,
                "post_shutdown": gate.empty_cache_receipt("v2ref01", "2026-10-12T00:01:00Z")}
        self.assertTrue(gate.cache_receipt_pair_valid(pair, "v2ref01"))
        with self.assertRaisesRegex(ValueError, "pre-start and post-shutdown"):
            gate.cache_receipt_pair_valid({"pre_start": receipt}, "v2ref01")
        receipt["mounts"].pop()
        with self.assertRaisesRegex(ValueError, "cache categories"):
            gate.validate_cache_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
