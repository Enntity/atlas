# SPDX-License-Identifier: AGPL-3.0-only
import copy
import unittest

import numpy as np
from parity_protocol import token_sha
from statistical_gate import (average_ranks, evaluate, merge_run, negative_controls,
                              ulp_perturbed, validate_runs)
from test_parity import provenance

VOCAB = 64
CASES = [{"id": "a", "prompt_ids": [1, 2, 3], "continuation_ids": [4, 5, 6]},
         {"id": "b", "prompt_ids": [7, 8], "continuation_ids": [9, 10, 11]}]


def manifest(engine, serve="serve"):
    prov = provenance(engine)
    prov["serve_flags"] = [serve]
    return {"provenance": prov, "cases": copy.deepcopy(CASES), "tokens_sha256": token_sha(CASES),
            "vocab": VOCAB, "request_policy": {"temperature": 0, "seed": 42, "max_tokens": 1},
            "generated_text": {c["id"]: ["t"] * 4 for c in CASES}}


BASE = np.random.default_rng(0).normal(0, 3, (len(CASES), 4, VOCAB))
BASE[:, :, 0] += 20  # clear argmax margin so noise never flips it


def run(rng, sigma=0.01, engine="vllm", bias=None, serve="serve"):
    arrays = {}
    for index, case in enumerate(CASES):
        rows = BASE[index] + rng.normal(0, sigma, BASE[index].shape)
        if bias is not None:
            rows = rows + bias[index]
        arrays[case["id"]] = rows.astype(np.float32)
    return merge_run([(manifest(engine, serve), arrays)])


class GateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.refs = [run(self.rng) for _ in range(5)]

    def test_exchangeable_candidate_fails_at_about_one_over_n_plus_one(self):
        rng, n, trials, fails = np.random.default_rng(2), 4, 200, 0
        for _ in range(trials):
            runs = [run(rng) for _ in range(n + 1)]
            fails += not evaluate(runs[:n], runs[n])["passed"]
        rate = fails / trials
        self.assertGreater(rate, 0.10)
        self.assertLess(rate, 0.32)  # expected 1/(n+1) = 0.2

    def test_shifted_candidate_fails_rank_and_catastrophic_rules(self):
        bias = np.zeros_like(BASE)
        bias[:, :, 5] = 8.0
        result = evaluate(self.refs, run(self.rng, engine="atlas", bias=bias))
        self.assertFalse(result["passed"])
        self.assertTrue(result["catastrophic_failures"])
        self.assertGreater(result["T_candidate"], max(result["T_references"]))

    def test_single_catastrophic_row_is_reported(self):
        bias = np.zeros_like(BASE)
        bias[1, 2, 7] = 12.0
        result = evaluate(self.refs, run(self.rng, engine="atlas", bias=bias))
        self.assertFalse(result["passed"])
        rows = {(tuple(f["case"]), f["row"]) for f in result["catastrophic_failures"]}
        self.assertIn(((0, "b"), 2), rows)

    def test_reference_outlier_makes_rule_unverifiable(self):
        bias = np.zeros_like(BASE)
        bias[0, 1, 3] = 15.0
        refs = self.refs[:4] + [run(self.rng, bias=bias)]
        result = evaluate(refs, run(self.rng, engine="atlas"))
        self.assertFalse(result["catastrophic_rule_verifiable"])
        self.assertIn("catastrophic-row rule unverifiable: a leave-one-out reference fails it",
                      result["reasons"])

    def test_average_ranks_ties(self):
        self.assertEqual(average_ranks([1.0, 3.0, 1.0, 2.0]).tolist(), [1.5, 4.0, 1.5, 3.0])

    def test_text_rules_unanimous_and_observed_set(self):
        candidate = run(self.rng, engine="atlas")
        candidate["text"][(0, "a")] = ["t", "x", "t", "t"]
        failures = evaluate(self.refs, candidate)["discrete_failures"]
        self.assertEqual([(f["rule"], f["row"]) for f in failures], [("text", 1)])
        self.refs[2]["text"][(0, "a")] = ["t", "x", "t", "t"]
        self.assertFalse(evaluate(self.refs, candidate)["discrete_failures"])

    def test_argmax_outside_observed_set_fails(self):
        bias = np.zeros_like(BASE)
        bias[0, 0, 9] = 40.0
        failures = evaluate(self.refs, run(self.rng, engine="atlas", bias=bias))["discrete_failures"]
        self.assertIn(("argmax", 0), [(f["rule"], f["row"]) for f in failures])

    def test_controls(self):
        candidate = run(self.rng, engine="atlas")
        controls = negative_controls(self.refs, candidate)
        self.assertTrue(controls["passed"])
        self.assertTrue(controls["swapped_tokens_detected"])
        self.assertTrue(controls["shifted_vocabulary_detected"])
        perturbed = ulp_perturbed(self.refs[0])
        self.assertTrue(evaluate(self.refs[1:], perturbed)["passed"])

    def test_validation_refuses_bad_inputs(self):
        cand = run(self.rng, engine="atlas")
        with self.assertRaisesRegex(ValueError, "at least 3"):
            validate_runs(self.refs[:2], [cand])
        with self.assertRaisesRegex(ValueError, "mix serve"):
            validate_runs(self.refs + [run(self.rng, serve="other")], [cand])
        mixed_references = copy.deepcopy(self.refs)
        mixed_references[-1]["manifests"][0]["provenance"]["environment"]["TRACE_HOOK"] = "enabled"
        with self.assertRaisesRegex(ValueError, "mix runtime environments"):
            validate_runs(mixed_references, [cand])
        second_candidate = copy.deepcopy(cand)
        second_candidate["manifests"][0]["provenance"]["environment"]["CANDIDATE_CAPTURE"] = "enabled"
        with self.assertRaisesRegex(ValueError, "mix runtime environments"):
            validate_runs(self.refs, [cand, second_candidate])
        separate_engine_profile = copy.deepcopy(cand)
        separate_engine_profile["manifests"][0]["provenance"]["environment"] = {"CAPTURE": "candidate-only"}
        validate_runs(self.refs, [separate_engine_profile, copy.deepcopy(separate_engine_profile)])
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            validate_runs(self.refs, [cand, run(self.rng, engine="atlas")])
        other = copy.deepcopy(cand)
        other["manifests"][0]["cases"][0]["continuation_ids"] = [4, 5, 7]
        other["manifests"][0]["tokens_sha256"] = token_sha(other["manifests"][0]["cases"])
        with self.assertRaisesRegex(ValueError, "differ in cases"):
            validate_runs(self.refs, [other])
        bad = copy.deepcopy(cand)
        bad["rows"][(0, "a")][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_runs(self.refs, [bad])
        validate_runs(self.refs, [cand, copy.deepcopy(cand)])

    def test_build_and_capture_hook_hashes_are_pinned_per_engine(self):
        mixed_build = copy.deepcopy(self.refs)
        mixed_build[-1]["manifests"][0]["provenance"]["engine_build"]["source_sha256"] = "different-source"
        with self.assertRaisesRegex(ValueError, "mix engine builds"):
            validate_runs(mixed_build, [run(self.rng, engine="atlas")])

        candidate = run(self.rng, engine="atlas")
        mixed_candidate_build = copy.deepcopy(candidate)
        mixed_candidate_build["manifests"][0]["provenance"]["engine_build"]["binary_sha256"] = "different-binary"
        with self.assertRaisesRegex(ValueError, "mix engine builds"):
            validate_runs(self.refs, [candidate, mixed_candidate_build])

        mixed_hooks = copy.deepcopy(self.refs)
        mixed_hooks[-1]["manifests"][0]["provenance"]["capture_hooks_sha256"] = {"sitecustomize.py": "different-hook"}
        with self.assertRaisesRegex(ValueError, "mix capture hook hashes"):
            validate_runs(mixed_hooks, [candidate])

        candidate["manifests"][0]["provenance"]["capture_hooks_sha256"] = {"collector.py": "candidate-hook"}
        validate_runs(self.refs, [candidate, copy.deepcopy(candidate)])


if __name__ == "__main__":
    unittest.main()
