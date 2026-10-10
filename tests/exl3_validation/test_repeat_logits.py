# SPDX-License-Identifier: AGPL-3.0-only
import copy
import unittest

import numpy as np
from test_parity import manifest
from repeat_logits import compare_repeats


class RepeatTests(unittest.TestCase):
    def setUp(self):
        self.manifest = manifest("vllm")
        self.rows = {"case": np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.float32)}

    def test_identical_repeats_pass_with_every_pair_checked(self):
        runs = [(copy.deepcopy(self.manifest), self.rows) for _ in range(3)]
        runs[1][0]["provenance"]["utc"] = "later"
        result = compare_repeats(runs)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["pairs"]), 3)
        self.assertTrue(all(p["cases"][0]["bitwise_equal"] for p in result["pairs"]))

    def test_same_argmax_does_not_hide_changed_distribution(self):
        altered = {"case": self.rows["case"].copy()}
        altered["case"][0, 0] = 0.5
        result = compare_repeats([(self.manifest, self.rows), (self.manifest, altered)])
        self.assertFalse(result["passed"])
        case = result["pairs"][0]["cases"][0]
        self.assertTrue(case["argmax_matches"])
        self.assertFalse(case["bitwise_equal"])
        self.assertGreater(case["max_kl"], 0)

    def test_changed_profile_or_inputs_fail_closed(self):
        for location, field, value in (("provenance", "engine", "atlas"),
                                       ("provenance", "environment", {"HOOK": "1"}),
                                       (None, "request_policy", {}),
                                       (None, "tokens_sha256", "bad")):
            changed = copy.deepcopy(self.manifest)
            (changed if location is None else changed[location])[field] = value
            with self.assertRaises(ValueError):
                compare_repeats([(self.manifest, self.rows), (changed, self.rows)])
        with self.assertRaises(ValueError):
            compare_repeats([(self.manifest, self.rows)])
        with self.assertRaises(ValueError):
            compare_repeats([(self.manifest, self.rows), (self.manifest, {"case": np.full((2, 4), np.nan)})])


if __name__ == "__main__":
    unittest.main()
