# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate only the launcher's pure validation prefix; never contact nodes."""
import subprocess
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "start-glm53-ep2.sh").read_text()
PREFIX = SOURCE.split("MODEL_MAX_SEQ_LEN=", 1)[0]
FLAGS = ["GLM_M5_ROUTER_BN4", "GLM_M5_ROUTER_BN4_VERIFY",
         "GLM_M5_SHARED_M16", "GLM_M5_SHARED_M16_VERIFY", "GLM_MTP_K5_LEDGER",
         "GLM_TARGET_SHARED_FP8", "GLM_TARGET_SHARED_FP8_VERIFY", "GLM_MTP_HIDDEN_TRACE"]


class M5LauncherTests(unittest.TestCase):
    def prefix(self, **changes):
        env = {"PATH": "/usr/bin:/bin", **changes}
        return subprocess.run(["/bin/bash", "-c", PREFIX], env=env,
                              capture_output=True, text=True, timeout=5)

    def test_default_is_valid_and_flags_are_strict(self):
        self.assertEqual(self.prefix().returncode, 0)
        for flag in FLAGS:
            with self.subTest(flag=flag):
                result = self.prefix(**{flag: "true"})
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"{flag} must be 0 or 1", result.stderr)

    def test_each_oracle_requires_only_its_own_feature(self):
        for feature in ("GLM_M5_ROUTER_BN4", "GLM_M5_SHARED_M16"):
            oracle = feature + "_VERIFY"
            result = self.prefix(**{oracle: "1", "GLM_TP_VERIFY_GRAPH": "0"})
            self.assertEqual(result.returncode, 2)
            self.assertIn(f"{oracle} requires {feature}=1", result.stderr)
            self.assertEqual(self.prefix(**{oracle: "1", feature: "1",
                                           "GLM_TP_VERIFY_GRAPH": "0"}).returncode, 0)

    def test_oracle_launch_explicitly_requires_eager_m5(self):
        for feature in ("GLM_M5_ROUTER_BN4", "GLM_M5_SHARED_M16"):
            result = self.prefix(**{feature: "1", feature + "_VERIFY": "1"})
            self.assertEqual(result.returncode, 2)
            self.assertIn("GLM_TP_VERIFY_GRAPH=0", result.stderr)
            self.assertEqual(self.prefix(**{feature: "1"}).returncode, 0)

    def test_both_ranks_receive_flags_and_defaults_are_off(self):
        for flag in FLAGS:
            self.assertEqual(SOURCE.count(f"-e ATLAS_{flag}="), 2)
            operator = "-" if flag == "GLM_MTP_HIDDEN_TRACE" else ":-"
            self.assertIn(f'{flag}="${{{flag}{operator}0}}"', PREFIX)

    def test_hidden_trace_requires_existing_repair_profile_but_not_target_graph_disable(self):
        for value in ("", " ", "2", "true"):
            self.assertEqual(self.prefix(GLM_MTP_HIDDEN_TRACE=value).returncode, 2)
        result = self.prefix(GLM_MTP_HIDDEN_TRACE="1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("GLM_MTP_HIDDEN_TRACE requires GLM_MTP_REPAIR=1", result.stderr)
        profile = dict(GLM_MTP_HIDDEN_TRACE="1", GLM_MTP_REPAIR="1", SPECULATIVE="1",
                       TP_SIZE="2", MAX_BATCH_SIZE="1", MAX_NUM_SEQS="1", NUM_DRAFTS="4",
                       MTP_GATE_FORCE="1", MTP_SPEC_THINK="1", MTP_SINGLE_DEPTH_ADAPT="0",
                       GLM_MTP_DISTRIBUTED="1", MTP_PREFILL_ONLY="1", SWAP_SPACE_GB="0",
                       GLM_TP_VERIFY_GRAPH="1")
        self.assertEqual(self.prefix(**profile).returncode, 0)
        for field, value in (("NUM_DRAFTS", "3"), ("MAX_NUM_SEQS", "4"),
                             ("GLM_MTP_DISTRIBUTED", "0"), ("MTP_PREFILL_ONLY", "0")):
            self.assertEqual(self.prefix(**{**profile, field: value}).returncode, 2)

    def test_shared_cache_requires_bounded_prefill_and_consuming_path(self):
        self.assertEqual(self.prefix(GLM_TARGET_SHARED_FP8="1").returncode, 0)
        for changes in ({"MAX_PREFILL_TOKENS": "1025"},
                        {"MAX_PREFILL_TOKENS": "0"},
                        {"MAX_PREFILL_TOKENS": "1+1"},
                        {"MAX_PREFILL_TOKENS": "18446744073709551617"},
                        {"MAX_PREFILL_TOKENS": "010"},
                        {"GLM_K5_BATCHED_SHARED": "1"}):
            with self.subTest(changes=changes):
                self.assertEqual(self.prefix(GLM_TARGET_SHARED_FP8="1", **changes).returncode, 2)
        self.assertEqual(self.prefix(GLM_TARGET_SHARED_FP8_VERIFY="1",
                                     GLM_TP_VERIFY_GRAPH="0").returncode, 2)
        self.assertEqual(self.prefix(GLM_TARGET_SHARED_FP8="1",
                                     GLM_TARGET_SHARED_FP8_VERIFY="1").returncode, 2)
        self.assertEqual(self.prefix(GLM_TARGET_SHARED_FP8="1",
                                     GLM_TARGET_SHARED_FP8_VERIFY="1",
                                     GLM_TP_VERIFY_GRAPH="0").returncode, 2)
        self.assertEqual(self.prefix(GLM_TARGET_SHARED_FP8="1",
                                     GLM_TARGET_SHARED_FP8_VERIFY="1",
                                     GLM_TP_VERIFY_GRAPH="0",
                                     MOE_SHARED_REDUCE_OVERLAP="0").returncode, 0)


if __name__ == "__main__":
    unittest.main()
