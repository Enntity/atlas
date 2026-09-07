# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate only the launcher's pure validation prefix; never contact nodes."""
import subprocess
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "start-glm53-ep2.sh").read_text()
PREFIX = SOURCE.split("MODEL_MAX_SEQ_LEN=", 1)[0]
FLAGS = ["GLM_M5_ROUTER_BN4", "GLM_M5_ROUTER_BN4_VERIFY",
         "GLM_M5_SHARED_M16", "GLM_M5_SHARED_M16_VERIFY", "GLM_MTP_K5_LEDGER"]


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
            self.assertIn(f'{flag}="${{{flag}:-0}}"', PREFIX)


if __name__ == "__main__":
    unittest.main()
