# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only launcher policy tests; execute only its pure validation prefix."""

import subprocess
import unittest
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "start-glm53-ep2.sh").read_text()
PREFIX = SOURCE.split('if [[ "$MTP_SPEC_THINK"', 1)[0]


class Batch4LauncherTests(unittest.TestCase):
    def launch(self, **changes):
        env = dict(PATH="/usr/bin:/bin", GLM_C4_DECODE="1", TP_SIZE="2",
                   MAX_BATCH_SIZE="4", MAX_NUM_SEQS="4", MAX_SEQ_LEN="2048",
                   GLM_MLA_BATCH4="1", SPECULATIVE="0")
        env.update(changes)
        return subprocess.run(["/bin/bash", "-c", PREFIX], env=env,
                              capture_output=True, text=True, timeout=5)

    def test_valid_c4_and_default_off_legacy(self):
        self.assertEqual(self.launch().returncode, 0)
        self.assertEqual(self.launch(GLM_MLA_BATCH4="0").returncode, 0)
        self.assertEqual(self.launch(GLM_C4_DECODE="0", GLM_MLA_BATCH4="0",
                                     MAX_BATCH_SIZE="3", MAX_NUM_SEQS="3").returncode, 0)

    def test_flag_requires_c4_and_binary_value(self):
        self.assertNotEqual(self.launch(GLM_C4_DECODE="0", MAX_BATCH_SIZE="3",
                                        MAX_NUM_SEQS="3").returncode, 0)
        for flag in ("true", "2", "-1"):
            self.assertNotEqual(self.launch(GLM_MLA_BATCH4=flag).returncode, 0)

    def test_both_ranks_receive_default_off_flag(self):
        self.assertIn('GLM_MLA_BATCH4="${GLM_MLA_BATCH4:-0}"', SOURCE)
        self.assertEqual(SOURCE.count("-e ATLAS_GLM_MLA_BATCH4="), 2)


if __name__ == "__main__":
    unittest.main()
