# SPDX-License-Identifier: AGPL-3.0-only
"""CPU only: evaluate the launcher's validation prefix, never external actions."""
import subprocess
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "start-glm53-ep2.sh").read_text()
PREFIX = SOURCE.split('MODEL_MAX_SEQ_LEN=', 1)[0]

class RepairLauncherTests(unittest.TestCase):
    def run_prefix(self, **changes):
        env = dict(PATH="/usr/bin:/bin", GLM_MTP_REPAIR="1", SPECULATIVE="1",
                   TP_SIZE="2", MAX_BATCH_SIZE="1", MAX_NUM_SEQS="1", NUM_DRAFTS="4",
                   MTP_GATE_FORCE="1", MTP_SPEC_THINK="1", MTP_SINGLE_DEPTH_ADAPT="0",
                   GLM_MTP_DISTRIBUTED="1", MTP_PREFILL_ONLY="1", SWAP_SPACE_GB="0")
        env.update(changes)
        return subprocess.run(["/bin/bash", "-c", PREFIX], env=env,
                              capture_output=True, text=True, timeout=5)

    def test_valid_and_default_off(self):
        self.assertEqual(self.run_prefix().returncode, 0)
        self.assertEqual(self.run_prefix(GLM_MTP_REPAIR="0", MTP_PREFILL_ONLY="0").returncode, 0)

    def test_malformed_or_noncontinuous_rejected(self):
        for field, value in [("GLM_MTP_REPAIR", "true"), ("NUM_DRAFTS", "3"),
                             ("MTP_GATE_FORCE", "0"), ("MTP_PREFILL_ONLY", "0"),
                             ("MAX_NUM_SEQS", "2"), ("MTP_SINGLE_DEPTH_ADAPT", "1"),
                             ("GLM_MTP_DISTRIBUTED", "0"), ("SPECULATIVE", "0"),
                             ("SWAP_SPACE_GB", "3")]:
            with self.subTest(field=field):
                self.assertNotEqual(self.run_prefix(**{field: value}).returncode, 0)

    def test_both_ranks_receive_checked_switches(self):
        for field in ["GLM_MTP_REPAIR", "GLM_MTP_KV_REPAIR_VERIFY",
                      "MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE",
                      "GLM_MOE_GATE_UP_M16", "GLM_MOE_GATE_UP_M16_VERIFY"]:
            self.assertEqual(SOURCE.count(f"-e ATLAS_{field}="), 2)

if __name__ == "__main__":
    unittest.main()
