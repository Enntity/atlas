# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only template transformation checks; no generator, HTTP or node calls."""
import copy
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest

import materialize

HERE = Path(__file__).resolve().parent


def supplied(value):
    if isinstance(value, dict):
        return {k: supplied(v) for k, v in value.items()}
    if isinstance(value, list):
        return [supplied(v) for v in value]
    if isinstance(value, str) and "REQUIRED_" in value:
        if value.endswith("SHA256"):
            return "a" * 64
        return value.replace("REQUIRED_", "operator_")
    return value


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = supplied(json.loads((HERE / "recipe-input.example.json").read_text()))
        self.launch = supplied(json.loads((HERE / "launch.example.json").read_text()))

    def test_only_local_paths_and_hashes_are_derived(self):
        before = copy.deepcopy(self.recipe)
        launch_before = copy.deepcopy(self.launch)
        recipe, launch = materialize.inputs(self.recipe, self.launch, Path("/private/run"),
                                           Path("/checkout/workload.py"), b"pinned bytes")
        self.assertEqual(recipe["ranks"], before["ranks"])
        self.assertEqual(recipe["output_directory"], "/private/run/recipes")
        self.assertEqual(launch["policy"], launch_before["policy"])
        self.assertEqual(launch["controller"], launch_before["controller"])
        self.assertEqual(launch["workload"]["input_file"], "/private/run/workload.input")
        self.assertEqual(launch["workload"]["input_sha256"], hashlib.sha256(b"pinned bytes").hexdigest())
        self.assertEqual(self.recipe, before)
        self.assertEqual(self.launch, launch_before)

    def test_unresolved_operator_field_refused(self):
        self.recipe["ranks"][1]["image_sha256"] = "REQUIRED_IMAGE_SHA256"
        with self.assertRaises(ValueError):
            materialize.inputs(self.recipe, self.launch, Path("/private/run"), Path("/workload.py"), b"x")

    def test_private_workload_bytes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "checkout.py"
            source.write_bytes(b"exact workload\n\x00")
            source.chmod(0o664)
            target = Path(directory) / "workload.input"
            data = materialize.read(source, 65536)
            materialize.save_bytes(target, data)
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            self.assertEqual(target.stat().st_nlink, 1)
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o664)
            with self.assertRaises(FileExistsError):
                materialize.save_bytes(target, b"replacement")
            self.assertEqual(target.read_bytes(), data)

    def test_capacity_cli_mismatch_refused(self):
        self.recipe["ranks"][1]["argv"].remove("--max-num-seqs=8")
        self.recipe["ranks"][1]["argv"].append("--max-num-seqs=4")
        with self.assertRaises(ValueError):
            materialize.inputs(self.recipe, self.launch, Path("/private/run"), Path("/workload.py"), b"x")


if __name__ == "__main__":
    unittest.main()
