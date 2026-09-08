# SPDX-License-Identifier: AGPL-3.0-only
"""CPU source closure and PTX-checker tests; never numerical CUDA evidence."""
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import check_glm_moe_btile_ptx as checker_module
from check_glm_moe_btile_ptx import check_ptx

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "kernels/gb10/deepseek-v4-flash/nvfp4"
SCRIPTS = ROOT / "scripts/dev"
# Frozen pre-promotion bodies, excluding only full-line comments/blank lines.
BODIES = {
    "glm_moe_btile.cuh": "25ba037dbcf1a400d502e267d584adfd2598dc28d4df3669e71ddd120a0cf2d2",
    "glm_moe_btile_m64.cuh": "54649d6279c2c1734c97ccbddaa638497490b855583045af76cd0202807762eb",
    "glm_moe_btile_m64_bounds.h": "b80d66724243797e34fc8f15dbc567789b8d3faad22ea88691c401417585104f",
    "glm_moe_btile_decode_register.cuh": "0a72b942571490505d0af9a630ba68f8f64de52de3904d1992693770084a434e",
    "glm_moe_btile_native_repack.cuh": "17b51248e039e3fe50bd8eca05c8be49ee7d95b7f3f0e74f26165648fad6211e",
    "glm_moe_btile_native_layout.h": "fdfe7afe7cbdf8d5fbfb009dbb91b77bdbc4d9ce154535bb382715ee816e144a",
}


def body(source):
    return "".join(line for line in source.splitlines(keepends=True)
                   if line.strip() and not line.lstrip().startswith("//"))


def fixture_ptx():
    # Independently spelled ABI signatures, not imported checker constants.
    pointer = "u64"
    scalar = [pointer] * 10 + [pointer, pointer, "f32", pointer] * 2 + ["u32"] * 3
    dense = [pointer] * 8 + ["u32"] * 3
    compact = dense + [pointer, pointer, "u32"]
    fused = [pointer] * 12 + ["u32"] * 3 + [pointer, pointer, "u32"]
    specs = {
        "grouped": [("glm_moe_gate_up_btile" + suffix, fused)
                    for suffix in ("", "_vecscale", "_m64", "_m64_vecscale")]
                   + [("glm_moe_btile_m64_" + suffix, types)
                      for suffix, types in (("dense", dense), ("vecscale_dense", dense),
                                            ("compact", compact), ("vecscale_compact", compact))],
        "decode": [(f"glm_btile_decode_{variant}{rows}", scalar)
                   for variant in ("word", "vec") for rows in (1, 2, 3)],
        "repack": [("glm_native_to_btile_u8", [pointer, pointer, "u32", "u32"])],
    }
    return {module: ".version 9.0\n.target sm_121a\n.address_size 64\n" + "\n".join(
        ".visible .entry " + name + "(\n" + ",\n".join(
            f".param .{kind} p{index}" for index, kind in enumerate(types)) + "\n) { ret; }"
        for name, types in entries) for module, entries in specs.items()}


class SourceClosureTests(unittest.TestCase):
    def test_promoted_helpers_are_unchanged_single_sources_with_shims(self):
        for name, expected in BODIES.items():
            with self.subTest(name=name):
                destination = PRODUCTION / name
                self.assertTrue(destination.is_file(), f"missing production helper {name}")
                self.assertEqual(hashlib.sha256(body(destination.read_text()).encode()).hexdigest(), expected)
                self.assertEqual((SCRIPTS / name).read_text().splitlines(), [
                    "// SPDX-License-Identifier: AGPL-3.0-only",
                    '#include "../../kernels/gb10/deepseek-v4-flash/nvfp4/' + name + '"'])

    def test_actual_translation_units_register_all_promoted_families(self):
        grouped = (PRODUCTION / "moe_w4a16_grouped_gemm.cu").read_text()
        for header in ("glm_moe_btile.cuh", "glm_moe_btile_m64.cuh"):
            self.assertEqual(grouped.count('#include "' + header + '"'), 1)
        for name, headers in {
            "glm_moe_btile_decode.cu": ["../../common/moe_shared_expert_fused_t.cu",
                                        "glm_moe_btile_decode_register.cuh"],
            "glm_moe_btile_native_repack.cu": ["glm_moe_btile_native_repack.cuh"],
        }.items():
            path = PRODUCTION / name
            self.assertTrue(path.is_file(), f"missing production TU {name}")
            self.assertEqual(re.findall(r'^#include "([^"]+)"$', path.read_text(), re.M), headers)
        m64 = (PRODUCTION / "glm_moe_btile_m64_bounds.h").read_text()
        self.assertIn("glm_btile_m64_max_rows = 1088", m64)

    def test_target_inheritance_fmad_and_common_source_unchanged(self):
        for name, expected in {
            "kernels/gb10/common/moe_shared_expert_fused_t.cu":
                "6dbd77b11400302b8fcbed0d64c5b3cf2d02ab62874116d8a260f43868a62513",
            "kernels/gb10/deepseek-v4-flash/nvfp4/KERNEL.toml":
                "057917bfc97b11468f80f0e81a159795e4383c10a7895c9c1b989385e1c8d899",
            "kernels/gb10/glm-5.3-flash-nvfp4/MODEL.toml":
                "354bd187a0c0d8e8b0bbc1699b9a1727f785b099b322a6f74c6c132c9ef9f8a1",
        }.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)


class PtxCheckerTests(unittest.TestCase):
    def test_complete_three_module_signature_evidence(self):
        result = check_ptx(fixture_ptx())
        self.assertEqual(set(result), {"grouped", "decode", "repack"})
        self.assertEqual([len(result[name]) for name in ("grouped", "decode", "repack")], [8, 6, 1])
        self.assertEqual(result["repack"]["glm_native_to_btile_u8"], ["u64", "u64", "u32", "u32"])

    def test_missing_empty_duplicate_and_wrong_module_reject(self):
        for fault in range(5):
            texts = fixture_ptx()
            if fault == 0:
                del texts["decode"]
            elif fault == 1:
                texts["decode"] = ""
            elif fault == 2:
                texts["grouped"] = texts["grouped"].replace("glm_moe_gate_up_btile(", "missing(")
            elif fault == 3:
                texts["repack"] += "\n.visible .entry glm_native_to_btile_u8() {}"
            else:
                texts["repack"] = texts["decode"]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                check_ptx(texts)

    def test_type_width_arity_target_and_commented_exports_reject(self):
        for old, new in ((".u64 p0", ".u32 p0"), (".u32 p3", ".f32 p3"),
                         (",\n.param .u32 p3", ""), ("sm_121a", "sm_121"),
                         (".address_size 64", ".address_size 32"),
                         (".visible .entry", "// .visible .entry")):
            texts = fixture_ptx()
            texts["repack"] = texts["repack"].replace(old, new)
            with self.subTest(new=new), self.assertRaises(ValueError):
                check_ptx(texts)

    def test_cli_requires_real_explicit_files_and_reports_success(self):
        checker = SCRIPTS / "check_glm_moe_btile_ptx.py"
        with tempfile.TemporaryDirectory() as directory:
            paths = {key: Path(directory) / (key + ".ptx") for key in fixture_ptx()}
            command = [sys.executable, str(checker)]
            for key, path in paths.items():
                command += ["--" + key, str(path)]
            missing = subprocess.run(command, capture_output=True, text=True, timeout=5)
            self.assertEqual(missing.returncode, 2)
            for key, source in fixture_ptx().items():
                paths[key].write_text(source)
            complete = subprocess.run(command, capture_output=True, text=True, timeout=5)
            self.assertEqual(complete.returncode, 0, complete.stderr)
            paths["decode"].write_text("// no exported kernels\n")
            empty = subprocess.run(command, capture_output=True, text=True, timeout=5)
            self.assertEqual(empty.returncode, 2)

    def test_export_declarations_without_visible_definitions_reject(self):
        for old, new in ((".visible .entry", ".extern .entry"),
                         (".visible .entry", ".entry"), ("{ ret; }", ";"),
                         (".u64 p1", ".u64 p0")):
            texts = fixture_ptx()
            texts["repack"] = texts["repack"].replace(old, new)
            with self.subTest(new=new), self.assertRaises(ValueError):
                check_ptx(texts)

    def test_bounded_input_and_normal_launch_directives(self):
        with patch.object(checker_module, "MAX_PTX_BYTES", 32):
            with self.assertRaises(ValueError):
                check_ptx(fixture_ptx())
        texts = fixture_ptx()
        texts["repack"] = texts["repack"].replace(") {", ")\n.maxntid 256, 1, 1\n{")
        self.assertEqual(len(check_ptx(texts)["repack"]), 1)

    def test_target_line_whitespace_and_compiler_comment_are_valid(self):
        for target in (".target sm_121a \t", ".target sm_121a // exact target"):
            texts = fixture_ptx()
            texts["repack"] = texts["repack"].replace(".target sm_121a", target)
            self.assertEqual(len(check_ptx(texts)["repack"]), 1)


if __name__ == "__main__":
    unittest.main()
