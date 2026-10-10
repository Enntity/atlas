# SPDX-License-Identifier: AGPL-3.0-only
"""CPU acceptance oracle for the EXL3 reference-policy KDA core.

Wraps the validated NumPy emulation of the vLLM/FLA KDA reference policy
(``emulate_fla_ext.prefill`` / ``emulate_fla_ext.decode``) into reusable
functions and asserts the recorded bit-exact results against the captured
reference tensors. The CUDA kernel work uses these tests as its runnable
acceptance oracle; see docs/design/exl3-kda-reference-core.md.

Everything here is CPU-only (pure NumPy). The emulation scripts and fixtures
live outside the repository at

    <root>/arithmetic-audit/kda-core-divergence/
      emulate_fla_chunk.py, emulate_fla_ext.py, atlas-l0-kda-params.json,
      fla-emulation-ext.json, raw/ffn-L0-rank{0,1}-{000016,000324,000325}-*

with <root> defaulting to /home/colum/Workspace/atlas-exl3-validation-20261010
(the scripts hard-code that root). When the directory or a fixture is absent
the tests skip with a clear message instead of failing.

Expected results (evidence: fla-emulation-ext.json, findings.md):
  M5 prefill (inv 000324), both ranks:     0 / 0 mismatches vs reference
  decode (inv 000325), both ranks:         0 / 0 (f32 state, f32 l2norm)
  M128 prefill (inv 000016), both ranks:   1728 / 585 of 524288 documented
      residual (not a tolerance: the exact counts are asserted)
"""
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path('/home/colum/Workspace/atlas-exl3-validation-20261010')
DIVERGENCE_SUBDIR = 'arithmetic-audit/kda-core-divergence'
RANKS = (0, 1)
M5_INVOCATION, DECODE_INVOCATION, M128_INVOCATION = 324, 325, 16
M5_TOKENS, DECODE_TOKENS, M128_TOKENS, M128_CHUNK = 5, 1, 128, 64
HL, DIM = 32, 128
# Frozen evidence from fla-emulation-ext.json (M128 default policy).
M128_EXPECTED = {0: {'total': 1728, 'chunk1': 745, 'chunk2': 983},
                 1: {'total': 585, 'chunk1': 190, 'chunk2': 395}}


def divergence_dir() -> Path:
    override = os.environ.get('ATLAS_KDA_DIVERGENCE_DIR')
    return Path(override) if override else DEFAULT_ROOT / DIVERGENCE_SUBDIR


def require(root: Path, names) -> None:
    missing = [n for n in names if not (root / n).is_file()]
    if missing:
        raise unittest.SkipTest(
            f'KDA divergence artefacts unavailable under {root}: missing '
            f'{", ".join(sorted(missing))}; restore the validation artefact '
            f'directory or point ATLAS_KDA_DIVERGENCE_DIR at a copy that '
            f'includes emulate_fla_ext.py and raw/')


_EMULATION = None


def emulation():
    """Import emulate_fla_ext once; skip cleanly when it is unavailable."""
    global _EMULATION
    if _EMULATION is not None:
        return _EMULATION
    root = divergence_dir()
    require(root, ['emulate_fla_chunk.py', 'emulate_fla_ext.py',
                   'atlas-l0-kda-params.json'])
    name = 'emulate_fla_ext'
    if name in sys.modules:
        _EMULATION = sys.modules[name]
        return _EMULATION
    try:
        spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except (OSError, ImportError) as error:
        sys.modules.pop(name, None)
        raise unittest.SkipTest(
            f'cannot load the KDA reference emulation from {root}: {error}')
    _EMULATION = module
    return module


def load_case(rank: int, invocation: int, tokens: int):
    """Captured reference inputs and output for one (rank, invocation)."""
    m = emulation()
    root = divergence_dir() / 'raw'
    stem = f'ffn-L0-rank{rank}-{invocation:06d}-kda-core'
    require(root, [f'{stem}-{s}.bin' for s in
                    ('input-qkv_proj_states', 'input-g1', 'input-beta', 'output')])
    return m.load_case(rank, invocation, tokens)


def prefill_reference(m, rank: int, case):
    """Reference-policy prefill over raw inputs; returns (output, final_state)."""
    x, g1, beta, _ = case
    return m.prefill(x, g1, beta, rank, dict(m.DEFAULT))


def decode_reference(m, rank: int, m5_case, decode_case):
    """One-row fused_recurrent decode after the M5 prefill.

    Initial state = the emulated M5 prefill final state kept in f32 (a BF16
    state is a documented mismatch, 725/752); conv history = the last three
    raw qkv rows of the M5 prefill.
    """
    x5 = m5_case[0]
    _, final_state = prefill_reference(m, rank, m5_case)
    xd, gd, beta_d, _ = decode_case
    return m.decode(xd, gd, beta_d, x5[-3:], final_state, rank, dict(m.DEFAULT))


def mismatch_count(m, output, reference) -> int:
    """BF16-bit mismatches between a pre-store f32 output and captured BF16."""
    return m.mism(output, reference)


class Tests(unittest.TestCase):
    def test_m5_prefill_matches_reference_bit_exact_both_ranks(self):
        m = emulation()
        for rank in RANKS:
            case = load_case(rank, M5_INVOCATION, M5_TOKENS)
            output, _ = prefill_reference(m, rank, case)
            self.assertEqual(
                mismatch_count(m, output, case[3]), 0,
                f'M5 prefill rank{rank} must be bit-exact vs the captured '
                f'reference (single chunk, zero state)')

    def test_decode_after_m5_prefill_matches_reference_bit_exact_both_ranks(self):
        m = emulation()
        for rank in RANKS:
            m5 = load_case(rank, M5_INVOCATION, M5_TOKENS)
            step = load_case(rank, DECODE_INVOCATION, DECODE_TOKENS)
            output = decode_reference(m, rank, m5, step)
            self.assertEqual(
                mismatch_count(m, output, step[3]), 0,
                f'decode rank{rank} must be bit-exact vs the captured '
                f'reference (f32 state, f32 l2norm)')

    def test_m128_matches_recorded_residual_both_ranks(self):
        m = emulation()
        for rank in RANKS:
            case = load_case(rank, M128_INVOCATION, M128_TOKENS)
            output, _ = prefill_reference(m, rank, case)
            flips = m.to_bf16(output).view(np.uint32) != case[3].view(np.uint32)
            expected = M128_EXPECTED[rank]
            self.assertEqual(int(flips.sum()), expected['total'],
                             f'M128 rank{rank} total residual must equal the '
                             f'frozen evidence (not a tolerance)')
            self.assertEqual(int(flips[:M128_CHUNK].sum()), expected['chunk1'],
                             f'M128 rank{rank} chunk-1 residual')
            self.assertEqual(int(flips[M128_CHUNK:].sum()), expected['chunk2'],
                             f'M128 rank{rank} chunk-2 residual')

    def test_recorded_report_states_frozen_expectations(self):
        report = divergence_dir() / 'fla-emulation-ext.json'
        if not report.is_file():
            self.skipTest(f'{report} absent; evidence cross-check skipped')
        data = json.loads(report.read_text())
        for rank in RANKS:
            key = f'rank{rank}'
            self.assertEqual(data['M5_sanity_000324'][key]['vs_ref'], 0)
            decode = data['decode_000325'][key]
            self.assertEqual(decode['state_f32_silu_f64_acc_f32']['vs_ref'], 0)
            self.assertNotEqual(decode['state_bf16_silu_f64_acc_f32']['vs_ref'], 0,
                                'a BF16 state must remain a documented mismatch')
            default = data['M128_000016'][key]['default']
            self.assertEqual(default['vs_ref'], M128_EXPECTED[rank]['total'])
            self.assertEqual(default['vs_ref_chunk1'], M128_EXPECTED[rank]['chunk1'])
            self.assertEqual(default['vs_ref_chunk2'], M128_EXPECTED[rank]['chunk2'])


if __name__ == '__main__':
    unittest.main()
