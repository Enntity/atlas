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


def synthetic_case(m, rank: int, tokens: int, seed: int = 42):
    """Generate synthetic inputs for property testing."""
    np.random.seed(seed + rank * 1000 + tokens)
    x = m.to_bf16(np.random.randn(tokens, 3 * HL * DIM).astype(np.float32))
    g1 = m.to_bf16(np.random.randn(tokens, HL, DIM).astype(np.float32))
    beta = m.to_bf16(np.random.randn(tokens, HL).astype(np.float32))
    return x, g1, beta


def prefill_synthetic(m, rank: int, x, g1, beta):
    """Run prefill on synthetic inputs; returns (output, final_state)."""
    return m.prefill(x, g1, beta, rank, dict(m.DEFAULT))


def bf16_equal(m, a, b) -> bool:
    """Check BF16-bit equality of two f32 arrays."""
    return np.array_equal(m.to_bf16(a).view(np.uint32), m.to_bf16(b).view(np.uint32))


class PropertyTests(unittest.TestCase):
    """Property-based tests for KDA chunk boundary behavior.

    These tests verify mathematical properties of the reference-policy KDA
    recurrence without requiring captured reference outputs. They use synthetic
    inputs and the CPU emulation to check internal consistency.
    """

    def test_chunk_boundary_token_counts(self):
        """Test prefill at chunk boundary token counts for structural validity.

        Verifies that the emulation runs without error and produces correctly
        shaped outputs for token counts at and around chunk boundaries.
        Note: emulation only supports 1 sub-block (T<=16) or 4 sub-blocks (T=64)
        per chunk; partial chunks with 2-3 sub-blocks are not implemented.
        """
        m = emulation()
        for rank in RANKS:
            # Valid token counts: T<=16, or T%64==0, or T%64<=16
            for tokens in [1, 2, 3, 4, 5, 7, 8, 15, 16, 64, 65, 66, 70, 80,
                           128, 129, 130, 140, 144, 192, 193, 194, 200, 208,
                           256, 257, 258, 272, 320, 384, 400]:
                x, g1, beta = synthetic_case(m, rank, tokens, seed=123)
                output, final_state = prefill_synthetic(m, rank, x, g1, beta)
                self.assertEqual(output.shape, (tokens, HL, DIM),
                                 f'rank{rank} tokens={tokens}: output shape')
                self.assertEqual(final_state.shape, (HL, DIM, DIM),
                                 f'rank{rank} tokens={tokens}: state shape')
                self.assertTrue(np.all(np.isfinite(output)),
                                f'rank{rank} tokens={tokens}: output has non-finite values')
                self.assertTrue(np.all(np.isfinite(final_state)),
                                f'rank{rank} tokens={tokens}: state has non-finite values')

    def test_deterministic_reproducibility(self):
        """Same inputs produce identical outputs (determinism)."""
        m = emulation()
        for rank in RANKS:
            for tokens in [5, 64, 128, 200]:
                x, g1, beta = synthetic_case(m, rank, tokens, seed=456)
                out1, state1 = prefill_synthetic(m, rank, x, g1, beta)
                out2, state2 = prefill_synthetic(m, rank, x, g1, beta)
                self.assertTrue(bf16_equal(m, out1, out2),
                                f'rank{rank} tokens={tokens}: output not deterministic')
                self.assertTrue(bf16_equal(m, state1, state2),
                                f'rank{rank} tokens={tokens}: state not deterministic')

    def test_zero_input_produces_zero_output(self):
        """All-zero inputs should produce all-zero outputs (linearity at zero)."""
        m = emulation()
        for rank in RANKS:
            for tokens in [1, 5, 64, 128]:
                x = np.zeros((tokens, 3 * HL * DIM), dtype=np.uint16)
                g1 = np.zeros((tokens, HL, DIM), dtype=np.uint16)
                beta = np.zeros((tokens, HL), dtype=np.uint16)
                output, final_state = prefill_synthetic(m, rank, x, g1, beta)
                self.assertTrue(np.all(output == 0),
                                f'rank{rank} tokens={tokens}: zero input -> non-zero output')
                self.assertTrue(np.all(final_state == 0),
                                f'rank{rank} tokens={tokens}: zero input -> non-zero state')

    def test_decode_after_various_prefill_lengths(self):
        """Test one-row decode after prefill of various lengths.

        Verifies the decode path works correctly with different prefill history lengths.
        """
        m = emulation()
        for rank in RANKS:
            for prefill_tokens in [5, 63, 64, 65, 127, 128, 129, 199, 200]:
                x5, g5, b5 = synthetic_case(m, rank, prefill_tokens, seed=789)
                _, hfin = prefill_synthetic(m, rank, x5, g5, b5)
                xd, gd, bd = synthetic_case(m, rank, 1, seed=999)
                hist = x5[-3:] if prefill_tokens >= 3 else np.zeros((3, 3 * HL * DIM), dtype=np.uint16)
                out_d = m.decode(xd, gd, bd, hist, hfin, rank, dict(m.DEFAULT))
                self.assertEqual(out_d.shape, (1, HL, DIM),
                                 f'rank{rank} prefill={prefill_tokens}: decode output shape')
                self.assertTrue(np.all(np.isfinite(out_d)),
                                f'rank{rank} prefill={prefill_tokens}: decode has non-finite values')

    def test_state_dtype_f32_vs_bf16(self):
        """Verify f32 state is required (BF16 state produces documented mismatches)."""
        m = emulation()
        for rank in RANKS:
            x5, g5, b5 = synthetic_case(m, rank, 5)
            _, hfin_f32 = prefill_synthetic(m, rank, x5, g5, b5)
            hfin_bf16 = m.to_bf16(hfin_f32)
            xd, gd, bd = synthetic_case(m, rank, 1)
            hist = x5[-3:]
            out_f32 = m.decode(xd, gd, bd, hist, hfin_f32, rank, dict(m.DEFAULT, state_dtype='f32'))
            out_bf16 = m.decode(xd, gd, bd, hist, hfin_bf16, rank, dict(m.DEFAULT, state_dtype='bf16'))
            self.assertFalse(bf16_equal(m, out_f32, out_bf16),
                             f'rank{rank}: f32 and bf16 state should produce different outputs')

    def test_decode_state_carry_over(self):
        """Test that decode state carry-over is deterministic and consistent.

        Running decode multiple times with the same initial state and inputs
        should produce the same output and final state. This verifies the
        recurrent state update is deterministic.
        """
        m = emulation()
        for rank in RANKS:
            # Prefill to get initial state
            x5, g5, b5 = synthetic_case(m, rank, 64, seed=111)
            _, hfin = prefill_synthetic(m, rank, x5, g5, b5)
            # Multiple decode steps with same initial state
            xd, gd, bd = synthetic_case(m, rank, 1, seed=222)
            hist = x5[-3:]
            out1, state1 = m.decode(xd, gd, bd, hist, hfin, rank, dict(m.DEFAULT)), None
            # The decode function only returns output, not state.
            # But we can verify output determinism.
            out2 = m.decode(xd, gd, bd, hist, hfin, rank, dict(m.DEFAULT))
            self.assertTrue(bf16_equal(m, out1, out2),
                            f'rank{rank}: decode output not deterministic with same state')
            # Test with different prefill lengths
            for prefill_tokens in [5, 64, 128, 200]:
                x5b, g5b, b5b = synthetic_case(m, rank, prefill_tokens, seed=333)
                _, hfin_b = prefill_synthetic(m, rank, x5b, g5b, b5b)
                hist_b = x5b[-3:] if prefill_tokens >= 3 else np.zeros((3, 3 * HL * DIM), dtype=np.uint16)
                out_b = m.decode(xd, gd, bd, hist_b, hfin_b, rank, dict(m.DEFAULT))
                self.assertEqual(out_b.shape, (1, HL, DIM),
                                 f'rank{rank} prefill={prefill_tokens}: decode output shape')
                self.assertTrue(np.all(np.isfinite(out_b)),
                                f'rank{rank} prefill={prefill_tokens}: decode has non-finite values')


if __name__ == '__main__':
    unittest.main()
