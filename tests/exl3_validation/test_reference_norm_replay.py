# SPDX-License-Identifier: AGPL-3.0-only
import math
import unittest

from reference_norm_replay import replay_reference_norm


class Tensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(values),)
        self.dtype = 'f32-fixture'

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values)

    def zero_(self):
        self.values[:] = [0] * len(self.values)
        return self

    def equal(self, other):
        return self.values == other.values


class Norm:
    def __init__(self):
        self.eps = 0.125
        self._parameters = {'weight': Tensor([2, 3]), 'bias': None}

    @property
    def weight(self):
        return self._parameters['weight']

    @property
    def bias(self):
        return self._parameters['bias']

    def forward_cuda(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        scale = math.sqrt(sum(v*v for v in x.values) / len(x.values) + self.eps)
        x.values[:] = [v / scale * w / (1 + math.exp(-gate))
                       for v, w, gate in zip(x.values, self.weight.values, g.values)]
        return x


class Tests(unittest.TestCase):
    def run_replay(self, norm, *, captured=None, flags=None):
        snapshots = {'input-x': Tensor([1, 2]), 'input-g': Tensor([3, 4]),
                     'weight': norm.weight.clone()}
        if captured is None:
            captured = norm.forward_cuda(snapshots['input-x'].clone(), snapshots['input-g'].clone())
        writes, receipts = [], []
        replay_reference_norm(norm, norm.forward_cuda, snapshots, captured,
                              {'eps': norm.eps, 'bias_present': False,
                               'elementwise_affine': True, 'call':
                               {'prenorm': False, 'residual_in_fp32': False,
                                'residual_present': False} if flags is None else flags},
                              context=(0, 1, 7),
                              write_tensor=lambda *args: writes.append(args),
                              write_metadata=lambda *args: receipts.append(args))
        return snapshots, writes, receipts

    def test_self_reproduction_and_operand_controls_preserve_resident_state(self):
        norm = Norm()
        original_weight, parameters = norm.weight, norm._parameters
        snapshots, writes, receipts = self.run_replay(norm)
        self.assertIs(norm._parameters, parameters)
        self.assertIs(norm.weight, original_weight)
        self.assertEqual(norm.weight.values, [2, 3])
        self.assertEqual(snapshots['input-x'].values, [1, 2])
        receipt = receipts[0][1]
        self.assertTrue(receipt['self_reproduction_exact'])
        self.assertEqual(receipt['controls'], {'input-zero': True, 'gate-zero': True,
                                               'weight-zero': True})
        self.assertEqual(len(writes), 4)
        self.assertEqual(receipts[0][2], (0, 1, 7))

    def test_failed_self_reproduction_is_retained_and_stops_controls(self):
        norm = Norm()
        snapshots, writes, receipts = self.run_replay(norm, captured=Tensor([99, 99]))
        self.assertFalse(receipts[0][1]['self_reproduction_exact'])
        self.assertEqual(receipts[0][1]['controls'], {})
        self.assertEqual(len(writes), 1)

    def test_residual_prenorm_are_explicitly_unsupported(self):
        for field in ('prenorm', 'residual_present', 'residual_in_fp32'):
            flags = {'prenorm': False, 'residual_present': False, 'residual_in_fp32': False}
            flags[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_replay(Norm(), flags=flags)

    def test_capture_constant_clones_bind_replay_after_resident_change(self):
        norm = Norm()
        snapshots = {'input-x': Tensor([1, 2]), 'input-g': Tensor([3, 4]),
                     'weight': norm.weight.clone()}
        captured = norm.forward_cuda(snapshots['input-x'].clone(), snapshots['input-g'].clone())
        norm.weight.values[:] = [8, 9]
        receipts = []
        replay_reference_norm(norm, norm.forward_cuda, snapshots, captured,
                              {'eps': norm.eps, 'bias_present': False, 'elementwise_affine': True,
                               'call': {'prenorm': False, 'residual_in_fp32': False, 'residual_present': False}},
                              context=(0, 0, 0), write_tensor=lambda *args: None,
                              write_metadata=lambda *args: receipts.append(args))
        self.assertTrue(receipts[0][1]['self_reproduction_exact'])
        self.assertEqual(norm.weight.values, [8, 9])


if __name__ == '__main__':
    unittest.main()
