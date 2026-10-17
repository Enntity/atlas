# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import numpy as np
from native_gated_norm_replay import validate_operands, replay, perturbations, KDA_CONTRACT, device_properties


class GatedNormReplayTests(unittest.TestCase):
    def operands(self):
        return (np.full((2, 32, 128), 0x3f80, dtype='<u2'),
                np.full((2, 4096), 0x3f80, dtype='<u2'),
                np.full((128,), 0x3f80, dtype='<u2'))

    def test_rejects_wrong_geometry_dtype_nonfinite_and_epsilon(self):
        x, g, w = self.operands()
        for operands, eps in (((x[:, :31], g, w), 1e-5),
                              ((x, g.reshape(2, 32, 128), w), 1e-5),
                              ((x, g, w.astype(np.float32)), 1e-5),
                              ((x, g, w), 0), ((x, g, w), float('nan')),
                              ((x, g, w), 1e100)):
            with self.subTest(eps=eps, shape=operands[0].shape), self.assertRaises(ValueError):
                validate_operands(*operands, eps)
        x[0, 0, 0] = 0x7f80
        with self.assertRaises(ValueError): validate_operands(x, g, w, 1e-5)

    def test_launch_abi_and_input_clones(self):
        x, g, w = self.operands()
        snapshots = [v.copy() for v in (x, g, w)]
        launches = []
        class Stream:
            def synchronize(self): pass
        class Cuda:
            def get_current_stream(self): return Stream()
        class Backend:
            uint16 = np.uint16
            cuda = Cuda()
            def array(self, value, **kwargs): return np.array(value, **kwargs)
            def full(self, *args, **kwargs): return np.full(*args, **kwargs)
            def asnumpy(self, value): return value.copy()
        def kernel(grid, block, args):
            launches.append((grid, block, args))
            args[3][:x.size] = 0x3f80
        out = replay(Backend(), kernel, x, g, w, 1e-5)
        self.assertEqual(out.shape, x.shape)
        grid, block, args = launches[0]
        self.assertEqual(grid, (64,)); self.assertEqual(block, (128,))
        self.assertEqual(tuple(args[4:6]), (32, 128))
        self.assertIsInstance(args[6], np.float32)
        for value, snapshot in zip((x, g, w), snapshots): np.testing.assert_array_equal(value, snapshot)

    def test_perturbations_are_fresh_and_change_only_declared_operand(self):
        operands = self.operands()
        controls = perturbations(*operands)
        self.assertEqual(set(controls), {'input', 'gate', 'weight'})
        for index, name in enumerate(('input', 'gate', 'weight')):
            for i, (original, changed) in enumerate(zip(operands, controls[name])):
                self.assertFalse(np.shares_memory(original, changed))
                self.assertEqual(np.array_equal(original, changed), i != index)
        self.assertEqual(KDA_CONTRACT['kda']['argument_counts']['kda_sigmoid_gated_rms_norm'], 7)

    def test_device_properties_preserve_non_utf8_and_ascii_bytes(self):
        import json
        properties = {'uuid': bytes([0xff, 0x00, 0xa1]), 'name': b'NVIDIA GB10',
                      'nested': {'luid': bytes([0x80, 0xff])}, 'dimensions': (1, 2, 3)}
        captured = device_properties(properties)
        decoded = json.loads(json.dumps(captured))
        self.assertEqual(bytes.fromhex(decoded['uuid']['value']), properties['uuid'])
        self.assertEqual(bytes.fromhex(decoded['name']['value']), properties['name'])
        self.assertEqual(decoded['nested']['luid']['encoding'], 'hex')
        self.assertEqual(decoded['dimensions'], [1, 2, 3])

    def test_guard_or_operand_mutation_fails(self):
        x, g, w = self.operands()
        class Stream:
            def synchronize(self): pass
        class Cuda:
            def get_current_stream(self): return Stream()
        class Backend:
            uint16 = np.uint16; cuda = Cuda()
            def array(self, value, **kwargs): return np.array(value, **kwargs)
            def full(self, *args, **kwargs): return np.full(*args, **kwargs)
            def asnumpy(self, value): return value.copy()
        for slot in (0, 1, 2, 3):
            def kernel(grid, block, args): args[slot][-1] = 0
            with self.subTest(slot=slot), self.assertRaises(RuntimeError):
                replay(Backend(), kernel, x, g, w, 1e-5)

if __name__ == '__main__': unittest.main()
