# SPDX-License-Identifier: AGPL-3.0-only
from collections import namedtuple
import unittest
from reference_norm_launch_binding import install_launch_binding


def source_function(): pass


class JIT:
    fn = staticmethod(source_function)
    def __call__(self): pass


class Heuristics:
    def __init__(self, result): self.fn = JIT(); self.result = result; self.calls = []
    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs)); return self.result


class Compiled:
    name = 'actual_kernel'; hash = 'actual_compiler_hash'
    metadata = namedtuple('Metadata', 'enable_fp_fusion num_warps')(True, 8)
    asm = {'ptx': '// actual ptx', 'cubin': b'actual cubin'}


KWARGS = dict(eps=1e-5, T=640, H=32, g_stride_n=4096, D=128, BD=128,
              BT=16, ACTIVATION='sigmoid', IS_RMS_NORM=True, num_warps=8,
              launch_pdl=False, grid=(40,))


class LaunchBindingTests(unittest.TestCase):
    def install(self, kernel, ctx):
        blobs = []; metadata = []
        restore = install_launch_binding(kernel, context=lambda: ctx,
            write_blob=lambda *args: blobs.append(args),
            write_metadata=lambda *args: metadata.append(args),
            read_source=lambda path: b'actual source bytes')
        return restore, blobs, metadata

    def test_unmarked_bypasses_every_capture_requirement(self):
        kernel = Heuristics(None)
        restore, blobs, metadata = self.install(kernel, None)
        sentinel = object()
        self.assertIsNone(kernel.run(sentinel, opaque=sentinel))
        self.assertIs(kernel.calls[0][0][0], sentinel)
        self.assertEqual(blobs, []); self.assertEqual(metadata, [])
        restore()

    def test_actual_launch_result_artifacts_metadata_and_counter_bound(self):
        result = Compiled(); kernel = Heuristics(result)
        original = kernel.run
        restore, blobs, metadata = self.install(kernel, (0, 1, 12))
        sentinel = object()
        for _ in range(5): self.assertIs(kernel.run(sentinel, **KWARGS), result)
        self.assertEqual(len(blobs), 10)
        self.assertEqual([item[0] for item in metadata],
                         ['kda-norm-launch-%03d' % i for i in range(5)])
        captured = metadata[0][1]
        self.assertTrue(captured['compiled_metadata']['enable_fp_fusion'])
        self.assertEqual(captured['launch']['eps'], 1e-5)
        self.assertEqual(captured['name'], result.name)
        self.assertEqual(captured['compiler_hash'], result.hash)
        self.assertEqual(blobs[0][1], b'// actual ptx')
        self.assertEqual(blobs[1][1], b'actual cubin')
        self.assertIs(kernel.calls[0][0][0], sentinel)
        self.assertEqual(kernel.calls[0][1], KWARGS)
        restore(); self.assertEqual(kernel.run, original)

    def test_installed_contiguous_gate_launch_records_absent_optional_fields(self):
        kernel = Heuristics(Compiled())
        _, _, metadata = self.install(kernel, (0, 0, 0))
        kwargs = {k: v for k, v in KWARGS.items() if k not in ('H', 'g_stride_n', 'launch_pdl')}
        kwargs['num_warps'] = 4
        kernel.run(**kwargs)
        value = metadata[0][1]
        self.assertEqual(value['launch']['num_warps'], 4)
        self.assertEqual(value['absent_launch_options'], ['H', 'g_stride_n', 'launch_pdl'])
        for name in value['absent_launch_options']:
            self.assertNotIn(name, value['launch'])

    def test_marked_missing_compiled_metadata_or_artifact_fails_closed(self):
        for result in (None, object(), type('NoFusion', (),
                       {'metadata': {}, 'asm': Compiled.asm})(),
                       type('NoCubin', (), {'metadata': Compiled.metadata,
                            'name': 'kernel', 'hash': 'hash', 'asm': {'ptx': 'ptx'}})()):
            with self.subTest(result=result):
                kernel = Heuristics(result)
                self.install(kernel, (0, 0, 0))
                with self.assertRaises(ValueError): kernel.run(**KWARGS)

    def test_missing_launch_argument_and_duplicate_install_rejected(self):
        kernel = Heuristics(Compiled()); self.install(kernel, (0, 0, 0))
        with self.assertRaises(ValueError): kernel.run(eps=1e-5)
        with self.assertRaises(ValueError): self.install(kernel, (0, 0, 0))

if __name__ == '__main__': unittest.main()
