# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the probe sitecustomize bootstrap (no vLLM, no GPU)."""
import atexit
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

import probe_sitecustomize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MODEL_SOURCE = '''# SPDX-License-Identifier: AGPL-3.0-only
class Glm5NextMoE:
    def __init__(self, prefix=''):
        self.prefix = prefix
'''
RUNNER_SOURCE = '''# SPDX-License-Identifier: AGPL-3.0-only
class GPUModelRunner:
    def execute_model(self, scheduler_output, intermediate_tensors=None,
                      dummy_run=False):
        return None
'''
COMM_SOURCE = '''# SPDX-License-Identifier: AGPL-3.0-only
class CudaCommunicator:
    def __init__(self, unique_name=''):
        self.unique_name = unique_name
'''


class StubLoader(importlib.abc.Loader):
    """Delegate that executes the real module then marks it, proving chaining."""

    def __init__(self, flag, inner=None):
        self.flag = flag
        self.inner = inner

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.inner is not None:
            self.inner.exec_module(module)
        module.executed_by = self.flag


class StubFinder(importlib.abc.MetaPathFinder):
    """Stands in for the lineage finder: wraps the real loader like lineage."""

    def __init__(self, fullname, flag):
        self.fullname = fullname
        self.flag = flag

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.fullname:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        spec.loader = StubLoader(self.flag, spec.loader)
        return spec


class LoadConfigTests(unittest.TestCase):
    def config(self, **overrides):
        value = {'model_module': 'm', 'runner_module': 'r',
                 'communicator_module': 'c',
                 'source_sha256': {'m': 'a' * 64, 'r': 'b' * 64,
                                   'c': 'c' * 64},
                 'backends': [],
                 'source_targets': {'x': 'json:loads'}}
        value.update(overrides)
        return value

    def with_env(self, config):
        saved = dict(os.environ)
        os.environ[probe_sitecustomize.DIR_ENV] = '/tmp/probe-out'
        if config is None:
            os.environ.pop(probe_sitecustomize.CONFIG_ENV, None)
        else:
            os.environ[probe_sitecustomize.CONFIG_ENV] = config
        self.addCleanup(lambda: (os.environ.clear(),
                                 os.environ.update(saved)))

    def test_missing_config_env_rejected(self):
        self.with_env(None)
        with self.assertRaisesRegex(ValueError, 'required'):
            probe_sitecustomize.load_config()

    def test_wrong_config_keys_rejected(self):
        self.with_env(json.dumps({'model_module': 'm'}))
        with self.assertRaisesRegex(ValueError, 'explicit probe config keys'):
            probe_sitecustomize.load_config()

    def test_source_sha_must_cover_exactly_the_targets(self):
        self.with_env(json.dumps(self.config(
            source_sha256={'m': 'a' * 64, 'r': 'b' * 64})))
        with self.assertRaisesRegex(ValueError, 'exactly the three target'):
            probe_sitecustomize.load_config()

    def test_source_sha_must_be_hex(self):
        self.with_env(json.dumps(self.config(
            source_sha256={'m': 'z' * 64, 'r': 'b' * 64, 'c': 'c' * 64})))
        with self.assertRaisesRegex(ValueError, 'SHA256 hex'):
            probe_sitecustomize.load_config()

    def test_source_targets_must_be_module_attribute_references(self):
        self.with_env(json.dumps(self.config(source_targets={'x': 'json'})))
        with self.assertRaisesRegex(ValueError, 'module:attribute'):
            probe_sitecustomize.load_config()

    def test_backends_must_be_a_list(self):
        self.with_env(json.dumps(self.config(backends={})))
        with self.assertRaisesRegex(ValueError, 'explicit list'):
            probe_sitecustomize.load_config()

    def test_valid_config_accepted(self):
        self.with_env(json.dumps(self.config()))
        value = probe_sitecustomize.load_config()
        self.assertEqual(value['model_module'], 'm')


class InstallInactiveTests(unittest.TestCase):
    def test_install_is_a_noop_without_the_env(self):
        saved = os.environ.pop(probe_sitecustomize.DIR_ENV, None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, probe_sitecustomize.DIR_ENV, saved)
        self.assertIsNone(probe_sitecustomize.install())

    def test_off_arm_still_installs_the_lineage_observer(self):
        # The off arm replaces the campaign observer on PYTHONPATH; without
        # the lineage chain it would capture no logits at all.
        saved = os.environ.pop(probe_sitecustomize.DIR_ENV, None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, probe_sitecustomize.DIR_ENV, saved)
        path = Path(tempfile.mkdtemp(prefix='atlas-lineage-'))
        (path / 'lineage_sitecustomize.py').write_text(
            'import importlib.abc, sys\n'
            'class _Finder(importlib.abc.MetaPathFinder):\n'
            '    def find_spec(self, fullname, path=None, target=None):\n'
            '        return None\n'
            'FINDER = _Finder()\n'
            'sys.meta_path.insert(0, FINDER)\n')
        sys.path.insert(0, str(path))
        self.addCleanup(sys.path.remove, str(path))
        self.addCleanup(sys.modules.pop, 'lineage_sitecustomize', None)
        self.assertIsNone(probe_sitecustomize.install())
        finder = sys.modules['lineage_sitecustomize'].FINDER
        self.assertIn(finder, sys.meta_path)
        sys.meta_path.remove(finder)

    def test_top_level_import_is_clean_without_the_env(self):
        result = subprocess.run(
            [sys.executable, '-c', 'import probe_sitecustomize'],
            capture_output=True, cwd=Path(__file__).parent)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


class BootstrapFailureTests(unittest.TestCase):
    def test_bad_config_exits_78_when_active(self):
        env = dict(os.environ,
                   ATLAS_PROBE_TELEMETRY_DIR='/tmp/probe-bootstrap-test',
                   ATLAS_PROBE_CONFIG='{}')
        result = subprocess.run(
            [sys.executable, '-c', 'import probe_sitecustomize'],
            capture_output=True, env=env, cwd=Path(__file__).parent)
        self.assertEqual(result.returncode, 78, result.stderr.decode())
        self.assertIn(b'probe sitecustomize failed', result.stderr)


class FinderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-probe-finder-')
        self.addCleanup(self.tmp.cleanup)
        sys.path.insert(0, self.tmp.name)
        self.addCleanup(sys.path.remove, self.tmp.name)

    def write_module(self, name, source):
        path = Path(self.tmp.name) / (name + '.py')
        path.write_text(source)
        return path

    def test_fallback_to_pathfinder_and_patch_after_delegate(self):
        self.write_module('probe_target_a', 'MARKER = 1\n')
        calls = []
        finder = probe_sitecustomize._ProbeFinder(
            {'probe_target_a': lambda module: calls.append(module.__name__)})
        spec = finder.find_spec('probe_target_a')
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.MARKER, 1)
        self.assertEqual(calls, ['probe_target_a'])
        self.assertIsNone(finder.find_spec('probe_other'))

    def test_chains_with_lineage_wrapped_loader(self):
        self.write_module('probe_target_b', 'MARKER = 2\n')
        calls = []
        lineage = StubFinder('probe_target_b', 'lineage')
        finder = probe_sitecustomize._ProbeFinder(
            {'probe_target_b': lambda module: calls.append(module.__name__)},
            lineage=lineage)
        spec = finder.find_spec('probe_target_b')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.MARKER, 2)
        self.assertEqual(module.executed_by, 'lineage')
        self.assertEqual(calls, ['probe_target_b'])


class LineageImportTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop('lineage_sitecustomize', None)

    def test_missing_lineage_returns_none(self):
        self.assertIsNone(probe_sitecustomize._import_lineage())

    def test_lineage_without_finder_returns_none(self):
        path = Path(tempfile.mkdtemp(prefix='atlas-lineage-'))
        (path / 'lineage_sitecustomize.py').write_text('VALUE = 1\n')
        sys.path.insert(0, str(path))
        try:
            self.assertIsNone(probe_sitecustomize._import_lineage())
        finally:
            sys.path.remove(str(path))

    def test_lineage_finder_is_captured(self):
        path = Path(tempfile.mkdtemp(prefix='atlas-lineage-'))
        (path / 'lineage_sitecustomize.py').write_text(
            'import importlib.abc\n'
            'class _Finder(importlib.abc.MetaPathFinder):\n'
            '    def find_spec(self, fullname, path=None, target=None):\n'
            '        return None\n'
            'sys_meta_path_inserted = True\n'
            'import sys\n'
            'sys.meta_path.insert(0, _Finder())\n')
        sys.path.insert(0, str(path))
        try:
            finder = probe_sitecustomize._import_lineage()
            self.assertIsInstance(finder, importlib.abc.MetaPathFinder)
            sys.meta_path.remove(finder)
        finally:
            sys.path.remove(str(path))


class FullInstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-probe-install-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.out = root / 'telemetry'
        sources = {'probe_fixture_model': MODEL_SOURCE,
                   'probe_fixture_runner': RUNNER_SOURCE,
                   'probe_fixture_comm': COMM_SOURCE}
        hashes = {}
        for name, source in sources.items():
            path = root / (name + '.py')
            path.write_text(source)
            hashes[name] = sha(path)
        sys.path.insert(0, str(root))
        self.addCleanup(sys.path.remove, str(root))
        config = {'model_module': 'probe_fixture_model',
                  'runner_module': 'probe_fixture_runner',
                  'communicator_module': 'probe_fixture_comm',
                  'source_sha256': hashes,
                  'backends': [{'name': 'X', 'attr': 'pynccl_comm',
                                'method': 'all_reduce', 'required': False}],
                  'source_targets': {'loads': 'json:loads'}}
        saved = dict(os.environ)
        os.environ[probe_sitecustomize.DIR_ENV] = str(self.out)
        os.environ[probe_sitecustomize.CONFIG_ENV] = json.dumps(config)
        self.addCleanup(lambda: (os.environ.clear(),
                                 os.environ.update(saved)))
        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)

    def tearDown(self):
        state = probe_sitecustomize._state
        finder = probe_sitecustomize._finder
        probe_sitecustomize._state = None
        probe_sitecustomize._finder = None
        if finder is not None and finder in sys.meta_path:
            sys.meta_path.remove(finder)
        if state is not None:
            atexit.unregister(state.close)
            state.close()
        for name in ('probe_fixture_model', 'probe_fixture_runner',
                     'probe_fixture_comm'):
            sys.modules.pop(name, None)

    def test_install_registers_chained_finder_and_receipt(self):
        finder = probe_sitecustomize.install()
        self.assertIs(finder, sys.meta_path[0])
        self.assertIs(finder, probe_sitecustomize._finder)
        state = probe_sitecustomize._state
        self.assertIsNotNone(state)
        receipt = state.install_receipts[0]
        self.assertEqual(receipt['kind'], 'sitecustomize')
        self.assertEqual(receipt['lineage_chained'], False)
        self.assertEqual(len(receipt['config_sha256']), 64)
        self.assertEqual(sorted(receipt['targets']),
                         ['probe_fixture_comm', 'probe_fixture_model',
                          'probe_fixture_runner'])

    def test_target_imports_are_patched_through_the_finder(self):
        probe_sitecustomize.install()
        model = importlib.import_module('probe_fixture_model')
        self.assertNotEqual(model.Glm5NextMoE.__init__.__name__, '__init__')
        runner = importlib.import_module('probe_fixture_runner')
        self.assertNotEqual(runner.GPUModelRunner.execute_model.__name__,
                            'execute_model')
        comm = importlib.import_module('probe_fixture_comm')
        self.assertNotEqual(comm.CudaCommunicator.__init__.__name__, '__init__')

    def test_targets_imported_before_install_are_patched_in_place(self):
        # The reference image's startup .pth hooks import the communicator
        # before sitecustomize runs; the finder alone would never see it.
        comm = importlib.import_module('probe_fixture_comm')
        self.assertEqual(comm.CudaCommunicator.__init__.__name__, '__init__')
        probe_sitecustomize.install()
        self.assertNotEqual(comm.CudaCommunicator.__init__.__name__, '__init__')
        receipt = probe_sitecustomize._state.install_receipts[0]
        self.assertEqual(receipt['patched_after_import'], ['probe_fixture_comm'])
        self.assertEqual(sorted(receipt['targets']),
                         ['probe_fixture_comm', 'probe_fixture_model',
                          'probe_fixture_runner'])
        model = importlib.import_module('probe_fixture_model')
        self.assertNotEqual(model.Glm5NextMoE.__init__.__name__, '__init__')

    def test_tampered_target_source_fails_the_import(self):
        probe_sitecustomize.install()
        root = Path(self.tmp.name)
        # Rewrite one target with content whose hash differs from the config.
        (root / 'probe_fixture_model.py').write_text(MODEL_SOURCE + '# tampered\n')
        with self.assertRaisesRegex(ValueError, 'installed source differs'):
            importlib.import_module('probe_fixture_model')
        sys.modules.pop('probe_fixture_model', None)


if __name__ == '__main__':
    unittest.main()
