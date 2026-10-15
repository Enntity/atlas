# SPDX-License-Identifier: AGPL-3.0-only
"""Probe sitecustomize: import-time bootstrap for the runtime-mechanism probe.

This file is copied into the evidence-root probe overlay as
``sitecustomize.py`` (the package assembler renames it), alongside a verbatim
copy of the lineage sitecustomize and the frozen harness modules. In the
package directory it:

1. installs only the lineage (campaign observer) chain unless
   ``ATLAS_PROBE_TELEMETRY_DIR`` is set (the arm difference between off/on
   attempts is this environment, not file content);
2. validates the explicit ``ATLAS_PROBE_CONFIG`` JSON (no defaults, PCND);
3. chains with the lineage sitecustomize when present, so L1-style lineage
   hooks and probe hooks can coexist on the same module (probe wraps the
   already-lineage-wrapped loader);
4. preflights each installed module source against its registered SHA256 at
   loader-exec time, before any instrumentation, and patches the model, runner
   and communicator modules through ``probe_wiring``;
5. registers the SIGTERM flush so container teardown persists the last
   request's buffer.

Python swallows sitecustomize top-level exceptions, so any bootstrap failure
writes stderr and exits the process with code 78 (the lineage precedent):
an attempt must fail loudly instead of running uninstrumented.
"""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import signal
import sys

CONFIG_ENV = 'ATLAS_PROBE_CONFIG'
DIR_ENV = 'ATLAS_PROBE_TELEMETRY_DIR'
CONFIG_KEYS = {'model_module', 'runner_module', 'communicator_module',
               'source_sha256', 'backends', 'source_targets'}
EXIT_CODE = 78

_state = None
_finder = None


def config_digest(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def load_config():
    raw = os.environ.get(CONFIG_ENV)
    if raw is None:
        raise ValueError(CONFIG_ENV + ' is required when ' + DIR_ENV + ' is set')
    config = json.loads(raw)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError('explicit probe config keys required: '
                         + str(sorted(CONFIG_KEYS)))
    for key in ('model_module', 'runner_module', 'communicator_module'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError('explicit module name required: ' + key)
    hashes = config['source_sha256']
    targets = (config['model_module'], config['runner_module'],
               config['communicator_module'])
    if not isinstance(hashes, dict) or set(hashes) != set(targets):
        raise ValueError('source_sha256 must cover exactly the three target modules')
    for digest in hashes.values():
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('source_sha256 values must be lowercase SHA256 hex')
    if not isinstance(config['backends'], list):
        raise ValueError('backends must be an explicit list')
    targets = config['source_targets']
    if not isinstance(targets, dict) or not targets or any(
            not isinstance(key, str) or not isinstance(value, str)
            or ':' not in value for key, value in targets.items()):
        raise ValueError('source_targets must map labels to module:attribute '
                         'references')
    return config


class _ProbeLoader(importlib.abc.Loader):
    def __init__(self, delegate, patch):
        self.delegate = delegate
        self.patch = patch

    def create_module(self, spec):
        return self.delegate.create_module(spec)

    def exec_module(self, module):
        self.delegate.exec_module(module)
        self.patch(module)


class _ProbeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, patches, lineage=None):
        self.patches = patches
        self.lineage = lineage

    def find_spec(self, fullname, path=None, target=None):
        patch = self.patches.get(fullname)
        if patch is None:
            return None
        spec = None
        if self.lineage is not None:
            spec = self.lineage.find_spec(fullname, path, target)
        if spec is None:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _ProbeLoader(spec.loader, patch)
        return spec


def _import_lineage():
    """Import the package's lineage sitecustomize copy; return its finder."""
    before = list(sys.meta_path)
    try:
        import lineage_sitecustomize  # noqa: F401  (package member)
    except ImportError:
        return None
    added = [entry for entry in sys.meta_path if entry not in before]
    finders = [entry for entry in added
               if isinstance(entry, importlib.abc.MetaPathFinder)]
    if not finders:
        # Lineage copy present but inactive (its own env gates unset):
        # nothing to chain with.
        return None
    if len(finders) != 1:
        raise ValueError('lineage sitecustomize must register at most one finder, '
                         'found ' + str(len(finders)))
    return finders[0]


def _sigterm(signum, frame):
    # Container teardown must persist the last request's buffer before exit.
    raise SystemExit(143)


def install():
    global _state, _finder
    out_dir = os.environ.get(DIR_ENV)
    if not out_dir:
        # Off arm: this package replaces the campaign observer on PYTHONPATH,
        # so the lineage observer must still run or the off arm captures no
        # logits. Only the telemetry differs between arms.
        _import_lineage()
        return None
    import probe_wiring as wiring
    config = load_config()
    state = wiring.ProbeState(out_dir)
    patches = {
        config['model_module']: lambda module: wiring.patch_moe(
            module, state, config['source_targets'],
            config['source_sha256'][config['model_module']]),
        config['runner_module']: lambda module: wiring.patch_runner(
            module, state, config['source_sha256'][config['runner_module']]),
        config['communicator_module']: lambda module: wiring.patch_communicator(
            module, state, config['backends'], config['source_targets'],
            config['source_sha256'][config['communicator_module']]),
    }
    lineage = _import_lineage()
    # Startup .pth hooks in the reference image import some targets (e.g.
    # the communicator) before sitecustomize runs, so the finder never sees
    # them. Patch those resident modules now; their classes are patched
    # before any instance exists because instances are created at runtime.
    already = sorted(name for name in patches if name in sys.modules)
    for name in already:
        patches.pop(name)(sys.modules[name])
    finder = _ProbeFinder(patches, lineage)
    sys.meta_path.insert(0, finder)
    state.record_install('sitecustomize', {
        'config_sha256': config_digest(config),
        'telemetry_dir': str(out_dir),
        'lineage_chained': lineage is not None,
        'targets': sorted(patches) + already,
        'patched_after_import': already})
    signal.signal(signal.SIGTERM, _sigterm)
    _state, _finder = state, finder
    return finder


try:
    install()
except SystemExit:
    raise
except BaseException as error:
    if os.environ.get(DIR_ENV):
        sys.stderr.write('EXL3 probe sitecustomize failed: %s\n' % error)
        sys.stderr.flush()
        os._exit(EXIT_CODE)
    raise
