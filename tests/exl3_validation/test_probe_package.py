# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the probe overlay assembler (evidence-root package)."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import probe_package


REPO = Path(__file__).resolve().parent
REGISTRATION = (REPO.parents[1] / 'docs' / 'design'
                / 'exl3-runtime-mechanism-probe-registration.md')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RegistrationTripwireTests(unittest.TestCase):
    def test_registered_trio_matches_current_repo_files(self):
        """A repo edit to a frozen file must fail here (re-freeze first)."""
        trio = probe_package.frozen_trio(REGISTRATION)
        self.assertEqual(set(trio), probe_package.FROZEN_TRIO)
        for name, digest in trio.items():
            self.assertEqual(sha(REPO / name), digest,
                             'frozen harness changed without re-freeze: ' + name)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-probe-pkg-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.evidence = self.root / 'evidence'
        self.evidence.mkdir()
        self.lineage = self.root / 'lineage'
        (self.lineage / '__pycache__').mkdir(parents=True)
        (self.lineage / 'sitecustomize.py').write_text(
            '# lineage sitecustomize\n')
        (self.lineage / 'reference_ffn_trace.py').write_text(
            '# lineage ffn trace\n')
        (self.lineage / 'package-binding.json').write_text(
            '{"schema": "atlas-upstream-hook-package-context-v5"}\n')
        (self.lineage / '__pycache__' / 'junk.pyc').write_bytes(b'\x00\x01')
        (self.lineage / '.pytest_cache' / 'v' / 'cache').mkdir(parents=True)
        (self.lineage / '.pytest_cache' / 'v' / 'cache' / 'lastfailed').write_text(
            '{}\n')
        self.config = {'schema': 'atlas-probe-package-config-v1',
                       'out': str(self.evidence / 'probe-overlay'),
                       'evidence_root': str(self.evidence),
                       'registration': {'path': str(REGISTRATION),
                                        'sha256': sha(REGISTRATION)},
                       'lineage_overlay': {'path': str(self.lineage)}}
        self.config_path = self.root / 'package.json'
        self.write()

    def write(self):
        self.config_path.write_text(json.dumps(self.config))

    def assemble(self):
        return probe_package.assemble(self.config_path)


class AssembleTests(Fixture):
    def test_package_binds_trio_glue_and_lineage(self):
        result = self.assemble()
        out = Path(result['out'])
        binding = json.loads((out / 'package-binding.json').read_text())
        self.assertEqual(binding['schema'], 'atlas-probe-hook-package-v1')
        self.assertEqual(result['package_sha256'],
                         sha(out / 'package-binding.json'))
        # Lineage copy with the renamed sitecustomize; no cache junk.
        self.assertTrue((out / 'lineage_sitecustomize.py').is_file())
        self.assertTrue((out / 'reference_ffn_trace.py').is_file())
        self.assertFalse((out / '.pytest_cache').exists())
        self.assertFalse(any('__pycache__' in name or '.pytest_cache' in name
                             for name in binding['file_sha256']))
        # Probe members: our sitecustomize plus wiring and the trio dependency.
        self.assertEqual(sha(out / 'sitecustomize.py'),
                         sha(REPO / 'probe_sitecustomize.py'))
        self.assertFalse((out / 'probe_sitecustomize.py').exists())
        self.assertEqual(sha(out / 'probe_wiring.py'),
                         sha(REPO / 'probe_wiring.py'))
        self.assertEqual(sha(out / 'reference_norm_binding.py'),
                         sha(REPO / 'reference_norm_binding.py'))
        # Frozen trio verified against the registration table.
        for name, digest in binding['registration']['frozen_trio'].items():
            self.assertEqual(sha(out / name), digest)
        # Every recorded file hash re-verifies; nothing unrecorded.
        recorded = set(binding['file_sha256'])
        actual = {str(path.relative_to(out)) for path in out.rglob('*')
                  if path.is_file() and path.name != 'package-binding.json'}
        self.assertEqual(recorded, actual)
        for name, digest in binding['file_sha256'].items():
            self.assertEqual(sha(out / name), digest)
        self.assertEqual(binding['lineage']['package_binding_sha256'],
                         sha(self.lineage / 'package-binding.json'))
        self.assertEqual(result['files'], len(binding['file_sha256']))

    def test_fresh_output_required(self):
        Path(self.config['out']).mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh directory'):
            self.assemble()

    def test_output_must_live_inside_evidence_root(self):
        self.config['out'] = str(self.root / 'elsewhere')
        self.write()
        with self.assertRaisesRegex(ValueError, 'inside the evidence root'):
            self.assemble()

    def test_registration_binding_mismatch_rejected(self):
        self.config['registration']['sha256'] = '0' * 64
        self.write()
        with self.assertRaisesRegex(ValueError, 'registration binding differs'):
            self.assemble()

    def test_lineage_without_sitecustomize_rejected(self):
        (self.lineage / 'sitecustomize.py').unlink()
        with self.assertRaisesRegex(ValueError, 'sitecustomize.py'):
            self.assemble()

    def test_frozen_trio_mismatch_fails_before_copying(self):
        rows = '\n'.join(
            f'| `tests/exl3_validation/{name}` | `{"0" * 64}` |'
            for name in sorted(probe_package.FROZEN_TRIO))
        registration = self.root / 'tampered-registration.md'
        registration.write_text(
            '## Harness freeze\n\n| Frozen harness file | SHA256 |\n'
            '|---|---|\n' + rows + '\n')
        self.config['registration'] = {'path': str(registration),
                                       'sha256': sha(registration)}
        self.write()
        with self.assertRaisesRegex(ValueError, 're-freeze'):
            self.assemble()

    def test_registration_must_freeze_exactly_the_trio(self):
        registration = self.root / 'partial-registration.md'
        registration.write_text('## Harness freeze\n\nnothing here\n')
        with self.assertRaisesRegex(ValueError, 'expected harness trio'):
            probe_package.frozen_trio(registration)


if __name__ == '__main__':
    unittest.main()
