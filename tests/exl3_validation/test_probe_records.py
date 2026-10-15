# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the probe pre-Spark records (attestation, profiler, NCCL)."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import probe_records


REPO = Path(__file__).resolve().parent
REGISTRATION = (REPO.parents[1] / 'docs' / 'design'
                / 'exl3-runtime-mechanism-probe-registration.md')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeRunner:
    def __init__(self, rc=0, stdout=b''):
        self.rc = rc
        self.stdout = stdout
        self.calls = []

    def run(self, argv, *, env=None, timeout):
        self.calls.append(list(argv))
        return {'rc': self.rc, 'stdout': self.stdout, 'stderr': b''}


class AttemptTests(unittest.TestCase):
    def test_registered_attempts_match_the_code_freeze(self):
        """Editing the registration's attempt list must fail here."""
        ids = probe_records.registered_attempts(REGISTRATION)
        self.assertEqual(ids, tuple(name for name, _ in probe_records.ATTEMPTS))

    def test_attestation_binds_registration_hash(self):
        value = probe_records.attestation(REGISTRATION)
        self.assertEqual(value['schema'], 'atlas-probe-attempt-attestation-v1')
        self.assertEqual([row['arm'] for row in value['attempts']],
                         ['off', 'on', 'on', 'off', 'off', 'on'])
        self.assertEqual(value['registration']['sha256'], sha(REGISTRATION))

    def test_tampered_attempt_list_rejected(self):
        with tempfile.TemporaryDirectory(prefix='atlas-probe-attest-') as tmp:
            path = Path(tmp) / 'registration.md'
            path.write_text(
                '## Attempt list (frozen)\n\nsome prose\n\n'
                '    off01, on01, on02, off02, off03, off04\n\n'
                '## Profile (frozen)\n')
            with self.assertRaisesRegex(ValueError, 'code-side freeze'):
                probe_records.registered_attempts(path)

    def test_stamp_writes_once_into_a_receipt_directory(self):
        with tempfile.TemporaryDirectory(prefix='atlas-probe-stamp-') as tmp:
            receipt = Path(tmp)
            value = probe_records.stamp(REGISTRATION, receipt)
            self.assertEqual(value['schema'], 'atlas-probe-attempt-attestation-v1')
            written = json.loads((receipt / 'attestation.json').read_text())
            self.assertEqual(written, value)
            with self.assertRaisesRegex(ValueError, 'already carries'):
                probe_records.stamp(REGISTRATION, receipt)

    def test_stamp_requires_a_directory(self):
        with tempfile.TemporaryDirectory(prefix='atlas-probe-stamp-') as tmp:
            with self.assertRaisesRegex(ValueError, 'directory required'):
                probe_records.stamp(REGISTRATION, Path(tmp) / 'absent')


class ProfilerTests(unittest.TestCase):
    def test_bind_records_version_binary_and_invocation(self):
        with tempfile.TemporaryDirectory(prefix='atlas-probe-nsys-') as tmp:
            binary = Path(tmp) / 'nsys'
            binary.write_text('#!/bin/sh\necho nsys\n')
            out = Path(tmp) / 'profiler.json'
            runner = FakeRunner(stdout=b'nsys (x) 2025.3.1\n')
            value = probe_records.profiler_bind(binary, out, runner)
            self.assertTrue(value['available'])
            self.assertEqual(value['version'], 'nsys (x) 2025.3.1')
            self.assertEqual(value['binary']['sha256'], sha(binary))
            self.assertEqual(value['invocation'],
                             ['nsys', 'profile', '--trace=cuda,nvtx',
                              '--sample=none', '--cpuctxsw=none',
                              '-o', '{output}'])
            self.assertEqual(value['invocation_sha256'],
                             probe_records.invocation_sha256())
            self.assertEqual(runner.calls[0], [str(binary), '--version'])
            self.assertEqual(json.loads(out.read_text()), value)

    def test_failed_version_probe_rejected(self):
        with tempfile.TemporaryDirectory(prefix='atlas-probe-nsys-') as tmp:
            binary = Path(tmp) / 'nsys'
            binary.write_text('broken\n')
            with self.assertRaisesRegex(ValueError, '--version failed'):
                probe_records.profiler_bind(binary, Path(tmp) / 'o.json',
                                            FakeRunner(rc=1))

    def test_missing_binary_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            probe_records.profiler_bind('/nonexistent/nsys', Path('/tmp/o.json'))

    def test_unavailable_records_the_limitation_with_same_invocation(self):
        with tempfile.TemporaryDirectory(prefix='atlas-probe-nsys-') as tmp:
            out = Path(tmp) / 'profiler.json'
            value = probe_records.profiler_unavailable(
                'nsys not installed on either Spark', out)
            self.assertFalse(value['available'])
            self.assertIsNone(value['binary'])
            self.assertEqual(value['limitation'], 'nsys not installed on either Spark')
            self.assertEqual(value['invocation_sha256'],
                             probe_records.invocation_sha256())
            with self.assertRaisesRegex(ValueError, 'reason'):
                probe_records.profiler_unavailable('   ', out)


class NcclPriorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-probe-nccl-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.loaded = {}
        for rank, digest in (('0', 'aa' * 32), ('1', 'bb' * 32)):
            path = self.root / f'spark{rank}-native-loaded-nccl.sha256'
            path.write_text(f'{digest}  /usr/lib/aarch64-linux-gnu/libnccl.so.2.32.3\n')
            self.loaded[rank] = path
        self.runtime = self.root / 'reference-nccl-runtime-evidence.json'
        self.runtime.write_text(json.dumps(
            {'status': 0, 'version_code': 23007,
             'library': '/usr/local/lib/python3.12/dist-packages/nvidia/nccl/'
                        'lib/libnccl.so.2',
             'sha256': 'cc' * 32}))
        self.config = {'schema': 'atlas-probe-nccl-prior-config-v1',
                       'entries': [
                           {'path': str(self.loaded['0']), 'sha256': sha(self.loaded['0']),
                            'rank': '0', 'kind': 'loaded-sha256'},
                           {'path': str(self.loaded['1']), 'sha256': sha(self.loaded['1']),
                            'rank': '1', 'kind': 'loaded-sha256'},
                           {'path': str(self.runtime), 'sha256': sha(self.runtime),
                            'rank': 'reference', 'kind': 'runtime-evidence'}]}
        self.config_path = self.root / 'prior.json'
        self.write()

    def write(self):
        self.config_path.write_text(json.dumps(self.config))

    def prior(self, out=None):
        out = out or (self.root / 'nccl.json')
        return probe_records.nccl_prior(self.config_path, out)

    def test_prior_binding_covers_both_ranks_and_reference(self):
        value = self.prior()
        self.assertEqual(value['schema'], 'atlas-probe-nccl-binding-v1')
        self.assertIn('live re-bind', value['scope'])
        ranks = sorted(entry['rank'] for entry in value['entries'])
        self.assertEqual(ranks, ['0', '1', 'reference'])
        by_rank = {entry['rank']: entry for entry in value['entries']}
        self.assertEqual(by_rank['0']['libraries'][0]['sha256'], 'aa' * 32)
        self.assertEqual(by_rank['reference']['libraries'][0]['version_code'], 23007)

    def test_both_ranks_required(self):
        self.config['entries'] = [entry for entry in self.config['entries']
                                  if entry['rank'] != '1']
        self.write()
        with self.assertRaisesRegex(ValueError, 'both ranks'):
            self.prior()

    def test_tampered_evidence_rejected(self):
        self.loaded['0'].write_text('tampered\n')
        with self.assertRaisesRegex(ValueError, 'bound prior evidence differs'):
            self.prior()

    def test_malformed_loaded_line_rejected(self):
        self.loaded['1'].write_text('not-a-hash  /usr/lib/libnccl.so.2\n')
        self.config['entries'][1]['sha256'] = sha(self.loaded['1'])
        self.write()
        with self.assertRaisesRegex(ValueError, 'unparseable'):
            self.prior()


class NcclLiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-probe-live-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proc = self.root / 'proc-root'
        library = self.proc / 'usr/lib/aarch64-linux-gnu'
        library.mkdir(parents=True)
        (library / 'libnccl.so.2.32.3').write_bytes(b'nccl-bytes')
        self.maps = self.root / 'maps'
        self.maps.write_text(
            '7f0000000000-r-xp 00000000 fd:00 42 /usr/lib/'
            'aarch64-linux-gnu/libnccl.so.2.32.3\n'
            '7f1000000000-r--p 00000000 fd:00 43 /usr/lib/libc.so.6\n')

    def live(self, out=None):
        out = out or (self.root / 'live.json')
        return probe_records.nccl_live(self.maps, self.proc, '0', out)

    def test_live_binding_hashes_mapped_library_under_proc_root(self):
        value = self.live()
        entry = value['entries'][0]
        self.assertEqual(entry['rank'], '0')
        self.assertEqual(entry['kind'], 'loaded-maps')
        self.assertEqual(entry['libraries'],
                         [{'path': '/usr/lib/aarch64-linux-gnu/libnccl.so.2.32.3',
                           'sha256': sha(self.proc / 'usr/lib/aarch64-linux-gnu'
                                         / 'libnccl.so.2.32.3')}])
        self.assertEqual(value['scope'], 'live loaded-maps binding')

    def test_missing_mapping_rejected(self):
        self.maps.write_text('7f00-r-xp 0000 fd:00 1 /usr/lib/libc.so.6\n')
        with self.assertRaisesRegex(ValueError, 'no loaded libnccl'):
            self.live()

    def test_missing_file_under_proc_root_rejected(self):
        (self.proc / 'usr/lib/aarch64-linux-gnu/libnccl.so.2.32.3').unlink()
        with self.assertRaisesRegex(ValueError, 'missing under proc root'):
            self.live()

    def test_bad_rank_rejected(self):
        with self.assertRaisesRegex(ValueError, 'rank must be'):
            probe_records.nccl_live(self.maps, self.proc, '2',
                                    self.root / 'o.json')


if __name__ == '__main__':
    unittest.main()
