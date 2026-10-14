# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the ablation variant compile lanes (D5)."""
import hashlib
import json
import sys

import pytest

from gated_norm_variants import KERNEL_ENTRY
from variant_compile import (LANES, compile_variant, compiler_id, nvrtc_record,
                             parse_lane, production_binding)

COMPILER = 'Build cuda_13.0.r13.0/compiler.36260728_0'

_FAKE_NVCC = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
if args == ['--version']:
    print('nvcc: NVIDIA (R) Cuda compiler driver')
    print(COMPILER_LINE)
    sys.exit(0)
open(ARGS_FILE, 'w').write('\\n'.join(args))
if EXIT_CODE != 0:
    sys.stderr.write('simulated nvcc failure\\n')
    sys.exit(EXIT_CODE)
out = args[args.index('-o') + 1]
open(out, 'w').write('.version 8.0\\n.target PTX_TARGET\\n.entry ENTRY( .param .u64 a )\\n')
"""


def _fake_nvcc(tmp_path, compiler_line=COMPILER, target='sm_121f',
               entry=KERNEL_ENTRY, exit_code=0, ptx_target=None):
    args_file = tmp_path / 'nvcc-args.txt'
    script = tmp_path / 'nvcc'
    script.write_text(_FAKE_NVCC
                      .replace('COMPILER_LINE', repr(compiler_line))
                      .replace('ARGS_FILE', repr(str(args_file)))
                      .replace('EXIT_CODE', str(exit_code))
                      .replace('PTX_TARGET', ptx_target or target)
                      .replace('ENTRY', entry))
    script.chmod(0o755)
    return script, args_file


def _manifest(tmp_path, arch='sm_121f', compiler=COMPILER, sha='a' * 64,
              schema='atlas-exl3-build-manifest-v1', flags=None, module_flags=None,
              drop_flags=False, drop_compiler=False, drop_embedded=False):
    compiler_flags = {'arch': arch}
    if not drop_compiler:
        compiler_flags['compiler'] = compiler
    if not drop_flags:
        compiler_flags['flags'] = (flags if flags is not None else
                                  ['--fmad=false', '-DTQ_PLUS_SIGNS',
                                   '--expt-relaxed-constexpr'])
    if module_flags is not None:
        compiler_flags['module_flags'] = module_flags
    manifest = {'schema': schema, 'compiler_flags': compiler_flags}
    if not drop_embedded:
        manifest['embedded_ptx'] = {'kda': {'sha256': sha}}
    path = tmp_path / 'build-manifest.json'
    path.write_text(json.dumps(manifest))
    return path


def _attested_ptx(tmp_path, target='sm_121f'):
    path = tmp_path / 't0__kda.ptx'
    path.write_text('.version 8.0\n.target ' + target + '\n.entry '
                    + KERNEL_ENTRY + '( .param .u64 a )\n')
    return path


def _binding(sha='a' * 64):
    return {'modules': {'kda': {'sha256': sha, 'file': 't0__kda.ptx'}}}


def test_parse_lane_requires_explicit_lane_and_matching_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    assert parse_lane('nvrtc', None) == 'nvrtc'
    assert parse_lane('nvcc-production', manifest) == 'nvcc-production'
    for lane, build_manifest in ((None, None), ('', None), ('nvcc', None),
                                 ('nvcc-production', None),
                                 ('nvrtc', manifest)):
        with pytest.raises(ValueError):
            parse_lane(lane, build_manifest)


def test_production_binding_binds_manifest_to_attested_ptx(tmp_path):
    manifest = _manifest(tmp_path)
    record = production_binding(manifest, _binding(), _attested_ptx(tmp_path))
    assert record['schema'] == 'atlas-variant-compile-production-v1'
    assert record['lane'] == 'nvcc-production'
    assert record['module'] == 'kda'
    assert record['arch'] == 'sm_121f'
    assert record['compiler'] == COMPILER
    assert record['effective_flags'] == [
        '--fmad=false', '-DTQ_PLUS_SIGNS', '--expt-relaxed-constexpr']
    assert record['build_manifest_file'] == str(manifest)
    assert len(record['build_manifest_sha256']) == 64


def test_production_binding_module_override_replaces_base_flags(tmp_path):
    manifest = _manifest(tmp_path, module_flags={
        'kda': ['--fmad=true', '-DTQ_PLUS_SIGNS', '--expt-relaxed-constexpr'],
        'exl3_mhc': ['--fmad=true']})
    record = production_binding(manifest, _binding(), _attested_ptx(tmp_path))
    assert record['effective_flags'][0] == '--fmad=true'


def test_production_binding_fail_closed_conditions(tmp_path):
    with pytest.raises(ValueError, match='schema mismatch'):
        production_binding(_manifest(tmp_path, schema='other'),
                           _binding(), _attested_ptx(tmp_path))
    with pytest.raises(ValueError, match='embedded PTX differs'):
        production_binding(_manifest(tmp_path, sha='b' * 64),
                           _binding(), _attested_ptx(tmp_path))
    with pytest.raises(ValueError, match='arch differs'):
        production_binding(_manifest(tmp_path), _binding(),
                           _attested_ptx(tmp_path, target='sm_999'))
    with pytest.raises(ValueError, match='effective flags invalid'):
        production_binding(_manifest(tmp_path, drop_flags=True),
                           _binding(), _attested_ptx(tmp_path))
    with pytest.raises(ValueError, match='effective flags invalid'):
        production_binding(_manifest(tmp_path, flags=[]),
                           _binding(), _attested_ptx(tmp_path))
    with pytest.raises(ValueError, match='compiler identity missing'):
        production_binding(_manifest(tmp_path, drop_compiler=True),
                           _binding(), _attested_ptx(tmp_path))
    with pytest.raises(ValueError, match='embedded PTX differs'):
        production_binding(_manifest(tmp_path, drop_embedded=True),
                           _binding(), _attested_ptx(tmp_path))


def test_compiler_id_is_last_nonempty_version_line(tmp_path):
    script, _ = _fake_nvcc(tmp_path)
    assert compiler_id(script) == COMPILER


def test_compile_variant_argv_parity_and_receipt(tmp_path):
    script, args_file = _fake_nvcc(tmp_path)
    common = tmp_path / 'common'
    common.mkdir()
    source = tmp_path / 'v24-minimal.cu'
    source.write_text('// variant\n')
    out_ptx = tmp_path / 'v24-minimal.ptx'
    production = {'arch': 'sm_121f', 'compiler': COMPILER,
                  'effective_flags': ['--fmad=false', '-DTQ_PLUS_SIGNS']}
    record = compile_variant(script, source, out_ptx, common, production, KERNEL_ENTRY)
    assert args_file.read_text().splitlines() == [
        '--ptx', '-arch=sm_121f', '-O3', '--fmad=false', '-DTQ_PLUS_SIGNS',
        '--Werror', 'all-warnings', '-I', str(common),
        str(source), '-o', str(out_ptx)]
    assert record['compiler'] == COMPILER
    assert record['ptx_sha256'] == hashlib.sha256(out_ptx.read_bytes()).hexdigest()
    assert record['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record['ptx_file'] == str(out_ptx)


def test_compile_variant_rejects_wrong_compiler_identity(tmp_path):
    script, args_file = _fake_nvcc(tmp_path, compiler_line='Build other')
    production = {'arch': 'sm_121f', 'compiler': COMPILER,
                  'effective_flags': ['--fmad=false']}
    with pytest.raises(ValueError, match='local compiler differs'):
        compile_variant(script, tmp_path / 'v.cu', tmp_path / 'v.ptx',
                        tmp_path, production, KERNEL_ENTRY)
    assert not args_file.exists()  # refused before any compile


def test_compile_variant_rejects_failed_compile_and_bad_output(tmp_path):
    production = {'arch': 'sm_121f', 'compiler': COMPILER,
                  'effective_flags': ['--fmad=false']}
    source = tmp_path / 'v.cu'
    source.write_text('// variant\n')
    failing, _ = _fake_nvcc(tmp_path, exit_code=1)
    with pytest.raises(ValueError, match='variant compile failed'):
        compile_variant(failing, source, tmp_path / 'v.ptx', tmp_path,
                        production, KERNEL_ENTRY)
    wrong_target, _ = _fake_nvcc(tmp_path, ptx_target='sm_999')
    with pytest.raises(ValueError, match='target differs'):
        compile_variant(wrong_target, source, tmp_path / 'w.ptx', tmp_path,
                        production, KERNEL_ENTRY)
    wrong_entry, _ = _fake_nvcc(tmp_path, entry='other_entry')
    with pytest.raises(ValueError, match='missing the kernel entry'):
        compile_variant(wrong_entry, source, tmp_path / 'x.ptx', tmp_path,
                        production, KERNEL_ENTRY)


def test_compile_variant_windows_std_gate(monkeypatch, tmp_path):
    script, args_file = _fake_nvcc(tmp_path)
    source = tmp_path / 'v.cu'
    source.write_text('// variant\n')
    production = {'arch': 'sm_121f', 'compiler': COMPILER,
                  'effective_flags': ['--fmad=false']}
    monkeypatch.setattr(sys, 'platform', 'win32')
    compile_variant(script, source, tmp_path / 'w.ptx', tmp_path,
                    production, KERNEL_ENTRY)
    assert args_file.read_text().splitlines()[:4] == [
        '--ptx', '-arch=sm_121f', '-O3', '-std=c++17']


def test_nvrtc_record_shape():
    record = nvrtc_record(['--std=c++17', '--fmad=false'],
                          {'v24-minimal': ('code', 'diff', 'b' * 64)})
    assert record == {'schema': 'atlas-variant-compile-nvrtc-v1', 'lane': 'nvrtc',
                      'toolchain': 'NVRTC', 'options': ['--std=c++17', '--fmad=false'],
                      'variants': {'v24-minimal': {'source_sha256': 'b' * 64}}}
    assert set(LANES) == {'nvrtc', 'nvcc-production'}
