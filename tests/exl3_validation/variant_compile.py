# SPDX-License-Identifier: AGPL-3.0-only
"""Compile lanes for gated-norm ablation variants (docs/plans/l1-escalation-tooling-spec.md D5).

`nvrtc` is the frozen historical diagnostic lane (native_ptx.NVRTC_OPTIONS).
`nvcc-production` replays the recorded production toolchain and effective
flags from an `atlas-exl3-build-manifest-v1` (B11), with argv parity to the
atlas-kernels NVIDIA build (crates/atlas-kernels/build_target.rs), tied to
the attested embedded PTX of the same image. Fail-closed throughout; every
lane records its toolchain, flags and per-variant source/PTX hashes so a
v24 claim can never be read across lanes (B6).
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

LANES = ('nvrtc', 'nvcc-production')
MANIFEST_SCHEMA = 'atlas-exl3-build-manifest-v1'
MODULE = 'kda'
COMPILE_TIMEOUT_SECONDS = 600


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_lane(variant_compile, build_manifest):
    """Explicit lane selection only: the production lane requires the build
    manifest it replays, and the diagnostic lane takes none (B5)."""
    _require(variant_compile in LANES,
             'explicit variant compile lane required: ' + str(variant_compile))
    if variant_compile == 'nvcc-production':
        _require(build_manifest is not None,
                 'nvcc-production lane requires --build-manifest')
    else:
        _require(build_manifest is None,
                 'the nvrtc diagnostic lane takes no --build-manifest')
    return variant_compile


def compiler_id(nvcc):
    """Build-target compiler identity: last non-empty `--version` line,
    whitespace-collapsed (atlas-kernels/build_target.rs compiler_version)."""
    out = subprocess.run([str(nvcc), '--version'], capture_output=True, text=True,
                         timeout=COMPILE_TIMEOUT_SECONDS)
    _require(out.returncode == 0, 'compiler identity unavailable: ' + str(nvcc))
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    _require(bool(lines), 'compiler identity output empty: ' + str(nvcc))
    return ' '.join(lines[-1].split())


def production_binding(build_manifest, binding, attested_ptx):
    """Effective production compile policy for the MODULE kernel, tied to the
    attested embedded PTX the same build produced. Returns a receipt record."""
    manifest_path = Path(build_manifest)
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get('schema') == MANIFEST_SCHEMA,
             'build manifest schema mismatch: ' + str(manifest_path))
    flags = manifest.get('compiler_flags')
    _require(isinstance(flags, dict), 'build manifest compiler_flags missing')
    arch, compiler = flags.get('arch'), flags.get('compiler')
    _require(isinstance(arch, str) and arch.startswith('sm_'),
             'build manifest arch invalid')
    _require(isinstance(compiler, str) and compiler,
             'build manifest compiler identity missing')
    effective = flags.get('module_flags', {}).get(MODULE, flags.get('flags'))
    _require(isinstance(effective, list) and bool(effective)
             and all(isinstance(flag, str) and flag for flag in effective),
             'build manifest effective flags invalid')
    embedded = manifest.get('embedded_ptx', {}).get(MODULE, {})
    attested = binding.get('modules', {}).get(MODULE, {})
    _require(isinstance(embedded.get('sha256'), str)
             and embedded.get('sha256') == attested.get('sha256'),
             'build manifest embedded PTX differs from the attested binding')
    target = re.search(r'\.target\s+(\S+)', Path(attested_ptx).read_text())
    _require(target is not None and target[1] == arch,
             'build manifest arch differs from the attested PTX target')
    return {'schema': 'atlas-variant-compile-production-v1',
            'lane': 'nvcc-production', 'module': MODULE,
            'arch': arch, 'compiler': compiler,
            'effective_flags': list(effective),
            'build_manifest_file': str(manifest_path),
            'build_manifest_sha256': _sha(manifest_path)}


def compile_variant(nvcc, source, out_ptx, common, production, entry):
    """Compile one variant .cu to PTX with production argv parity
    (build_target.rs: --ptx -arch=<arch> -O3 [strict warnings] <flags>
    <source> -o <out>; -std=c++17 only on Windows). Validates the local
    compiler identity first and the produced PTX after."""
    identity = compiler_id(nvcc)
    _require(identity == production['compiler'],
             'local compiler differs from the recorded build compiler')
    argv = [str(nvcc), '--ptx', '-arch=' + production['arch'], '-O3']
    if sys.platform == 'win32':
        argv.append('-std=c++17')
    argv += list(production['effective_flags'])
    # Strict kernel validation mirrors the production default
    # (build_target.rs: --Werror all-warnings unless ATLAS_KERNEL_NO_STRICT=1;
    # the suppress list is empty today — a future nonempty list would show up
    # here as a compile failure rather than silent drift).
    argv += ['--Werror', 'all-warnings', '-I', str(common),
             str(source), '-o', str(out_ptx)]
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=COMPILE_TIMEOUT_SECONDS)
    _require(done.returncode == 0, 'variant compile failed: ' + done.stderr[-2000:])
    text = Path(out_ptx).read_text()
    target = re.search(r'\.target\s+(\S+)', text)
    _require(target is not None and target[1] == production['arch'],
             'compiled variant PTX target differs from the recorded arch')
    _require(re.search(r'\.entry\s+' + re.escape(entry) + r'\s*\(', text),
             'compiled variant PTX is missing the kernel entry: ' + entry)
    return {'compiler': identity, 'argv': argv,
            'ptx_file': str(out_ptx), 'ptx_sha256': _sha(out_ptx),
            'source_file': str(source), 'source_sha256': _sha(source)}


def nvrtc_record(options, variants):
    """Receipt record for the frozen NVRTC diagnostic lane."""
    return {'schema': 'atlas-variant-compile-nvrtc-v1', 'lane': 'nvrtc',
            'toolchain': 'NVRTC', 'options': list(options),
            'variants': {name: {'source_sha256': digest}
                         for name, (_, _, digest) in variants.items()}}
