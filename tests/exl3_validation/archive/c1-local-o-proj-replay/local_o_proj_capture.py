# SPDX-License-Identifier: AGPL-3.0-only
# ARCHIVED 2026-10-12 — DO NOT USE. The C1 BF16 common-space replay premise
# is invalid (see README.md in this directory). Importing or running this
# module fails unless ATLAS_ALLOW_ARCHIVED_C1=1 is set for forensic reading.
import os as _os
if _os.environ.get("ATLAS_ALLOW_ARCHIVED_C1") != "1":
    raise ImportError("archived C1 o_proj replay harness: premise invalid; see README.md")
"""Experiment C1: validate engine capture records and assemble binding rows.

Both engines emit an `atlas-local-o-proj-binding-v1` capture record for the
layer-0 local (pre-TP) o_proj:

* candidate — `ffn-L0-rank{r}-{inv:06}-o_proj_binding.json` envelope written
  by the Rust forward capture (`UpstreamTrace::capture_o_proj_binding`);
* reference — `ffn-L0-rank{r}-{inv:06}-o-proj-binding.json` written by the
  reference overlay (`ref-overlay/o_proj_binding_router.py`).

`read_capture_record` fail-closes on the record format (schema / engine /
rank / layer / invocation / tokens identity, the complete projection config,
the `files.weight` entry, the `resident` sub-record and the GEMM receipt).
`assemble_binding_row` then builds the immutable five-file
`rank{r}-row{n}/` rows that `local_o_proj_replay.validate_binding` consumes,
re-validating every hash on the way and finishing with a full
`validate_binding` self-check. The BF16 `weight` recorded here is the common
comparison space declared by the binding spec — not a claim about the exact
bits each engine's native GEMM multiplies; that claim is what the C1
self-anchor tests at GPU time.
"""
import json
import shutil
from pathlib import Path

import numpy as np

import gated_norm_ablation
from local_o_proj_replay import (BINDING_SCHEMA, PROJECTION_DTYPES,
                                 WEIGHT_LAYOUTS, validate_binding)
from native_ptx import sha

DEFAULT_OVERLAY_DIR = (
    '/home/colum/Workspace/atlas-exl3-validation-20261010/'
    'q2-c1-execution-20261012/ref-overlay')
RESIDENT_KEYS = ('form', 'file', 'dtype', 'sha256')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonempty_str(value):
    return isinstance(value, str) and bool(value)


def _sha256_hex(value):
    if not _nonempty_str(value) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_capture_record(binding, receipt, *, engine, rank, row, tokens):
    """Fail closed on every `atlas-local-o-proj-binding-v1` capture field and
    on the GEMM receipt that must accompany it."""
    _require(isinstance(binding, dict), 'capture record must be an object')
    _require(binding.get('schema') == BINDING_SCHEMA, 'capture record schema mismatch')
    for key, expected, label, kind in (
            ('engine', engine, 'engine', str), ('rank', rank, 'rank', int),
            ('layer', 0, 'layer', int), ('invocation', row, 'invocation', int),
            ('tokens', tokens, 'token', int)):
        _require(type(binding.get(key)) is kind and binding.get(key) == expected,
                 f'capture record {label} mismatch')
    projection = binding.get('projection')
    _require(isinstance(projection, dict), 'capture projection missing')
    _require(_nonempty_str(projection.get('module')),
             'capture projection module missing')
    _require(projection.get('tp_pre_reduce') is True,
             'capture projection must bind the local pre-TP output '
             '(tp_pre_reduce=true)')
    _require(projection.get('weight_layout') in WEIGHT_LAYOUTS,
             'capture projection weight_layout mismatch')
    for key, expected in PROJECTION_DTYPES.items():
        _require(type(projection.get(key)) is str
                 and projection.get(key) == expected,
                 f'capture projection {key} mismatch')
    files = binding.get('files')
    _require(isinstance(files, dict) and set(files) == {'weight'},
             'capture record must hold exactly the files.weight entry')
    weight = files['weight']
    _require(isinstance(weight, dict), 'capture weight entry must be an object')
    for key in ('file', 'dtype', 'sha256'):
        _require(_nonempty_str(weight.get(key)), f'capture weight {key} missing')
    _require(isinstance(weight.get('shape'), list) and len(weight['shape']) == 2
             and all(type(n) is int and n > 0 for n in weight['shape']),
             'capture weight shape mismatch')
    _require(weight['file'].endswith('.npy'),
             'capture weight file must be the declared .npy')
    _require(weight['dtype'] == 'bf16-le', 'capture weight dtype mismatch')
    _require(_sha256_hex(weight['sha256']), 'capture weight sha256 mismatch')
    resident = binding.get('resident')
    _require(isinstance(resident, dict), 'capture resident sub-record missing')
    for key in RESIDENT_KEYS:
        _require(_nonempty_str(resident.get(key)),
                 f'capture resident {key} missing')
    _require(_sha256_hex(resident['sha256']), 'capture resident sha256 mismatch')
    recipe = resident.get('recipe')
    _require(isinstance(recipe, dict), 'capture resident recipe missing')
    _require(_nonempty_str(recipe.get('name')),
             'capture resident recipe name missing')
    _require(isinstance(recipe.get('inputs'), dict),
             'capture resident recipe inputs missing')
    _require(isinstance(receipt, dict), 'capture GEMM receipt missing')
    # Candidate receipts nest the provider under `dispatch` (o_proj_receipt.rs);
    # reference receipts carry it at the top level. Accept either envelope
    # style, still fail-closed on a missing/empty provider identity.
    dispatch = receipt.get('dispatch')
    provider = receipt.get('provider')
    if not _nonempty_str(provider) and isinstance(dispatch, dict):
        provider = dispatch.get('provider')
    _require(_nonempty_str(provider), 'capture GEMM receipt provider missing')
    placement = receipt.get('placement')
    _require(isinstance(placement, dict),
             'capture GEMM receipt placement missing')
    _require(placement.get('tp_pre_reduce') is True
             and _nonempty_str(placement.get('capture_point')),
             'capture GEMM receipt must bind the pre-reduce placement')
    return binding


def read_capture_record(path, *, engine, rank, row, tokens):
    """Parse either capture envelope style and validate it fail-closed."""
    path = Path(path)
    _require(path.is_file(), f'capture record missing: {path}')
    try:
        envelope = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'capture record is not valid JSON: {path}') from error
    _require(isinstance(envelope, dict), f'capture record must be an object: {path}')
    binding = envelope.get('binding', envelope)
    receipt = envelope.get('receipt')
    try:
        validate_capture_record(binding, receipt, engine=engine, rank=rank,
                                row=row, tokens=tokens)
    except ValueError as error:
        raise ValueError(f'{path}: {error}') from error
    return {'binding': binding, 'receipt': receipt, 'capture_dir': str(path.parent)}


def assemble_binding_row(*, rank, row, engine, capture_path, fixture_row_dir,
                         own_output_bin, v24_path, out_dir):
    """Build one immutable `rank{r}-row{n}/` binding row from an engine
    capture record, the norm fixture row and the captured local output;
    finishes with the harness `validate_binding` as the fail-closed gate."""
    fixture, arrays = gated_norm_ablation.validate_fixture(
        Path(fixture_row_dir), row, rank)
    tokens = int(arrays['expected'].shape[0])
    capture = read_capture_record(capture_path, engine=engine, rank=rank,
                                  row=row, tokens=tokens)
    binding = json.loads(json.dumps(capture['binding']))
    capture_dir = Path(capture['capture_dir'])
    fixture_dir = Path(fixture_row_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    files = {}

    weight_entry = binding['files']['weight']
    weight_src = capture_dir / weight_entry['file']
    _require(weight_src.is_file(), f'captured weight missing: {weight_src}')
    _require(sha(weight_src) == weight_entry['sha256'],
             'captured weight hash mismatch')
    weight = np.load(weight_src, allow_pickle=False)
    _require(weight.dtype.str == '<u2' and weight.flags.c_contiguous,
             'weight must be contiguous little-endian BF16 bits')
    _require(list(weight.shape) == weight_entry['shape'],
             'captured weight geometry mismatch')
    _require(not np.any((weight & 0x7f80) == 0x7f80),
             'weight must contain finite BF16 values')
    shutil.copyfile(weight_src, out / 'weight.npy')
    files['weight'] = dict(weight_entry, file='weight.npy')

    layout = binding['projection']['weight_layout']
    out_features = int(weight.shape[0] if layout == 'out-in'
                       else weight.shape[1])
    own_src = Path(own_output_bin)
    _require(own_src.is_file(), f'own-output capture missing: {own_src}')
    raw = own_src.read_bytes()
    _require(len(raw) == tokens * out_features * 2,
             'own-output byte count disagrees with the bound geometry')
    own = np.frombuffer(raw, dtype='<u2').reshape(tokens, out_features)
    np.save(out / 'own-output.npy', own, allow_pickle=False)
    files['own-output'] = {'file': 'own-output.npy',
                           'shape': [tokens, out_features], 'dtype': 'bf16-le',
                           'sha256': sha(out / 'own-output.npy')}

    variants = (('candidate-original', fixture_dir / 'expected.npy', None),
                ('reference-exact', fixture_dir / 'reference-expected.npy', None),
                ('v24-corrected', Path(v24_path), arrays['expected'].shape))
    for name, source, required_shape in variants:
        _require(source.is_file(), f'{name} source missing: {source}')
        shutil.copyfile(source, out / (name + '.npy'))
        value = np.load(out / (name + '.npy'), allow_pickle=False)
        expected_shape = required_shape if required_shape is not None \
            else arrays['expected' if name == 'candidate-original'
                        else 'reference-expected'].shape
        _require(value.dtype.str == '<u2' and value.flags.c_contiguous
                 and list(value.shape) == list(expected_shape)
                 and not np.any((value & 0x7f80) == 0x7f80),
                 f'{name} must be the declared finite BF16 fixture array')
        files[name] = {'file': name + '.npy', 'shape': list(value.shape),
                       'dtype': 'bf16-le', 'sha256': sha(out / (name + '.npy'))}
    binding['files'] = files

    resident = binding['resident']
    if resident['file'] == weight_entry['file']:
        # Reference-style capture: the resident bytes ARE the weight file, so
        # the row keeps a single copy instead of duplicating the matrix.
        _require(resident['sha256'] == weight_entry['sha256'],
                 'resident and weight hashes disagree for a shared file')
        binding['resident'] = dict(resident, file='weight.npy')
    else:
        resident_src = capture_dir / resident['file']
        _require(resident_src.is_file(), f'resident capture missing: {resident_src}')
        _require(sha(resident_src) == resident['sha256'],
                 'resident capture hash mismatch')
        shutil.copyfile(resident_src, out / resident['file'])

    (out / 'binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    (out / 'receipt.json').write_text(
        json.dumps(capture['receipt'], indent=2) + '\n')
    # Final fail-closed gate: the assembled row must satisfy the harness.
    validate_binding(out, rank, row, engine, arrays)
    return binding
