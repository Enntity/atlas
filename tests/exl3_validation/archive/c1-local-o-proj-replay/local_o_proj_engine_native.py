# SPDX-License-Identifier: AGPL-3.0-only
# ARCHIVED 2026-10-12 — DO NOT USE. The C1 BF16 common-space replay premise
# is invalid (see README.md in this directory). Importing or running this
# module fails unless ATLAS_ALLOW_ARCHIVED_C1=1 is set for forensic reading.
import os as _os
if _os.environ.get("ATLAS_ALLOW_ARCHIVED_C1") != "1":
    raise ImportError("archived C1 o_proj replay harness: premise invalid; see README.md")
"""Engine-native local o_proj replay drivers for experiment C1.

Two independent engine-native code paths behind the C1 self-anchor, plus the
supplemental `atlas-local-o-proj-algorithm-receipt-v1` derivation:

* candidate — replays the EXL3 packed-linear kernel sequence exactly as
  `Exl3LinearWorkspace::launch_bf16` (the `project_hot -> launch_local_bf16`
  dispatch) submits it: `exl3_input_hadamard_f16` -> `exl3_mul1_matmul_mma`
  -> `exl3_output_hadamard`, over the verbatim resident shard bytes carried
  by the binding row's `resident` sub-record. Kernel identity is the frozen
  candidate image's own embedded PTX, sliced from the image's `spark` binary
  at the offsets the frozen build manifest records and hash-attested to that
  manifest — the same attestation discipline as the reviewed Q2 native gated
  norm replay. No candidate rebuild and no recapture is required.
* reference — `torch.nn.functional.linear` over the captured BF16 weight and
  input, run where torch lives (the frozen
  `glm53-flash-mikeroz:dt-carry20` container) with the captured runtime
  policy applied explicitly (allow_tf32 off; PCND — no implicit defaults).

The `atlas-local-o-proj-binding-v1` rows stay immutable: algorithm receipts
are derived at replay time from the existing per-row capture receipts plus
container introspection and written into the replay output only.

Q2 scope (do not overclaim): literal same-weight cross-engine comparison is
impossible for this pair — the EXL3 trellis cannot represent the reference
BF16 matrix, and the bound weights differ at 6.3% of elements. The achievable
decomposition is the dequant-recipe delta (via the `resident` sub-record)
plus the GEMM-arithmetic delta at bound weights; engine-native cross entries
measure that combined delta at identical input bytes.
"""
import ctypes
import json
import os
from pathlib import Path

import numpy as np

from native_ptx import load_module, sha, validate_binding

CANDIDATE_KERNEL_SEQUENCE = ('exl3_input_hadamard_f16', 'exl3_mul1_matmul_mma',
                             'exl3_output_hadamard')
# Entry ABI contracts for the two embedded PTX modules the candidate o_proj
# dispatch loads; argument counts mirror the CUDA signatures in
# kernels/gb10/common/{exl3.cu,exl3_linear_mma.cu} (verified fail-closed
# against the actual PTX text by native_ptx.validate_binding).
EXL3_PTX_CONTRACT = {
    'exl3': {'entries': ('exl3_input_hadamard_f16', 'exl3_output_hadamard'),
             'argument_counts': {'exl3_input_hadamard_f16': 5,
                                 'exl3_output_hadamard': 5},
             'sources': ('exl3.cu', 'exl3_common.cuh')},
    'exl3_linear_mma': {'entries': ('exl3_mul1_matmul_mma',),
                        'argument_counts': {'exl3_mul1_matmul_mma': 13},
                        'sources': ('exl3_linear_mma.cu', 'exl3_mma.cuh',
                                    'exl3_common.cuh')},
}
RESIDENT_BUFFER_ORDER = ('trellis', 'input_scale', 'output_scale')
# The bound C1 layer-0 TP2 local o_proj shard geometry. Any other geometry
# fails closed: the numerics profile below is an explicit mirror of
# glm53_tp2_packed_numerics for THIS shard, never a silent default.
CANDIDATE_GEOMETRY = (4096, 4096, 6)  # (k, physical_n, bits)
GUARD = 64
GUARD_VALUE = np.uint16(0x5a5a)
RECEIPT_SCHEMA = 'atlas-local-o-proj-algorithm-receipt-v1'
# Row buckets of Exl3LinearNumerics::profile_for_rows (reference_min_rows=3,
# reference_pad=true) for the pinned (4096, 4096, 4096, 6) geometry:
# padded rows 3..=4 -> (32,256,48); 5..=8 -> (32,256,48); 9..=128 -> (32,128,48).
# Source: crates/spark-runtime/src/weights/exl3/native/numerics.rs +
# crates/spark-model/src/weight_loader/glm5/exl3_numerics.rs.
def pinned_row_profile(rows):
    """Bucket mirror of Exl3LinearNumerics::profile_for_rows with
    reference_min_rows=3 / reference_pad=true for the pinned geometry:
    padded rows 3..=4 and 5..=8 -> (32,256,48); rows >= 9 -> (32,128,48);
    rows 1..=2 pad up to the 3..=4 bucket."""
    if type(rows) is not int or rows <= 0 or rows > 65535:
        raise ValueError(f'unsupported engine-native row count: {rows}')
    padded = max(rows, 3)
    if padded <= 8:
        return (32, 256, 48)
    return (32, 128, 48)


def bf16_bits_to_f32(bits):
    if bits.dtype.str != '<u2' or not bits.flags.c_contiguous:
        raise ValueError('contiguous little-endian uint16 BF16 bits required')
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_bits_to_bf16(values):
    """Exact round trip of finite BF16-representable F32 values to BF16 bits."""
    if not np.isfinite(values).all():
        raise RuntimeError('engine-native replay produced nonfinite F32')
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    bits = (words >> np.uint32(16)).astype(np.uint16)
    if np.any((bits & 0x7f80) == 0x7f80):
        raise RuntimeError('engine-native replay produced nonfinite BF16')
    return bits


def candidate_numerics(rows, k, n, bits):
    """Explicit mirror of the pinned GLM-5.3 TP2 packed numerics for the bound
    o_proj shard; fail-closed on any geometry or row count outside it."""
    if (k, n, bits) != CANDIDATE_GEOMETRY:
        raise ValueError('unsupported engine-native o_proj geometry: '
                         f'{(k, n, bits)} != {CANDIDATE_GEOMETRY}')
    if k % 128 or n % 128:
        raise ValueError('EXL3 native dimensions must be H128 multiples')
    tile_k, tile_n, slices = pinned_row_profile(rows)
    return {'logical_n': n, 'logical_offset': 0, 'tile_k': tile_k,
            'tile_n': tile_n, 'slices': slices, 'half_partial': 0, 'bits': bits}


def split_resident(manifest, resident_bytes, k, n, bits):
    """Split the binding row's verbatim resident shard into the three device
    buffers `Exl3DeviceShard::buffers()` uploads, fail-closed on sizes."""
    buffers = manifest.get('resident', {}).get('buffers')
    if not isinstance(buffers, list) or [b.get('name') for b in buffers] != list(
            RESIDENT_BUFFER_ORDER):
        raise ValueError('resident buffers must be trellis/input_scale/output_scale')
    expected = [(k // 16) * (n // 16) * 32 * bits, k * 2, n * 2]
    if [b.get('bytes') for b in buffers] != expected:
        raise ValueError('resident buffer sizes differ from shard geometry: '
                         f'{[b.get("bytes") for b in buffers]} != {expected}')
    if len(resident_bytes) != sum(expected):
        raise ValueError('resident file size differs from recorded buffers')
    offset = 0
    views = {}
    for name, size in zip(RESIDENT_BUFFER_ORDER, expected):
        chunk = resident_bytes[offset:offset + size]
        views[name] = (np.frombuffer(chunk, dtype='<u4').copy()
                       if name == 'trellis'
                       else np.frombuffer(chunk, dtype='<u2').copy())
        offset += size
    return views


def launch_plan(k, n, rows, bits, numerics):
    """The exact three-kernel launch sequence of
    `Exl3LinearWorkspace::launch_output` for the cooperative F16-input
    o_proj path (grid/block/argument order mirror native.rs)."""
    return [
        {'entry': 'exl3_input_hadamard_f16', 'grid': [k // 128, rows, 1],
         'block': [128, 1, 1], 'args': ['input', 'input_scale', 'scratch_in',
                                        'rows', 'k']},
        {'entry': 'exl3_mul1_matmul_mma', 'grid': [(n + 127) // 128, rows, 1],
         'block': [128, 1, 1],
         'args': ['scratch_in', 'trellis', 'scratch_out', 'rows', 'n', 'k',
                  'bits', 'logical_n', 'logical_offset', 'tile_k', 'tile_n',
                  'slices', 'half_partial']},
        {'entry': 'exl3_output_hadamard', 'grid': [n // 128, rows, 1],
         'block': [128, 1, 1],
         'args': ['scratch_out', 'output_scale', 'output', 'rows', 'n']},
    ]


def replay_candidate_exl3(cp, kernels, resident, inp_bits, rows, k, n, bits,
                          numerics):
    """Replay the candidate's own EXL3 kernel sequence over the resident shard
    and captured input. `kernels` maps entry name -> callable(grid, block,
    args); guards prove operand immutability and finite output."""
    if rows <= 0 or rows > 0xFFFFFFFF // max(k, n):
        raise ValueError('unsupported engine-native row count')
    if inp_bits.shape != (rows, k) or inp_bits.dtype.str != '<u2':
        raise ValueError(f'input must be BF16 bits [{rows}, {k}]')
    if np.any((inp_bits & 0x7f80) == 0x7f80):
        raise ValueError('finite BF16 input required')
    u32 = np.uint32
    x = cp.asarray(inp_bits.reshape(-1), dtype=cp.uint16, copy=True)
    su = cp.asarray(resident['input_scale'], dtype=cp.uint16, copy=True)
    tre = cp.asarray(resident['trellis'], dtype=cp.uint32, copy=True)
    sv = cp.asarray(resident['output_scale'], dtype=cp.uint16, copy=True)
    snapshots = [cp.asnumpy(value).copy() for value in (x, su, tre, sv)]
    scratch_in = cp.zeros(rows * k, dtype=cp.float32)
    scratch_out = cp.zeros(rows * n, dtype=cp.float32)
    output = cp.full(rows * n + GUARD, GUARD_VALUE, dtype=cp.uint16)
    kernels['exl3_input_hadamard_f16'](
        (k // 128, rows), (128,),
        (x, su, scratch_in, np.uint32(rows), np.uint32(k)))
    kernels['exl3_mul1_matmul_mma'](
        ((n + 127) // 128, rows), (128,),
        (scratch_in, tre, scratch_out, np.uint32(rows), np.uint32(n),
         np.uint32(k), np.uint32(bits), np.uint32(numerics['logical_n']),
         np.uint32(numerics['logical_offset']), np.uint32(numerics['tile_k']),
         np.uint32(numerics['tile_n']), np.uint32(numerics['slices']),
         np.uint32(numerics['half_partial'])))
    kernels['exl3_output_hadamard'](
        (n // 128, rows), (128,),
        (scratch_out, sv, output[:rows * n], np.uint32(rows), np.uint32(n)))
    cp.cuda.get_current_stream().synchronize()
    result = cp.asnumpy(output)
    if not np.all(result[rows * n:] == GUARD_VALUE):
        raise RuntimeError('EXL3 output guard overwritten')
    for device, snapshot in zip((x, su, tre, sv), snapshots):
        if not np.array_equal(cp.asnumpy(device), snapshot):
            raise RuntimeError('EXL3 replay mutated an operand')
    bits_out = result[:rows * n].reshape(rows, n)
    if np.any((bits_out & 0x7f80) == 0x7f80):
        raise RuntimeError('EXL3 replay produced nonfinite BF16 output')
    return bits_out


def build_ptx_binding(manifest_path, spark_binary, common, image, out_dir):
    """Slice the two attested EXL3 PTX modules out of the frozen candidate
    image's `spark` binary at the frozen build-manifest offsets, verify every
    hash against the manifest, and write a native_ptx binding for them."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('schema') != 'atlas-exl3-build-manifest-v1':
        raise ValueError('unsupported build manifest schema')
    if not isinstance(image, str) or not image:
        raise ValueError('candidate image identity required')
    common = Path(common)
    binary_entry = manifest.get('binaries', {}).get('spark', {})
    binary_sha = binary_entry.get('sha256')
    if not isinstance(binary_sha, str) or len(binary_sha) != 64:
        raise ValueError('manifest must bind the spark binary sha256')
    blob = Path(spark_binary).read_bytes()
    if sha(spark_binary) != binary_sha:
        raise ValueError('spark binary differs from the frozen build manifest')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modules = {}
    for name, contract in EXL3_PTX_CONTRACT.items():
        entry = manifest.get('embedded_ptx', {}).get(name)
        if not isinstance(entry, dict):
            raise ValueError(f'manifest must embed PTX module {name}')
        offset = entry.get('embedded_byte_offsets', {}).get('spark')
        size = entry.get('bytes')
        if not isinstance(offset, int) or not isinstance(size, int):
            raise ValueError(f'manifest must record spark offset/bytes for {name}')
        piece = blob[offset:offset + size]
        if len(piece) != size:
            raise ValueError(f'truncated embedded PTX slice for {name}')
        import hashlib
        if hashlib.sha256(piece).hexdigest() != entry.get('sha256'):
            raise ValueError(f'embedded PTX bytes differ from manifest for {name}')
        (out_dir / entry['file']).write_bytes(piece)
        sources = {}
        for source in contract['sources']:
            sources[source] = sha(common / source)
        modules[name] = {'file': entry['file'], 'sha256': entry['sha256'],
                         'contained_in_binary': True,
                         'embedded_byte_offset': offset,
                         'entry_argument_counts': dict(
                             contract['argument_counts']),
                         'sources': sources}
    binding = {'image': image,
               'build_source_sha256': manifest['source_snapshot_sha256'],
               'embedded_binary_sha256': binary_sha,
               'modules': modules}
    binding_path = out_dir / 'binding.json'
    binding_path.write_text(json.dumps(binding, indent=2) + '\n')
    validate_binding(out_dir, binding_path, common, required=EXL3_PTX_CONTRACT)
    return binding, binding_path


def load_candidate_kernels(cp, ptx_dir, binding_path, common):
    """Load the two attested PTX modules and resolve the three o_proj entries."""
    validate_binding(ptx_dir, binding_path, common, required=EXL3_PTX_CONTRACT)
    modules = {name: load_module(cp, common, name, ptx_dir=ptx_dir,
                                 binding=binding_path)
               for name in EXL3_PTX_CONTRACT}
    return {entry: modules[module].get_function(entry)
            for module, contract in EXL3_PTX_CONTRACT.items()
            for entry in contract['entries']}


def bf16_bits_to_torch(torch, bits):
    """Exact BF16-bits -> torch.bfloat16 conversion (BF16 is a subset of F32)."""
    return torch.from_numpy(bf16_bits_to_f32(bits)).to(torch.bfloat16)


def torch_to_bf16_bits(tensor):
    values = tensor.detach().to('float32').cpu().numpy()
    return np.ascontiguousarray(f32_bits_to_bf16(values))


def project_reference_native(torch, layout, inp_bits, weight_bits,
                             blas_trace=None):
    """The reference engine's own GEMM: torch linear (cuBLAS) over the bound
    BF16 weight, with the captured runtime policy applied explicitly."""
    if layout != 'out-in':
        raise ValueError('engine-native reference replay requires out-in weight')
    rows, k = inp_bits.shape
    if weight_bits.ndim != 2 or weight_bits.shape[1] != k:
        raise ValueError('reference weight must be [out, in] matching the input')
    torch.backends.cuda.matmul.allow_tf32 = False
    x = bf16_bits_to_torch(torch, inp_bits)
    w = bf16_bits_to_torch(torch, weight_bits)

    def linear():
        with torch.no_grad():
            return torch.nn.functional.linear(x, w)

    result = linear() if blas_trace is None else blas_trace.run('linear', linear)
    torch.cuda.synchronize()
    return torch_to_bf16_bits(result)


def cublas_version():
    """Cheap in-process cuBLAS version probe; honest None when unobservable."""
    try:
        lib = ctypes.CDLL('libcublas.so.13')
        p, i = ctypes.c_void_p, ctypes.c_int
        lib.cublasCreate_v2.argtypes = [ctypes.POINTER(p)]
        lib.cublasCreate_v2.restype = i
        lib.cublasGetVersion_v2.argtypes = [p, ctypes.POINTER(i)]
        lib.cublasGetVersion_v2.restype = i
        lib.cublasDestroy_v2.argtypes = [p]
        lib.cublasDestroy_v2.restype = i
        handle = p()
        if lib.cublasCreate_v2(ctypes.byref(handle)) != 0:
            return None
        version = i()
        status = lib.cublasGetVersion_v2(handle, ctypes.byref(version))
        lib.cublasDestroy_v2(handle)
        return version.value if status == 0 else None
    except (OSError, AttributeError):
        return None


def mapped_libraries(names):
    """Hash the actual mapped BLAS libraries of this process (attestation of
    what this replay actually linked, not what it was supposed to link)."""
    import hashlib
    found = {}
    try:
        maps = Path('/proc/self/maps').read_text()
    except OSError:
        return found
    for line in maps.splitlines():
        path = line.split()[-1] if len(line.split()) >= 6 else ''
        if not path.startswith('/'):
            continue
        for name in names:
            if Path(path).name.startswith(name) and name not in found:
                try:
                    found[name] = {'path': path,
                                   'sha256': hashlib.sha256(
                                       Path(path).read_bytes()).hexdigest()}
                except OSError:
                    found[name] = {'path': path, 'sha256': None}
    return found


def reference_runtime_identity(torch):
    """Observable identity of the reference GEMM environment at replay time.
    Only cheaply observable facts are recorded; anything unavailable is an
    honest null, never a guess."""
    precision = getattr(torch, 'get_float32_matmul_precision', None)
    return {'torch': getattr(torch, '__version__', None),
            'cuda': getattr(getattr(torch, 'version', None), 'cuda', None),
            'cublas': cublas_version(),
            'mapped_libraries': mapped_libraries(
                ('libcublas.so', 'libcublasLt.so')),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'deterministic_algorithms': bool(
                getattr(torch, 'are_deterministic_algorithms_enabled',
                        lambda: False)()),
            'matmul_precision': precision() if callable(precision) else None}


def validate_row_receipt(row_dir, engine, rank, row, tokens):
    """Fail closed on the per-row capture receipt the algorithm receipt extends."""
    path = Path(row_dir) / 'receipt.json'
    if not path.is_file():
        raise ValueError('binding row capture receipt missing: ' + str(row_dir))
    receipt = json.loads(path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError('capture receipt must be an object: ' + str(row_dir))
    placement = receipt.get('placement')
    if not isinstance(placement, dict) or placement.get('tp_pre_reduce') is not True:
        raise ValueError('capture receipt must bind the local pre-TP output: '
                         + str(row_dir))
    if engine == 'candidate':
        dispatch = receipt.get('dispatch')
        if not isinstance(dispatch, dict) or dispatch.get('provider') != 'exl3-native':
            raise ValueError('candidate receipt must bind the exl3-native dispatch: '
                             + str(row_dir))
        if 'launch_local_bf16' not in str(dispatch.get('dispatch_path', '')):
            raise ValueError('candidate receipt dispatch path must name '
                             'launch_local_bf16: ' + str(row_dir))
        # The recorded scope limit: the exl3-native arm carries function-level
        # identity only. Attest it as recorded; never invent kernel handles.
        for key in ('kernel_handle', 'kernel_module', 'kernel_symbol'):
            if dispatch.get(key) is not None:
                raise ValueError('exl3-native dispatch must record null ' + key
                                 + ': ' + str(row_dir))
        if receipt.get('tokens') != tokens:
            raise ValueError('candidate receipt token mismatch: ' + str(row_dir))
        if not isinstance(receipt.get('weight_basis'), str) or not receipt['weight_basis']:
            raise ValueError('candidate receipt weight_basis missing: ' + str(row_dir))
    elif engine == 'reference':
        if receipt.get('provider') != 'vllm-Glm53PackedLinearMethod':
            raise ValueError('reference receipt must bind Glm53PackedLinearMethod: '
                             + str(row_dir))
        if not isinstance(receipt.get('quant_method'), str) or not receipt['quant_method']:
            raise ValueError('reference receipt quant_method missing: ' + str(row_dir))
        policy = receipt.get('runtime_policy')
        for key in ('torch', 'cuda', 'allow_tf32', 'deterministic_algorithms',
                    'compilation_mode'):
            if not isinstance(policy, dict) or key not in policy:
                raise ValueError('reference receipt runtime_policy missing ' + key
                                 + ': ' + str(row_dir))
    else:
        raise ValueError('unsupported engine: ' + str(engine))
    return receipt


def candidate_algorithm_receipt(row_receipt, binding, resident_sha, ptx_binding,
                                numerics):
    """Supplemental candidate algorithm receipt: the captured function-level
    dispatch attestation plus the replay's own kernel identity."""
    return {'schema': RECEIPT_SCHEMA, 'engine': 'candidate',
            'gemm_identity_scope': 'function-level dispatch '
            '(project_hot -> launch_local_bf16); kernel_handle/module/symbol '
            'are null by recorded design of the exl3-native arm',
            'captured_dispatch': row_receipt['dispatch'],
            'weight_basis': row_receipt['weight_basis'],
            'resident_sha256': resident_sha,
            'kernel_identity': {
                'provider': 'frozen-image-embedded-ptx-replay',
                'image': ptx_binding['image'],
                'embedded_binary_sha256': ptx_binding['embedded_binary_sha256'],
                'build_source_sha256': ptx_binding['build_source_sha256'],
                'modules': {name: {'file': entry['file'],
                                   'sha256': entry['sha256'],
                                   'embedded_byte_offset':
                                       entry['embedded_byte_offset']}
                            for name, entry in ptx_binding['modules'].items()},
                'entries': list(CANDIDATE_KERNEL_SEQUENCE)},
            'numerics': dict(numerics)}


def reference_algorithm_receipt(row_receipt, image, runtime, dispatch_observed):
    """Supplemental reference algorithm receipt: the captured quant_method
    identity plus the observable runtime identity at replay time."""
    return {'schema': RECEIPT_SCHEMA, 'engine': 'reference',
            'quant_method': row_receipt['quant_method'],
            'algorithm': row_receipt['algorithm'],
            'captured_runtime_policy': row_receipt['runtime_policy'],
            'image': image,
            'runtime': runtime,
            'dispatch_observed': dispatch_observed}
