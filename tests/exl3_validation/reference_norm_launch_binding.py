# SPDX-License-Identifier: AGPL-3.0-only
"""Capture actual reference Triton launch and returned compilation artifacts.

Install on the actual installed ``layer_norm_gated_fwd_kernel`` Heuristics
object. ``context`` returns (layer, rank, invocation) only for marked calls.
I/O belongs to callbacks: write_blob(stage, bytes, ctx),
write_metadata(stage, JSON-compatible-dict, ctx), and read_source(path).
The installed run method receives original argument objects and the caller
receives its original result. This does not alter arithmetic or compile options.
"""
import dataclasses
import functools
import hashlib
import math
from reference_norm_binding import _identity

_FIELDS = ('eps', 'T', 'D', 'BD', 'BT', 'ACTIVATION', 'IS_RMS_NORM', 'num_warps')
_OPTIONAL_LAUNCH_FIELDS = ('H', 'g_stride_n', 'launch_pdl')


def _json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value): raise ValueError('Nonfinite launch metadata')
        return value
    if hasattr(value, '_asdict'): return _json(value._asdict())
    if dataclasses.is_dataclass(value): return _json(dataclasses.asdict(value))
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError('Compilation metadata keys must be strings')
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_json(v) for v in value]
    raise ValueError(f'Unbound compilation metadata value: {type(value)!r}')


def install_launch_binding(kernel, *, context, write_blob, write_metadata, read_source):
    """Wrap actual Heuristics.run, returning an explicit restore hook.

    Capture actual generated PTX/cubin, returned metadata including
    enable_fp_fusion, and all arithmetic-relevant launch kwargs. Each marked
    context has its own launch counter, distinguishing original and replay runs.
    Missing returned compilation artifacts fail the marked diagnostic.
    """
    if getattr(kernel, '_atlas_norm_launch_binding', None) is not None:
        raise ValueError('Normalization launch already instrumented')
    original = kernel.run
    had_instance_run = 'run' in vars(kernel)
    instance_run = vars(kernel).get('run')
    counts = {}

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        ctx = context()
        if ctx is None:
            return original(*args, **kwargs)
        if (not isinstance(ctx, tuple) or len(ctx) != 3 or
                any(type(v) is not int or v < 0 for v in ctx)):
            raise ValueError('Explicit layer/rank/invocation context required')
        if any(name not in kwargs for name in _FIELDS):
            raise ValueError('Actual reference norm launch kwargs missing required fields')
        launch = {name: _json(kwargs[name]) for name in _FIELDS + _OPTIONAL_LAUNCH_FIELDS
                  if name in kwargs}
        absent_launch_options = [name for name in _OPTIONAL_LAUNCH_FIELDS if name not in kwargs]
        # Bind optional compile controls only when actually passed. Effective
        # compiler settings come from the returned CompiledKernel metadata.
        optional = {name: _json(kwargs[name]) for name in
                    ('grid', 'warmup', 'enable_fp_fusion', 'num_stages', 'num_ctas',
                     'maxnreg', 'launch_cooperative_grid') if name in kwargs}
        index = counts.get(ctx, 0)
        counts[ctx] = index + 1
        stage = f'kda-norm-launch-{index:03d}'
        sources = _identity(kernel, read_source)
        result = original(*args, **kwargs)
        metadata = _json(getattr(result, 'metadata', None))
        if not isinstance(metadata, dict) or type(metadata.get('enable_fp_fusion')) is not bool:
            raise ValueError('Actual CompiledKernel enable_fp_fusion metadata required')
        name, compiler_hash = getattr(result, 'name', None), getattr(result, 'hash', None)
        if not isinstance(name, str) or not name or not isinstance(compiler_hash, str) or not compiler_hash:
            raise ValueError('Actual returned CompiledKernel name/hash required')
        asm = getattr(result, 'asm', None)
        if not isinstance(asm, dict):
            raise ValueError('Actual returned compiled artifact dictionary required')
        artifacts = {}
        for kind in ('ptx', 'cubin'):
            payload = asm.get(kind)
            if kind == 'ptx' and isinstance(payload, str): payload = payload.encode('utf-8')
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f'Actual compiled {kind} artifact required')
            blob_stage = stage + '-' + kind
            write_blob(blob_stage, payload, ctx)
            artifacts[kind] = {'stage': blob_stage, 'sha256': hashlib.sha256(payload).hexdigest(),
                               'bytes': len(payload)}
        write_metadata(stage, {'schema': 'reference-norm-compiled-launch-v1',
                              'launch_index': index, 'launch': launch,
                              'absent_launch_options': absent_launch_options,
                              'explicit_compile_options': optional,
                              'compiled_metadata': metadata, 'name': name,
                              'compiler_hash': compiler_hash, 'sources': sources,
                              'artifacts': artifacts}, ctx)
        return result

    kernel.run = wrapped
    kernel._atlas_norm_launch_binding = wrapped

    def restore():
        if kernel.run is not wrapped:
            raise ValueError('Reference launch method changed after instrumentation')
        if had_instance_run: kernel.run = instance_run
        else: del kernel.run
        del kernel._atlas_norm_launch_binding

    return restore
