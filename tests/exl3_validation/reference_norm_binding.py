# SPDX-License-Identifier: AGPL-3.0-only
"""Bind resident reference norm operands and selected CustomOp dispatch.

Install on ``self.o_norm`` after the KDA constructor, before its first call.
``context`` must return the enclosing KDA hook's (layer, rank, invocation),
only for a marked request, and None otherwise. Tensor/metadata/source I/O goes
through supplied router callbacks. ``source_targets`` must explicitly include
kernel and dispatch infrastructure objects relevant to the installed runtime;
this module never guesses an implementation from an extracted source file.

The selected method receives the original argument objects. Captures clone all
inputs before calling it because the CUDA implementation can return x itself.
This helper changes diagnostics only, not normalization arithmetic.
"""
import functools
import hashlib
import inspect
import math


_FIELDS = ('eps', 'activation', 'elementwise_affine', 'hidden_size', 'weight', 'bias')
_ARGUMENTS = {'x', 'g', 'residual', 'prenorm', 'residual_in_fp32'}


def _source_identity(target, read_source):
    try:
        path = inspect.getsourcefile(target)
    except (TypeError, OSError) as error:
        raise ValueError(f'Cannot bind installed source for {target!r}') from error
    if path is None:
        raise ValueError(f'Cannot bind installed source for {target!r}')
    source = read_source(path)
    if not isinstance(source, bytes) or not source:
        raise ValueError(f'Empty or non-byte installed source: {path}')
    code = getattr(getattr(target, '__func__', target), '__code__', None)
    return {'module': target.__module__, 'qualname': target.__qualname__,
            'source_file': path, 'source_sha256': hashlib.sha256(source).hexdigest(),
            'first_line': code.co_firstlineno if code is not None else None}


def _identity(target, read_source):
    """Bind explicit decorator/JIT chains and their ultimate Python callable.

    Triton JITFunction.fn can itself be another JITFunction. Preserve every
    callable wrapper's actual class source instead of selecting a guessed
    extracted kernel. Cycles, missing fn targets and uninspectable terminals
    reject the binding.
    """
    wrappers = []
    seen = set()
    while True:
        if id(target) in seen:
            raise ValueError('Installed source wrapper chain contains a cycle')
        seen.add(id(target))
        if inspect.isclass(target):
            terminal = _source_identity(target, read_source)
            break
        via = None
        if hasattr(target, '__wrapped__'):
            via = '__wrapped__'
        elif hasattr(target, 'fn'):
            via = 'fn'
        if via is None:
            if not (inspect.isfunction(target) or inspect.ismethod(target)):
                raise ValueError(f'Unbound installed callable source: {target!r}')
            terminal = _source_identity(target, read_source)
            break
        wrapper_source = target if inspect.isfunction(target) or inspect.ismethod(target) else type(target)
        wrappers.append({'via': via, 'identity': _source_identity(wrapper_source, read_source)})
        target = getattr(target, via)
        if target is None or not callable(target):
            raise ValueError('Installed source wrapper does not identify a callable target')
    if wrappers:
        terminal['wrappers'] = wrappers
    return terminal


def _tensor_info(value):
    if value is None:
        return None
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'device': str(value.device)}


def install_norm_binding(norm, *, context, write_tensor, write_metadata,
                         read_source, runtime_policy, source_targets, replay):
    """Wrap a resident CustomOp selected method; return an explicit restore hook.

    Metadata records the actual effective Python defaults from the selected
    method's signature, not independently maintained defaults. A missing runtime
    field, source binding or unexpected signature fails the marked capture.
    Sources are read only during marked calls. Tensor hashes/bytes belong to the
    tensor I/O router; metadata and tensor stages share the exact same context.
    """
    if replay is not None and not callable(replay):
        raise ValueError('Replay must be explicitly disabled or callable')
    if not isinstance(source_targets, dict) or not source_targets:
        raise ValueError('Explicit installed kernel/dispatch source targets are required')
    if getattr(norm, '_atlas_norm_binding', None) is not None:
        raise ValueError('Resident normalization already has a binding wrapper')
    original = norm._forward_method
    signature = inspect.signature(original)
    if set(signature.parameters) != _ARGUMENTS:
        raise ValueError(f'Unsupported selected norm signature: {signature}')

    @functools.wraps(original)
    def selected(*args, **kwargs):
        ctx = context()
        if ctx is None:
            return original(*args, **kwargs)
        if (not isinstance(ctx, tuple) or len(ctx) != 3 or
                any(type(item) is not int or item < 0 for item in ctx)):
            raise ValueError('Explicit nonnegative layer/rank/invocation context required')
        for field in _FIELDS:
            if not hasattr(norm, field):
                raise ValueError(f'Resident norm is missing runtime field {field}')
        if not isinstance(norm.eps, (int, float)) or not math.isfinite(norm.eps) or norm.eps <= 0:
            raise ValueError('Resident normalization epsilon must be finite and positive')
        if norm.activation not in ('swish', 'silu', 'sigmoid'):
            raise ValueError('Unsupported resident normalization activation')
        if type(norm.elementwise_affine) is not bool:
            raise ValueError('Resident elementwise_affine must be explicit boolean')
        if norm.elementwise_affine != (norm.weight is not None):
            raise ValueError('Resident affine flag and actual weight disagree')
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = bound.arguments
        if type(call['prenorm']) is not bool or type(call['residual_in_fp32']) is not bool:
            raise ValueError('Norm invocation flags must be explicit booleans')
        if call['x'] is None or call['g'] is None:
            raise ValueError('Normalization requires actual x and g tensors')
        policy = runtime_policy()
        if not isinstance(policy, dict) or not policy:
            raise ValueError('Explicit runtime arithmetic/compilation policy required')
        fallback = (original.__name__ == 'forward_native' and
                    (call['residual'] is not None or call['prenorm']))
        metadata = {
            'schema': 'reference-norm-runtime-binding-v1',
            'class': _identity(type(norm), read_source),
            'dispatch': _identity(original, read_source),
            'source_targets': {name: _identity(target, read_source)
                               for name, target in source_targets.items()},
            'eps': norm.eps, 'activation': norm.activation,
            'elementwise_affine': norm.elementwise_affine,
            'hidden_size': norm.hidden_size, 'bias_present': norm.bias is not None,
            'call': {'prenorm': call['prenorm'],
                     'residual_in_fp32': call['residual_in_fp32'],
                     'residual_present': call['residual'] is not None},
            'runtime_policy': policy,
            'tensors': {name: _tensor_info(value) for name, value in
                        (('x', call['x']), ('g', call['g']),
                         ('residual', call['residual']), ('weight', norm.weight),
                         ('bias', norm.bias))},
            'native_residual_fallback_requested': fallback,
            'fallback_dispatch': _identity(norm.forward_cuda, read_source) if fallback else None,
        }
        # Do this before dispatch: CUDA can alias output to x; a downstream
        # writer that delays serialization must still receive stable snapshots.
        snapshots = {name: value.detach().clone() for name, value in
                     (('input-x', call['x']), ('input-g', call['g']),
                      ('residual', call['residual']), ('weight', norm.weight),
                      ('bias', norm.bias)) if value is not None}
        write_metadata('kda-norm-runtime-binding', metadata, ctx)
        for name, snapshot in snapshots.items():
            write_tensor('kda-norm-bound-' + name, snapshot, ctx)
        output = original(*args, **kwargs)
        if call['prenorm']:
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError('Prenorm output must bind normalized and residual tensors')
            tensors = (('output', output[0]), ('residual-output', output[1]))
        else:
            tensors = (('output', output),)
        output_snapshots = {name: value.detach().clone() for name, value in tensors}
        for name, value in output_snapshots.items():
            write_tensor('kda-norm-bound-' + name, value, ctx)
        if replay is not None:
            replay(norm, original, snapshots, output_snapshots['output'], metadata,
                   context=ctx, write_tensor=write_tensor, write_metadata=write_metadata)
        return output

    norm._forward_method = selected
    norm._atlas_norm_binding = selected

    def restore():
        if norm._forward_method is not selected:
            raise ValueError('Selected normalization dispatch changed after instrumentation')
        norm._forward_method = original
        del norm._atlas_norm_binding

    return restore
