# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in exact-byte reference MLA attention diagnostics (vLLM side).

Installed next to reference_ffn_trace.py in the reference hook package and
enabled by ATLAS_EXL3_ATTN_DUMP_LAYERS=<csv>. For each selected layer it records
every tensor output of every `self_attn` submodule that actually runs (so fused
paths simply produce fewer stages) plus the `o_proj` input, using the FFN
trace's sink and `(layer, rank, invocation)` context. Original outputs are
returned unchanged.
"""
import inspect
import os
import re
import threading

_PATTERN = re.compile(r'(?:^|\.)layers\.(\d+)\.self_attn$')
_COUNTERS = {}
_LOCK = threading.Lock()
_LOCAL = threading.local()


def _selected():
    raw = os.environ.get('ATLAS_EXL3_ATTN_DUMP_LAYERS')
    if raw is None:
        return set()
    if not raw or any(not x.isdigit() for x in raw.split(',')):
        raise RuntimeError('attention diagnostics require explicit nonnegative layer CSV')
    return {int(x) for x in raw.split(',')}


def _tensors(value):
    if hasattr(value, 'detach'):
        return [('', value)]
    if isinstance(value, (tuple, list)):
        return [(f'.{i}', v) for i, v in enumerate(value) if hasattr(v, 'detach')]
    return []


def patch(module, sink, *, rank):
    layers = _selected()
    if not layers:
        return
    cls = module.Glm5NextMLAAttention
    original = cls.__init__
    signature = inspect.signature(original)

    def context():
        return getattr(_LOCAL, 'context', None)

    def start(mod, args, kwargs):
        if not sink._enabled():
            return
        owner = rank()
        with _LOCK:
            invoke = _COUNTERS.get((mod._atlas_attn_layer, owner), 0)
            _COUNTERS[mod._atlas_attn_layer, owner] = invoke + 1
        _LOCAL.context = (mod._atlas_attn_layer, owner, invoke)
        bound = inspect.signature(mod.forward).bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            for suffix, tensor in _tensors(value):
                sink._write(f'attn-input-{name}{suffix}', tensor, _LOCAL.context)

    def finish(mod, args, kwargs, output):
        ctx = context()
        if ctx is not None:
            for suffix, tensor in _tensors(output):
                sink._write(f'attn-output{suffix}', tensor, ctx)
        _LOCAL.context = None

    def record(path):
        def hook(mod, args, kwargs, output):
            ctx = context()
            if ctx is not None:
                for suffix, tensor in _tensors(output):
                    sink._write(f'attn-{path}{suffix}', tensor, ctx)
        return hook

    def record_input(path):
        def hook(mod, args, kwargs):
            ctx = context()
            if ctx is not None and args:
                for suffix, tensor in _tensors(args[0]):
                    sink._write(f'attn-{path}-input{suffix}', tensor, ctx)
        return hook

    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original(self, *args, **kwargs)
        match = _PATTERN.search(bound.arguments.get('prefix', ''))
        if not match or int(match.group(1)) not in layers:
            return
        self._atlas_attn_layer = int(match.group(1))
        self.register_forward_pre_hook(start, with_kwargs=True)
        self.register_forward_hook(finish, with_kwargs=True, always_call=True)
        for path, sub in self.named_modules():
            if not path:
                continue
            safe = path.replace('.', '_')
            sub.register_forward_hook(record(safe), with_kwargs=True)
            if path.endswith('o_proj'):
                sub.register_forward_pre_hook(record_input(safe), with_kwargs=True)

    cls.__init__ = initialize
