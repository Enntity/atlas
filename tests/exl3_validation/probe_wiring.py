# SPDX-License-Identifier: AGPL-3.0-only
"""Probe observer glue: wire the frozen router/collective harnesses into the
resident reference process with buffered, engagement-counted telemetry.

Diagnostics only; this module changes no routing or collective arithmetic.
It implements the overlay-author obligations recorded in
``docs/design/exl3-runtime-mechanism-probe-registration.md``:

- ``context()`` is side-effect-free (it only reads forward state) and returns
  the ``(layer, rank, invocation, request)`` 4-tuple for a marked request,
  None otherwise; invocation advances only at forward entry.
- ``write_tensor``/``write_metadata`` clone to bytes immediately (no tensor
  references are retained) and BUFFER file I/O; the flush boundary is a
  model step (``execute_model`` exit), request transition, process exit,
  SIGTERM (registered by the sitecustomize), or a coverage violation (which forces an immediate flush before the frozen
  harness raises).
- The collective coverage-violation raise propagates unmasked: an attempt
  that hits it is failing, never retried.
- Install operations are buffered receipt records, so a process with no
  flush has no engagement evidence (an incomplete chain, by design).

Assumptions inherited from the frozen harnesses: one forward per process at
a time (the serial TP=2 / batch-1 comparison profile), and the collective
frame stack is not thread-safe. Per-request counters are valid within one
flush window; an analyzer sums them across a request's flush windows.
"""
import atexit
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import threading

FLUSH_SCHEMA = 'atlas-probe-flush-v1'
VIOLATION = 'collective-coverage-violation'
LAYER_PATTERN = re.compile(r'(?:^|\.)layers\.(\d+)\.mlp(?:\.|$)')
TENSOR_DTYPES = {'float16': ('f16-le', '<f2'),
                 'float32': ('f32-le', '<f4'),
                 'int32': ('i32-le', '<i4'),
                 'int64': ('i64-le', '<i8'),
                 'bfloat16': ('bf16-le', '<u2')}
ID_PREFIXES = (('router-gate-logits-', 'gate_logits_written'),
               ('router-topk-weights-', 'weights_written'),
               ('router-topk-ids-', 'ids_written'))
BACKEND_KEYS = {'name', 'attr', 'method', 'required'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path):
    return Path(path).read_bytes()


def tensor_bytes(value):
    """Byte snapshot of an already-cloned capture tensor; retains no tensor.

    ``bfloat16`` needs torch's view trick; every other dtype is a plain numpy
    conversion, so CPU contract tests exercise this without torch.
    """
    tensor = value.detach().contiguous()
    kind = str(tensor.dtype).removeprefix('torch.')
    require(kind in TENSOR_DTYPES, 'unsupported probe tensor dtype: ' + kind)
    label, numpy_dtype = TENSOR_DTYPES[kind]
    if kind == 'bfloat16':
        import torch
        raw = tensor.view(torch.uint16).cpu().numpy()
    else:
        raw = tensor.cpu().numpy()
    raw = raw.astype(numpy_dtype, copy=False)
    return raw.tobytes(), label, list(tensor.shape)


class ProbeState:
    """Marked-request state, buffered writers and the flush boundary."""

    def __init__(self, out_dir, *, register_atexit=True):
        self.out_dir = Path(out_dir)
        self._lock = threading.RLock()
        self._mark = None
        self._current = None
        self._counters = {}
        self._records = []
        self._flushes = 0
        self._stats = {}
        self.absent_backends = []
        self.install_receipts = []
        self._flushed_installs = 0
        self._flushed_absent = 0
        if register_atexit:
            atexit.register(self.close)

    # --- marked request and forward state (context reads only) -------------

    def mark_request(self, request_id):
        """Serial-profile marking: flush the previous request at transition."""
        require(isinstance(request_id, str) and request_id,
                'probe mark requires a nonempty request id')
        with self._lock:
            if self._mark == request_id:
                return
            if self._mark is not None:
                self._flush()
            self._mark = request_id

    def unmark_request(self):
        """Empty prepare (no active request): flush and clear the mark."""
        with self._lock:
            if self._mark is not None:
                self._flush()
            self._mark = None

    def begin_forward(self, layer, rank):
        with self._lock:
            require(isinstance(layer, int) and layer >= 0
                    and isinstance(rank, int) and rank >= 0,
                    'probe forward requires explicit layer/rank')
            if self._mark is None:
                self._current = None
                return
            index = self._counters.get((layer, rank), 0)
            self._counters[layer, rank] = index + 1
            self._current = {'layer': layer, 'rank': rank,
                             'invocation': index, 'request': self._mark}

    def end_forward(self):
        with self._lock:
            self._current = None

    def router_context(self, layer, rank):
        def context():
            with self._lock:
                if self._current is None:
                    return None
                if self._current['layer'] != layer or self._current['rank'] != rank:
                    raise ValueError('probe forward context does not match the '
                                     'instrumented router instance')
                return (layer, rank, self._current['invocation'],
                        self._current['request'])
        return context

    def collective_context(self):
        def context():
            with self._lock:
                if self._current is None:
                    return None
                return (self._current['layer'], self._current['rank'],
                        self._current['invocation'], self._current['request'])
        return context

    # --- buffered writers (called inside routing/dispatch frames) -----------

    def write_tensor(self, name, tensor, ctx):
        payload, dtype, shape = tensor_bytes(tensor)
        require(isinstance(name, str) and name and '/' not in name,
                'probe tensor stage name must be a plain filename stem')
        with self._lock:
            self._records.append({'kind': 'tensor', 'name': name, 'dtype': dtype,
                                  'shape': shape, 'ctx': list(ctx),
                                  'request': ctx[3], '_bytes': payload})

    def write_metadata(self, name, record, ctx):
        require(isinstance(name, str) and name and '/' not in name,
                'probe metadata stage name must be a plain filename stem')
        with self._lock:
            self._records.append({'kind': 'metadata', 'name': name,
                                  'record': copy.deepcopy(record),
                                  'ctx': list(ctx) if ctx is not None else None,
                                  'request': ctx[3] if ctx is not None else None})
            if name == VIOLATION:
                # Obligation: the failing branch's evidence hits disk before
                # the frozen harness raises out of the dispatch frame.
                self._flush()

    def register_stats(self, name, provider):
        with self._lock:
            key = name
            suffix = 1
            while key in self._stats:
                suffix += 1
                key = '%s-%d' % (name, suffix)
            self._stats[key] = provider
            return key

    def record_install(self, kind, details):
        with self._lock:
            self.install_receipts.append({'kind': kind, **details})

    def record_absent_backend(self, spec, reason):
        with self._lock:
            self.absent_backends.append({'name': spec['name'],
                                         'attr': spec['attr'],
                                         'method': spec['method'],
                                         'required': spec['required'],
                                         'reason': reason})

    # --- flush --------------------------------------------------------------

    @staticmethod
    def counters(records):
        per_request = {}
        for record in records:
            if record['kind'] != 'tensor' or record['request'] is None:
                continue
            bucket = per_request.setdefault(
                record['request'],
                {label: 0 for _, label in ID_PREFIXES})
            for prefix, label in ID_PREFIXES:
                if record['name'].startswith(prefix):
                    bucket[label] += 1
        return per_request

    def _flush(self):
        if not self._records:
            return
        self._flushes += 1
        window = list(self._records)
        self._records = []
        # Every Python process in a rank container imports the probe (API
        # server, engine core, workers) and they share one telemetry dir:
        # key by the flushing pid, read at flush time so forks diverge.
        root = self.out_dir / ('pid-%d' % os.getpid()) / ('flush-%06d' % self._flushes)
        root.mkdir(parents=True, exist_ok=False)
        serialized = []
        for position, record in enumerate(window):
            entry = {key: value for key, value in record.items() if key != '_bytes'}
            if record['kind'] == 'tensor':
                # Stage counters are per (context, stage), so every routed
                # layer in one step reuses e.g. router-gate-logits-000: the
                # window position keeps file names unique; ctx says which.
                entry['file'] = '%06d-%s.bin' % (position, record['name'])
                with (root / entry['file']).open('xb') as target:
                    target.write(record['_bytes'])
            serialized.append(entry)
        payload = {'schema': FLUSH_SCHEMA, 'flush': self._flushes,
                   'counters': self.counters(window),
                   'cumulative_stats': {name: provider()
                                        for name, provider in self._stats.items()}}
        if len(self.install_receipts) > self._flushed_installs:
            payload['install_receipts'] = list(self.install_receipts)
            self._flushed_installs = len(self.install_receipts)
        if len(self.absent_backends) > self._flushed_absent:
            payload['absent_backends'] = list(self.absent_backends)
            self._flushed_absent = len(self.absent_backends)
        payload['records'] = serialized
        with (root / 'records.json').open('x') as target:
            target.write(json.dumps(payload, indent=2, allow_nan=False) + '\n')

    def flush(self):
        with self._lock:
            self._flush()

    def close(self):
        with self._lock:
            self._flush()


# --- reference-process patching ---------------------------------------------

def resolve_targets(spec):
    """Resolve explicit ``label: 'module:attribute'`` source-target entries.

    The frozen harnesses bind installed source identity via ``__module__`` /
    ``__qualname__``, so targets must be functions/classes/methods, never
    bare modules.
    """
    require(isinstance(spec, dict) and spec, 'explicit source-target map required')
    resolved = {}
    for label, reference in spec.items():
        require(isinstance(reference, str) and ':' in reference,
                'source target must be module:attribute: ' + str(label))
        module_name, _, attribute = reference.partition(':')
        require(module_name and attribute,
                'source target must be module:attribute: ' + str(label))
        target = importlib.import_module(module_name)
        for part in attribute.split('.'):
            require(part, 'source target attribute must be nonempty: ' + label)
            target = getattr(target, part)
        resolved[label] = target
    return resolved


def preflight_source(module, expected_sha):
    actual = sha_file(module.__file__)
    require(actual == expected_sha,
            'installed source differs from registration: ' + module.__name__)
    return {'module': module.__name__, 'file': str(module.__file__),
            'sha256': actual}


def _resident_routers(target):
    """Routers reachable from a MoE module, by interface, deduplicated.

    The pinned reference router (``FusedMoERouter``, an ABC) is not an
    ``nn.Module``: ``FusedMoEFactory`` returns a ``MoERunner`` module that
    holds it as a plain ``router`` attribute, so ``named_modules()`` alone
    never yields it. Check each module and its ``router`` attribute.
    """
    found, seen = [], set()
    for name, child in target.named_modules():
        attr_name = name + '.router' if name else 'router'
        for label, candidate in ((name, child),
                                 (attr_name, getattr(child, 'router', None))):
            if candidate is None or id(candidate) in seen:
                continue
            if callable(getattr(candidate, 'set_capture_fn', None)) \
                    and callable(getattr(candidate, '_select_experts', None)):
                seen.add(id(candidate))
                found.append((label, candidate))
    return found


def patch_moe(module, state, source_targets, expected_sha):
    """Bind layer/rank forward state and the resident router per MoE module.

    Layer identity comes from the construction prefix (the same
    ``layers.<n>.mlp`` contract the registered FFN overlay uses); instances
    whose prefix does not match stay uninstrumented (registered-overlay
    precedent) and engagement counters catch a total miss. The router is
    discovered on first forward by interface (``set_capture_fn`` +
    ``_select_experts``) and must be unique, else the attempt fails closed.
    """
    from reference_router_capture import install_router_capture
    preflight_source(module, expected_sha)
    cls = module.Glm5NextMoE
    original_init = cls.__init__
    signature = inspect.signature(original_init)

    def start(target, args, kwargs):
        if getattr(target, '_atlas_probe_router_bound', False):
            state.begin_forward(target._atlas_probe_layer, target.tp_rank)
            return
        candidates = _resident_routers(target)
        require(len(candidates) == 1,
                'probe requires exactly one resident router per MoE '
                'module, found ' + str(len(candidates)))
        router_name, router = candidates[0]
        binding = install_router_capture(
            router,
            context=state.router_context(target._atlas_probe_layer,
                                         target.tp_rank),
            write_tensor=state.write_tensor,
            write_metadata=state.write_metadata,
            read_source=read_source,
            source_targets=resolve_targets(source_targets))
        state.record_install('router', {
            'layer': target._atlas_probe_layer, 'rank': target.tp_rank,
            'router_name': router_name,
            'router_class': type(router).__qualname__,
            'router_module': type(router).__module__,
            'stats_key': state.register_stats(
                'router-L%d' % target._atlas_probe_layer, binding.stats)})
        target._atlas_probe_router_bound = True
        state.begin_forward(target._atlas_probe_layer, target.tp_rank)

    def finish(target, args, kwargs, output):
        state.end_forward()

    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original_init(self, *args, **kwargs)
        prefix = bound.arguments.get('prefix', '')
        match = LAYER_PATTERN.search(prefix) if prefix.endswith('.mlp') else None
        if match is None:
            return
        self._atlas_probe_layer = int(match.group(1))
        self.register_forward_pre_hook(start, with_kwargs=True)
        self.register_forward_hook(finish, with_kwargs=True, always_call=True)

    cls.__init__ = initialize
    return {'class': cls.__qualname__, 'layer_pattern': LAYER_PATTERN.pattern}


def patch_runner(module, state, expected_sha):
    """Mark the serial request per model step; flush at every step boundary.

    The seam is ``GPUModelRunner.execute_model``, not ``prepare_inputs``: the
    chained campaign observer (lineage chunk trace) owns ``prepare_inputs``
    and fails capture if that attribute is rewrapped after its install.
    Every step flushes on exit, so the last request's telemetry reaches disk
    even when a forked worker exits via ``os._exit`` (no atexit) or the
    executor replaces the SIGTERM handler.
    """
    from reference_router_capture import _bound_identity
    source = preflight_source(module, expected_sha)
    original = module.GPUModelRunner.execute_model
    signature = inspect.signature(original)
    if list(signature.parameters)[:4] != \
            ['self', 'scheduler_output', 'intermediate_tensors', 'dummy_run']:
        raise ValueError('installed v2 execute_model signature differs from '
                         'the registered serial profile')
    identity = _bound_identity(original, read_source,
                               'GPUModelRunner.execute_model')

    def execute(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        scheduler_output = bound.arguments['scheduler_output']
        scheduled = getattr(scheduler_output, 'num_scheduled_tokens', None) or {}
        if bound.arguments['dummy_run'] or not scheduled:
            state.unmark_request()
            return original(*bound.args, **bound.kwargs)
        if len(scheduled) != 1:
            raise ValueError('probe requires exactly one scheduled serial '
                             'request per step (concurrency is outside the '
                             'registered diagnostic profile)')
        state.mark_request(next(iter(scheduled)))
        try:
            return original(*bound.args, **bound.kwargs)
        finally:
            state.flush()

    module.GPUModelRunner.execute_model = execute
    receipt = {'kind': 'runner', 'source': source, 'execute_identity': identity,
               'flush_boundary': 'every execute_model step'}
    state.record_install('runner', receipt)
    return receipt


def patch_communicator(module, state, backends, source_targets, expected_sha):
    """Install the frozen collective trace on each resident communicator."""
    from reference_collective_trace import install_collective_trace
    preflight_source(module, expected_sha)
    require(isinstance(backends, list), 'backends must be an explicit list')
    for spec in backends:
        require(set(spec) == BACKEND_KEYS
                and isinstance(spec['name'], str) and spec['name']
                and isinstance(spec['attr'], str) and spec['attr']
                and isinstance(spec['method'], str) and spec['method']
                and isinstance(spec['required'], bool),
                'backend specs need {name, attr, method, required}: '
                + str(spec))
    cls = module.CudaCommunicator
    original_init = cls.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if getattr(self, '_atlas_probe_dispatch_bound', False):
            return
        resolved = []
        for spec in backends:
            target = getattr(self, spec['attr'], None)
            if target is None or not callable(getattr(target, spec['method'], None)):
                state.record_absent_backend(
                    spec, 'attribute or method missing on resident communicator')
                require(spec['required'] is False,
                        'required collective backend absent: ' + spec['name'])
                continue
            resolved.append((spec['name'], target, spec['method']))
        binding = install_collective_trace(
            self, context=state.collective_context(),
            write_metadata=state.write_metadata,
            read_source=read_source,
            source_targets=resolve_targets(source_targets),
            backends=resolved)
        group = {field: str(getattr(self, field, None))
                 for field in ('unique_name', 'rank', 'world_size')}
        state.record_install('dispatch', {
            'group': group,
            'backends': [name for name, _, _ in resolved],
            'stats_key': state.register_stats('dispatch', binding.stats)})
        self._atlas_probe_dispatch_bound = True

    cls.__init__ = initialize
    return {'class': cls.__qualname__}
