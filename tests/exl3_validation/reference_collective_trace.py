# SPDX-License-Identifier: AGPL-3.0-only
"""Bind resident reference collective dispatch and its selected backend calls.

Diagnostics only: this module changes no collective arithmetic or backend
selection. Install on a resident CudaCommunicator-like dispatch object
BEFORE its first collective call. Deliberate deviation from the norm
binding's zero-I/O-unmarked rule: collectives are process-global, so their
call-sequence completeness cannot be request-attributed; ALL dispatches
record an open record and ``ctx`` None identifies an unmarked call.
Each record binds op, monotonic sequence, context/request, first tensor
argument, group fields and stream, plus the backend events observed while
the dispatch frame was on the stack. After the dispatch original returns
normally, a record with ZERO backend events means an unknown branch
executed: the binding writes 'collective-coverage-violation' and fails
closed. A dispatch exception writes its record with dispatch_status
'raised' and re-raises the ORIGINAL exception unmasked. Backend methods
called outside any dispatch frame are recorded as standalone writes.
"""
import functools
import itertools
import math

from reference_norm_binding import _tensor_info
from reference_router_capture import _bound_identity, _validated_context

_GUARD = '_atlas_collective_trace'
_GROUP_FIELDS = ('unique_name', 'rank', 'world_size')


def _restore_attr(obj, name, had_instance, instance_value):
    if had_instance:
        setattr(obj, name, instance_value)
    else:
        delattr(obj, name)


def _first_tensor(args, kwargs):
    for value in itertools.chain(args, kwargs.values()):
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            info = _tensor_info(value)
            info['numel'] = math.prod(value.shape)
            return info
    return None


class _CollectiveTraceBinding:
    """Installed collective-trace state exposing stats and an explicit restore."""

    def __init__(self, stats, restore_fn, sources):
        self._stats = stats
        self._restore_fn = restore_fn
        self.sources = sources
        self._done = False

    def stats(self):
        return dict(self._stats)

    def restore(self):
        if self._done:
            return
        self._restore_fn()
        self._done = True


def install_collective_trace(dispatch, *, context, write_metadata, read_source,
                             source_targets, backends, stream_fn=None):
    """Wrap dispatch ops and backend methods; return an explicit restore hook.

    ``backends`` must be an iterable of (name, target, method_name) triples;
    the op is derived from the method-name suffix so the wiring is explicit
    (e.g. ('CUSTOM', ca_comm, 'custom_all_reduce')). Duplicate names, an
    unknown suffix, a target that is the dispatch itself, an unbindable
    installed source or a missing dispatch op all fail the install closed.
    """
    if getattr(dispatch, _GUARD, None) is not None:
        raise ValueError('Reference communicator already has a collective trace')
    if not isinstance(source_targets, dict) or not source_targets:
        raise ValueError('Explicit collective source targets are required')
    if stream_fn is not None and not callable(stream_fn):
        raise ValueError('Stream callback must be explicitly disabled or callable')
    dispatch_reduce = getattr(dispatch, 'all_reduce', None)
    dispatch_gather = getattr(dispatch, 'all_gather', None)
    if not callable(dispatch_reduce) or not callable(dispatch_gather):
        raise ValueError('Reference communicator must expose callable '
                         'all_reduce and all_gather')

    entries = []
    names = set()
    pairs = set()
    for entry in backends:
        try:
            name, target, method_name = entry
        except (TypeError, ValueError) as error:
            raise ValueError('Collective backends must be '
                             '(name, target, method_name) triples') from error
        if not isinstance(name, str) or not name or not isinstance(method_name, str):
            raise ValueError('Collective backend names and methods must be explicit strings')
        if name in names:
            raise ValueError(f'Duplicate collective backend name {name}')
        if method_name.endswith('all_reduce'):
            op = 'all_reduce'
        elif method_name.endswith('all_gather'):
            op = 'all_gather'
        else:
            raise ValueError(f'Unsupported collective backend method {method_name}')
        if target is dispatch:
            raise ValueError('Collective backend would double-wrap the dispatch')
        if (id(target), method_name) in pairs:
            raise ValueError(f'Collective backend method {method_name} registered twice')
        if not callable(getattr(target, method_name, None)):
            raise ValueError(f'Collective backend {name} lacks callable {method_name}')
        names.add(name)
        pairs.add((id(target), method_name))
        entries.append((name, target, method_name, op))

    sources = {'dispatch_all_reduce': _bound_identity(dispatch_reduce, read_source,
                                                      'dispatch.all_reduce'),
               'dispatch_all_gather': _bound_identity(dispatch_gather, read_source,
                                                      'dispatch.all_gather')}
    for name, target, method_name, op in entries:
        sources[f'backend:{name}'] = _bound_identity(getattr(target, method_name),
                                                     read_source, f'backend {name}')
    sources['source_targets'] = {name: _bound_identity(target, read_source, name)
                                 for name, target in source_targets.items()}

    stats = {'all_reduce_calls': 0, 'all_gather_calls': 0,
             'backend_calls': 0, 'standalone_backend_calls': 0}
    stack = []
    seq_counter = itertools.count()
    op_counters = {}
    standalone_counters = {}

    def backend_event(name, op, status):
        if stack:
            stats['backend_calls'] += 1
            stack[-1]['events'].append({'backend': name, 'op': op, 'status': status})
            return
        stats['standalone_backend_calls'] += 1
        index = standalone_counters.get(op, 0)
        standalone_counters[op] = index + 1
        write_metadata('collective-backend-%s-%03d' % (op, index),
                       {'backend': name, 'op': op, 'status': status,
                        'standalone': True}, None)

    def make_backend_wrapper(original, name, op):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
            except BaseException:
                backend_event(name, op, 'raised')
                raise
            backend_event(name, op, 'returned')
            return result
        return wrapped

    def make_dispatch_wrapper(original, op):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            ctx = _validated_context(context)
            stats[op + '_calls'] += 1
            seq = next(seq_counter)
            index = op_counters.get(op, 0)
            op_counters[op] = index + 1
            tensor = _first_tensor(args, kwargs)
            missing = [name for name in _GROUP_FIELDS
                       if not hasattr(dispatch, name)]
            record = {
                'op': op,
                'seq': seq,
                'ctx': ctx,
                'request': ctx[3] if ctx is not None else None,
                'tensor': tensor,
                'tensor_reason': None if tensor is not None else
                                  'no tensor argument exposed',
                'group': {name: getattr(dispatch, name, None)
                          for name in _GROUP_FIELDS},
                'group_reason': ('absent on dispatch: ' + ', '.join(missing))
                                if missing else None,
                'stream': stream_fn() if stream_fn is not None else None,
                'stream_reason': None if stream_fn is not None else
                                 'stream_fn not supplied',
            }
            frame = {'events': []}
            stack.append(frame)
            try:
                result = original(*args, **kwargs)
            except BaseException:
                stack.pop()
                record['dispatch_status'] = 'raised'
                record['backends'] = list(frame['events'])
                write_metadata('collective-%s-%03d' % (op, index), record, ctx)
                raise
            stack.pop()
            record['dispatch_status'] = 'returned'
            record['backends'] = list(frame['events'])
            if not record['backends']:
                write_metadata('collective-coverage-violation', record, ctx)
                raise ValueError(f'Unknown collective branch: {op} sequence {seq} '
                                 'executed no registered backend')
            write_metadata('collective-%s-%03d' % (op, index), record, ctx)
            return result
        return wrapped

    wrapped_reduce = make_dispatch_wrapper(dispatch_reduce, 'all_reduce')
    wrapped_gather = make_dispatch_wrapper(dispatch_gather, 'all_gather')
    originals = [(dispatch, 'all_reduce', dispatch_reduce, wrapped_reduce),
                 (dispatch, 'all_gather', dispatch_gather, wrapped_gather)]
    for name, target, method_name, op in entries:
        originals.append((target, method_name,
                          getattr(target, method_name),
                          make_backend_wrapper(getattr(target, method_name),
                                               name, op)))
    saved = [(target, method_name, original, wrapped,
              method_name in vars(target), vars(target).get(method_name))
             for target, method_name, original, wrapped in originals]
    applied = 0
    try:
        for target, method_name, original, wrapped in originals:
            setattr(target, method_name, wrapped)
            applied += 1
    except Exception as error:
        for target, method_name, original, wrapped, had, value in saved[:applied]:
            _restore_attr(target, method_name, had, value)
        raise ValueError('Collective instrumentation cannot wrap '
                         'communicator methods') from error

    def restore():
        for target, method_name, original, wrapped in originals:
            if getattr(target, method_name) is not wrapped:
                raise ValueError(f'Collective dispatch {method_name} changed '
                                 'after instrumentation')
        for target, method_name, original, wrapped, had, value in saved:
            _restore_attr(target, method_name, had, value)
        delattr(dispatch, _GUARD)

    binding = _CollectiveTraceBinding(stats, restore, sources)
    setattr(dispatch, _GUARD, binding)
    return binding
