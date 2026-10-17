# SPDX-License-Identifier: AGPL-3.0-only
"""Bind a resident reference router's routing stages and supported capture hook.

Diagnostics only: this module changes no routing arithmetic. Install on a
resident reference router (BaseRouter subclass) before its first
``_select_experts`` call. Stage-split rationale: the supported
``set_capture_fn`` hook fires inside ``_select_experts`` after
``_compute_routing`` and before EPLB mapping, so it owns the pre-EPLB logical
``topk_ids`` bytes AND proves callback engagement (zero callback counts fail
the plan's engagement check); the instance wrap of ``_compute_routing`` owns
the gate-logit and ``topk_weights`` snapshots without touching ID bytes, so
the three stages are non-redundant. The ``apply_exl3_experts``/physical-ID
boundary is deliberately NOT wrapped because with the recorded EPLB status
'disabled' or 'absent' logical IDs equal physical IDs; add that wrap only if
a live run records EPLB enabled. ``context`` must return
(layer, rank, invocation, request) for a marked request and None otherwise;
``context`` is treated as side-effect-free and all counters live here.
"""
import functools

from reference_norm_binding import _source_identity, _tensor_info

_GUARD = '_atlas_router_capture'


def _validated_context(context):
    """Return the marked 4-tuple context, None, or reject the value."""
    ctx = context()
    if ctx is None:
        return None
    if (not isinstance(ctx, tuple) or len(ctx) != 4 or
            any(type(v) is not int or v < 0 for v in ctx[:3]) or
            not isinstance(ctx[3], str) or not ctx[3]):
        raise ValueError('Explicit nonnegative layer/rank/invocation and '
                         'request context required')
    return ctx


def _bound_identity(target, read_source, name):
    """Bind a callable's installed source, converting read failures to ValueError."""
    try:
        return _source_identity(target, read_source)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f'Cannot bind installed source for {name}') from error


def _next_index(counts, ctx, family):
    key = (ctx, family)
    index = counts.get(key, 0)
    counts[key] = index + 1
    return index


class _RouterCaptureBinding:
    """Installed router-capture state exposing stats and an explicit restore."""

    def __init__(self, router, wrapper, had_instance, instance_value,
                 callback, stats):
        self._router = router
        self._wrapper = wrapper
        self._had_instance = had_instance
        self._instance_value = instance_value
        self._callback = callback
        self._stats = stats
        self._done = False

    def stats(self):
        return dict(self._stats)

    def restore(self):
        if self._done:
            return
        if self._router._compute_routing is not self._wrapper:
            raise ValueError('Router compute routing changed after instrumentation')
        if self._router.capture_fn is not self._callback:
            raise ValueError('Router capture callback changed after instrumentation')
        if self._had_instance:
            self._router._compute_routing = self._instance_value
        else:
            del self._router._compute_routing
        self._router.set_capture_fn(None)
        delattr(self._router, _GUARD)
        self._done = True


def install_router_capture(router, *, context, write_tensor, write_metadata,
                           read_source, source_targets):
    """Wrap router routing stages and register the supported capture hook.

    Install metadata records the concrete router class, the actual EPLB
    state status ('absent'/'disabled'/class name; logical IDs are physical
    only for absent/disabled) and the source identities of the bound router
    methods plus every explicit source target. The wrapper receives the
    original argument objects, snapshots gate logits BEFORE calling the
    original (the original may mutate its input in place), returns the
    original result object unchanged, and never writes topk_ids bytes — the
    supported capture callback owns that stage. An unknown/unbindable
    installed source, a pre-existing capture callback, a double install or a
    misbehaving ``set_capture_fn`` all fail the install closed.
    """
    if getattr(router, _GUARD, None) is not None:
        raise ValueError('Reference router already has a capture binding')
    if not isinstance(source_targets, dict) or not source_targets:
        raise ValueError('Explicit router/dispatch source targets are required')
    set_capture_fn = getattr(router, 'set_capture_fn', None)
    if not callable(set_capture_fn):
        raise ValueError('Reference router lacks callable set_capture_fn')
    original = getattr(router, '_compute_routing', None)
    if not callable(original):
        raise ValueError('Reference router lacks instance _compute_routing')
    if not callable(getattr(router, '_select_experts', None)):
        raise ValueError('Reference router lacks _select_experts')
    if not hasattr(router, 'capture_fn'):
        raise ValueError('Reference router lacks readable capture_fn')
    if router.capture_fn is not None:
        raise ValueError('Reference router capture_fn already registered')
    sources = {name: _bound_identity(target, read_source, name) for name, target in
               (('compute_routing', router._compute_routing),
                ('select_experts', router._select_experts),
                ('set_capture_fn', router.set_capture_fn))}
    targets = {name: _bound_identity(target, read_source, name)
               for name, target in source_targets.items()}
    if not hasattr(router, 'eplb_state'):
        eplb_status, physical = 'absent', True
    elif router.eplb_state is None:
        eplb_status, physical = 'disabled', True
    else:
        eplb_status, physical = type(router.eplb_state).__name__, False
    write_metadata('router-binding', {
        'class': {'module': type(router).__module__,
                  'qualname': type(router).__qualname__},
        'eplb_status': eplb_status,
        'logical_ids_are_physical': physical,
        'sources': sources,
        'source_targets': targets,
    }, None)

    stats = {'compute_routing_calls': 0, 'capture_callback_calls': 0,
             'gate_logits_written': 0, 'weights_written': 0, 'ids_written': 0}
    counts = {}

    def capture_callback(topk_ids):
        # Engagement counter: incremented on every callback, marked or not.
        stats['capture_callback_calls'] += 1
        ctx = _validated_context(context)
        if ctx is None:
            return
        clone = topk_ids.detach().clone()
        index = _next_index(counts, ctx, 'topk-ids')
        write_tensor('router-topk-ids-%03d' % index, clone, ctx)
        stats['ids_written'] += 1

    set_capture_fn(capture_callback)
    if router.capture_fn is not capture_callback:
        set_capture_fn(None)  # fail closed without leaving a zombie callback
        raise ValueError('Reference router did not retain the capture callback')

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        stats['compute_routing_calls'] += 1
        ctx = _validated_context(context)
        if ctx is None:
            return original(*args, **kwargs)
        if len(args) > 1:
            logits = args[1]
        elif 'router_logits' in kwargs:
            logits = kwargs['router_logits']
        else:
            raise ValueError('Marked routing requires router_logits')
        try:
            _tensor_info(logits)
        except (AttributeError, TypeError) as error:
            raise ValueError('Marked router_logits must expose tensor metadata') from error
        # Snapshot before the original call: it may mutate logits in place.
        clone = logits.detach().clone()
        index = _next_index(counts, ctx, 'gate-logits')
        write_tensor('router-gate-logits-%03d' % index, clone, ctx)
        stats['gate_logits_written'] += 1
        result = original(*args, **kwargs)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError('Routing must return (topk_weights, topk_ids)')
        weights = result[0].detach().clone()
        index = _next_index(counts, ctx, 'topk-weights')
        write_tensor('router-topk-weights-%03d' % index, weights, ctx)
        stats['weights_written'] += 1
        # Return the ORIGINAL result object unchanged; the capture hook owns IDs.
        return result

    had_instance = '_compute_routing' in vars(router)
    instance_value = vars(router).get('_compute_routing')
    router._compute_routing = wrapped
    binding = _RouterCaptureBinding(router, wrapped, had_instance,
                                    instance_value, capture_callback, stats)
    setattr(router, _GUARD, binding)
    return binding
