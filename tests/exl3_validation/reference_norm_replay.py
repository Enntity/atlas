# SPDX-License-Identifier: AGPL-3.0-only
"""Online reference self replay through the actual selected resident method.

This is diagnostic only. Every trial owns fresh input/gate/constant clones;
parameter dictionaries are copied so trials cannot replace resident weights.
The supported marked invocation is affine norm without residual or prenorm.
I/O is supplied by the caller's tensor and metadata router.
"""
import copy


def _independent_norm(norm, snapshots, binding):
    replica = copy.copy(norm)
    # torch.nn.Module stores parameters in a mutable map. A shallow module copy
    # alone would replace the resident model's parameters during perturbations.
    if hasattr(norm, '_parameters'):
        object.__setattr__(replica, '_parameters', norm._parameters.copy())
    for name in ('weight', 'bias'):
        if name == 'weight' or binding['bias_present']:
            value = snapshots[name].detach().clone()
        else:
            value = None
        if hasattr(replica, '_parameters') and name in replica._parameters:
            replica._parameters[name] = value
        else:
            object.__setattr__(replica, name, value)
    replica.eps = binding['eps']
    return replica


def _equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if hasattr(left, 'untyped_storage'):
        # Numeric equal treats signed zeros as equal and NaNs as unequal.
        # Self reproduction must compare actual operand/output bytes instead.
        from torch import uint8
        return bool(left.detach().contiguous().view(uint8).equal(
            right.detach().contiguous().view(uint8)))
    return bool(left.equal(right))


def replay_reference_norm(norm, original, snapshots, captured, binding, *,
                          context, write_tensor, write_metadata):
    """Preserve baseline/control outputs and a self-reproduction receipt.

    Failed self reproduction is recorded and controls are skipped. Control
    engagement is recorded honestly: a zero perturbation that leaves output
    unchanged fails that control rather than being counted as proof.
    """
    flags = binding['call']
    if any(flags[name] is not False for name in
           ('prenorm', 'residual_present', 'residual_in_fp32')):
        raise ValueError('Reference norm replay supports no residual/prenorm path only')
    if binding['elementwise_affine'] is not True or 'weight' not in snapshots:
        raise ValueError('Reference norm replay requires bound affine weights')
    function = getattr(original, '__func__', None)
    if function is None:
        raise ValueError('Replay requires the actual selected bound Python method')
    output_snapshot = captured.detach().clone()
    receipt = {'schema': 'reference-norm-self-replay-v1',
               'selected_method': original.__module__ + '.' + original.__qualname__,
               'self_reproduction_exact': False, 'controls': {},
               'fresh_operand_clones_per_trial': True,
               'independent_parameter_map': True}

    def trial(name):
        replica = _independent_norm(norm, snapshots, binding)
        x = snapshots['input-x'].detach().clone()
        g = snapshots['input-g'].detach().clone()
        if name == 'input-zero':
            x.zero_()
        elif name == 'gate-zero':
            g.zero_()
        elif name == 'weight-zero':
            replica.weight.zero_()
        selected = function.__get__(replica, type(replica))
        result = selected(x, g, residual=None, prenorm=False, residual_in_fp32=False)
        stable = result.detach().clone()
        write_tensor('kda-norm-replay-' + name, stable, context)
        return stable

    baseline = trial('baseline')
    receipt['self_reproduction_exact'] = _equal(baseline, output_snapshot)
    if receipt['self_reproduction_exact']:
        for name in ('input-zero', 'gate-zero', 'weight-zero'):
            receipt['controls'][name] = not _equal(trial(name), baseline)
    write_metadata('kda-norm-self-replay', receipt, context)
    return receipt
