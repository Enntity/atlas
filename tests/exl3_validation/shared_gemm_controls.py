# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit same-input component controls, separate from qualification."""


def independent_gemm_controls(input_bits, gate_up_bits, reference_activation_bits,
                              down_bits, libraries, gemm):
    if set(libraries) != {'candidate', 'reference'} or any(not p for p in libraries.values()):
        raise ValueError('explicit candidate and reference Lt library bindings required')
    arrays, metadata = {}, {}
    for label, library in libraries.items():
        for stage, activation, weight in [('gate_up', input_bits, gate_up_bits),
                                         ('independent_down', reference_activation_bits, down_bits)]:
            name = label + '_' + stage
            arrays[name], metadata[name] = gemm(activation, weight, library)
    return arrays, metadata


def reduction_controls(get_flag, set_flag, run):
    original = get_flag()
    try:
        result = {}
        for flag in (False, True):
            set_flag(flag)
            result[flag] = run(flag)
        return result
    finally:
        set_flag(original)
