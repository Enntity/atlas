# SPDX-License-Identifier: AGPL-3.0-only
"""Numerical boundary replay, not an emulation of CUDA reduction order."""
import numpy as np


def round_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('finite F32 values required')
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def normalize_collapse(collapse, weight, epsilon, *, square_sum):
    """Select the measured source contract explicitly; both round collapse BF16.

    ``unrounded`` measures the original F32 collapse; ``rounded`` measures
    the BF16 collapse emitted by a separate pre-mix kernel. NumPy reductions
    cannot prove equality to either installed CUDA reduction implementation.
    """
    collapse = np.asarray(collapse, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if (collapse.ndim != 2 or min(collapse.shape) <= 0
            or weight.shape != (collapse.shape[1],)
            or not np.isfinite(collapse).all() or not np.isfinite(weight).all()
            or not np.isfinite(epsilon) or epsilon <= 0
            or square_sum not in ('unrounded', 'rounded')):
        raise ValueError('explicit finite mHC normalization contract required')
    rounded = round_bf16(collapse)
    measure = collapse if square_sum == 'unrounded' else rounded
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        inverse = np.float32(1.) / np.sqrt(np.mean(measure * measure, axis=-1,
                                                  keepdims=True) + np.float32(epsilon))
        return round_bf16(rounded * inverse * round_bf16(weight))
