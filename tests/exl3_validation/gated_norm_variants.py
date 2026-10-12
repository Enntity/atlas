# SPDX-License-Identifier: AGPL-3.0-only
"""Derive single-feature gated-norm arithmetic ablation variants (pure Python).

Each variant is the frozen `kda_sigmoid_gated_rms_norm` body with exactly one
arithmetic feature changed toward the reference engine's compiled PTX
arithmetic, so a <=1-ULP cross-engine difference can be attributed per
feature. Transforms are fail-closed verbatim snippet replacements: a snippet
that is not found exactly once raises ValueError instead of producing a
silently divergent body. Composed probes (v24-minimal, v-all) change the
documented sets of features. The entry name and 7-argument ABI are unchanged
in every variant (separate compilation modules).
"""
import difflib
import hashlib
import re
from pathlib import Path

KERNEL_ENTRY = 'kda_sigmoid_gated_rms_norm'
_DERIVATION_FILE = 'gated_norm_variants.py'


class ExactReplace:
    """Fail-closed verbatim snippet replacement transform."""

    def __init__(self, old, new):
        self.old, self.new = old, new

    def __call__(self, body):
        if body.count(self.old) != 1:
            raise ValueError('variant transform snippet not found verbatim exactly once')
        return body.replace(self.old, self.new)


def _verbatim(body):
    return body


# v1's `i < dim` loop equals the frozen 128-slot tree only because
# validate_operands enforces dim == 128.
# v1-reduction-order: shared-memory halving tree -> serial sum-of-squares.
_REDUCTION_OLD = """    red[d] = x * x;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) red[d] += red[d + stride];
        __syncthreads();
    }"""
_REDUCTION_NEW = """    red[d] = x * x;
    __syncthreads();
    if (d == 0) {
        float sum = red[0];
        for (unsigned int i = 1; i < dim; ++i) sum += red[i];
        red[0] = sum;
    }
    __syncthreads();"""

# v2-rsqrt-sqrt-div: rsqrtf -> sqrt.approx.ftz.f32 + div.full.f32 reciprocal.
_RSQRT_OLD = "    if (d == 0) inv = rsqrtf(red[0] / (float)dim + eps);"
_RSQRT_NEW = """    if (d == 0) {
        float ms = red[0] / (float)dim + eps;
        float root;
        asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(root) : "f"(ms));
        float inv_reg;
        asm volatile("div.full.f32 %0, %1, %2;" : "=f"(inv_reg) : "f"(1.0f), "f"(root));
        inv = inv_reg;
    }"""

# v3/v4 share the sigmoid product line as their documented region.
_SIGMOID_OLD = "    float y = x * inv * (float)weight[d] / (1.0f + expf(-g));"
# v3-sigmoid-exp: Cody-Waite expf -> ex2.approx.f32 of (-g * log2e).
_SIGMOID_EXP_NEW = """    float e;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(-g * 1.4426950408889634f));
    float y = x * inv * (float)weight[d] / (1.0f + e);"""
# v4-sigmoid-reciprocal-mul: prod/(1+e) -> prod * (div.full.f32(1,1+e)).
_SIGMOID_RECIPROCAL_NEW = """    float den = 1.0f + expf(-g);
    float sig;
    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(sig) : "f"(1.0f), "f"(den));
    float y = x * inv * (float)weight[d] * sig;"""

# v24-minimal: composition of the two active features only (v2 + v4), the
# minimal-fix probe for analysis-20261012.md; v1/v3 proven output-inert.
_MIDSPAN = """
    __syncthreads();
    float g = (float)gate[off];
"""

# v-all: explicit composition of v1+v2+v3+v4 (they overlap one line, so the
# composed replacement spans reduction through the sigmoid product verbatim).
_COMPOSED_OLD = """    red[d] = x * x;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) red[d] += red[d + stride];
        __syncthreads();
    }
    if (d == 0) inv = rsqrtf(red[0] / (float)dim + eps);
    __syncthreads();
    float g = (float)gate[off];
    float y = x * inv * (float)weight[d] / (1.0f + expf(-g));"""
_COMPOSED_NEW = """    red[d] = x * x;
    __syncthreads();
    if (d == 0) {
        float sum = red[0];
        for (unsigned int i = 1; i < dim; ++i) sum += red[i];
        red[0] = sum;
    }
    __syncthreads();
    if (d == 0) {
        float ms = red[0] / (float)dim + eps;
        float root;
        asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(root) : "f"(ms));
        float inv_reg;
        asm volatile("div.full.f32 %0, %1, %2;" : "=f"(inv_reg) : "f"(1.0f), "f"(root));
        inv = inv_reg;
    }
    __syncthreads();
    float g = (float)gate[off];
    float e;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(-g * 1.4426950408889634f));
    float den = 1.0f + e;
    float sig;
    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(sig) : "f"(1.0f), "f"(den));
    float y = x * inv * (float)weight[d] * sig;"""

VARIANTS = {
    'v0-copy': _verbatim,
    'v1-reduction-order': ExactReplace(_REDUCTION_OLD, _REDUCTION_NEW),
    'v2-rsqrt-sqrt-div': ExactReplace(_RSQRT_OLD, _RSQRT_NEW),
    'v3-sigmoid-exp': ExactReplace(_SIGMOID_OLD, _SIGMOID_EXP_NEW),
    'v4-sigmoid-reciprocal-mul': ExactReplace(_SIGMOID_OLD, _SIGMOID_RECIPROCAL_NEW),
    'v24-minimal': ExactReplace(_RSQRT_OLD + _MIDSPAN + _SIGMOID_OLD,
                                _RSQRT_NEW + _MIDSPAN + _SIGMOID_RECIPROCAL_NEW),
    'v-all': ExactReplace(_COMPOSED_OLD, _COMPOSED_NEW),
}


def frozen_body(kda_cu_path):
    """Extract the exact kda_sigmoid_gated_rms_norm definition body."""
    text = Path(kda_cu_path).read_text()
    if len(re.findall(r'\b' + KERNEL_ENTRY + r'\b', text)) != 1:
        raise ValueError('frozen kernel entry must appear exactly once: ' + KERNEL_ENTRY)
    marker = 'extern "C" __global__ void ' + KERNEL_ENTRY + '('
    if text.count(marker) != 1:
        raise ValueError('frozen kernel entry definition must appear exactly once')
    start = text.index(marker)
    opening = text.index('{', start)
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == '{':
            depth += 1
        elif text[position] == '}':
            depth -= 1
            if depth == 0:
                return text[start:position + 1]
    raise ValueError('unterminated frozen kernel body: ' + KERNEL_ENTRY)


def render_variant(name, body):
    """Wrap a variant body as standalone .cu text for NVRTC compilation."""
    return ('// SPDX-License-Identifier: AGPL-3.0-only\n'
            '// Autogenerated by ' + _DERIVATION_FILE + ': variant ' + name
            + ' of ' + KERNEL_ENTRY + ' derived from frozen kda.cu.\n'
            '#include "atlas_pdl.cuh"\n'
            '#include <cuda_bf16.h>\n\n'
            + body + '\n')


def derive(name, kda_cu_path):
    """Return (source_text, unified_diff frozen->variant, source sha256)."""
    if name not in VARIANTS:
        raise ValueError('unknown gated-norm variant: ' + str(name))
    frozen = frozen_body(kda_cu_path)
    variant = VARIANTS[name](frozen)
    source = render_variant(name, variant)
    diff = ''.join(difflib.unified_diff(
        frozen.splitlines(keepends=True), variant.splitlines(keepends=True),
        fromfile='frozen/' + KERNEL_ENTRY, tofile='variant/' + name))
    return source, diff, hashlib.sha256(source.encode('utf-8')).hexdigest()
