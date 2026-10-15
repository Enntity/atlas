# SPDX-License-Identifier: AGPL-3.0-only
"""First-differing-tensor localization between two reference dump runs.

Reads two context-v4 dump directories (``ffn-L<l>-rank<r>-<inv>-<stage>``
JSON + bin pairs from ``reference_ffn_trace``/``reference_mhc_trace``), joins
every tensor to its request through the embedded ``chunk_position`` (never by
counters, which differ between FFN and mHC hooks), orders each request's
tensors in decoder dataflow order, and reports per (request, rank) the first
tensor whose bytes differ between the runs.

Dataflow order per layer L:
  mHC attention side (``pre`` at layer 0, else the first ``fused`` call):
    inputs, then outputs (residual-output, post, comb, layer-input)
  [attention: internals not captured]
  mHC FFN side (the second ``fused`` call): inputs (``x`` is the attention
    output), then outputs (``layer-input`` is the FFN input)
  FFN: input, router-logits, route-ids, route-weights, shared-output,
    shared-local-down, routed-weighted-f32, routed-output (rank-local, before
    the reduce; the fused kernel's FP32 output precedes apply()'s return),
    post-tp-output (after the reduce). Dense layers: input,
    gate-up, activation, dense-local-down, post-tp-output.

Self-checks, which fail the analysis instead of reporting a localization:
FFN ``input`` must equal the FFN-side mHC ``layer-input`` bytes, and every
request must carry the same tensor keys in both runs.

Usage:
    hsl_localize.py --a DUMP_A --b DUMP_B --rows ROWS_A.json ROWS_B.json --out F
ROWS_X.json maps collector row names to response ids (``{row: [id, n]}``).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

STEM = re.compile(r'^ffn-L(\d+)-rank(\d+)-(\d{6})-(.+)\.json$')
FFN_ORDER = ['input', 'router-logits', 'route-ids', 'route-weights', 'shared-output',
             'shared-local-down', 'routed-weighted-f32', 'routed-output',
             'gate-up', 'activation', 'dense-local-down', 'post-tp-output']
MHC_OUTPUTS = ['residual-output', 'post', 'comb', 'layer-input']
KDA_ORDER = ['kda-qkvbfg-input0', 'kda-qkvbfg', 'kda-g1-input0', 'kda-g1',
             'kda-g2-input0', 'kda-g2', 'kda-gated-norm-input0',
             'kda-gated-norm-input1', 'kda-gated-norm', 'kda-o-proj-input0',
             'kda-o-proj']
# MLA attention trace (reference_attn_trace): dataflow order inside the
# attention block. The indexer's top-k output precedes sparse attention
# (mla_attn_mla_attn, rank-local); the wrapper output mla_attn is
# o_proj(attn_out) after the row-parallel reduce (mla.py forward), so it
# follows o_proj.
ATTN_ORDER = ['attn-input-hidden_states', 'attn-input-positions',
              'attn-fused_qkv_a_proj.0', 'attn-q_b_proj.0', 'attn-indexer_wq_b.0',
              'attn-indexer_wk_weights_proj.0', 'attn-indexer_indexer_op',
              'attn-indexer', 'attn-mla_attn_mla_attn', 'attn-o_proj-input',
              'attn-o_proj.0', 'attn-mla_attn', 'attn-output']
LOCAL_STAGES = {'attn-indexer_indexer_op', 'attn-indexer', 'attn-mla_attn_mla_attn',
                'attn-o_proj-input',
                'shared-local-down', 'routed-output', 'routed-weighted-f32',
                'dense-local-down', 'gate-up', 'activation', 'shared-output'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def index(root, row_ids, ignore_unknown=False):
    """{(row, rank): {(chunk, layer, stage): (order, sha)}}.

    ``chunk`` is the prefill chunk's ``computed_before`` token offset (from
    ``chunk_position``), so multi-chunk long prompts keep one tensor per
    (chunk, layer, stage) and order chunk-major.
    """
    root = Path(root)
    rid_to_row = {v[0]: k for k, v in row_ids.items()}
    raw = {}
    for path in root.glob('ffn-L*-rank*-*.json'):
        m = STEM.match(path.name)
        if m is None:
            continue
        meta = json.loads(path.read_text())
        if 'file' not in meta or 'chunk_position' not in meta:
            continue  # runtime-semantics / policy / source receipts
        layer, rank, inv, stage = int(m[1]), int(m[2]), int(m[3]), m[4]
        rid = meta['chunk_position']['request_id']
        row = next((rid_to_row[x] for x in rid_to_row if rid.startswith(x)), None)
        if row is None and ignore_unknown:
            continue
        require(row is not None, 'dump tensor for an unknown request: ' + rid)
        digest = hashlib.sha256((root / meta['file']).read_bytes()).hexdigest()
        chunk = int(meta['chunk_position']['computed_before'])
        raw.setdefault((row, rank), []).append((chunk, layer, inv, stage, digest))
    out = {}
    for (row, rank), items in raw.items():
        # mHC calls per (layer, kind): the lowest invocation is the attention
        # side, except layer 0 where 'pre' is attention and 'fused' is FFN.
        fused_inv = {}
        for chunk, layer, inv, stage, _ in items:
            if stage.startswith('mhc-fused-'):
                fused_inv.setdefault((chunk, layer), set()).add(inv)
        keyed = {}
        for chunk, layer, inv, stage, digest in items:
            if stage.startswith('mhc-'):
                kind = stage.split('-')[1]
                if kind == 'pre':
                    side = 'attn'
                else:
                    invs = sorted(fused_inv[chunk, layer])
                    require(len(invs) in (1, 2), 'unexpected mHC fused call count at L%d' % layer)
                    side = 'ffn' if (len(invs) == 1 or inv == invs[1]) else 'attn'
                    if layer > 0 and len(invs) == 1:
                        raise ValueError('layer %d has one fused mHC call' % layer)
                name = stage.split('-', 2)[2]
                is_input = name.startswith('input-')
                part = name[len('input-'):] if is_input else name
                rank_in_group = (0, part) if is_input else (1, MHC_OUTPUTS.index(part))
                order = (layer, 0 if side == 'attn' else 1) + rank_in_group
                key = (chunk, layer, 'mhc-%s-%s' % (side, name))
            elif stage == 'upstream-embedding':
                order, key = (layer, -1, 0, ''), (chunk, layer, stage)
            elif stage.startswith(('kda-', 'attn-')):
                # Attention internals (KDA/attention trace dumps), when present.
                # Listed stages in dataflow order; other internals after them.
                listed = KDA_ORDER + ATTN_ORDER
                position = listed.index(stage) if stage in listed else len(listed)
                order, key = (layer, 0.5, position, stage), (chunk, layer, stage)
            else:
                require(stage in FFN_ORDER, 'unknown FFN stage ' + stage)
                order = (layer, 2, FFN_ORDER.index(stage), '')
                key = (chunk, layer, stage)
            order = (chunk,) + order
            require(key not in keyed, 'duplicate tensor %s for %s' % (key, row))
            keyed[key] = (order, digest)
        out[row, rank] = keyed
    return out


def self_check(keyed, label):
    for (row, rank), tensors in keyed.items():
        for (chunk, layer, stage), (_, digest) in tensors.items():
            if stage == 'input':
                twin = tensors.get((chunk, layer, 'mhc-ffn-layer-input'))
                if twin is not None:
                    require(twin[1] == digest, '%s %s rank%d L%d: FFN input differs from '
                            'mHC FFN-side layer-input (join is wrong)' % (label, row, rank, layer))


def localize(a, b):
    require(set(a) == set(b), 'runs cover different (row, rank) sets')
    results = {}
    for row, rank in sorted(a):
        ta, tb = a[row, rank], b[row, rank]
        require(set(ta) == set(tb), '%s rank%d: tensor key sets differ' % (row, rank))
        ordered = sorted(ta, key=lambda k: ta[k][0])
        diffs = [k for k in ordered if ta[k][1] != tb[k][1]]
        first = diffs[0] if diffs else None
        # (chunk, layer) of the first differing router logits / FFN input.
        first_gate = next(((k[0], k[1]) for k in ordered if k[2] == 'router-logits'
                           and ta[k][1] != tb[k][1]), None)
        first_input = next(((k[0], k[1]) for k in ordered if k[2] == 'input'
                            and ta[k][1] != tb[k][1]), None)
        if first_gate is not None:
            require(first_input is not None and first_input <= first_gate,
                    '%s rank%d: router logits differ before their input' % (row, rank))
        results['%s|rank%d' % (row, rank)] = {
            'row': row, 'rank': rank, 'tensors': len(ordered), 'differing': len(diffs),
            'first_chunk': first[0] if first else None,
            'first_layer': first[1] if first else None,
            'first_stage': first[2] if first else None,
            'first_is_rank_local': (first[2] in LOCAL_STAGES) if first else None,
            'first_ffn_input_layer': first_input[1] if first_input else None,
            'first_router_logits_layer': first_gate[1] if first_gate else None,
            'first_five': [list(k) for k in diffs[:5]]}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True, type=Path)
    parser.add_argument('--b', required=True, type=Path)
    parser.add_argument('--rows', required=True, nargs=2, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--ignore-unknown-requests', action='store_true',
                        help='skip dump tensors of requests absent from ROWS '
                             '(validation on mixed historical dumps only)')
    args = parser.parse_args()
    rows_a, rows_b = (json.loads(p.read_text()) for p in args.rows)
    a = index(args.a, rows_a, args.ignore_unknown_requests)
    b = index(args.b, rows_b, args.ignore_unknown_requests)
    self_check(a, 'A')
    self_check(b, 'B')
    results = localize(a, b)
    summary = {}
    for value in results.values():
        key = 'identical' if value['first_stage'] is None else value['first_stage']
        summary[key] = summary.get(key, 0) + 1
    with args.out.open('x') as target:
        target.write(json.dumps({'schema': 'atlas-hsl-localization-v1', 'a': str(args.a),
                                 'b': str(args.b), 'first_stage_counts': summary,
                                 'results': results}, indent=2) + '\n')
    print(json.dumps({'first_stage_counts': summary}, indent=2))


if __name__ == '__main__':
    main()
