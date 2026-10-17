# SPDX-License-Identifier: AGPL-3.0-only
"""Byte classes of reference final-logit rows: campaign v2ref draws vs probe.

Reference-only and descriptive (registration step 4/5): for every
(suite, case, row) it groups the campaign's 15 reference draws and each
collected probe attempt by exact FP32 byte identity, then reports class
memberships, whether each probe row joins an existing campaign class, and
where the H3/H4 failing rows (v2ref14 short16/longer/r3, v2ref15
short16/code/r1) fall. It never scores candidates or applies gate rules.

Usage:
    probe_classes.py --out report.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EV = Path('/home/colum/Workspace/atlas-exl3-validation-20261010')
CAMPAIGN = EV / 'gate-v2-qualification-20261013-luna1' / 'attempts'
PROBE = EV / 'probe-runtime-mechanism-20261014'
REFS = ['v2ref%02d' % i for i in range(1, 16)]
TARGETS = [('short16', 'longer', 3, 'v2ref14'), ('short16', 'code', 1, 'v2ref15'),
           ('short16', 'longer', 0, None)]


def rows(npz_path):
    with np.load(npz_path) as z:
        return {case: [hashlib.sha256(np.ascontiguousarray(z[case][i]).tobytes())
                       .hexdigest() for i in range(z[case].shape[0])]
                for case in z.files}


def probe_sources():
    out = {}
    for attempt in ('off01', 'on01', 'on02', 'off02', 'off03', 'on03'):
        base = PROBE / 'attempts' / attempt / 'extracted' / 'gate' / 'attempts' / (
            'probe-rm-' + attempt)
        if (base / 'short16' / 'logits.npz').is_file():
            out[attempt] = base
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    sources = {name: CAMPAIGN / name / 'attempt-files' for name in REFS}
    sources.update(probe_sources())
    digests = {}
    for name, base in sources.items():
        for suite in ('short16', 'long'):
            path = base / suite / 'logits.npz'
            if path.is_file():
                for case, hashes in rows(path).items():
                    for index, value in enumerate(hashes):
                        digests.setdefault((suite, case, index), {})[name] = value
    table = []
    for (suite, case, index), members in sorted(digests.items()):
        classes = {}
        for name, value in members.items():
            classes.setdefault(value, []).append(name)
        ordered = sorted(classes.values(), key=lambda m: (-sum(n in REFS for n in m), m))
        label = {n: k for k, m in enumerate(ordered) for n in m}
        probe = {n: {'class': label[n],
                     'joins_campaign_class': any(x in REFS for x in ordered[label[n]])}
                 for n in members if n not in REFS}
        table.append({'suite': suite, 'case': case, 'row': index,
                      'classes': ordered, 'probe': probe})
    targets = []
    for suite, case, index, failing in TARGETS:
        entry = next((t for t in table if (t['suite'], t['case'], t['row'])
                      == (suite, case, index)), None)
        if entry is None:
            continue
        cls = None
        if failing is not None:
            cls = next(k for k, m in enumerate(entry['classes']) if failing in m)
        targets.append({'suite': suite, 'case': case, 'row': index,
                        'failing_draw': failing, 'failing_class': cls,
                        'failing_class_members': entry['classes'][cls] if cls is not None else None,
                        'classes': entry['classes'], 'probe': entry['probe']})
    summary = {}
    for name in sources:
        if name in REFS:
            continue
        joined = [t['probe'][name]['joins_campaign_class'] for t in table if name in t['probe']]
        summary[name] = {'rows': len(joined), 'rows_joining_a_campaign_class': sum(joined)}
    args.out.write_text(json.dumps({'schema': 'atlas-probe-byte-classes-v1',
                                    'sources': {k: str(v) for k, v in sources.items()},
                                    'summary': summary, 'targets': targets,
                                    'rows': table}, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'targets': [
        {k: t[k] for k in ('suite', 'case', 'row', 'failing_draw', 'failing_class')}
        | {'n_classes': len(t['classes']),
           'probe': {n: v['class'] for n, v in t['probe'].items()}} for t in targets]},
        indent=2))


if __name__ == '__main__':
    main()
