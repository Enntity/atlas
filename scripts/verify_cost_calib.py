#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Calibrate the expert-aware DFlash verify cost model (ATLAS_GLM_VERIFY_COST_MODEL).

The scheduler prices a verify step of `owners` sequences x `rows` rows as

    ms = c0 + c_owner*owners + c_row*owners*rows + c_exp*U(D) + c_lone2*[owners==1 and rows==2]
    U(D) = E * (1 - (1 - K/E)**D)          expected distinct routed experts per MoE layer
    D    = owners + a*novel + b*repeat      effective independent top-K draws

where `novel` / `repeat` count the rows past each owner's first whose token is
new to the step / repeats an earlier row's token (see
crates/spark-server/src/scheduler/verify_cost.rs). This script fits the seven
coefficients from per-step timings and prints the env line to deploy.

Subcommands
  load  URL KIND N TOKENS [OUT]  N concurrent greedy decodes of distinct prompts
                                 (KIND prose|code; URL .../v1/chat/completions);
                                 OUT saves the texts as JSON for equality diffs.
  fit   LOG [LOG...]             fit from serve logs of a sweep run
  table                          fit the built-in 2026-09-28 prose table (the
                                 defaults compiled into verify_cost.rs)

The sweep run needs, on top of the production profile:
  ATLAS_GLM_VERIFY_COST_SWEEP=1   round-robin the verify width 1..7 per step
  ATLAS_DFLASH_WIDTH_LOG=1        one `DFLASH VERIFY owners= rows= novel= repeat=` line per step
The step cost of line i is the time to line i+1 when both have the same owner
count (as experiments/2026-09-28-adaptive-width/wsweep.sh measured the table).
Pure Python, no dependencies.
"""
import json
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

EXPERTS, TOP_K = 288.0, 8.0

# Median scheduler step interval (ms), GLM-5.3 Flash TP2 GB10, prose,
# 2026-09-28, ATLAS_DFLASH_FIXED_WIDTH sweeps (dflash_width.rs step_ms).
# Rows 2..=8 per owner, owners 1..=8; None = over the 32-row verify budget.
TABLE = [
    [83.8, 73.5, 81.7, 88.7, 97.0, 104.1, 111.2],
    [93.2, 109.4, 123.1, 144.0, 155.3, 165.2, 175.9],
    [114.1, 141.0, 161.2, 176.3, 191.6, 202.7, 216.3],
    [132.1, 163.1, 186.2, 206.4, 222.1, 236.2, 248.4],
    [167.9, 189.5, 218.6, 241.4, 259.3, None, None],
    [184.0, 210.2, 240.4, 263.6, None, None, None],
    [197.8, 229.0, 262.6, None, None, None, None],
    [210.2, 246.1, 279.6, None, None, None, None],
]

PROMPTS = {
    "prose": [
        "Explain how a hash map works, in flowing prose, at length. No code, no lists.",
        "Describe the history of the printing press and its effect on Europe, in prose.",
        "Write a long reflective essay about a coastal town in winter. No lists.",
        "Explain how vaccines train the immune system, in plain flowing prose.",
        "Tell the story of a lighthouse keeper who finds a message in a bottle.",
        "Explain the causes of inflation to a curious teenager, in prose.",
        "Describe how a sourdough starter works and how bakers care for it.",
        "Write a travel essay about crossing a desert by train. No lists.",
    ],
    "code": [
        "Write a red-black tree in Python with insert, delete and rebalancing. Code only.",
        "Write a Rust LRU cache with a HashMap and a doubly linked list. Code only.",
        "Write a C function set implementing a growable ring buffer with tests. Code only.",
        "Write a TypeScript Express server with CRUD routes for todos. Code only.",
        "Write a Go worker pool with context cancellation and a demo main. Code only.",
        "Write a Python tokenizer and recursive-descent parser for arithmetic. Code only.",
        "Write a Java implementation of Dijkstra with a binary heap. Code only.",
        "Write a bash script that rotates and compresses log files safely. Code only.",
    ],
}


def load(url, kind, n, tokens, out=None):
    """N concurrent greedy decodes, one distinct prompt each (real serving is
    not N copies of one prompt: cross-owner expert overlap must be realistic).
    `out`: write the completions as JSON (greedy-equality diffs)."""

    def one(i):
        body = {
            "model": "glm-5.3-flash-atlas",
            "messages": [{"role": "user", "content": PROMPTS[kind][i % len(PROMPTS[kind])]}],
            "max_tokens": tokens,
            "min_tokens": tokens,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        }
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        return r["usage"]["completion_tokens"], r["choices"][0]["message"]["content"]

    t = time.time()
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(one, range(n)))
    toks = [r[0] for r in res]
    print(f"{kind} n={n} tokens={toks} {sum(toks) / (time.time() - t):.1f} tok/s aggregate")
    if out:
        json.dump([r[1] for r in res], open(out, "w"), indent=1)


ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(
    r"^(\S+)\s.*DFLASH VERIFY owners=(\d+) rows=(\d+)(?: novel=(\d+) repeat=(\d+))?"
)


def parse(paths):
    """Per-step samples (owners, rows, novel, repeat, ms) from serve logs."""
    steps = []
    for path in paths:
        prev = None
        for raw in open(path, errors="replace"):
            m = LINE.search(ANSI.sub("", raw))
            if not m:
                continue
            ts = datetime.fromisoformat(m.group(1).rstrip("Z"))
            owners, rows = int(m.group(2)), int(m.group(3))
            # Without the cost-model fields every row is taken as novel.
            novel = int(m.group(4)) if m.group(4) else owners * (rows - 1)
            repeat = int(m.group(5)) if m.group(5) else 0
            if prev and prev[1] == owners:
                steps.append((*prev[1:], (ts - prev[0]).total_seconds() * 1000.0))
            prev = (ts, owners, rows, novel, repeat)
    # Drop stalls (prefill interleave, request turnover): > 3x the cell median.
    cells = {}
    for s in steps:
        cells.setdefault(s[:2], []).append(s[4])
    med = {k: statistics.median(v) for k, v in cells.items()}
    return [s for s in steps if s[4] <= 3.0 * med[s[:2]]]


def table_samples():
    return [
        (o + 1, r + 2, (o + 1) * (r + 1), 0, ms)
        for o, row in enumerate(TABLE)
        for r, ms in enumerate(row)
        if ms is not None
    ]


def features(s, a, b):
    owners, rows, novel, repeat, _ = s
    d = owners + a * novel + b * repeat
    u = EXPERTS * (1.0 - (1.0 - TOP_K / EXPERTS) ** d)
    return [1.0, owners, owners * rows, u, 1.0 if (owners == 1 and rows == 2) else 0.0]


def solve(x, y):
    """Least squares by the normal equations (Gauss-Jordan, partial pivot);
    a column with no variation (e.g. no lone 2-row steps) is pinned to 0."""
    n = len(x[0])
    ata = [[sum(r[i] * r[j] for r in x) for j in range(n)] for i in range(n)]
    aty = [sum(r[i] * v for r, v in zip(x, y)) for i in range(n)]
    for i in range(n):
        if ata[i][i] == 0.0:
            ata[i][i], aty[i] = 1.0, 0.0
    m = [row + [v] for row, v in zip(ata, aty)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[c][c]:
                f = m[r][c] / m[c][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def nnls(x, y):
    """Least squares with the cost coefficients (all but c_lone2, a path
    quirk that may go either way) held non-negative: pin the most negative
    to zero and refit (a small active set; cost terms are few)."""
    pinned = set()
    while True:
        keep = [j for j in range(len(x[0])) if j not in pinned]
        sub = solve([[r[j] for j in keep] for r in x], y)
        c = [0.0] * len(x[0])
        for j, v in zip(keep, sub):
            c[j] = v
        neg = [j for j in range(4) if c[j] < 0.0]
        if not neg:
            return c
        pinned.add(min(neg, key=lambda j: c[j]))


def fit(samples):
    """Grid over (a, b), b <= a; non-negative least squares for the linear
    coefficients at each point; keep the smallest squared error."""
    grid = [i / 20 for i in range(1, 21)]
    y = [s[4] for s in samples]
    best = None
    for a in grid:
        for b in [0.0] + [g for g in grid if g <= a]:
            x = [features(s, a, b) for s in samples]
            c = nnls(x, y)
            sse = sum((sum(ci * xi for ci, xi in zip(c, r)) - v) ** 2 for r, v in zip(x, y))
            if best is None or sse < best[0] - 1e-9:
                best = (sse, a, b, c)
    return best


def report(samples):
    sse, a, b, c = fit(samples)
    rms = (sse / len(samples)) ** 0.5
    coeffs = [*c, a, b]
    print(f"samples={len(samples)} rms_ms={rms:.2f}")
    print("c0 c_owner c_row c_exp c_lone2 a b =", " ".join(f"{v:.4g}" for v in coeffs))
    print("ATLAS_GLM_VERIFY_COST_COEFFS=" + ",".join(f"{v:.4g}" for v in coeffs))
    cells = {}
    for s in samples:
        cells.setdefault(s[:2], []).append(s)
    print("owners rows  n  measured_ms  model_ms  mean_novel_frac")
    for (o, r), ss in sorted(cells.items()):
        meas = statistics.median(s[4] for s in ss)
        pred = statistics.median(
            sum(ci * xi for ci, xi in zip(c, features(s, a, b))) for s in ss
        )
        frac = statistics.mean(s[2] / max(1, s[2] + s[3]) for s in ss)
        print(f"{o:6d} {r:4d} {len(ss):3d} {meas:11.1f} {pred:9.1f} {frac:10.2f}")


def main(argv):
    if len(argv) >= 5 and argv[0] == "load":
        load(argv[1], argv[2], int(argv[3]), int(argv[4]), argv[5] if len(argv) > 5 else None)
    elif len(argv) >= 2 and argv[0] == "fit":
        report(parse(argv[1:]))
    elif argv[:1] == ["table"]:
        report(table_samples())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
