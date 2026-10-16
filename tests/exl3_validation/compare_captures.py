# SPDX-License-Identifier: AGPL-3.0-only
"""Byte-compare per-step logits dumps of two bench_tp2 --capture runs (rank 0)."""
import hashlib, subprocess, sys
a, b = sys.argv[1], sys.argv[2]
cmd = "cd {0}/rank0 && for f in logits-*.f32; do sha256sum $f; done"
ha = subprocess.run(['/home/colum/.local/bin/spark0', cmd.format(a)], capture_output=True, text=True).stdout.split('\n')
hb = subprocess.run(['/home/colum/.local/bin/spark0', cmd.format(b)], capture_output=True, text=True).stdout.split('\n')
A = dict(reversed(l.split()) for l in ha if l.strip()); B = dict(reversed(l.split()) for l in hb if l.strip())
same = sum(A[k] == B.get(k) for k in A)
diff = sorted(k for k in A if k in B and A[k] != B[k])
print(f'steps a={len(A)} b={len(B)} identical={same} differing={len(diff)} first_diff={diff[:3]}')
sys.exit(0 if A and A == B else 1)
