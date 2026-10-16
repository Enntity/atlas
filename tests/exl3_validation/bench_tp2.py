# SPDX-License-Identifier: AGPL-3.0-only
"""Uninstrumented TP=2 decode/prefill bench on both Sparks (P0/P1 A/B loop).

Creates one candidate container per rank from the registered collector-only
descriptors, with an explicit image and optional --env overrides (the only
variables an A/B may change), waits for health, runs perf_baseline.py
workloads through an SSH tunnel, then stops and removes the bench containers.
Decode rates come from perf_baseline's TPOT (prefill never leaks into them).

  python3 tests/exl3_validation/bench_tp2.py --image IMAGE --label base \
      --out $EV/perf-ab-20261015 [--env KEY=VALUE ...] [--workload decode_prose]
"""
import argparse
import json
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESCRIPTORS = Path('/home/colum/Workspace/atlas-exl3-validation-20261010/'
                   'statistical-gate-20261011/collector-only-profile')
DESCRIPTOR_IMAGE = 'atlas-exl3-validation:src-8b6b47f14d5b-rust1.97.1'
OBSERVER_ENV = 'ATLAS_EXL3_LOGITS_DIR'
HOSTS = {0: Path.home()/'.local/bin/spark0', 1: Path.home()/'.local/bin/spark1'}
MODEL = ('/root/.cache/huggingface/hub/models--MikeRoz--GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3/'
         'snapshots/dda99c7e935920fa6e235ce410e6f806e1c02720')
SSH_TUNNEL = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-J', 'colum@192.168.1.50',
              '-N', '-L']


def remote(rank, command, check=True, timeout=300):
    result = subprocess.run([str(HOSTS[rank]), command], text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'spark{rank}: {command}\n{result.stdout}\n{result.stderr}')
    return result.stdout


def create_argv(rank, image, name, extra_env):
    argv = json.loads((DESCRIPTORS/f'cand-rank{rank}-create-command.json').read_text())
    out, i = [], 0
    while i < len(argv):
        if argv[i] == '--env' and argv[i+1].startswith(OBSERVER_ENV+'='):
            i += 2; continue
        if argv[i] in ('--volume', '-v') and argv[i+1].endswith(':/tmp/capture'):
            i += 2; continue
        out.append(argv[i]); i += 1
    out[out.index('--name')+1] = name
    at = out.index(DESCRIPTOR_IMAGE)
    overrides = []
    for item in extra_env:
        key = item.split('=', 1)[0]
        # replace an existing --env KEY=... rather than duplicating it
        for j in range(len(out)-1):
            if out[j] == '--env' and out[j+1].split('=', 1)[0] == key:
                out[j+1] = item; break
        else:
            overrides += ['--env', item]
    out[at:at+1] = overrides + [image]
    return out


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--workload', action='append', default=None)
    ap.add_argument('--workloads', default=str(HERE/'perf_workloads_ab.json'))
    ap.add_argument('--health-timeout', type=float, default=1800)
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(args.out)/f'{args.label}-{stamp}'
    run.mkdir(parents=True)
    names = {r: f'atlas-bench-{args.label}-{stamp.lower()}-r{r}' for r in (0, 1)}
    argvs = {r: create_argv(r, args.image, names[r], args.env) for r in (0, 1)}
    meta = {'label': args.label, 'image': args.image, 'env_overrides': args.env, 'argv': argvs,
            'utc': stamp, 'classification': 'unqualified-uninstrumented'}
    created, tunnel = [], None
    try:
        for r in (0, 1):
            busy = remote(r, "docker ps --format '{{.Names}}' | grep -vx atlas-dev-builder; nvidia-smi --query-compute-apps=pid --format=csv,noheader", check=False).strip()
            if busy:
                raise RuntimeError(f'spark{r} not idle: {busy}')
        for r in (0, 1):
            meta[f'image_id_r{r}'] = remote(r, f'docker image inspect --format {{{{.Id}}}} {shlex.quote(args.image)}').strip()
            remote(r, shlex.join(argvs[r])); created.append(r)
        for r in (0, 1):
            remote(r, f'docker start {names[r]}')
        port = free_port()
        tunnel = subprocess.Popen(SSH_TUNNEL + [f'127.0.0.1:{port}:127.0.0.1:8011', 'colum@10.50.0.10'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        endpoint = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + args.health_timeout
        while True:
            probe = subprocess.run(['curl', '-fsS', '--max-time', '3', endpoint+'/v1/models'],
                                   capture_output=True, text=True)
            if probe.returncode == 0 and MODEL in probe.stdout:
                break
            for r in (0, 1):
                state = remote(r, f"docker inspect --format '{{{{.State.Status}}}}' {names[r]}").strip()
                if state != 'running':
                    raise RuntimeError(f'spark{r} container {state}')
            if time.monotonic() > deadline:
                raise TimeoutError('health')
            time.sleep(5)
        meta['ready_utc'] = datetime.now(timezone.utc).isoformat()
        cmd = [sys.executable, str(HERE/'perf_baseline.py'), '--endpoint', endpoint, '--model', MODEL,
               '--workloads', args.workloads, '--raw-out', str(run/'raw.jsonl'),
               '--summary-out', str(run/'summary.json'), '--initial-state', 'unknown',
               '--unqualified', '--fingerprint', str(run/'fingerprint.json')]
        for w in args.workload or []:
            cmd += ['--workload', w]
        (run/'fingerprint.json').write_text(json.dumps({
            'image_id': meta['image_id_r0'], 'source_commit': args.image, 'serve_flags': argvs[0],
            'environment': args.env, 'hosts': ['spark0', 'spark1'],
            'checkpoint_revision': 'dda99c7e935920fa6e235ce410e6f806e1c02720'}))
        result = subprocess.run(cmd, capture_output=True, text=True)
        (run/'perf_baseline.log').write_text(result.stdout + result.stderr)
        meta['perf_rc'] = result.returncode
    finally:
        if tunnel:
            tunnel.terminate()
        for r in created:
            try:
                (run/f'container-r{r}.log').write_text(remote(r, f'docker logs --tail 400 {names[r]} 2>&1', check=False))
                remote(r, f'docker rm -f {names[r]}', check=False)
            except Exception as exc:  # keep tearing down the other rank
                meta[f'teardown_error_r{r}'] = str(exc)
        (run/'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(run)


if __name__ == '__main__':
    main()
