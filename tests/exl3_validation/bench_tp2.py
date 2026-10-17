# SPDX-License-Identifier: AGPL-3.0-only
"""Uninstrumented TP=2 decode/prefill bench on both Sparks (P0/P1 A/B loop).

Creates one candidate container per rank from the registered collector-only
descriptors, with an explicit image and optional --env overrides (the only
variables an A/B may change), waits for health, runs perf_baseline.py
workloads through an SSH tunnel, then stops and removes the bench containers.
Decode rates come from perf_baseline's TPOT (prefill never leaks into them).

  python3 tests/exl3_validation/bench_tp2.py --image IMAGE --label base \
      --out $EV/perf-ab-20261015 [--env KEY=VALUE ...] [--workload decode_prose]

--profile glm53-nvfp4-ep2 serves the GLM-5.3-Flash NVFP4 checkpoint in upstream's
dual-Spark layout (EP=2, TP=1) for upstream non-regression A/Bs. It keeps the
descriptors' docker/NCCL/RDMA arguments, drops every ATLAS_* env and the EXL3
driver-cache mount (upstream defaults only), mounts the hosts' HF cache and
replaces the serve command. Run it under `flock /tmp/atlas-bench.lock`;
--print-argv shows both ranks' argv without touching the Sparks.
"""
import argparse
import gzip
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# ATLAS_BENCH_DESCRIPTORS selects an alternate collector-only profile directory
# (same layout, different serve args) so serve-flag A/Bs need no code edits.
DESCRIPTORS = Path(os.environ.get(
    'ATLAS_BENCH_DESCRIPTORS',
    '/home/colum/Workspace/atlas-exl3-validation-20261010/'
    'statistical-gate-20261011/collector-only-profile'))
DESCRIPTOR_IMAGE = 'atlas-exl3-validation:src-8b6b47f14d5b-rust1.97.1'
OBSERVER_ENV = 'ATLAS_EXL3_LOGITS_DIR'
HOSTS = {0: Path.home()/'.local/bin/spark0', 1: Path.home()/'.local/bin/spark1'}
MODEL = ('/root/.cache/huggingface/hub/models--MikeRoz--GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3/'
         'snapshots/dda99c7e935920fa6e235ce410e6f806e1c02720')
# Serve profiles. The EXL3 TP2 profile is the descriptors' serve command as-is.
NVFP4_HF_ROOT = '/home/colum/.cache/huggingface'
NVFP4_REVISION = '36c184c6cda000a481711306df5adde42f63321a'
NVFP4_MODEL = ('/root/.cache/huggingface/hub/models--RedHatAI--GLM-5.3-Flash-NVFP4/snapshots/'
               + NVFP4_REVISION)


def nvfp4_ep2_serve(rank):
    # Upstream's GLM-5.3-Flash dual-Spark layout; KV dtype and LM head come from
    # MODEL.toml (bf16 KV). No --language-model-only: upstream's CLI lacks it.
    return ['serve', NVFP4_MODEL, '--world-size', '2', '--tp-size', '1', '--ep-size', '2',
            '--rank', str(rank), '--master-addr', '192.168.100.1', '--master-port', '29721',
            '--bind', '0.0.0.0', '--port', '8011', '--max-seq-len', '4096',
            '--max-batch-size', '1', '--max-num-seqs', '1', '--gpu-memory-utilization', '0.88',
            '--swap-space-gb', '0', '--oom-guard-mb', '1024', '--request-timeout', '1200']


PROFILES = {
    'glm53-exl3-tp2': {'model': MODEL, 'revision': 'dda99c7e935920fa6e235ce410e6f806e1c02720',
                       'serve': None, 'hf_root': None},
    'glm53-nvfp4-ep2': {'model': NVFP4_MODEL, 'revision': NVFP4_REVISION,
                        'serve': nvfp4_ep2_serve, 'hf_root': NVFP4_HF_ROOT},
}
DEFAULT_PROFILE = 'glm53-exl3-tp2'
SSH_TUNNEL = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-J', 'colum@192.168.1.50',
              '-N', '-L']

# Bytes: a full container log is archived as-is below 1 MiB and gzipped above it.
# NCCL_DEBUG=INFO transports emit far more than that, and the run directory is
# copied off the Sparks wholesale, so the gz threshold keeps archives bounded.
ARCHIVE_GZIP_THRESHOLD = 1024 * 1024
# Seconds: pulling the full log can take minutes where the 400-line tail could not.
ARCHIVE_LOG_TIMEOUT = 900


def remote(rank, command, check=True, timeout=300):
    for attempt in range(4):  # ssh exit 255 = connection failure via the jump host; retry
        result = subprocess.run([str(HOSTS[rank]), command], text=True, capture_output=True, timeout=timeout)
        if result.returncode != 255:
            break
        time.sleep(5 * (attempt + 1))
    if check and result.returncode:
        raise RuntimeError(f'spark{rank}: {command}\n{result.stdout}\n{result.stderr}')
    return result.stdout


def create_argv(rank, image, name, extra_env, binary=None, capture=None, cpuset=None,
                profile=DEFAULT_PROFILE):
    prof = PROFILES[profile]
    argv = json.loads((DESCRIPTORS/f'cand-rank{rank}-create-command.json').read_text())
    out, i = [], 0
    while i < len(argv):
        if argv[i] == '--env' and argv[i+1].startswith(OBSERVER_ENV+'='):
            i += 2; continue
        if argv[i] in ('--volume', '-v') and argv[i+1].endswith(':/tmp/capture'):
            i += 2; continue
        if prof['serve']:  # another model: keep only the docker/NCCL/RDMA arguments
            if argv[i] == '--env' and argv[i+1].split('=', 1)[0].startswith(('ATLAS_', 'CUDA_CACHE_PATH')):
                i += 2; continue
            if argv[i] in ('--volume', '-v') and argv[i+1].split(':')[1] in (
                    '/tmp/cuda-driver-cache', '/root/.cache/huggingface'):
                i += 2; continue
        out.append(argv[i]); i += 1
    out[out.index('--name')+1] = name
    at = out.index(DESCRIPTOR_IMAGE)
    if prof['serve']:
        out[at:] = ['--volume', f"{prof['hf_root']}:/root/.cache/huggingface:ro", DESCRIPTOR_IMAGE] \
            + prof['serve'](rank)
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
    if cpuset:  # host CPUs 0-4,10-14 are isolcpus (no balancing); see gb10-fleet-facts
        overrides += ['--cpuset-cpus', cpuset]
    if binary:  # dev-built spark mounted over the image's binary (same runtime libs)
        overrides += ['--volume', f'{binary}:/usr/local/bin/spark:ro']
    if capture:  # per-step final-logit dumps for bit-identity checks
        overrides += ['--env', f'{OBSERVER_ENV}=/tmp/capture', '--volume', f'{capture}/rank{rank}:/tmp/capture']
    out[at:at+1] = overrides + [image]
    return out


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]


def archive_container_log(rank, text, run_dir):
    """Write rank's full container stdout+stderr into run_dir.

    Additive to the 400-line tail the caller keeps in container-r{rank}.log:
    the tail is what consumers read today, the archive carries the NCCL
    init/transport lines that scroll past any tail (NCCL_DEBUG=INFO). Gzipped
    above ARCHIVE_GZIP_THRESHOLD bytes. Returns the archive path.
    """
    path = Path(run_dir)/f'container-r{rank}.full.log'
    if len(text.encode('utf-8', 'replace')) > ARCHIVE_GZIP_THRESHOLD:
        path = path.with_name(path.name + '.gz')
        with gzip.open(path, 'wt', encoding='utf-8', errors='replace', compresslevel=6) as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding='utf-8')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--workload', action='append', default=None)
    ap.add_argument('--workloads', default=str(HERE/'perf_workloads_ab.json'))
    ap.add_argument('--health-timeout', type=float, default=1800)
    ap.add_argument('--binary', help='host path of a spark binary present on both Sparks')
    ap.add_argument('--binary-source-commit',
                    help='source commit of the mounted binary, required with --binary')
    ap.add_argument('--cpuset', help='docker --cpuset-cpus for both ranks')
    ap.add_argument('--capture', action='store_true', help='dump per-step logits under /home/colum/atlas-dev/capture/<run>')
    ap.add_argument('--profile', choices=sorted(PROFILES), default=DEFAULT_PROFILE,
                    help='model and serve layout (default: the EXL3 TP2 descriptors)')
    ap.add_argument('--print-argv', action='store_true',
                    help='print both ranks\' docker create argv as JSON and exit')
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    if args.capture and prof['serve']:
        ap.error('--capture uses the EXL3 logits observer; it needs the EXL3 profile')
    if bool(args.binary) != bool(args.binary_source_commit):
        ap.error('--binary and --binary-source-commit must be supplied together')
    if args.binary_source_commit and not re.fullmatch(r'[0-9a-fA-F]{7,40}', args.binary_source_commit):
        ap.error('--binary-source-commit must be a 7-40 character hexadecimal commit ID')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    names = {r: f'atlas-bench-{args.label}-{stamp.lower()}-r{r}' for r in (0, 1)}
    capture = f'/home/colum/atlas-dev/capture/{args.label}-{stamp}' if args.capture else None
    argvs = {r: create_argv(r, args.image, names[r], args.env, args.binary, capture, args.cpuset,
                            args.profile) for r in (0, 1)}
    if args.print_argv:
        print(json.dumps(argvs, indent=2))
        return 0
    run = Path(args.out)/f'{args.label}-{stamp}'
    run.mkdir(parents=True)
    meta = {'label': args.label, 'profile': args.profile, 'image': args.image,
            'env_overrides': args.env, 'argv': argvs,
            'binary': args.binary, 'binary_source_commit': args.binary_source_commit,
            'capture': capture,
            'utc': stamp, 'classification': 'unqualified-uninstrumented'}
    created, tunnel = [], None
    try:
        for r in (0, 1):
            busy = remote(r, "docker ps --format '{{.Names}}' | grep -vxE 'atlas-(dev|pr)-builder[0-9]*|atlas-spin'; nvidia-smi --query-compute-apps=pid --format=csv,noheader", check=False).strip()
            if busy:
                raise RuntimeError(f'spark{r} not idle: {busy}')
        for r in (0, 1):
            meta[f'image_id_r{r}'] = remote(r, f'docker image inspect --format {{{{.Id}}}} {shlex.quote(args.image)}').strip()
            if args.binary:
                meta[f'binary_sha256_r{r}'] = remote(r, f'sha256sum {shlex.quote(args.binary)}').split()[0]
        if meta['image_id_r0'] != meta['image_id_r1']:
            raise RuntimeError('rank image IDs differ; refusing an unmatched benchmark')
        if args.binary and meta['binary_sha256_r0'] != meta['binary_sha256_r1']:
            raise RuntimeError('rank binary hashes differ; refusing an unmatched benchmark')
        for r in (0, 1):
            if capture:
                remote(r, f'mkdir -p {capture}/rank{r}')
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
            if probe.returncode == 0 and prof['model'] in probe.stdout:
                break
            for r in (0, 1):
                state = remote(r, f"docker inspect --format '{{{{.State.Status}}}}' {names[r]}").strip()
                if state != 'running':
                    raise RuntimeError(f'spark{r} container {state}')
            if time.monotonic() > deadline:
                raise TimeoutError('health')
            time.sleep(5)
        meta['ready_utc'] = datetime.now(timezone.utc).isoformat()
        if capture:
            for r in (0, 1):
                remote(r, f'touch {capture}/rank{r}/capture-enabled')
        cmd = [sys.executable, str(HERE/'perf_baseline.py'), '--endpoint', endpoint, '--model', prof['model'],
               '--workloads', args.workloads, '--raw-out', str(run/'raw.jsonl'),
               '--summary-out', str(run/'summary.json'), '--initial-state', 'unknown',
               '--unqualified', '--fingerprint', str(run/'fingerprint.json')]
        for w in args.workload or []:
            cmd += ['--workload', w]
        (run/'fingerprint.json').write_text(json.dumps({
            'image_id': meta['image_id_r0'], 'source_commit': args.binary_source_commit,
            'binary': args.binary,
            'binary_sha256_r0': meta.get('binary_sha256_r0'),
            'binary_sha256_r1': meta.get('binary_sha256_r1'),
            'serve_flags': argvs[0],
            'environment': args.env, 'hosts': ['spark0', 'spark1'],
            'checkpoint_revision': prof['revision']}))
        result = subprocess.run(cmd, capture_output=True, text=True)
        (run/'perf_baseline.log').write_text(result.stdout + result.stderr)
        meta['perf_rc'] = result.returncode
    finally:
        if tunnel:
            tunnel.terminate()
        for r in created:
            # Archive before removal: docker logs dies with the container. A failed
            # archive must not cost us the tail log or the teardown, hence the split.
            try:
                full = remote(r, f'docker logs {names[r]} 2>&1', check=False, timeout=ARCHIVE_LOG_TIMEOUT)
                meta[f'log_archive_r{r}'] = str(archive_container_log(r, full, run))
            except Exception as exc:  # archiving is additive; keep tearing down
                meta[f'log_archive_error_r{r}'] = str(exc)
            try:
                (run/f'container-r{r}.log').write_text(remote(r, f'docker logs --tail 400 {names[r]} 2>&1', check=False))
                remote(r, f'docker rm -f {names[r]}', check=False)
            except Exception as exc:  # keep tearing down the other rank
                meta[f'teardown_error_r{r}'] = str(exc)
        (run/'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(run)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
