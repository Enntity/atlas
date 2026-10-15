# SPDX-License-Identifier: AGPL-3.0-only
"""Append-only runner for the registered hidden-state localization attempts.

Executes ``docs/design/exl3-hidden-state-localization-registration.md``:
three fresh-process reference attempts of the gate-v2 campaign reference
profile, with the campaign's own context-v4 observer turned up to exact-byte
FFN and mHC boundary dumps on every layer (three environment variables), a
short16-only collection, and an isolated copy of the probe's frozen cache
snapshot per attempt. Lifecycle helpers are shared with ``probe_runner``.

Usage:
    hsl_runner.py mounts                 # freeze fixed mount sources (once)
    hsl_runner.py attempt NAME [--dry-run]
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from image_cycle import sha, require
import probe_runner as pr

BASE = pr.EV / 'hidden-state-localization-20261015'
HERE = Path(__file__).resolve().parent
REGISTRATION = (HERE.parent.parent / 'docs' / 'design'
                / 'exl3-hidden-state-localization-registration.md')
FINALIZER = BASE / 'driver' / 'finalize-hsl-reference.py'
REMOTE = '/home/colum/hsl-20261015'
ATTEMPTS = ('hsl01', 'hsl02', 'hsl03')
LAYERS = ','.join(str(i) for i in range(45))
DUMP_DIR = '/tmp/ffn-capture'
DUMP_ENV = {'ATLAS_EXL3_FFN_DUMP_DIR': DUMP_DIR,
            'ATLAS_EXL3_FFN_DUMP_LAYERS': LAYERS,
            'ATLAS_EXL3_MHC_DUMP_LAYERS': LAYERS}
# Dumps larger than this stay on the Spark (durable home) with a hash list.
# hsl01 (5.1 GiB/rank) was copied locally; local disk cannot hold three, so
# later attempts keep dumps on the Sparks, where pairwise analysis runs.
LOCAL_COPY_LIMIT = 0


def registered_attempts(path=REGISTRATION):
    body = Path(path).read_text()
    section = re.search(r'^## Attempt list.*?(?=^## )', body, re.S | re.M)
    require(section is not None, 'registration lacks the attempt-list section')
    rows = re.findall(r'^\s{4}([a-z0-9]+(?:,\s*[a-z0-9]+)+)\s*$', section.group(0), re.M)
    require(len(rows) == 1 and tuple(x.strip() for x in rows[0].split(',')) == ATTEMPTS,
            'registered attempt list differs from the code-side freeze')
    return ATTEMPTS


def hsl_command(command, *, attempt, rank):
    """Campaign reference command + dump env and attempt-scoped mounts."""
    require(attempt in ATTEMPTS, 'unregistered attempt ' + attempt)
    c = list(command)
    c[c.index('--name') + 1] = 'hsl-%s-ref-%d' % (attempt, rank)
    ra = '%s/attempts/%s' % (REMOTE, attempt)
    vols = pr._volume_index(c)
    pr.cache_sources(command)
    replace = {'/tmp/reference-logits': ra + '/capture:/tmp/reference-logits',
               DUMP_DIR: ra + '/dump:' + DUMP_DIR,
               '/root/.cache/vllm': ra + '/cache/vllm:/root/.cache/vllm',
               '/root/.triton/cache': ra + '/cache/vllm/triton:/root/.triton/cache',
               '/root/.tilelang/cache': ra + '/cache/vllm/tilelang:/root/.tilelang/cache',
               '/root/.nv/ComputeCache': ra + '/cache/vllm/nv:/root/.nv/ComputeCache',
               '/tmp/reference-autotune': ra + '/cache/tune:/tmp/reference-autotune'}
    for dest, spec in replace.items():
        require(dest in vols, 'campaign command lacks volume ' + dest)
        c[vols[dest]] = spec
    require('/opt/atlas-logits-patch' in vols, 'campaign observer mount missing')
    envs = {c[i + 1].partition('=')[0] for i, a in enumerate(c[:-1]) if a == '--env'}
    require(not envs & set(DUMP_ENV), 'campaign command already sets a dump variable')
    require(c.count(pr.IMAGE) == 1 and c[-2:] == [pr.IMAGE, '/start.sh'],
            'campaign command must end with the reference image and /start.sh')
    extra = []
    for key, value in DUMP_ENV.items():
        extra += ['--env', key + '=' + value]
    at = c.index(pr.IMAGE)
    return c[:at] + extra + c[at:]


def hsl_draft(campaign_draft):
    draft = json.loads(json.dumps(campaign_draft))
    draft['environment'] = dict(draft['environment'], **DUMP_ENV)
    return draft


def fixed_mount_sources(command, attempt):
    ra = '%s/attempts/%s/' % (REMOTE, attempt)
    return sorted(command[i + 1].split(':')[0] for i, a in enumerate(command[:-1])
                  if a == '--volume' and not command[i + 1].startswith(ra))


def mount_manifest(rank, command, attempt):
    m = pr.remote_python(rank, pr.MOUNT_SOURCE,
                         {'sources': fixed_mount_sources(command, attempt)})
    missing = [src for src, v in m.items() if v['type'] == 'missing']
    require(not missing, 'rank%d mount sources missing: %s' % (rank, missing))
    return m


def commands_for(attempt):
    return [hsl_command(pr.campaign_command(r), attempt=attempt, rank=r) for r in (0, 1)]


def cmd_mounts():
    out_dir = BASE / 'mount-freeze'
    out_dir.mkdir(parents=True, exist_ok=False)
    for rank, command in enumerate(commands_for(ATTEMPTS[0])):
        m = mount_manifest(rank, command, ATTEMPTS[0])
        pr.write(out_dir / ('rank%d.json' % rank), {'rank': rank, 'utc': pr.now(), 'mounts': m})
        print('rank%d: %d fixed mount sources frozen' % (rank, len(m)))


def verify_next(attempt):
    registered_attempts()
    progress = BASE / 'progress.json'
    done = json.loads(progress.read_text())['completed'] if progress.exists() else []
    require(len(done) < len(ATTEMPTS) and ATTEMPTS[len(done)] == attempt,
            'order violation: next is %s; got %s' % (
                ATTEMPTS[len(done)] if len(done) < len(ATTEMPTS) else 'none', attempt))
    return progress, done


def preflight(attempt, commands):
    ra = '%s/attempts/%s' % (REMOTE, attempt)
    checks = {'collector_inputs': pr.collector_inputs_ok()}
    for r in (0, 1):
        pr.gpu_idle(r)
        pr.spark(r, 'test ! -e ' + ra)
        frozen = json.loads((BASE / 'mount-freeze' / ('rank%d.json' % r)).read_text())
        require(mount_manifest(r, commands[r], attempt) == frozen['mounts'],
                'rank%d fixed mount sources differ from the frozen manifest' % r)
        free = int(pr.spark(r, "df -B1 --output=avail /home | tail -1").stdout.strip())
        require(free > 25 * 1024 ** 3, 'rank%d has under 25 GiB free' % r)
        checks['free_bytes_rank%d' % r] = free
    return checks


def collect_short16(attempt, caproot):
    name = 'hsl-' + attempt
    remote = '''set -euo pipefail
BASE="$HOME/gate/attempts/{name}"
mkdir "$BASE"
python3 "$HOME/gate/tests/exl3_validation/collect_logits.py" --endpoint http://127.0.0.1:{port} --model {model} --cases "$HOME/gate/cases-short16.json" --capture-dir "{cap}" --out "$BASE/short16" --provenance "$HOME/gate/{name}/provenance-finalization/prov-ref.json" --thresholds "$HOME/gate/tests/exl3_validation/thresholds.json" --timeout 1200 > "$BASE/short16.log" 2>&1
'''.format(name=name, port=pr.PORT, model=pr.SERVED_MODEL, cap=caproot)
    p = pr.spark(0, remote, check=False)
    return p.returncode, p.stdout, p.stderr


def cmd_attempt(attempt, dry_run):
    progress, done = verify_next(attempt)
    commands = commands_for(attempt)
    draft = hsl_draft(json.loads((pr.CAMPAIGN_DRIVER / 'prov-ref.json').read_text()))
    names = [c[c.index('--name') + 1] for c in commands]
    if dry_run:
        print(json.dumps({'attempt': attempt, 'commands': commands, 'containers': names},
                         indent=2))
        return
    checks = preflight(attempt, commands)
    root = BASE / 'attempts' / attempt
    root.mkdir(parents=True, exist_ok=False)
    receipt = {'attempt': attempt, 'status': 'started', 'start_utc': pr.now(),
               'containers': names, 'preflight': checks,
               'bindings': {'runner_sha256': sha(Path(__file__)),
                            'probe_runner_sha256': sha(Path(pr.__file__)),
                            'registration_sha256': sha(REGISTRATION),
                            'finalizer_sha256': sha(FINALIZER)},
               'stop_timeout_s': pr.STOP_TIMEOUT_S, 'steps': []}
    ra = '%s/attempts/%s' % (REMOTE, attempt)
    try:
        (root / 'profile').mkdir()
        for r in (0, 1):
            pr.write(root / 'profile' / ('ref-rank%d-create-command.json' % r), commands[r])
        pr.write(root / 'hsl-draft-prov-ref.json', draft)
        run(attempt, commands, names, draft, root, ra, receipt)
    except BaseException as exc:  # recorded; the attempt still tears down
        receipt['error'] = '%s: %s' % (type(exc).__name__, exc)
    finally:
        try:
            teardown(names, root, ra, receipt, attempt)
        except BaseException as exc:  # noqa: BLE001 - never leave order stuck
            receipt['teardown_ok'] = False
            receipt.setdefault('teardown_errors', []).append(
                'teardown: %s: %s' % (type(exc).__name__, exc))
        receipt['status'] = ('complete' if 'error' not in receipt
                             and receipt.get('teardown_ok') else 'failed')
        receipt['completed_utc'] = pr.now()
        pr.write(root / 'attempt.json', receipt)
        done.append(attempt)
        pr.write(progress, {'completed': done, 'last_attempt_status': receipt['status']})
    if receipt['status'] != 'complete':
        raise SystemExit('%s failed; evidence preserved; no retry. See %s' % (attempt, root))
    print(json.dumps({'attempt': attempt, 'status': 'complete', 'path': str(root)}))


def run(attempt, commands, names, draft, root, ra, receipt):
    def step(label, **extra):
        receipt['steps'].append(dict({'step': label, 'utc': pr.now()}, **extra))
        pr.write(root / 'attempt.json', receipt)

    step('preflight_passed')
    for r in (0, 1):
        pr.spark(r, 'mkdir -p {ra}/capture {ra}/dump {ra}/cache && '
                    'sudo -n cp -a {snap}/vllm {ra}/cache/vllm && '
                    'sudo -n cp -a {snap}/tune {ra}/cache/tune'.format(
                        ra=ra, snap=pr.REMOTE_SNAPSHOT), timeout=3600)
        pre = pr.manifest(r, {'vllm': ra + '/cache/vllm', 'tune': ra + '/cache/tune'})
        frozen = json.loads((pr.BASE / 'cache-freeze' / ('rank%d.json' % r)).read_text())
        for key in ('vllm', 'tune'):
            require(pre[key]['content_sha256']
                    == frozen['manifest']['snapshot_' + key]['content_sha256'],
                    'rank%d %s attempt cache differs from the frozen snapshot' % (r, key))
        pr.write(root / ('cache-pre-rank%d.json' % r), pre)
    step('cache_initialized')
    receipt['containers_may_exist'] = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        starts = list(pool.map(lambda r: pr.create_and_start(r, commands[r]), (0, 1)))
    step('servers_started', containers=starts)
    pr.wait_ready(names)
    step('api_ready')
    fin = root / 'finalizer'
    fin.mkdir()
    shutil.copy2(FINALIZER, fin / 'finalize_provenance.py')
    pr.write(fin / 'prov-ref.json', draft)
    shutil.copy2(pr.CAMPAIGN_DRIVER / 'reference-image-inspect.json', fin)
    env = dict(os.environ, GATE_V2_CONTAINER_PREFIX=names[0][:-2],
               PROBE_PROFILE_DIR=str(root / 'profile'))
    fp = subprocess.run([sys.executable, str(fin / 'finalize_provenance.py'), 'ref',
                         'hsl-' + attempt], env=env, cwd=fin, text=True, capture_output=True)
    (fin / 'finalizer.stdout').write_text(fp.stdout)
    (fin / 'finalizer.stderr').write_text(fp.stderr)
    require(fp.returncode == 0, 'provenance finalizer failed: ' + fp.stderr[-3000:])
    step('provenance_finalized')
    # Dump gate (proven q2-diagnostic mechanism): the collector only marks
    # rank0's logits dir, so both ranks' dump dirs are marked explicitly.
    for r in (0, 1):
        pr.spark(r, 'test ! -e {d}/capture-enabled && touch {d}/capture-enabled'.format(
            d=ra + '/dump'))
    step('dump_markers_set')
    rc, out, err = collect_short16(attempt, ra + '/capture')
    (root / 'collector.stdout').write_text(out)
    (root / 'collector.stderr').write_text(err)
    receipt['collection_exit_code'] = rc
    step('collected', exit_code=rc)
    require(rc == 0, 'collection failed with exit %d' % rc)


def teardown(names, root, ra, receipt, attempt):
    ok = True
    stops = []
    if receipt.get('containers_may_exist'):
        def stop(r):
            exists = pr.spark(r, 'docker inspect %s >/dev/null 2>&1' % names[r],
                              check=False).returncode == 0
            if not exists:
                return {'rank': r, 'container': names[r], 'exists': False}
            p = pr.guarded(receipt, 'stop rank%d' % r, lambda: pr.spark(
                r, 'docker stop --timeout %d %s' % (pr.STOP_TIMEOUT_S, names[r]),
                check=False, timeout=pr.STOP_TIMEOUT_S + 300))
            state = pr.guarded(receipt, 'state rank%d' % r, lambda: json.loads(pr.spark(
                r, "docker inspect --format '{{json .State}}' " + names[r]).stdout))
            logs = pr.guarded(receipt, 'logs rank%d' % r, lambda: pr.spark(
                r, 'docker logs %s 2>&1' % names[r], check=False, timeout=600).stdout, '')
            (root / ('container-rank%d.log' % r)).write_text(logs or '')
            return {'rank': r, 'container': names[r], 'exists': True,
                    'stop_exit': p.returncode if p is not None else None, 'state': state}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            stops = list(pool.map(stop, (0, 1)))
        for s_ in stops:
            st = s_.get('state') or {}
            ok &= s_.get('exists') is True and s_.get('stop_exit') == 0 \
                and st.get('Running') is False and st.get('OOMKilled') is False
    receipt['shutdown'] = stops
    pr.write(root / 'attempt.json', receipt)
    for r in (0, 1):
        post = pr.guarded(receipt, 'cache-post rank%d' % r, lambda: pr.manifest(
            r, {'vllm': ra + '/cache/vllm', 'tune': ra + '/cache/tune'}))
        if post is None:
            ok = False
        else:
            pr.write(root / ('cache-post-rank%d.json' % r), post)
            # The cache copy is reproducible from the snapshot; drop it once
            # its post manifest is recorded (spark0 disk is tight).
            pr.guarded(receipt, 'cache-drop rank%d' % r, lambda: pr.spark(
                r, 'sudo -n rm -rf -- %s/cache' % ra))
    ok &= bool(pr.guarded(receipt, 'archive', lambda: archive(attempt, root, ra, receipt), False))
    ok &= not receipt.get('teardown_errors')
    receipt['teardown_ok'] = bool(ok)
    pr.write(root / 'attempt.json', receipt)


def archive(attempt, root, ra, receipt):
    ok = True
    pulls = {0: [('gate', 'tar --ignore-failed-read -C $HOME/gate -cf - '
                  'attempts/hsl-{a} hsl-{a}'.format(a=attempt))], 1: []}
    for r in (0, 1):
        pulls[r].append(('capture', 'sudo -n tar -C %s -cf - capture' % ra))
        size = pr.spark(r, 'sudo -n du -sb %s/dump | cut -f1' % ra, check=False).stdout.strip()
        receipt['dump_bytes_rank%d' % r] = int(size) if size.isdigit() else None
        sums = pr.spark(r, 'cd %s/dump && sudo -n find . -type f -exec sha256sum {} + | '
                           'sort -k2' % ra, check=False, timeout=7200)
        (root / ('dump-rank%d-sha256.txt' % r)).write_text(sums.stdout)
        if size.isdigit() and int(size) <= LOCAL_COPY_LIMIT:
            pulls[r].append(('dump', 'sudo -n tar -C %s -cf - dump' % ra))
        else:
            receipt['dump_rank%d_retained_at' % r] = 'spark%d:%s/dump' % (r, ra)
    archives = {}
    for r, items in pulls.items():
        for label, command in items:
            target = root / ('rank%d-%s.tar' % (r, label))
            with target.open('xb') as out:
                p = subprocess.run(['spark%d' % r, command], stdout=out, stderr=subprocess.PIPE)
            archives[target.name] = {'exit': p.returncode, 'sha256': sha(target),
                                     'bytes': target.stat().st_size}
            ok &= p.returncode == 0
    receipt['archives'] = archives
    return ok


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('mounts')
    att = sub.add_parser('attempt')
    att.add_argument('name', choices=ATTEMPTS)
    att.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.cmd == 'mounts':
        cmd_mounts()
    else:
        cmd_attempt(args.name, args.dry_run)


if __name__ == '__main__':
    main()
