# SPDX-License-Identifier: AGPL-3.0-only
"""Append-only runner for registered reference-only experiments.

Generalizes the lifecycle of ``hsl_runner`` (kept unchanged as executed):
the gate-v2 campaign reference create commands with an explicit environment
delta, attempt-scoped capture/dump mounts and an isolated copy of the
probe's frozen cache snapshot, preflight before the point of no return, the
gate-v2 finalizer, a registered collection scope, then guarded teardown and
archive that always advance the order. Each experiment is one ``EXPERIMENTS``
entry whose registration document freezes its attempt list.

Usage:
    ref_experiment_runner.py EXPERIMENT mounts
    ref_experiment_runner.py EXPERIMENT attempt NAME [--dry-run]
"""
import argparse
import concurrent.futures
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from image_cycle import sha, require
import probe_runner as pr

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent / 'docs' / 'design'


@dataclass(frozen=True)
class Experiment:
    key: str
    base: Path
    remote: str
    registration: Path
    attempts: tuple
    env: dict            # variable -> value; replaces the campaign value or adds it
    suites: tuple        # ('short16',), ('short16', 'long') or ('long',)
    dumps: bool          # mark per-rank dump dirs before collection
    local_copy_limit: int
    observer: Path = None    # local observer package replacing the campaign mount
    long_cases: Path = None  # local long-case subset replacing cases-long.json
    min_free_gib: int = 15


EXPERIMENTS = {
    'det': Experiment(
        key='det',
        base=pr.EV / 'fused-moe-determinism-20261015',
        remote='/home/colum/det-20261015',
        registration=DOCS / 'exl3-fused-moe-determinism-registration.md',
        attempts=('det01', 'det02', 'det03'),
        env={'EXL3_FUSED_MOE': '0'},
        suites=('short16', 'long'),
        dumps=False,
        local_copy_limit=0),
    'idx': Experiment(
        key='idx',
        base=pr.EV / 'indexer-localization-20261015',
        remote='/home/colum/idx-20261015',
        registration=DOCS / 'exl3-indexer-localization-registration.md',
        attempts=('idx01', 'idx02', 'idx03'),
        env={'EXL3_FUSED_MOE': '0',
             'ATLAS_EXL3_FFN_DUMP_DIR': '/tmp/ffn-capture',
             'ATLAS_EXL3_FFN_DUMP_LAYERS': ','.join(str(i) for i in range(3, 45, 4)),
             'ATLAS_EXL3_ATTN_DUMP_LAYERS': ','.join(str(i) for i in range(3, 45, 4))},
        suites=('long',),
        dumps=True,
        local_copy_limit=0,
        observer=pr.EV / 'q2-l1-tooling-20261013' / 'ref-overlay',
        long_cases=pr.EV / 'indexer-localization-20261015' / 'driver' / 'cases-long-2051.json',
        min_free_gib=20),
}

OBSERVER_FILES = ('sitecustomize.py', 'reference_chunk_trace.py', 'reference_ffn_trace.py',
                  'reference_mhc_trace.py', 'reference_kda_trace.py', 'reference_attn_trace.py')


def remote_observer(exp):
    return exp.remote + '/observer'


def remote_long_cases(exp):
    return exp.remote + '/cases-long.json'


def observer_hashes(exp):
    return {name: sha(exp.observer / name) for name in OBSERVER_FILES
            if (exp.observer / name).is_file()}


def registered_attempts(exp):
    body = exp.registration.read_text()
    section = re.search(r'^## Attempt list.*?(?=^## )', body, re.S | re.M)
    require(section is not None, 'registration lacks the attempt-list section')
    rows = re.findall(r'^\s{4}([a-z0-9]+(?:,\s*[a-z0-9]+)+)\s*$', section.group(0), re.M)
    require(len(rows) == 1 and tuple(x.strip() for x in rows[0].split(',')) == exp.attempts,
            'registered attempt list differs from the code-side freeze')
    return exp.attempts


def env_items(command):
    return {command[i + 1].partition('=')[0]: i + 1
            for i, a in enumerate(command[:-1]) if a == '--env'}


def experiment_command(exp, command, *, attempt, rank):
    """Campaign reference command + registered env delta + scoped mounts."""
    require(attempt in exp.attempts, 'unregistered attempt ' + attempt)
    c = list(command)
    c[c.index('--name') + 1] = '%s-%s-ref-%d' % (exp.key, attempt, rank)
    ra = '%s/attempts/%s' % (exp.remote, attempt)
    vols = pr._volume_index(c)
    pr.cache_sources(command)
    replace = {'/tmp/reference-logits': ra + '/capture:/tmp/reference-logits',
               '/tmp/ffn-capture': ra + '/dump:/tmp/ffn-capture',
               '/root/.cache/vllm': ra + '/cache/vllm:/root/.cache/vllm',
               '/root/.triton/cache': ra + '/cache/vllm/triton:/root/.triton/cache',
               '/root/.tilelang/cache': ra + '/cache/vllm/tilelang:/root/.tilelang/cache',
               '/root/.nv/ComputeCache': ra + '/cache/vllm/nv:/root/.nv/ComputeCache',
               '/tmp/reference-autotune': ra + '/cache/tune:/tmp/reference-autotune'}
    for dest, spec in replace.items():
        require(dest in vols, 'campaign command lacks volume ' + dest)
        c[vols[dest]] = spec
    require('/opt/atlas-logits-patch' in vols, 'campaign observer mount missing')
    if exp.observer is not None:
        c[vols['/opt/atlas-logits-patch']] = remote_observer(exp) + ':/opt/atlas-logits-patch:ro'
    require(c.count(pr.IMAGE) == 1 and c[-2:] == [pr.IMAGE, '/start.sh'],
            'campaign command must end with the reference image and /start.sh')
    present = env_items(c)
    extra = []
    for key, value in exp.env.items():
        if key in present:
            c[present[key]] = key + '=' + value
        else:
            extra += ['--env', key + '=' + value]
    at = c.index(pr.IMAGE)
    return c[:at] + extra + c[at:]


def experiment_draft(exp, campaign_draft):
    draft = json.loads(json.dumps(campaign_draft))
    draft['environment'] = dict(draft['environment'], **exp.env)
    if exp.observer is not None:
        hashes = observer_hashes(exp)
        draft['capture_hooks_sha256'] = {name: hashes[name]
                                         for name in draft['capture_hooks_sha256']}
    return draft


def commands_for(exp, attempt):
    return [experiment_command(exp, pr.campaign_command(r), attempt=attempt, rank=r)
            for r in (0, 1)]


def fixed_mount_sources(exp, command, attempt):
    ra = '%s/attempts/%s/' % (exp.remote, attempt)
    return sorted(command[i + 1].split(':')[0] for i, a in enumerate(command[:-1])
                  if a == '--volume' and not command[i + 1].startswith(ra))


def mount_manifest(exp, rank, command, attempt):
    m = pr.remote_python(rank, pr.MOUNT_SOURCE,
                         {'sources': fixed_mount_sources(exp, command, attempt)})
    missing = [src for src, v in m.items() if v['type'] == 'missing']
    require(not missing, 'rank%d mount sources missing: %s' % (rank, missing))
    return m


def cmd_deploy(exp):
    """Copy the observer package and long-case subset to both Sparks, read-only."""
    require(exp.observer is not None or exp.long_cases is not None, 'nothing to deploy')
    for rank in (0, 1):
        if exp.observer is not None:
            files = sorted(OBSERVER_FILES)
            tar = subprocess.run(['tar', '-C', str(exp.observer), '-cf', '-', *[
                f for f in files if (exp.observer / f).is_file()]],
                capture_output=True, check=True).stdout
            p = subprocess.run(['spark%d' % rank, 'test ! -e {d} && mkdir -p {d} && tar -C {d} -xf - '
                                '&& chmod -R a-w {d}'.format(d=remote_observer(exp))],
                               input=tar, capture_output=True)
            require(p.returncode == 0, 'observer deploy rank%d: %s' % (rank, p.stderr[-500:]))
            m = pr.manifest(rank, {'o': remote_observer(exp)})['o']
            require({f['path']: f['sha256'] for f in m['files']} == observer_hashes(exp),
                    'rank%d deployed observer differs' % rank)
        if exp.long_cases is not None and rank == 0:
            pr.spark(0, 'test ! -e {f} && mkdir -p {d} && cat > {f} && chmod a-w {f}'.format(
                f=remote_long_cases(exp), d=exp.remote), input_text=exp.long_cases.read_text())
            got = pr.spark(0, 'sha256sum ' + remote_long_cases(exp)).stdout.split()[0]
            require(got == sha(exp.long_cases), 'deployed long cases differ')
        print('rank%d deployed' % rank)


def cmd_mounts(exp):
    out_dir = exp.base / 'mount-freeze'
    out_dir.mkdir(parents=True, exist_ok=False)
    for rank, command in enumerate(commands_for(exp, exp.attempts[0])):
        m = mount_manifest(exp, rank, command, exp.attempts[0])
        pr.write(out_dir / ('rank%d.json' % rank), {'rank': rank, 'utc': pr.now(), 'mounts': m})
        print('rank%d: %d fixed mount sources frozen' % (rank, len(m)))


def verify_next(exp, attempt):
    registered_attempts(exp)
    progress = exp.base / 'progress.json'
    done = json.loads(progress.read_text())['completed'] if progress.exists() else []
    require(len(done) < len(exp.attempts) and exp.attempts[len(done)] == attempt,
            'order violation: next is %s; got %s' % (
                exp.attempts[len(done)] if len(done) < len(exp.attempts) else 'none', attempt))
    return progress, done


def preflight(exp, attempt, commands):
    ra = '%s/attempts/%s' % (exp.remote, attempt)
    checks = {'collector_inputs': pr.collector_inputs_ok()}
    for r in (0, 1):
        pr.gpu_idle(r)
        pr.spark(r, 'test ! -e ' + ra)
        frozen = json.loads((exp.base / 'mount-freeze' / ('rank%d.json' % r)).read_text())
        require(mount_manifest(exp, r, commands[r], attempt) == frozen['mounts'],
                'rank%d fixed mount sources differ from the frozen manifest' % r)
        free = int(pr.spark(r, "df -B1 --output=avail /home | tail -1").stdout.strip())
        require(free > exp.min_free_gib * 1024 ** 3,
                'rank%d has under %d GiB free' % (r, exp.min_free_gib))
        checks['free_bytes_rank%d' % r] = free
    return checks


def collect(exp, attempt, caproot):
    name = '%s-%s' % (exp.key, attempt)
    script = ['set -euo pipefail', 'BASE="$HOME/gate/attempts/%s"' % name, 'mkdir "$BASE"']
    common = ('--endpoint http://127.0.0.1:{port} --model {model} --capture-dir "{cap}" '
              '--provenance "$HOME/gate/{name}/provenance-finalization/prov-ref.json" '
              '--thresholds "$HOME/gate/tests/exl3_validation/thresholds.json"').format(
                  port=pr.PORT, model=pr.SERVED_MODEL, cap=caproot, name=name)
    require(exp.suites in (('short16',), ('short16', 'long'), ('long',)),
            'unsupported suite scope')
    if 'short16' in exp.suites:
        script.append('python3 "$HOME/gate/tests/exl3_validation/collect_logits.py" %s '
                      '--cases "$HOME/gate/cases-short16.json" --out "$BASE/short16" '
                      '--timeout 1200 > "$BASE/short16.log" 2>&1' % common)
    long_cases = remote_long_cases(exp) if exp.long_cases is not None \
        else '$HOME/gate/cases-long.json'
    if 'long' in exp.suites:
        # Campaign long-suite procedure (gate-v2 run_attempt.py collect()).
        script.append('''python3 - "$BASE/long-snapshot.json" "{cap}" <<'PY2'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[2]); files={{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('logits-*')}}
json.dump({{'head_start_sequence':sum(1 for n in files if n.endswith('.json')),'prior_head_files':files}},open(sys.argv[1],'w'),indent=2)
PY2'''.format(cap=caproot))
        script.append('python3 "$HOME/gate/tests/exl3_validation/collect_phase_logits.py" %s '
                      '--cases "%s" --out "$BASE/long" '
                      '--phase-snapshot "$BASE/long-snapshot.json" --timeout 1800 '
                      '> "$BASE/long.log" 2>&1' % (common, long_cases))
    p = pr.spark(0, '\n'.join(script) + '\n', check=False)
    return p.returncode, p.stdout, p.stderr


def cmd_attempt(exp, attempt, dry_run):
    progress, done = verify_next(exp, attempt)
    commands = commands_for(exp, attempt)
    draft = experiment_draft(exp, json.loads((pr.CAMPAIGN_DRIVER / 'prov-ref.json').read_text()))
    names = [c[c.index('--name') + 1] for c in commands]
    if dry_run:
        print(json.dumps({'attempt': attempt, 'commands': commands, 'containers': names},
                         indent=2))
        return
    checks = preflight(exp, attempt, commands)
    root = exp.base / 'attempts' / attempt
    root.mkdir(parents=True, exist_ok=False)
    receipt = {'experiment': exp.key, 'attempt': attempt, 'status': 'started',
               'start_utc': pr.now(), 'containers': names, 'preflight': checks,
               'env_delta': exp.env, 'suites': list(exp.suites), 'dumps': exp.dumps,
               'bindings': {'runner_sha256': sha(Path(__file__)),
                            'probe_runner_sha256': sha(Path(pr.__file__)),
                            'registration_sha256': sha(exp.registration),
                            'finalizer_sha256': sha(exp.base / 'driver' / 'finalize-reference.py')},
               'stop_timeout_s': pr.STOP_TIMEOUT_S, 'steps': []}
    ra = '%s/attempts/%s' % (exp.remote, attempt)
    try:
        (root / 'profile').mkdir()
        for r in (0, 1):
            pr.write(root / 'profile' / ('ref-rank%d-create-command.json' % r), commands[r])
        pr.write(root / 'draft-prov-ref.json', draft)
        run(exp, attempt, commands, names, draft, root, ra, receipt)
    except BaseException as exc:  # recorded; the attempt still tears down
        receipt['error'] = '%s: %s' % (type(exc).__name__, exc)
    finally:
        try:
            teardown(exp, names, root, ra, receipt, attempt)
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


def run(exp, attempt, commands, names, draft, root, ra, receipt):
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
    shutil.copy2(exp.base / 'driver' / 'finalize-reference.py', fin / 'finalize_provenance.py')
    pr.write(fin / 'prov-ref.json', draft)
    shutil.copy2(pr.CAMPAIGN_DRIVER / 'reference-image-inspect.json', fin)
    env = dict(os.environ, GATE_V2_CONTAINER_PREFIX=names[0][:-2],
               PROBE_PROFILE_DIR=str(root / 'profile'))
    fp = subprocess.run([sys.executable, str(fin / 'finalize_provenance.py'), 'ref',
                         '%s-%s' % (exp.key, attempt)], env=env, cwd=fin, text=True,
                        capture_output=True)
    (fin / 'finalizer.stdout').write_text(fp.stdout)
    (fin / 'finalizer.stderr').write_text(fp.stderr)
    require(fp.returncode == 0, 'provenance finalizer failed: ' + fp.stderr[-3000:])
    step('provenance_finalized')
    if exp.dumps:
        for r in (0, 1):
            pr.spark(r, 'test ! -e {d}/capture-enabled && touch {d}/capture-enabled'.format(
                d=ra + '/dump'))
        step('dump_markers_set')
    rc, out, err = collect(exp, attempt, ra + '/capture')
    (root / 'collector.stdout').write_text(out)
    (root / 'collector.stderr').write_text(err)
    receipt['collection_exit_code'] = rc
    step('collected', exit_code=rc)
    require(rc == 0, 'collection failed with exit %d' % rc)


def teardown(exp, names, root, ra, receipt, attempt):
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
            pr.guarded(receipt, 'cache-drop rank%d' % r, lambda: pr.spark(
                r, 'sudo -n rm -rf -- %s/cache' % ra))
    ok &= bool(pr.guarded(receipt, 'archive', lambda: archive(exp, attempt, root, ra, receipt),
                          False))
    ok &= not receipt.get('teardown_errors')
    receipt['teardown_ok'] = bool(ok)
    pr.write(root / 'attempt.json', receipt)


def archive(exp, attempt, root, ra, receipt):
    ok = True
    name = '%s-%s' % (exp.key, attempt)
    pulls = {0: [('gate', 'tar --ignore-failed-read -C $HOME/gate -cf - '
                  'attempts/{n} {n}'.format(n=name))], 1: []}
    for r in (0, 1):
        pulls[r].append(('capture', 'sudo -n tar -C %s -cf - capture' % ra))
        if exp.dumps:
            size = pr.spark(r, 'sudo -n du -sb %s/dump | cut -f1' % ra,
                            check=False).stdout.strip()
            receipt['dump_bytes_rank%d' % r] = int(size) if size.isdigit() else None
            sums = pr.spark(r, 'cd %s/dump && sudo -n find . -type f -exec sha256sum {} + | '
                               'sort -k2' % ra, check=False, timeout=7200)
            (root / ('dump-rank%d-sha256.txt' % r)).write_text(sums.stdout)
            if size.isdigit() and int(size) <= exp.local_copy_limit:
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
    parser.add_argument('experiment', choices=sorted(EXPERIMENTS))
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('mounts')
    sub.add_parser('deploy')
    att = sub.add_parser('attempt')
    att.add_argument('name')
    att.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    exp = EXPERIMENTS[args.experiment]
    if args.cmd == 'mounts':
        cmd_mounts(exp)
    elif args.cmd == 'deploy':
        cmd_deploy(exp)
    else:
        cmd_attempt(exp, args.name, args.dry_run)


if __name__ == '__main__':
    main()
