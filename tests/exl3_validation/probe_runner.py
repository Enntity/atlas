# SPDX-License-Identifier: AGPL-3.0-only
"""Append-only runner for the registered runtime-mechanism probe attempts.

Executes ``docs/design/exl3-runtime-mechanism-probe-registration.md``: six
fresh-process reference attempts in the frozen order, the campaign's
reference profile (gate-v2 create commands, observer and collectors), an
isolated writable copy of one frozen per-rank cache snapshot per attempt,
and an arm difference limited to the probe telemetry environment. nsys is
not used in either arm: it records no kernels from fork-without-exec CUDA
children (vLLM's default worker start), a registered tooling limitation
(``tooling-nsys-fork-check*``). Shape follows the gate-v2 ``run_attempt.py``.

A started attempt is terminal: it is never retried, replaced or deleted.
Every failure still stops, logs and archives what exists, then advances
the order (a failed attempt is retained and reported).

Usage:
    probe_runner.py profiler            # bind nsys version on both ranks
    probe_runner.py freeze              # per-rank cache snapshot (once)
    probe_runner.py deploy              # copy + verify the overlay package
    probe_runner.py mounts              # freeze fixed mount sources (once)
    probe_runner.py attempt NAME [--dry-run]
"""
import argparse
import base64
import concurrent.futures
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from image_cycle import sha, require, text
import probe_records

EV = Path('/home/colum/Workspace/atlas-exl3-validation-20261010')
BASE = EV / 'probe-runtime-mechanism-20261014'
CAMPAIGN = EV / 'gate-v2-qualification-20261013-luna1'
CAMPAIGN_DRIVER = CAMPAIGN / 'driver'
HERE = Path(__file__).resolve().parent
REGISTRATION = (HERE.parent.parent / 'docs' / 'design'
                / 'exl3-runtime-mechanism-probe-registration.md')
PACKAGE = BASE / 'probe-overlay'
PROBE_CONFIG = BASE / 'probe-config.json'
FINALIZER = BASE / 'driver' / 'finalize-probe-reference.py'

REMOTE = '/home/colum/probe-rm-20261014'
REMOTE_OVERLAY = REMOTE + '/probe-overlay'
REMOTE_SNAPSHOT = REMOTE + '/cache-snapshot'
REMOTE_TOOLS = REMOTE + '/tools'
NSYS_DIR = '/opt/nvidia/nsight-systems/2025.3.2'
NSYS_BIN = NSYS_DIR + '/target-linux-sbsa-armv8/nsys'
IMAGE = 'glm53-flash-mikeroz:dt-carry20'
TELEMETRY = '/tmp/probe-telemetry'
PORT = 8012
SERVED_MODEL = 'GLM-5.3-Flash-Uncensored-EXL3-mul1'
# Same in both arms; vLLM's own shutdown completes well inside this.
STOP_TIMEOUT_S = 120
# Telemetry larger than this stays on the Spark (home, durable) with a hash.
LOCAL_COPY_LIMIT = 8 * 1024 ** 3
CACHE_DESTS = {'vllm': '/root/.cache/vllm', 'triton': '/root/.triton/cache',
               'tilelang': '/root/.tilelang/cache', 'nv': '/root/.nv/ComputeCache',
               'tune': '/tmp/reference-autotune'}
ARMS = dict(probe_records.ATTEMPTS)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# --- pure command/draft construction (unit tested) ---------------------------

def _volume_index(command):
    out = {}
    for i, arg in enumerate(command[:-1]):
        if arg == '--volume':
            parts = command[i + 1].split(':')
            require(len(parts) in (2, 3), 'unexpected volume spec: ' + command[i + 1])
            require(parts[1] not in out, 'duplicate volume destination ' + parts[1])
            out[parts[1]] = i + 1
    return out


def cache_sources(command):
    """The campaign's per-rank cache host paths (vllm parent + tune dir).

    triton/tilelang/nv must be subdirectories of the vllm cache parent, so
    copying the parent once captures every compiler cache category.
    """
    vols = _volume_index(command)
    src = {key: command[vols[dest]].split(':')[0]
           for key, dest in CACHE_DESTS.items()}
    for key in ('triton', 'tilelang', 'nv'):
        require(src[key] == src['vllm'] + '/' + key,
                key + ' cache is not a subdirectory of the vllm cache parent')
    return {'vllm': src['vllm'], 'tune': src['tune']}


def probe_command(command, *, attempt, rank, config_json):
    """Derive one rank's create command from the campaign reference command."""
    require(attempt in ARMS, 'unregistered attempt ' + attempt)
    arm = ARMS[attempt]
    c = list(command)
    c[c.index('--name') + 1] = 'probe-rm-%s-ref-%d' % (attempt, rank)
    ra = '%s/attempts/%s' % (REMOTE, attempt)
    vols = _volume_index(c)
    replace = {'/tmp/reference-logits': ra + '/capture:/tmp/reference-logits',
               '/opt/atlas-logits-patch': REMOTE_OVERLAY + ':/opt/atlas-logits-patch:ro',
               '/root/.cache/vllm': ra + '/cache/vllm:/root/.cache/vllm',
               '/root/.triton/cache': ra + '/cache/vllm/triton:/root/.triton/cache',
               '/root/.tilelang/cache': ra + '/cache/vllm/tilelang:/root/.tilelang/cache',
               '/root/.nv/ComputeCache': ra + '/cache/vllm/nv:/root/.nv/ComputeCache',
               '/tmp/reference-autotune': ra + '/cache/tune:/tmp/reference-autotune'}
    cache_sources(command)
    for dest, spec in replace.items():
        require(dest in vols, 'campaign command lacks volume ' + dest)
        c[vols[dest]] = spec
    require(c.count(IMAGE) == 1 and c[-1] == '/start.sh'
            and c.index(IMAGE) == len(c) - 2,
            'campaign command must end with the reference image and /start.sh')
    require(c[c.index('--entrypoint') + 1] == 'bash', 'campaign entrypoint must be bash')
    extra = ['--volume', ra + '/telemetry:' + TELEMETRY]
    if arm == 'on':
        extra += ['--env', 'ATLAS_PROBE_TELEMETRY_DIR=' + TELEMETRY,
                  '--env', 'ATLAS_PROBE_CONFIG=' + config_json]
    image_at = c.index(IMAGE)
    return c[:image_at] + extra + c[image_at:]


def probe_draft(campaign_draft, *, attempt, config_json, package_files,
                package_sha):
    """Provenance draft for the finalizer: campaign draft + probe deltas."""
    draft = json.loads(json.dumps(campaign_draft))
    if ARMS[attempt] == 'on':
        draft['environment'] = dict(draft['environment'],
                                    ATLAS_PROBE_TELEMETRY_DIR=TELEMETRY,
                                    ATLAS_PROBE_CONFIG=config_json)
    draft['capture_hooks_sha256'] = {
        name: package_files[name] for name in draft['capture_hooks_sha256']}
    draft['hook_package_sha256'] = package_sha
    return draft


def config_json_value(config):
    """The exact ATLAS_PROBE_CONFIG value (compact, key-sorted)."""
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


# --- remote plumbing ----------------------------------------------------------

def spark(rank, command, check=True, input_text=None, timeout=None):
    p = subprocess.run(['spark%d' % rank, command], input=input_text, text=True,
                       capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError('spark%d failed (%d): %s %s' % (
            rank, p.returncode, p.stderr[-3000:], p.stdout[-1000:]))
    return p


def remote_python(rank, source, payload, sudo=True, timeout=None):
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    script = base64.b64encode(source.encode()).decode()
    cmd = ("%spython3 -c 'import base64; exec(compile(base64.b64decode(\"%s\"),"
           "\"<probe-remote>\",\"exec\"))' %s" % ('sudo -n ' if sudo else '', script, encoded))
    return json.loads(spark(rank, cmd, timeout=timeout).stdout)


MANIFEST_SOURCE = r'''import base64,hashlib,json,os,sys
p=json.loads(base64.b64decode(sys.argv[1])); out={}
for label,root in p['roots'].items():
 files=[]; links=[]
 if not os.path.isdir(root): out[label]={'root':root,'state':'missing'}; continue
 for d,dirs,names in os.walk(root):
  dirs.sort()
  for n in sorted(names)+[x for x in dirs if os.path.islink(os.path.join(d,x))]:
   f=os.path.join(d,n); rel=os.path.relpath(f,root)
   if os.path.islink(f): links.append({'path':rel,'target':os.readlink(f)}); continue
   h=hashlib.sha256()
   with open(f,'rb') as fh:
    for b in iter(lambda:fh.read(1<<20),b''): h.update(b)
   files.append({'path':rel,'size':os.path.getsize(f),'sha256':h.hexdigest()})
 files.sort(key=lambda x:x['path']); links.sort(key=lambda x:x['path'])
 body=json.dumps({'files':files,'symlinks':links},sort_keys=True).encode()
 out[label]={'root':root,'state':'present','files':files,'symlinks':links,'content_sha256':hashlib.sha256(body).hexdigest()}
print(json.dumps(out))'''


def manifest(rank, roots):
    return remote_python(rank, MANIFEST_SOURCE, {'roots': roots})


def campaign_command(rank):
    return json.loads((CAMPAIGN_DRIVER / 'profile'
                       / ('ref-rank%d-create-command.json' % rank)).read_text())


def write(path, value):
    Path(path).write_text(text(value))


def gpu_idle(rank):
    p = spark(rank, "nvidia-smi --query-compute-apps=pid --format=csv,noheader; "
                    "docker ps --format '{{.Names}}'")
    require(p.stdout.strip() == '', 'spark%d is not idle: %s' % (rank, p.stdout))


# --- subcommands ----------------------------------------------------------------

def cmd_profiler():
    out_dir = BASE / 'profiler'
    out_dir.mkdir(exist_ok=False)
    for rank in (0, 1):
        spark(rank, 'mkdir -p %s' % REMOTE_TOOLS)
        for name in ('probe_records.py', 'image_cycle.py'):
            spark(rank, 'cat > %s/%s' % (REMOTE_TOOLS, name),
                  input_text=(HERE / name).read_text())
        remote_out = '%s/profiler-rank%d.json' % (REMOTE, rank)
        spark(rank, 'cd %s && python3 probe_records.py profiler bind --nsys %s --out %s'
              % (REMOTE_TOOLS, NSYS_BIN, remote_out))
        (out_dir / ('rank%d.json' % rank)).write_text(spark(rank, 'cat ' + remote_out).stdout)
    print(json.dumps({r: json.loads((out_dir / ('rank%d.json' % r)).read_text())
                      for r in (0, 1)}, indent=2))


def cmd_freeze():
    out_dir = BASE / 'cache-freeze'
    out_dir.mkdir(exist_ok=False)
    for rank in (0, 1):
        gpu_idle(rank)
        src = cache_sources(campaign_command(rank))
        spark(rank, 'sudo -n test ! -e %s && sudo -n mkdir -p %s && '
                    'sudo -n cp -a %s %s/vllm && sudo -n cp -a %s %s/tune'
              % (REMOTE_SNAPSHOT, REMOTE_SNAPSHOT, src['vllm'], REMOTE_SNAPSHOT,
                 src['tune'], REMOTE_SNAPSHOT), timeout=3600)
        m = manifest(rank, {'original_vllm': src['vllm'], 'original_tune': src['tune'],
                            'snapshot_vllm': REMOTE_SNAPSHOT + '/vllm',
                            'snapshot_tune': REMOTE_SNAPSHOT + '/tune'})
        for key in ('vllm', 'tune'):
            require(m['original_' + key]['state'] == 'present'
                    and m['original_' + key]['content_sha256']
                    == m['snapshot_' + key]['content_sha256'],
                    'rank%d %s snapshot differs from the original' % (rank, key))
        write(out_dir / ('rank%d.json' % rank),
              {'rank': rank, 'captured_utc': now(), 'sources': src, 'manifest': m})
        print('rank%d snapshot vllm=%s tune=%s' % (
            rank, m['snapshot_vllm']['content_sha256'], m['snapshot_tune']['content_sha256']))


def package_files():
    binding = json.loads((PACKAGE / 'package-binding.json').read_text())
    for name, digest in binding['file_sha256'].items():
        require(sha(PACKAGE / name) == digest, 'local package file differs: ' + name)
    return binding['file_sha256'], sha(PACKAGE / 'package-binding.json')


def cmd_deploy():
    files, package_sha = package_files()
    tar = subprocess.run(['tar', '-C', str(PACKAGE), '--exclude=__pycache__', '-cf', '-',
                          *sorted(files), 'package-binding.json'],
                         capture_output=True, check=True).stdout
    for rank in (0, 1):
        p = subprocess.run(['spark%d' % rank, 'test ! -e %s && mkdir -p %s && tar -C %s -xf -'
                            % (REMOTE_OVERLAY, REMOTE_OVERLAY, REMOTE_OVERLAY)],
                           input=tar, capture_output=True)
        require(p.returncode == 0, 'deploy rank%d failed: %s' % (rank, p.stderr[-2000:]))
        spark(rank, 'chmod -R a-w %s' % REMOTE_OVERLAY)
        m = manifest(rank, {'overlay': REMOTE_OVERLAY})['overlay']
        remote = {f['path']: f['sha256'] for f in m['files']}
        require(remote == dict(files, **{'package-binding.json': package_sha}),
                'rank%d deployed overlay differs from the package binding' % rank)
        print('rank%d overlay verified (%d files, binding %s)' % (rank, len(remote), package_sha))


def create_and_start(rank, command):
    source = r'''import base64,json,subprocess,sys
c=json.loads(base64.b64decode(sys.argv[1])); name=c[c.index('--name')+1]
if subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0: raise SystemExit('container name exists; refusing reuse: '+name)
subprocess.run(c,check=True,stdout=subprocess.DEVNULL); subprocess.run(['docker','start',name],check=True,stdout=subprocess.DEVNULL)
print(json.dumps({'container':name,'image':subprocess.check_output(['docker','inspect','--format','{{.Image}}',name],text=True).strip()}))'''
    encoded = base64.b64encode(json.dumps(command).encode()).decode()
    script = base64.b64encode(source.encode()).decode()
    return json.loads(spark(rank, "python3 -c 'import base64; exec(compile(base64.b64decode("
                                  "\"%s\"),\"<create>\",\"exec\"))' %s" % (script, encoded)).stdout)


def wait_ready(names, timeout=3600):
    end = time.time() + timeout
    while time.time() < end:
        if spark(0, 'curl -fsS --max-time 3 http://127.0.0.1:%d/v1/models >/dev/null' % PORT,
                 check=False).returncode == 0:
            return
        for rank, name in enumerate(names):
            state = spark(rank, "docker inspect --format '{{.State.Running}}' " + name,
                          check=False).stdout.strip()
            require(state == 'true', '%s stopped before the API became ready' % name)
        time.sleep(10)
    raise RuntimeError('API did not become ready on rank0 port %d' % PORT)


NCCL_PIDS = r'''import base64,json,os,subprocess,sys
p=json.loads(base64.b64decode(sys.argv[1]))
top=subprocess.check_output(['docker','top',p['name'],'-eo','pid'],text=True).split()[1:]
out=[]
for pid in top:
 try: maps=open('/proc/%s/maps'%pid).read()
 except OSError: continue
 if 'libnccl' in maps: out.append(int(pid))
print(json.dumps(sorted(out)))'''


def nccl_live(rank, name, attempt, root):
    pids = remote_python(rank, NCCL_PIDS, {'name': name})
    require(pids, 'rank%d: no container process maps libnccl' % rank)
    records = []
    for pid in pids:
        remote_out = '%s/attempts/%s/nccl-live-rank%d-pid%d.json' % (REMOTE, attempt, rank, pid)
        spark(rank, 'cd %s && sudo -n python3 probe_records.py nccl live --maps /proc/%d/maps '
                    '--proc-root /proc/%d/root --rank %d --out %s'
              % (REMOTE_TOOLS, pid, pid, rank, remote_out))
        records.append(json.loads(spark(rank, 'sudo -n cat ' + remote_out).stdout))
    write(root / ('nccl-live-rank%d.json' % rank), {'pids': pids, 'records': records})


def collect(attempt, caproot):
    name = 'probe-rm-' + attempt
    remote = '''set -euo pipefail
BASE="$HOME/gate/attempts/{name}"
mkdir "$BASE"
python3 "$HOME/gate/tests/exl3_validation/collect_logits.py" --endpoint http://127.0.0.1:{port} --model {model} --cases "$HOME/gate/cases-short16.json" --capture-dir "{cap}" --out "$BASE/short16" --provenance "$HOME/gate/{name}/provenance-finalization/prov-ref.json" --thresholds "$HOME/gate/tests/exl3_validation/thresholds.json" --timeout 1200 > "$BASE/short16.log" 2>&1
python3 - "$BASE/long-snapshot.json" "{cap}" <<'PY2'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[2]); files={{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('logits-*')}}
json.dump({{'head_start_sequence':sum(1 for n in files if n.endswith('.json')),'prior_head_files':files}},open(sys.argv[1],'w'),indent=2)
PY2
python3 "$HOME/gate/tests/exl3_validation/collect_phase_logits.py" --endpoint http://127.0.0.1:{port} --model {model} --cases "$HOME/gate/cases-long.json" --capture-dir "{cap}" --out "$BASE/long" --provenance "$HOME/gate/{name}/provenance-finalization/prov-ref.json" --thresholds "$HOME/gate/tests/exl3_validation/thresholds.json" --phase-snapshot "$BASE/long-snapshot.json" --timeout 1800 > "$BASE/long.log" 2>&1
'''.format(name=name, port=PORT, model=SERVED_MODEL, cap=caproot)
    p = spark(0, remote, check=False)
    return p.returncode, p.stdout, p.stderr


def collector_inputs_ok():
    ledger = json.loads((CAMPAIGN_DRIVER / 'input-sha256-ledger.json').read_text())['inputs']
    remote = spark(0, 'cd ~/gate && sha256sum cases-short16.json cases-long.json '
                      'tests/exl3_validation/collect_logits.py '
                      'tests/exl3_validation/collect_phase_logits.py '
                      'tests/exl3_validation/parity_protocol.py '
                      'tests/exl3_validation/thresholds.json').stdout
    got = {Path(line.split()[1]).name: line.split()[0] for line in remote.splitlines()}
    for name, digest in got.items():
        require(ledger[name]['sha256'] == digest, 'collector input differs from ledger: ' + name)
    return got


def verify_next(attempt):
    progress = BASE / 'progress.json'
    done = json.loads(progress.read_text())['completed'] if progress.exists() else []
    order = [name for name, _ in probe_records.ATTEMPTS]
    require(len(done) < len(order) and order[len(done)] == attempt,
            'order violation: next is %s; got %s' % (
                order[len(done)] if len(done) < len(order) else 'none', attempt))
    return progress, done


def plan(attempt):
    config = json.loads(PROBE_CONFIG.read_text())
    config_json = config_json_value(config)
    commands = [probe_command(campaign_command(r), attempt=attempt, rank=r,
                              config_json=config_json) for r in (0, 1)]
    files, package_sha = package_files()
    draft = probe_draft(json.loads((CAMPAIGN_DRIVER / 'prov-ref.json').read_text()),
                        attempt=attempt, config_json=config_json,
                        package_files=files, package_sha=package_sha)
    return config, commands, draft, package_sha


MOUNT_SOURCE = r'''import base64,hashlib,json,os,sys
p=json.loads(base64.b64decode(sys.argv[1])); out={}
for src in p['sources']:
 if os.path.isfile(src):
  h=hashlib.sha256(open(src,'rb').read()).hexdigest(); out[src]={'type':'file','sha256':h}
 elif os.path.isdir(src):
  n=sum(len(f) for _,_,f in os.walk(src)); out[src]={'type':'dir','files':n}
 else: out[src]={'type':'missing'}
print(json.dumps(out,sort_keys=True))'''


def fixed_mount_sources(command, attempt):
    """Volume sources the runner does not create (profile + overlay)."""
    ra = '%s/attempts/%s/' % (REMOTE, attempt)
    return sorted(command[i + 1].split(':')[0] for i, a in enumerate(command[:-1])
                  if a == '--volume' and not command[i + 1].startswith(ra))


def mount_manifest(rank, command, attempt):
    m = remote_python(rank, MOUNT_SOURCE,
                      {'sources': fixed_mount_sources(command, attempt)})
    missing = [src for src, v in m.items() if v['type'] == 'missing']
    require(not missing, 'rank%d mount sources missing (docker would create '
                         'empty dirs): %s' % (rank, missing))
    return m


def cmd_mounts():
    """Freeze every fixed mount source's type/hash once, before off01."""
    out_dir = BASE / 'mount-freeze'
    out_dir.mkdir(exist_ok=False)
    for rank in (0, 1):
        command = probe_command(campaign_command(rank), attempt='off01', rank=rank,
                                config_json='{}')
        m = mount_manifest(rank, command, 'off01')
        write(out_dir / ('rank%d.json' % rank), {'rank': rank, 'utc': now(), 'mounts': m})
        print('rank%d: %d fixed mount sources frozen' % (rank, len(m)))


def preflight(attempt, commands):
    """Everything checkable before any host change; failure consumes nothing."""
    ra = '%s/attempts/%s' % (REMOTE, attempt)
    checks = {'collector_inputs': collector_inputs_ok()}
    for r in (0, 1):
        gpu_idle(r)
        spark(r, 'test ! -e ' + ra)
        frozen = json.loads((BASE / 'mount-freeze' / ('rank%d.json' % r)).read_text())
        require(mount_manifest(r, commands[r], attempt) == frozen['mounts'],
                'rank%d fixed mount sources differ from the frozen manifest' % r)
        free = int(spark(r, "df -B1 --output=avail /home | tail -1").stdout.strip())
        require(free > 12 * 1024 ** 3, 'rank%d has under 12 GiB free' % r)
        checks['free_bytes_rank%d' % r] = free
    files, _ = package_files()
    for r in (0, 1):
        remote = {f['path']: f['sha256'] for f in
                  manifest(r, {'o': REMOTE_OVERLAY})['o']['files']}
        require(all(remote.get(k) == v for k, v in files.items()),
                'rank%d deployed overlay differs from the package' % r)
    return checks


def cmd_attempt(attempt, dry_run):
    progress, done = verify_next(attempt)
    config, commands, draft, package_sha = plan(attempt)
    names = [c[c.index('--name') + 1] for c in commands]
    if dry_run:
        print(json.dumps({'attempt': attempt, 'arm': ARMS[attempt], 'commands': commands,
                          'containers': names, 'draft_env_delta': sorted(
                              set(draft['environment']) - set(json.loads(
                                  (CAMPAIGN_DRIVER / 'prov-ref.json').read_text())['environment'])),
                          'capture_hooks_sha256': draft['capture_hooks_sha256']}, indent=2))
        return
    checks = preflight(attempt, commands)
    # Point of no return: from here the attempt is consumed whatever happens.
    root = BASE / 'attempts' / attempt
    root.mkdir(parents=True, exist_ok=False)
    probe_records.stamp(REGISTRATION, root)
    receipt = {'attempt': attempt, 'arm': ARMS[attempt], 'status': 'started',
               'start_utc': now(), 'containers': names, 'preflight': checks,
               'bindings': {'runner_sha256': sha(Path(__file__)),
                            'wiring_sha256': sha(HERE / 'probe_wiring.py'),
                            'registration_sha256': sha(REGISTRATION),
                            'package_binding_sha256': package_sha,
                            'probe_config_sha256': sha(PROBE_CONFIG),
                            'finalizer_sha256': sha(FINALIZER)},
               'stop_timeout_s': STOP_TIMEOUT_S, 'steps': []}
    ra = '%s/attempts/%s' % (REMOTE, attempt)
    try:
        (root / 'profile').mkdir()
        for r in (0, 1):
            write(root / 'profile' / ('ref-rank%d-create-command.json' % r), commands[r])
        write(root / 'probe-draft-prov-ref.json', draft)
        run_attempt(attempt, commands, names, draft, root, ra, receipt)
    except BaseException as exc:  # recorded; the attempt still tears down
        receipt['error'] = '%s: %s' % (type(exc).__name__, exc)
    finally:
        try:
            teardown(attempt, names, root, ra, receipt)
        except BaseException as exc:  # noqa: BLE001 - never leave order stuck
            receipt['teardown_ok'] = False
            receipt.setdefault('teardown_errors', []).append(
                'teardown: %s: %s' % (type(exc).__name__, exc))
        receipt['status'] = ('complete' if 'error' not in receipt
                             and receipt.get('teardown_ok') else 'failed')
        receipt['completed_utc'] = now()
        write(root / 'attempt.json', receipt)
        done.append(attempt)
        write(progress, {'completed': done, 'last_attempt_status': receipt['status']})
    if receipt['status'] != 'complete':
        raise SystemExit('%s failed; evidence preserved; no retry. See %s' % (attempt, root))
    print(json.dumps({'attempt': attempt, 'status': 'complete', 'path': str(root)}))


def run_attempt(attempt, commands, names, draft, root, ra, receipt):
    def step(label, **extra):
        receipt['steps'].append(dict({'step': label, 'utc': now()}, **extra))
        write(root / 'attempt.json', receipt)

    step('preflight_passed')
    for r in (0, 1):
        spark(r, 'mkdir -p {ra}/capture {ra}/telemetry {ra}/cache && '
                 'sudo -n cp -a {snap}/vllm {ra}/cache/vllm && '
                 'sudo -n cp -a {snap}/tune {ra}/cache/tune'.format(
                     ra=ra, snap=REMOTE_SNAPSHOT), timeout=3600)
        pre = manifest(r, {'vllm': ra + '/cache/vllm', 'tune': ra + '/cache/tune'})
        frozen = json.loads((BASE / 'cache-freeze' / ('rank%d.json' % r)).read_text())
        for key in ('vllm', 'tune'):
            require(pre[key]['content_sha256']
                    == frozen['manifest']['snapshot_' + key]['content_sha256'],
                    'rank%d %s attempt cache differs from the frozen snapshot' % (r, key))
        write(root / ('cache-pre-rank%d.json' % r), pre)
    step('cache_initialized')
    receipt['containers_may_exist'] = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        starts = list(pool.map(lambda r: create_and_start(r, commands[r]), (0, 1)))
    step('servers_started', containers=starts)
    wait_ready(names)
    step('api_ready')
    if attempt == 'off01':
        for r in (0, 1):
            nccl_live(r, names[r], attempt, root)
        step('nccl_live_bound')
    fin = root / 'finalizer'
    fin.mkdir()
    shutil.copy2(FINALIZER, fin / 'finalize_provenance.py')
    write(fin / 'prov-ref.json', draft)
    shutil.copy2(CAMPAIGN_DRIVER / 'reference-image-inspect.json', fin)
    env = dict(__import__('os').environ, GATE_V2_CONTAINER_PREFIX=names[0][:-2],
               PROBE_PROFILE_DIR=str(root / 'profile'))
    fp = subprocess.run([sys.executable, str(fin / 'finalize_provenance.py'), 'ref',
                         'probe-rm-' + attempt], env=env, cwd=fin, text=True,
                        capture_output=True)
    (fin / 'finalizer.stdout').write_text(fp.stdout)
    (fin / 'finalizer.stderr').write_text(fp.stderr)
    require(fp.returncode == 0, 'provenance finalizer failed: ' + fp.stderr[-3000:])
    step('provenance_finalized')
    rc, out, err = collect(attempt, ra + '/capture')
    (root / 'collector.stdout').write_text(out)
    (root / 'collector.stderr').write_text(err)
    receipt['collection_exit_code'] = rc
    step('collected', exit_code=rc)
    require(rc == 0, 'collection failed with exit %d' % rc)


def guarded(receipt, label, fn, default=None):
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 - recorded, never masked
        receipt.setdefault('teardown_errors', []).append(
            '%s: %s: %s' % (label, type(exc).__name__, exc))
        return default


def teardown(attempt, names, root, ra, receipt):
    ok = True
    stops = []
    if receipt.get('containers_may_exist'):
        def stop(r):
            exists = spark(r, 'docker inspect %s >/dev/null 2>&1' % names[r],
                           check=False).returncode == 0
            if not exists:
                return {'rank': r, 'container': names[r], 'exists': False}
            p = guarded(receipt, 'stop rank%d' % r, lambda: spark(
                r, 'docker stop --timeout %d %s' % (STOP_TIMEOUT_S, names[r]),
                check=False, timeout=STOP_TIMEOUT_S + 300))
            state = guarded(receipt, 'state rank%d' % r, lambda: json.loads(spark(
                r, "docker inspect --format '{{json .State}}' " + names[r]).stdout))
            logs = guarded(receipt, 'logs rank%d' % r, lambda: spark(
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
    write(root / 'attempt.json', receipt)
    for r in (0, 1):
        post = guarded(receipt, 'cache-post rank%d' % r, lambda: manifest(
            r, {'vllm': ra + '/cache/vllm', 'tune': ra + '/cache/tune'}))
        if post is None:
            ok = False
        else:
            write(root / ('cache-post-rank%d.json' % r), post)
    ok &= bool(guarded(receipt, 'archive', lambda: archive(attempt, root, ra, receipt), False))
    ok &= not receipt.get('teardown_errors')
    receipt['teardown_ok'] = bool(ok)
    write(root / 'attempt.json', receipt)


def archive(attempt, root, ra, receipt):
    ok = True
    pulls = {0: [('gate', 'tar --ignore-failed-read -C $HOME/gate -cf - attempts/probe-rm-{a} probe-rm-{a}'
                  .format(a=attempt))], 1: []}
    for r in (0, 1):
        pulls[r].append(('capture', 'sudo -n tar -C %s -cf - capture' % ra))
        size = spark(r, 'sudo -n du -sb %s/telemetry | cut -f1' % ra, check=False).stdout.strip()
        receipt['telemetry_bytes_rank%d' % r] = int(size) if size.isdigit() else None
        if size.isdigit() and int(size) <= LOCAL_COPY_LIMIT:
            pulls[r].append(('telemetry', 'sudo -n tar -C %s -cf - telemetry' % ra))
        else:
            sums = spark(r, 'cd %s/telemetry && sudo -n find . -type f -exec sha256sum {} + | '
                            'sort -k2' % ra, check=False, timeout=7200)
            (root / ('telemetry-rank%d-remote-sha256.txt' % r)).write_text(sums.stdout)
            receipt['telemetry_rank%d_retained_at' % r] = 'spark%d:%s/telemetry' % (r, ra)
    archives = {}
    for r, items in pulls.items():
        for label, command in items:
            target = root / ('rank%d-%s.tar' % (r, label))
            with target.open('xb') as out:
                p = subprocess.run(['spark%d' % r, command], stdout=out,
                                   stderr=subprocess.PIPE)
            archives[target.name] = {'exit': p.returncode, 'sha256': sha(target),
                                     'bytes': target.stat().st_size}
            ok &= p.returncode == 0
    receipt['archives'] = archives
    return ok


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('profiler')
    sub.add_parser('freeze')
    sub.add_parser('deploy')
    sub.add_parser('mounts')
    att = sub.add_parser('attempt')
    att.add_argument('name', choices=[n for n, _ in probe_records.ATTEMPTS])
    att.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.cmd == 'profiler':
        cmd_profiler()
    elif args.cmd == 'freeze':
        cmd_freeze()
    elif args.cmd == 'deploy':
        cmd_deploy()
    elif args.cmd == 'mounts':
        cmd_mounts()
    else:
        cmd_attempt(args.name, args.dry_run)


if __name__ == '__main__':
    main()
