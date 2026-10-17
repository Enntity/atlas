# SPDX-License-Identifier: AGPL-3.0-only
"""Long-running TP=2 GLM-5.3 EXL3 server with the full decode-performance stack.

  python3 tests/exl3_validation/serve_perf.py start [--exact] [--max-seq-len N]
  python3 tests/exl3_validation/serve_perf.py status
  python3 tests/exl3_validation/serve_perf.py stop

`start` takes the shared bench lock (/tmp/atlas-bench.lock) through a detached
holder process before touching the GPUs. It keeps the lock for the server's
lifetime, so A/B benches queue instead of colliding with the server. `stop`
removes the containers and releases the lock. Rank 0 serves the OpenAI and
Anthropic API on spark0 (10.50.0.10) port 8000.

Containers come from bench_tp2's registered descriptors (same NCCL/RoCE
setup), with a frozen copy of the dev binary bind-mounted. The default stack
is the fastest measured one, including the numerics-changing flags. --exact
drops those flags and leaves the output-identical stack. Prefix caching
(KV radix tree + KDA/SSM snapshots) and decode graphs are on.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_tp2  # noqa: E402

IMAGE = 'atlas-exl3-validation:src-4ee96c12b44c-rust1.97.1'
BINARY = '/home/colum/atlas-serve/spark.main-34d94f34'  # merged main 34d94f34, frozen copy
NAMES = {0: 'glm53-exl3-perf-0', 1: 'glm53-exl3-perf-1'}
HOLD = Path('/tmp/atlas-serve.hold')
LOCK = '/tmp/atlas-bench.lock'

GRAPHS = ['ATLAS_NO_DECODE_GRAPHS=0', 'ATLAS_NO_DECODE_GRAPHS_MULTISEQ=0', 'ATLAS_DEBUG_NO_GRAPH=0',
          'ATLAS_EP_GRAPHS=1', 'ATLAS_KDA_FLA_GRAPH=1']
EXACT = ['ATLAS_EXL3_SPLIT_MMA=1', 'ATLAS_EXL3_SPLIT_MHC=1', 'ATLAS_RDMA_ALLREDUCE=1',
         'ATLAS_EXL3_COOP_GEMM=1', 'ATLAS_EXL3_COOP_MOE=1', 'ATLAS_EXL3_COOP_FUSED_QKV=1',
         'ATLAS_EXL3_PAR_SMALL=1', 'ATLAS_EXL3_COOP_MOE_V2=1', 'ATLAS_EXL3_COOP_MOE_V3=1',
         'ATLAS_EXL3_COOP_MOE_V6=1', 'ATLAS_EXL3_COOP_MOE_V7=1', 'ATLAS_EXL3_SHARED_OVERLAP=1',
         'ATLAS_DECODE_ARENA_FILL=skip',
         # Grouped fat routed MoE for prefill: ~12x TTFT, output-identical (m1-srv-*).
         'ATLAS_EXL3_COOP_MOE_FAT_MULTIROW=1']
FLAGGED = ['ATLAS_EXL3_SHARED_PACKED=1', 'ATLAS_GLM_SPARSE_DECODE_FP8_SPLIT=1']


def serve_argv(rank, env, max_seq_len):
    argv = bench_tp2.create_argv(rank, IMAGE, NAMES[rank], env, binary=BINARY, cpuset='15-19')
    argv[argv.index('--port') + 1] = '8000'
    argv[argv.index('--max-seq-len') + 1] = str(max_seq_len)
    argv[argv.index('--request-timeout') + 1] = '3600'  # long prompts at 128-token prefill chunks
    return argv + ['--enable-prefix-caching']


def holder_running():
    return HOLD.exists()


def take_lock():
    if holder_running():
        return
    subprocess.Popen(['setsid', 'flock', LOCK, 'sh', '-c',
                      f'touch {HOLD}; while [ -e {HOLD} ]; do sleep 15; done'],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('waiting for the bench lock (queued A/Bs finish first)...', flush=True)
    while not HOLD.exists():
        time.sleep(5)


def start(args):
    env = GRAPHS + EXACT + ([] if args.exact else FLAGGED)
    if not args.no_mla_prefix_skip:
        # Lets prefix-cache hits skip MLA prefill. Off by default in Atlas, which
        # otherwise recomputes the whole prompt on GLM (cache never hits).
        env.append('ATLAS_MLA_PREFIX_SKIP=1')
    take_lock()
    for rank in (0, 1):
        bench_tp2.remote(rank, f'docker rm -f {NAMES[rank]} >/dev/null 2>&1; true')
        argv = serve_argv(rank, env, args.max_seq_len)
        bench_tp2.remote(rank, ' '.join(bench_tp2.shlex.quote(a) for a in argv))
    for rank in (1, 0):  # worker first, then the leader
        bench_tp2.remote(rank, f'docker start {NAMES[rank]}')
    print('started; waiting for health on spark0:8000 ...', flush=True)
    for _ in range(180):
        out = bench_tp2.remote(0, 'curl -sf -m 5 http://127.0.0.1:8000/v1/models || true', check=False)
        if out.strip():
            print('healthy:', out.strip()[:200])
            return
        for rank in (0, 1):
            state = bench_tp2.remote(rank, f"docker inspect -f '{{{{.State.Status}}}}' {NAMES[rank]}", check=False)
            if state.strip() not in ('running', ''):
                sys.exit(f'spark{rank} container {state.strip()}; see `docker logs {NAMES[rank]}`')
        time.sleep(10)
    sys.exit('no health after 30 min; check docker logs')


def stop(_):
    for rank in (0, 1):
        bench_tp2.remote(rank, f'docker rm -f {NAMES[rank]} >/dev/null 2>&1; true')
    HOLD.unlink(missing_ok=True)
    print('stopped; bench lock released')


def status(_):
    for rank in (0, 1):
        print(f'spark{rank}:', bench_tp2.remote(
            rank, f"docker ps -a --filter name={NAMES[rank]} --format '{{{{.Names}}}} {{{{.Status}}}}'",
            check=False).strip() or 'absent')
    print('lock held by server:', holder_running())
    print(bench_tp2.remote(0, 'curl -sf -m 5 http://127.0.0.1:8000/v1/models || echo unhealthy', check=False).strip())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('start')
    s.add_argument('--exact', action='store_true', help='omit the numerics-changing flags')
    s.add_argument('--max-seq-len', type=int, default=16384)
    s.add_argument('--no-mla-prefix-skip', action='store_true',
                   help='keep the default MLA full-recompute guard (prefix cache never hits on GLM)')
    sub.add_parser('stop')
    sub.add_parser('status')
    a = p.parse_args()
    {'start': start, 'stop': stop, 'status': status}[a.cmd](a)


if __name__ == '__main__':
    main()
