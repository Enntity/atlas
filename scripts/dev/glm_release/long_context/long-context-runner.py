#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Root-operated ordinary (NOT paired-T3) guarded long-context qualification.

Usage: --dry-run INPUT.json, --run INPUT.json, or --selftest. No builds.
Input schema is exercised in selftest() below. All fields are mandatory. Pin
immutable per-node image IDs and the actual ELF; never use a mutable image tag.
The two fixed-profile containers remain stopped/preserved after completion.
Node-local monitors/receipts remain in the reported exclusive /tmp directories.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from long_context_profile import ENV, LIMIT, MODEL, argv, create_args, digest, environment

HERE = Path(__file__).resolve().parent
NODE_SOURCE = (HERE / "long_context_node.py").read_text()
LOCAL_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
# Same native fault markers as scripts/dev/check_glm53_native_log.py, plus
# explicit upstream teardown failures; a zero process status alone is insufficient.
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FAULT = re.compile(r"^(?:\d{4}-\d{2}-\d{2}[T ][\d:.]+Z?\s+)?ERROR(?:\s|:|$)|"
                   r"unhealthy|panic|illegal memory|out of memory|HIDDEN_TRACE|K5_LEDGER|"
                   r"CUDA_ERROR_|CudaError\(|NCCL error|device-side assert|"
                   r"could not synchronise|teardown reported a failure", re.IGNORECASE)


def unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate JSON key " + k)
        result[k] = v
    return result


def fields(obj, names):
    if not isinstance(obj, dict) or set(obj) != set(names.split()):
        raise ValueError("unexpected/missing fields, required: " + names)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate(c, files=True):
    fields(c, "version context nodes server_sha256 fabric_head api_url port ssh_key known_hosts output_directory workload ssh_executable node_python paged_prefill_bf16_gemm")
    if c["version"] != 1 or not isinstance(c["nodes"], list) or len(c["nodes"]) != 2:
        raise ValueError("version1/two ranks required")
    digest(c["server_sha256"])
    for rank, n in enumerate(c["nodes"]):
        fields(n, "rank destination image_sha256 weights_host_path fabric")
        if n["rank"] != rank or not re.fullmatch(r"[a-z_][a-z0-9_-]*@[0-9.]+", n["destination"]):
            raise ValueError("ordered ranks and literal user@IPv4 required")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", n["weights_host_path"]) or ".." in Path(n["weights_host_path"]).parts:
            raise ValueError("unsafe weights path")
        digest(n["image_sha256"])
        environment(n, c["paged_prefill_bf16_gemm"])
        argv(c["context"], rank, c["fabric_head"], c["port"])
    if c["nodes"][0]["destination"] == c["nodes"][1]["destination"]:
        raise ValueError("nodes must differ")
    expected = "http://" + c["nodes"][0]["destination"].split("@", 1)[1] + ":" + str(c["port"])
    if c["api_url"] != expected:
        raise ValueError("API must bind exactly to supplied head host/port")
    for key in ("ssh_key", "known_hosts", "output_directory", "ssh_executable", "node_python"):
        if not isinstance(c[key], str) or not Path(c[key]).is_absolute():
            raise ValueError("absolute path required: " + key)
    w = c["workload"]
    fields(w, "argv files_sha256 timeout_seconds")
    if (not isinstance(w["argv"], list) or not 1 <= len(w["argv"]) <= 128
            or any(not isinstance(x, str) or "\0" in x for x in w["argv"])
            or not Path(w["argv"][0]).is_absolute()
            or type(w["timeout_seconds"]) is not int or not 1 <= w["timeout_seconds"] <= 1800):
        raise ValueError("bounded explicit workload argv/time required")
    if w["argv"][0] not in w["files_sha256"]:
        raise ValueError("workload executable must be pinned")
    for path, sha in w["files_sha256"].items():
        digest(sha)
        if not Path(path).is_absolute() or (files and file_hash(path) != sha):
            raise ValueError("workload file hash mismatch: " + path)
    if files:
        for a in w["argv"]:
            if Path(a).is_absolute() and Path(a).is_file() and a not in w["files_sha256"]:
                raise ValueError("unhashed argument file " + a)
        for p in (c["ssh_key"], c["known_hosts"]):
            if not Path(p).is_file():
                raise ValueError("missing SSH credential/known_hosts file")
        if not Path(c["ssh_executable"]).is_file():
            raise ValueError("missing explicit SSH executable")
    if "REQUIRED_" in json.dumps(c):
        raise ValueError("unresolved operator placeholder")
    return c


class Runner:
    def __init__(self, config):
        self.c = config
        self.session = secrets.token_hex(16)
        self.out = Path(config["output_directory"])
        self.out.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.lock = threading.Lock()
        self.count = 0
        self.total = 0
        self.failed = threading.Event()
        self.finished = threading.Event()
        self.prepared = [False, False]
        self.created = [None, None]
        self.last = [None, None]
        self.stopping = False
        self.monitor = None
        self.workload = None
        self.end = time.monotonic() + 2300
        self.record("input", config)
        self.record("source", {p.name: file_hash(p) for p in
                              (Path(__file__), HERE / "long_context_profile.py", HERE / "long_context_node.py")})

    def record(self, name, value):
        raw = json.dumps(value, sort_keys=True, indent=2).encode()
        with self.lock:
            self.total += len(raw)
            if len(raw) > 2 * 1024**2 or self.total > 64 * 1024**2:
                raise RuntimeError("controller receipt cap exceeded")
            p = self.out / f"{self.count:05d}-{name}.json"
            self.count += 1
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())

    def remote(self, rank, op):
        node = self.c["nodes"][rank]
        payload = dict(node, session=self.session, server_sha256=self.c["server_sha256"],
                       create_argv=create_args(node, self.session, self.c["context"],
                                               self.c["fabric_head"], self.c["port"],
                                               self.c["paged_prefill_bf16_gemm"]))
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        command = shlex.join([self.c["node_python"], "-c", "SOURCE=" + repr(NODE_SOURCE) + "\nexec(SOURCE)", op, encoded])
        args = [self.c["ssh_executable"], "-i", self.c["ssh_key"], "-o", "BatchMode=yes",
                "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "UserKnownHostsFile=" + self.c["known_hosts"], "-o", "ConnectTimeout=3",
                "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2",
                node["destination"], command]
        begin = time.monotonic()
        # Remote helper bounds every Docker call. Local stdout/stderr land on disk,
        # never an unbounded communicate() buffer. Dedicated process, no shell locally.
        with open(self.out / f"remote-{rank}-{op}-{time.monotonic_ns()}.log", "xb") as log:
            p = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 env=LOCAL_ENV, start_new_session=True)
            timeout = 45 if op == "create" else 25 if op in ("stop", "collect") else 18
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
                raise RuntimeError(f"SSH {rank}/{op} deadline; node watchdog remains armed")
            if log.tell() > 2 * 1024**2:
                raise RuntimeError("remote output cap")
            name = log.name
        raw = Path(name).read_text()
        if p.returncode:
            raise RuntimeError(f"SSH {rank}/{op} failed {p.returncode}: {raw[-4096:]}")
        value = json.loads(raw, object_pairs_hook=unique)
        if op == "status":
            value["controller_begin"] = begin
        self.record(f"rank{rank}-{op}", value)
        if op == "status" and time.monotonic() - begin > 10:
            raise RuntimeError("remote sample freshness exceeded")
        return value

    def both(self, op):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.remote, rank, op) for rank in (0, 1)]
            values, errors = [], []
            for f in futures:
                try:
                    values.append(f.result())
                except BaseException as e:
                    values.append(None)
                    errors.append(repr(e))
            if errors:
                raise RuntimeError(str(errors))
            return values

    def verify_created(self, rank, result):
        d = result["container"]
        n = self.c["nodes"][rank]
        expected = create_args(n, self.session, self.c["context"], self.c["fabric_head"],
                               self.c["port"], self.c["paged_prefill_bf16_gemm"])
        cmd = expected[expected.index("sha256:" + n["image_sha256"]) + 1:]
        h = d["HostConfig"]
        if (not re.fullmatch("[0-9a-f]{64}", d["Id"]) or d["State"]["Running"]
                or d["Image"] != "sha256:" + n["image_sha256"]
                or result["server_sha256"] != self.c["server_sha256"]
                or d["Config"]["Entrypoint"] != ["/usr/bin/env"] or d["Config"]["Cmd"] != cmd
                or h["Memory"] != LIMIT or h["MemorySwap"] != LIMIT
                or h["RestartPolicy"]["Name"] != "no" or h["IpcMode"] != "private"
                or h["ShmSize"] != 1024**3 or h["NetworkMode"] != "host"
                or h["CpusetCpus"] != "0-19" or d["Config"]["User"] != "0:0"):
            raise RuntimeError("created container profile mismatch")
        mounts = [m for m in d["Mounts"] if m["Destination"] == MODEL]
        if len(mounts) != 1 or mounts[0]["Source"] != n["weights_host_path"] or mounts[0]["RW"]:
            raise RuntimeError("weights mount differs")
        if (h["Runtime"] != "runc" or sorted(h["CapAdd"] or []) != ["CAP_IPC_LOCK", "CAP_SYS_NICE"]
                or set(h["SecurityOpt"] or []) != {"no-new-privileges=true", "seccomp=unconfined", "label=disable"}
                or h["Devices"] != [{"PathOnHost": "/dev/infiniband", "PathInContainer": "/dev/infiniband", "CgroupPermissions": "rwm"}]
                or h["Ulimits"] != [{"Name": "memlock", "Hard": -1, "Soft": -1}]):
            raise RuntimeError("RDMA/security/runtime profile differs")
        requests = h["DeviceRequests"]
        if (len(requests or []) != 1 or requests[0]["Driver"] != ""
                or requests[0]["Count"] != -1 or requests[0]["Capabilities"] != [["gpu"]]
                or requests[0].get("DeviceIDs") not in (None, []) or requests[0].get("Options") not in (None, {})):
            raise RuntimeError("GPU device request differs")
        self.created[rank] = d["Id"]

    def monitoring(self):
        try:
            while not self.finished.is_set():
                # Phase facts precede observation; a delayed pre-create/start
                # sample must not be judged against a newer main-thread phase.
                created_before = list(self.created)
                started_before = (self.out / "started").exists()
                samples = self.both("status")
                for rank, sample in enumerate(samples):
                    d = sample.get("container")
                    if created_before[rank] and (d is None or d["Id"] != created_before[rank]):
                        raise RuntimeError("container identity disappeared")
                    if d and (d["State"]["OOMKilled"] or (not d["State"]["Running"] and not self.stopping)):
                        # Stopped containers are expected until start has completed.
                        if started_before:
                            raise RuntimeError("unexpected serving exit")
                    self.last[rank] = sample
                self.finished.wait(1)
        except BaseException as e:
            self.failed.set()
            self.record("monitor-failure", {"error": repr(e)})

    def healthy(self):
        if self.failed.is_set() or time.monotonic() >= self.end:
            raise RuntimeError("latched monitor/campaign failure")
        if any(s is None or time.monotonic() - s["controller_begin"] >= 10 for s in self.last):
            raise RuntimeError("both-rank status absent or stale")

    def run(self):
        success = False
        try:
            # Each prepare arms an independent node watchdog BEFORE any container.
            self.both("prepare")
            self.prepared = [True, True]
            time.sleep(2)
            self.last = self.both("status")
            self.healthy()
            self.monitor = threading.Thread(target=self.monitoring, daemon=True)
            self.monitor.start()
            created = self.both("create")
            for rank, result in enumerate(created):
                self.verify_created(rank, result)
            self.healthy()
            self.both("start")
            (self.out / "started").touch(exist_ok=False)
            deadline = time.monotonic() + 300
            while True:
                self.healthy()
                try:
                    with urllib.request.urlopen(self.c["api_url"] + "/health", timeout=2) as r:
                        reply = r.read(4097)
                        if len(reply) > 4096:
                            raise RuntimeError("health reply exceeds cap")
                        health = json.loads(reply, object_pairs_hook=unique)
                        ready = r.status == 200 and health.get("status") == "ready" and health.get("model") == MODEL
                except (OSError, TimeoutError, ValueError):
                    ready = False
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("startup readiness deadline")
                time.sleep(1)
            self.record("ready", {"context": self.c["context"], "session": self.session})
            self.healthy()
            w = self.c["workload"]
            for p, sha in w["files_sha256"].items():
                if file_hash(p) != sha:
                    raise RuntimeError("workload changed after validation")
            with open(self.out / "workload.stdout", "xb") as out, open(self.out / "workload.stderr", "xb") as err:
                self.workload = subprocess.Popen(w["argv"], stdin=subprocess.DEVNULL, stdout=out,
                                                stderr=err, env=LOCAL_ENV, start_new_session=True)
                deadline = time.monotonic() + w["timeout_seconds"]
                while self.workload.poll() is None:
                    self.healthy()
                    if time.monotonic() >= deadline or out.tell() + err.tell() > 32 * 1024**2:
                        raise RuntimeError("workload time/output boundary")
                    time.sleep(0.2)
                if self.workload.returncode != 0:
                    raise RuntimeError("workload failed " + str(self.workload.returncode))
            self.healthy()
            success = True
        except BaseException as e:
            self.record("failure", {"error": repr(e)})
        finally:
            success = self.cleanup(success)
        self.record("result", {"passed": success, "session": self.session,
                               "claim": "ordinary operational watchdog; not paired T3 quiescence"})
        return 0 if success else 1

    def cleanup(self, healthy):
        if self.workload and self.workload.poll() is None:
            self.workload.terminate()
            try:
                self.workload.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.workload.kill()
                self.workload.wait(timeout=3)
        self.stopping = True
        # Mark BOTH expected stops, then signal head; worker follows EP shutdown.
        try:
            self.remote(1, "drain")
            self.remote(0, "drain")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                states = self.both("status")
                if all(s.get("container") and not s["container"]["State"]["Running"] for s in states):
                    break
                time.sleep(0.5)
            else:
                healthy = False
                self.both("stop")
        except BaseException as e:
            healthy = False
            self.record("cleanup-failure", {"error": repr(e)})
            try:
                self.both("stop")
            except BaseException as second:
                self.record("cleanup-uncertain", {"error": repr(second), "watchdogs_remain_armed": True})
        self.finished.set()
        if self.monitor:
            self.monitor.join(timeout=30)
            if self.monitor.is_alive():
                healthy = False
        try:
            final = self.both("collect")
            for rank, result in enumerate(final):
                d = result["container"]
                if (d is None or d["Id"] != self.created[rank] or d["State"]["Running"]
                        or d["State"]["ExitCode"] != 0 or d["State"]["OOMKilled"]
                        or "terminal.json" in result or "watch-cleanup.json" in result):
                    healthy = False
                faults = [line for line in result.get("logs", "").splitlines()
                          if FAULT.search(ANSI.sub("", line))]
                if faults:
                    self.record(f"rank{rank}-log-faults", faults)
                    healthy = False
        except BaseException as e:
            self.record("final-uncertain", {"error": repr(e)})
            healthy = False
        return healthy and not self.failed.is_set()


def selftest():
    from long_context_dry_tests import run_safety_tests
    run_safety_tests(Runner)
    c = dict(version=1, context=4096, server_sha256="a" * 64, fabric_head="192.0.2.10",
             api_url="http://192.0.2.10:8890", port=8890, ssh_key="/fixture/key",
             known_hosts="/fixture/known_hosts", output_directory="/fixture/output",
             ssh_executable="/fixture/ssh", node_python="/fixture/python3", paged_prefill_bf16_gemm=False,
             nodes=[dict(rank=i, destination="fixture@192.0.2." + str(10 + i),
                         image_sha256=str(i + 1) * 64, weights_host_path=MODEL,
                         fabric=dict(interface="fixture0", hca="fixture_hca", gid_index="7")) for i in (0, 1)],
             workload=dict(argv=["/fixture/workload"], files_sha256={"/fixture/workload": "b" * 64}, timeout_seconds=1800))
    for context in (4096, 8192, 16384):
        c["context"] = context
        validate(c, files=False)
        for rank in (0, 1):
            a = create_args(c["nodes"][rank], "c" * 32, context, c["fabric_head"], c["port"], c["paged_prefill_bf16_gemm"])
            assert "--speculative" not in a and "--glm-paired-mtp" not in a
            assert not any("GLM_PAIR_FD" in v for v in a)
            assert "--max-seq-len=" + str(context) in a and "-i" in a
            assert a[a.index("--memory") + 1] == str(LIMIT)
    for bad in (2048, 32768):
        c["context"] = bad
        try:
            validate(c, files=False)
        except ValueError:
            pass
        else:
            raise AssertionError("bad context accepted")
    try:
        json.loads('{"version":1,"version":2}', object_pairs_hook=unique)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON accepted")
    assert ENV["ATLAS_GLM_C4_SPARSE"] == "1" and ENV["ATLAS_GLM_INDEPENDENT_DECODE"] == "0"
    print("PASS: dry pure profile/strict-input tests; no SSH/Docker/HTTP/subprocess invoked")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path)
    group.add_argument("--dry-run", type=Path)
    group.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        source = args.run or args.dry_run
        if source.stat().st_size > 65536:
            raise ValueError("input JSON cap")
        config = validate(json.loads(source.read_text(), object_pairs_hook=unique))
        if args.dry_run:
            print(json.dumps({"config": config, "environment": [environment(n, config["paged_prefill_bf16_gemm"]) for n in config["nodes"]], "argv": [argv(config["context"], i, config["fabric_head"], config["port"]) for i in (0, 1)]}, indent=2))
        else:
            runner = Runner(config)
            signal.signal(signal.SIGTERM, lambda *_: runner.failed.set())
            signal.signal(signal.SIGINT, lambda *_: runner.failed.set())
            sys.exit(runner.run())
