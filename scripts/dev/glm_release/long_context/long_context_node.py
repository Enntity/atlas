# SPDX-License-Identifier: AGPL-3.0-only
"""Fixed remote helper, sent literally by long-context-runner.py; not a service.

CPU-only selftests never dispatch this file. On-node execution is root's later
explicit action. An independent detached watchdog retains the exact session and
container identity. Scheduling/kernel/Docker availability still bounds enforcement.
"""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

LIMIT = 114 * 1024**3
LEASE = 30.0
CAMPAIGN = 2400.0
MAX_OUTPUT = 2 * 1024**2


def now():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def command(args, timeout=8):
    with tempfile.TemporaryFile() as output:
        p = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=output,
                           stderr=subprocess.STDOUT, timeout=timeout,
                           env={"PATH": "/usr/local/bin:/usr/bin:/bin"})
        if output.tell() > MAX_OUTPUT:
            raise RuntimeError("command output cap exceeded")
        output.seek(0)
        text = output.read().decode("utf-8", errors="strict")
    if p.returncode:
        raise RuntimeError(f"command failed {args[:2]} status={p.returncode}: {text[:4096]}")
    return text


def write_new(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


def read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as f:
        return json.load(f)


def atomic(path, value):
    tmp = path.with_name(path.name + "." + str(os.getpid()))
    write_new(tmp, value)
    os.replace(tmp, path)


def memory():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.split()[0])
    result = {"available_kib": fields["MemAvailable"],
              "swap_used_kib": fields["SwapTotal"] - fields["SwapFree"]}
    if result["available_kib"] < 4194304 or result["swap_used_kib"] != 0:
        raise RuntimeError("host memory boundary: " + json.dumps(result))
    return result


def inspect(meta, required=True):
    # A name is used ONLY for discovery after an interrupted create. Every
    # action thereafter requires the full ID and independently checked labels/image.
    result = command(["docker", "container", "ls", "-a", "--no-trunc", "--filter",
                      "name=^/" + meta["name"] + "$", "--format", "{{.ID}}"])
    ids = result.split()
    if not ids and not required:
        return None
    if len(ids) != 1 or not re.fullmatch("[0-9a-f]{64}", ids[0]):
        raise RuntimeError("exact session container missing or ambiguous")
    data = json.loads(command(["docker", "inspect", ids[0]]))[0]
    if (data["Id"] != ids[0] or data["Image"] != "sha256:" + meta["image_sha256"]
            or data["Config"]["Labels"].get("atlas.longctx") != meta["session"]
            or data["Name"] != "/" + meta["name"]):
        raise RuntimeError("container identity mismatch; no mutation authorized")
    saved = Path(f"/tmp/atlas-longctx-{meta['session']}-r{meta['rank']}/container-id.json")
    if saved.exists() and read(saved) != data["Id"]:
        raise RuntimeError("published full container ID changed; no mutation authorized")
    return data


def snapshot(meta):
    started = now()
    result = {"boottime": started, **memory()}
    data = inspect(meta, False)
    if data is not None:
        result["container"] = {k: data[k] for k in ("Id", "Image", "State")}
        host = data["HostConfig"]
        if (host["Memory"] != LIMIT or host["MemorySwap"] != LIMIT
                or host["RestartPolicy"]["Name"] not in ("no", "")
                or data["State"]["OOMKilled"]):
            raise RuntimeError("container memory/restart/OOM boundary")
        pid = data["State"]["Pid"]
        if data["State"]["Running"]:
            # Unified cgroup path is correlated through this exact container init PID.
            cgroups = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
            unified = [v[3:] for v in cgroups if v.startswith("0::")]
            if len(unified) != 1 or ".." in Path(unified[0]).parts:
                raise RuntimeError("expected one safe cgroup-v2 path")
            group = Path("/sys/fs/cgroup") / unified[0].lstrip("/")
            cg = {k: (group / k).read_text().strip() for k in
                  ("memory.current", "memory.peak", "memory.max", "memory.swap.current",
                   "memory.swap.max", "memory.events")}
            if cg["memory.max"] != str(LIMIT) or cg["memory.swap.max"] != "0" or cg["memory.swap.current"] != "0":
                raise RuntimeError("actual cgroup limits or swap changed")
            events = dict(line.split() for line in cg["memory.events"].splitlines())
            if any(int(events.get(k, "0")) for k in ("oom", "oom_kill", "oom_group_kill")):
                raise RuntimeError("actual cgroup OOM event")
            result["cgroup"] = cg
    if now() - started > 10:
        raise RuntimeError("stale node snapshot")
    result["observed_at"] = now()
    return result


def stop_exact(meta, grace):
    data = inspect(meta, False)
    if data is None:
        return {"absent": True}
    cid = data["Id"]
    errors = []
    if data["State"]["Running"]:
        try:
            command(["docker", "stop", "--timeout", str(grace), cid], grace + 5)
        except Exception as e:
            errors.append(str(e))
        current = inspect(meta)
        if current["State"]["Running"]:
            try:
                command(["docker", "kill", "--signal=KILL", cid], 5)
            except Exception as e:
                errors.append(str(e))
    return {"container": inspect(meta), "errors": errors, "fallback": True}


def watch(directory):
    meta = read(directory / "meta.json")
    count = 0
    try:
        while True:
            with open(directory / "lock", "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = read(directory / "lease.json")
                if (directory / "terminal.json").exists():
                    raise RuntimeError("controller/node latched terminal")
                if now() >= min(state["deadline"], meta["expires"]):
                    raise RuntimeError("irrevocable lease/campaign expiry")
                # Read phase BEFORE observing Docker state. A subsequent start
                # cannot make an old stopped sample look like a serving exit.
                had_started = (directory / "started").exists()
            sample = snapshot(meta)
            atomic(directory / "latest.json", sample)
            # Finite campaign gives an upper bound to this 1-Hz append-only receipt.
            with open(directory / "monitor.jsonl", "a") as f:
                f.write(json.dumps(sample, sort_keys=True) + "\n")
            count += 1
            data = sample.get("container")
            if data is not None and not data["State"]["Running"]:
                if had_started:
                    if not (directory / "stopping").exists():
                        raise RuntimeError("unexpected exit during serving")
                    write_new(directory / "watch-complete.json", {"samples": count, "state": data["State"]})
                    return
            time.sleep(1)
    except BaseException as e:
        # Serialize terminal publication with create/start. After this latch,
        # neither a late create nor a delayed start can escape our cleanup.
        with open(directory / "lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (directory / "terminal.json").exists():
                write_new(directory / "terminal.json", {"error": repr(e), "at": now()})
        try:
            result = stop_exact(meta, 5)
        except BaseException as cleanup:
            result = {"cleanup_uncertain": repr(cleanup)}
        atomic(directory / "watch-cleanup.json", result)


def dispatch(op, payload):
    session, rank = payload["session"], payload["rank"]
    if not re.fullmatch("[0-9a-f]{32}", session) or rank not in (0, 1):
        raise ValueError("invalid exact session/rank")
    directory = Path(f"/tmp/atlas-longctx-{session}-r{rank}")
    if op == "prepare":
        os.mkdir(directory, 0o700)
        meta = dict(payload, expires=now() + CAMPAIGN,
                    name=f"atlas-longctx-{session}-r{rank}")
        write_new(directory / "meta.json", meta)
        write_new(directory / "lease.json", {"deadline": now() + LEASE})
        source = globals()["SOURCE"]
        fd = os.open(directory / "watch.py", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(source)
            f.flush()
            os.fsync(f.fileno())
        with open(directory / "watch.log", "xb") as log:
            p = subprocess.Popen([sys.executable, str(directory / "watch.py"), "watch", str(directory)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True, close_fds=True)
        return {"watch_pid": p.pid, "directory": str(directory), **memory()}
    meta = read(directory / "meta.json")
    if meta["session"] != session or meta["rank"] != rank:
        raise RuntimeError("session metadata mismatch")
    if op == "status":
        sample = snapshot(meta)
        with open(directory / "lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read(directory / "lease.json")
            t = now()
            if t >= min(state["deadline"], meta["expires"]) or (directory / "terminal.json").exists():
                raise RuntimeError("late renewal or terminal watchdog")
            latest = read(directory / "latest.json")
            if t - latest["observed_at"] > 10:
                raise RuntimeError("watchdog stopped making fresh progress")
            atomic(directory / "lease.json", {"deadline": min(t + LEASE, meta["expires"])})
        return sample
    if op == "create":
        memory()
        image = json.loads(command(["docker", "image", "inspect", "sha256:" + meta["image_sha256"]]))[0]
        if image["Id"] != "sha256:" + meta["image_sha256"] or image["Architecture"] != "arm64":
            raise RuntimeError("pinned ARM64 image mismatch")
        with open(directory / "lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read(directory / "lease.json")
            if now() >= min(state["deadline"], meta["expires"]) or (directory / "terminal.json").exists():
                raise RuntimeError("create after terminal/expiry")
            if inspect(meta, False) is not None:
                raise RuntimeError("exclusive container already exists")
            created = command(meta["create_argv"], 15).strip()
            data = inspect(meta)
            if created != data["Id"] or data["State"]["Running"]:
                raise RuntimeError("create identity or stopped-state mismatch")
            write_new(directory / "container-id.json", created)
        command(["docker", "cp", created + ":/usr/local/bin/spark", str(directory / "server.elf")], 15)
        with open(directory / "server.elf", "rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != meta["server_sha256"]:
            raise RuntimeError("actual container server ELF digest mismatch")
        return {"container": data, "server_sha256": actual}
    if op == "start":
        with open(directory / "lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = inspect(meta)
            memory()
            state = read(directory / "lease.json")
            if now() >= min(state["deadline"], meta["expires"]) or (directory / "terminal.json").exists():
                raise RuntimeError("start after expiry or terminal")
            if (directory / "started").exists():
                raise RuntimeError("start is one-shot")
            started = command(["docker", "start", data["Id"]], 15).strip()
            write_new(directory / "started", True)
            if now() >= min(state["deadline"], meta["expires"]):
                write_new(directory / "terminal.json", {"error": "start crossed deadline", "at": now()})
                raise RuntimeError("start crossed deadline; watchdog cleanup required")
        return {"started": started}
    if op == "drain":
        write_new(directory / "stopping", True)
        data = inspect(meta)
        if rank == 0 and data["State"]["Running"]:
            command(["docker", "kill", "--signal=TERM", data["Id"]])
        return {"draining": data["Id"]}
    if op == "stop":
        if not (directory / "stopping").exists():
            write_new(directory / "stopping", True)
        return stop_exact(meta, 10)
    if op == "collect":
        data = inspect(meta, False)
        result = {"container": data, "memory": memory()}
        if data:
            result["logs"] = command(["docker", "logs", data["Id"]], 10)
        for name in ("terminal.json", "watch-cleanup.json", "watch-complete.json", "latest.json"):
            if (directory / name).exists():
                result[name] = read(directory / name)
        result["monitor_path"] = str(directory / "monitor.jsonl")
        return result
    raise ValueError("unknown operation")


if __name__ == "__main__":
    if sys.argv[1] == "watch":
        watch(Path(sys.argv[2]))
    else:
        print(json.dumps(dispatch(sys.argv[1], json.loads(base64.b64decode(sys.argv[2]))), sort_keys=True))
