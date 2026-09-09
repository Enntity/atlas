// SPDX-License-Identifier: AGPL-3.0-only
//! Host observations rooted at Docker's actual init PID, not caller parentage.
use super::{error, NodeProcess};
use std::fs::File;
use std::io::{self, Read};
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::fs::MetadataExt;

fn small(path: &str) -> io::Result<String> {
    let mut bytes = Vec::new();
    File::open(path)?.take(32769).read_to_end(&mut bytes)?;
    if bytes.len() > 32768 {
        return Err(error("bounded proc record exceeded"));
    }
    String::from_utf8(bytes).map_err(|_| error("non-UTF8 proc record"))
}
#[derive(Debug, PartialEq, Eq)]
struct Process {
    parent: u32,
    namespace_pids: Vec<u32>,
    ticks: u64,
    namespace: (u64, u64),
    uid: u32,
    gid: u32,
}
fn ids(status: &str, key: &str) -> io::Result<Vec<u32>> {
    let text = status
        .lines()
        .find_map(|s| s.strip_prefix(key))
        .ok_or_else(|| error("missing proc field"))?;
    text.split_ascii_whitespace()
        .map(|s| s.parse().map_err(|_| error("invalid proc ID")))
        .collect()
}
fn observe(pid: u32) -> io::Result<Process> {
    let status = small(&format!("/proc/{pid}/status"))?;
    let parent = ids(&status, "PPid:")?;
    let namespace_pids = ids(&status, "NSpid:")?;
    let uid = ids(&status, "Uid:")?;
    let gid = ids(&status, "Gid:")?;
    if parent.len() != 1
        || namespace_pids.first() != Some(&pid)
        || namespace_pids.len() < 2
        || namespace_pids.len() > 32
        || uid.len() != 4
        || gid.len() != 4
        || uid.iter().chain(&gid).any(|&n| n != 0)
    {
        return Err(error("unsupported process namespace or credentials"));
    }
    let stat = small(&format!("/proc/{pid}/stat"))?;
    let (prefix, fields) = stat
        .rsplit_once(") ")
        .ok_or_else(|| error("invalid proc stat"))?;
    if prefix
        .split_once(" (")
        .and_then(|(p, _)| p.parse::<u32>().ok())
        != Some(pid)
    {
        return Err(error("proc stat PID differs"));
    }
    let ticks = fields
        .split_ascii_whitespace()
        .nth(19)
        .and_then(|n| n.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| error("invalid start ticks"))?;
    let ns = std::fs::metadata(format!("/proc/{pid}/ns/pid"))?;
    Ok(Process {
        parent: parent[0],
        namespace_pids,
        ticks,
        namespace: (ns.dev(), ns.ino()),
        uid: uid[0],
        gid: gid[0],
    })
}
fn hold(pid: u32) -> io::Result<OwnedFd> {
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}
fn boot() -> io::Result<[u8; 16]> {
    let text = small("/proc/sys/kernel/random/boot_id")?;
    let text = text.trim_end_matches('\n');
    if text.len() != 36 || [8, 13, 18, 23].iter().any(|&i| text.as_bytes()[i] != b'-') {
        return Err(error("invalid boot identity"));
    }
    let hex: String = text.chars().filter(|&c| c != '-').collect();
    if !hex.is_ascii() {
        return Err(error("invalid boot identity"));
    }
    let mut bytes = [0; 16];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|_| error("invalid boot identity"))?;
    }
    Ok(bytes)
}
fn children(pid: u32) -> io::Result<Vec<u32>> {
    small(&format!("/proc/{pid}/task/{pid}/children"))?
        .split_ascii_whitespace()
        .map(|n| n.parse().map_err(|_| error("invalid direct child")))
        .collect()
}
fn stable_zombie(pid: u32, parent: u32) -> io::Result<bool> {
    let record = || -> io::Result<(bool, u64, u32)> {
        let stat = small(&format!("/proc/{pid}/stat"))?;
        let (prefix, fields) = stat
            .rsplit_once(") ")
            .ok_or_else(|| error("invalid zombie stat"))?;
        if prefix
            .split_once(" (")
            .and_then(|(p, _)| p.parse::<u32>().ok())
            != Some(pid)
        {
            return Err(error("zombie stat PID differs"));
        }
        let fields: Vec<_> = fields.split_ascii_whitespace().collect();
        let ticks = fields
            .get(19)
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| error("invalid zombie ticks"))?;
        let ppid = fields
            .get(1)
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| error("invalid zombie parent"))?;
        Ok((fields.first() == Some(&"Z"), ticks, ppid))
    };
    let first = record()?;
    if !first.0 {
        return Ok(false);
    }
    if first.2 != parent || record()? != first {
        return Err(error("zombie child identity changed"));
    }
    Ok(true)
}
pub(super) fn pair(host_guard_pid: u32, allow_gated: bool) -> io::Result<Option<NodeProcess>> {
    if host_guard_pid <= 1 {
        return Err(error("Docker init must be an actual host process"));
    }
    let _guard = hold(host_guard_pid)?;
    let boot_id = boot()?;
    let guard = observe(host_guard_pid)?;
    if guard.namespace_pids.last() != Some(&1) {
        return Err(error("container guard is not namespace PID1"));
    }
    let child_ids = children(host_guard_pid)?;
    if child_ids.is_empty() && allow_gated {
        if observe(host_guard_pid)? != guard
            || boot()? != boot_id
            || !children(host_guard_pid)?.is_empty()
        {
            return Err(error("guard changed during socket observation"));
        }
        return Ok(None);
    }
    if child_ids.len() != 1 {
        return Err(error("exactly one actual guard child required"));
    }
    let host_child_pid = child_ids[0];
    let _child = hold(host_child_pid)?;
    if allow_gated && stable_zombie(host_child_pid, host_guard_pid)? {
        if observe(host_guard_pid)? != guard
            || children(host_guard_pid)? != child_ids
            || boot()? != boot_id
        {
            return Err(error("guard changed around zombie observation"));
        }
        // No live-child/renewal proof. Only the controller's already-delivered
        // release phase can accept this normal exit-before-parent-reap gap.
        return Ok(None);
    }
    let child = observe(host_child_pid)?;
    let depth = guard.namespace_pids.len();
    if child.parent != host_guard_pid
        || child.namespace != guard.namespace
        || child.namespace_pids.len() != depth
        || child.namespace_pids.last().copied().unwrap_or(0) <= 1
    {
        return Err(error("guard/child direct namespace relationship differs"));
    }
    if observe(host_guard_pid)? != guard
        || observe(host_child_pid)? != child
        || children(host_guard_pid)? != child_ids
        || boot()? != boot_id
    {
        return Err(error("process identity changed during observation"));
    }
    Ok(Some(NodeProcess {
        boot_id,
        pid_namespace_device: guard.namespace.0,
        pid_namespace_inode: guard.namespace.1,
        guard_pid: 1,
        child_pid: child.namespace_pids[depth - 1],
        guard_start_ticks: guard.ticks,
        child_start_ticks: child.ticks,
        host_guard_pid,
        host_child_pid,
        uid: child.uid,
        gid: child.gid,
    }))
}
pub(super) fn memory() -> io::Result<(u64, u64)> {
    let text = small("/proc/meminfo")?;
    let number = |key: &str| -> io::Result<u64> {
        let value = text
            .lines()
            .find_map(|s| s.strip_prefix(key))
            .ok_or_else(|| error("missing memory observation"))?;
        let fields: Vec<_> = value.split_ascii_whitespace().collect();
        if fields.len() != 2 || fields[1] != "kB" {
            return Err(error("unexpected memory units"));
        }
        fields[0]
            .parse()
            .map_err(|_| error("invalid memory observation"))
    };
    Ok((
        number("MemAvailable:")?,
        number("SwapTotal:")?
            .checked_sub(number("SwapFree:")?)
            .ok_or_else(|| error("invalid swap accounting"))?,
    ))
}
