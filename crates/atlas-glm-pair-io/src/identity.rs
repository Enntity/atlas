// SPDX-License-Identifier: AGPL-3.0-only

//! Local Linux observations. These are not a launch ticket or Model authority.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;

use sha2::{Digest, Sha256};

use crate::Credentials;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub fn boot_time_ms() -> io::Result<u64> {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut value) } != 0 {
        return Err(io::Error::last_os_error());
    }
    u64::try_from(value.tv_sec)
        .ok()
        .and_then(|s| s.checked_mul(1000))
        .and_then(|ms| ms.checked_add(u64::try_from(value.tv_nsec).ok()? / 1_000_000))
        .ok_or_else(|| invalid("invalid BOOTTIME"))
}

pub fn check_deadline(deadline_ms: u64) -> io::Result<()> {
    if boot_time_ms()? >= deadline_ms {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "identity/handshake deadline",
        ));
    }
    Ok(())
}

pub fn fresh_nonce() -> io::Result<[u8; 32]> {
    let mut bytes = [0; 32];
    // One nonblocking kernel request: unavailable entropy or EINTR fails closed.
    let n = unsafe { libc::getrandom(bytes.as_mut_ptr().cast(), bytes.len(), libc::GRND_NONBLOCK) };
    if n != bytes.len() as isize {
        return Err(if n < 0 {
            io::Error::last_os_error()
        } else {
            invalid("short getrandom")
        });
    }
    if bytes == [0; 32] {
        return Err(invalid("zero challenge"));
    }
    Ok(bytes)
}

fn small_file(path: &str) -> io::Result<String> {
    let mut bytes = Vec::new();
    File::open(path)?.take(8193).read_to_end(&mut bytes)?;
    if bytes.len() > 8192 {
        return Err(invalid("oversized proc record"));
    }
    String::from_utf8(bytes).map_err(|_| invalid("non-UTF8 proc record"))
}

fn start_ticks(text: &str, pid: u32) -> io::Result<u64> {
    let (prefix, fields) = text
        .rsplit_once(") ")
        .ok_or_else(|| invalid("malformed proc stat"))?;
    if prefix
        .split_once(" (")
        .and_then(|(v, _)| v.parse::<u32>().ok())
        != Some(pid)
    {
        return Err(invalid("proc stat PID mismatch"));
    }
    fields
        .split_ascii_whitespace()
        .nth(19)
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .ok_or_else(|| invalid("invalid proc start ticks"))
}

fn process_ticks(pid: u32) -> io::Result<u64> {
    start_ticks(&small_file(&format!("/proc/{pid}/stat"))?, pid)
}

fn parent_of(pid: u32) -> io::Result<u32> {
    small_file(&format!("/proc/{pid}/status"))?
        .lines()
        .find_map(|line| line.strip_prefix("PPid:"))
        .and_then(|value| value.trim().parse().ok())
        .ok_or_else(|| invalid("missing actual parent PID"))
}

fn process_credentials(pid: u32) -> io::Result<Credentials> {
    let status = small_file(&format!("/proc/{pid}/status"))?;
    let get = |key: &str| -> io::Result<u32> {
        let line = status
            .lines()
            .find_map(|s| s.strip_prefix(key))
            .ok_or_else(|| invalid("missing proc credentials"))?;
        let ids = line
            .split_ascii_whitespace()
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("invalid proc credentials"))?;
        if ids.len() != 4 || ids.iter().any(|v| *v != ids[0]) {
            return Err(invalid("credential transition is unsupported"));
        }
        Ok(ids[0])
    };
    Ok(Credentials {
        pid: i32::try_from(pid).map_err(|_| invalid("invalid PID"))?,
        uid: get("Uid:")?,
        gid: get("Gid:")?,
    })
}

fn boot_id() -> io::Result<[u8; 16]> {
    let raw = small_file("/proc/sys/kernel/random/boot_id")?;
    let raw = raw.trim_end_matches('\n');
    if raw.len() != 36 || [8, 13, 18, 23].iter().any(|&i| raw.as_bytes()[i] != b'-') {
        return Err(invalid("invalid boot ID"));
    }
    let hex: String = raw.chars().filter(|&c| c != '-').collect();
    if hex.len() != 32 || !hex.is_ascii() {
        return Err(invalid("invalid boot ID"));
    }
    let mut id = [0; 16];
    for (i, byte) in id.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|_| invalid("invalid boot ID"))?;
    }
    Ok(id)
}

/// Observed in the caller's PID namespace, never supplied by a ticket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalIdentity {
    pub boot_id: [u8; 16],
    pub pid_namespace_device: u64,
    pub pid_namespace_inode: u64,
    pub parent: Credentials,
    pub child: Credentials,
    pub parent_start_ticks: u64,
    pub child_start_ticks: u64,
}

impl LocalIdentity {
    pub fn observe() -> io::Result<Self> {
        let child_pid = unsafe { libc::getpid() };
        let parent_pid = unsafe { libc::getppid() };
        if child_pid <= 1 || parent_pid <= 0 {
            return Err(invalid("invalid direct parent/child"));
        }
        Self::observe_pair(parent_pid as u32, child_pid as u32)
    }

    /// The caller separately retains the pidfd obtained for its actual fork.
    pub fn observe_child(child_pid: u32) -> io::Result<Self> {
        Self::observe_pair(unsafe { libc::getpid() } as u32, child_pid)
    }

    fn observe_pair(parent_pid: u32, child_pid: u32) -> io::Result<Self> {
        // A host-mounted /proc is not the caller's private PID namespace view.
        let caller_pid = unsafe { libc::getpid() } as u32;
        let caller_ticks = start_ticks(&small_file("/proc/self/stat")?, caller_pid)?;
        if process_ticks(caller_pid)? != caller_ticks {
            return Err(invalid("proc mount does not identify the actual caller"));
        }
        if parent_pid == 0 || child_pid <= 1 || parent_of(child_pid)? != parent_pid {
            return Err(invalid("process is not the actual direct child"));
        }
        let ns = std::fs::metadata(format!("/proc/{child_pid}/ns/pid"))?;
        let parent_ns = std::fs::metadata(format!("/proc/{parent_pid}/ns/pid"))?;
        if (ns.dev(), ns.ino()) != (parent_ns.dev(), parent_ns.ino()) {
            return Err(invalid("parent PID namespace differs"));
        }
        let value = Self {
            boot_id: boot_id()?,
            pid_namespace_device: ns.dev(),
            pid_namespace_inode: ns.ino(),
            parent: process_credentials(parent_pid)?,
            child: process_credentials(child_pid)?,
            parent_start_ticks: process_ticks(parent_pid)?,
            child_start_ticks: process_ticks(child_pid)?,
        };
        if parent_of(child_pid)? != parent_pid
            || process_ticks(parent_pid)? != value.parent_start_ticks
            || process_ticks(child_pid)? != value.child_start_ticks
        {
            return Err(invalid("process identity changed while observing"));
        }
        Ok(value)
    }

    pub fn require_guard_parent(&self) -> io::Result<()> {
        let mut signal: libc::c_int = 0;
        if unsafe { libc::prctl(libc::PR_GET_PDEATHSIG, &mut signal) } != 0 {
            return Err(io::Error::last_os_error());
        }
        if self.parent.pid != 1 || unsafe { libc::getppid() } != 1 || signal != libc::SIGKILL {
            return Err(invalid("requires direct PID1 parent and PDEATHSIG SIGKILL"));
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
struct FileIdentity {
    dev: u64,
    ino: u64,
    len: u64,
    mode: u32,
    uid: u32,
    gid: u32,
    mtime: (i64, i64),
    ctime: (i64, i64),
}
impl FileIdentity {
    fn read(file: &File) -> io::Result<Self> {
        let m = file.metadata()?;
        if !m.is_file() || m.mode() & (libc::S_ISUID | libc::S_ISGID) != 0 {
            return Err(invalid("non-regular or credential-changing executable"));
        }
        Ok(Self {
            dev: m.dev(),
            ino: m.ino(),
            len: m.len(),
            mode: m.mode(),
            uid: m.uid(),
            gid: m.gid(),
            mtime: (m.mtime(), m.mtime_nsec()),
            ctime: (m.ctime(), m.ctime_nsec()),
        })
    }
}

/// Keeps the exact opened ELF inode alive across the handshake.
pub struct PinnedExecutable {
    file: File,
    identity: FileIdentity,
    digest: [u8; 32],
}
impl PinnedExecutable {
    pub fn open_process(pid: u32, max_bytes: u64, deadline_ms: u64) -> io::Result<Self> {
        check_deadline(deadline_ms)?;
        Self::from_file(
            File::open(format!("/proc/{pid}/exe"))?,
            max_bytes,
            deadline_ms,
        )
    }

    /// Hash the already-open executable selected for fexecve, not its pathname.
    pub fn from_file(mut file: File, max_bytes: u64, deadline_ms: u64) -> io::Result<Self> {
        check_deadline(deadline_ms)?;
        file.seek(SeekFrom::Start(0))?;
        let identity = FileIdentity::read(&file)?;
        if max_bytes == 0 || identity.len < 4 || identity.len > max_bytes {
            return Err(invalid("executable size outside explicit bound"));
        }
        let caps = unsafe {
            libc::fgetxattr(
                file.as_raw_fd(),
                c"security.capability".as_ptr(),
                std::ptr::null_mut(),
                0,
            )
        };
        if caps >= 0 || io::Error::last_os_error().raw_os_error() != Some(libc::ENODATA) {
            return Err(invalid(
                "file capabilities or unverifiable executable capabilities",
            ));
        }
        let mut hash = Sha256::new();
        let mut buffer = [0; 65536];
        let mut total = 0u64;
        loop {
            check_deadline(deadline_ms)?;
            let n = file.read(&mut buffer)?;
            check_deadline(deadline_ms)?;
            if n == 0 {
                break;
            }
            if total == 0 && !buffer[..n].starts_with(b"\x7fELF") {
                return Err(invalid("process executable is not ELF"));
            }
            total = total
                .checked_add(n as u64)
                .ok_or_else(|| invalid("ELF length overflow"))?;
            if total > max_bytes {
                return Err(invalid("executable exceeded explicit bound"));
            }
            hash.update(&buffer[..n]);
        }
        if total != identity.len || FileIdentity::read(&file)? != identity {
            return Err(invalid("executable changed during hash"));
        }
        Ok(Self {
            file,
            identity,
            digest: hash.finalize().into(),
        })
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn revalidate(&self) -> io::Result<()> {
        if FileIdentity::read(&self.file)? != self.identity {
            return Err(invalid("pinned executable changed"));
        }
        Ok(())
    }

    pub fn revalidate_process(&self, pid: u32) -> io::Result<()> {
        let current = File::open(format!("/proc/{pid}/exe"))?;
        if FileIdentity::read(&self.file)? != self.identity
            || FileIdentity::read(&current)? != self.identity
        {
            return Err(invalid("pinned process executable changed"));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
