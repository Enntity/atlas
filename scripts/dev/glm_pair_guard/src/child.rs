// SPDX-License-Identifier: AGPL-3.0-only

use crate::linux::{error, owned, Io};
use std::ffi::CString;
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};

pub struct Spec {
    executable: OwnedFd,
    argv: Vec<CString>,
    env: Vec<CString>,
}

impl Spec {
    pub fn new(args: &[String]) -> io::Result<Self> {
        if args.is_empty() || args.len() > 64 || args.iter().any(|x| x.len() > 4096) {
            return Err(error("bounded executable and argv required"));
        }
        let argv: Vec<_> = args
            .iter()
            .map(|s| CString::new(s.as_bytes()))
            .collect::<Result<_, _>>()
            .map_err(|_| error("NUL in argument"))?;
        let executable = owned(unsafe {
            libc::open(
                argv[0].as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        })?;
        let mut stat = unsafe { std::mem::zeroed::<libc::stat>() };
        if unsafe { libc::fstat(executable.as_raw_fd(), &mut stat) } != 0
            || stat.st_mode & libc::S_IFMT != libc::S_IFREG
            || stat.st_mode & 0o6000 != 0
            || stat.st_mode & 0o111 == 0
        {
            return Err(error(
                "executable must be regular, executable and not set-ID",
            ));
        }
        let mut magic = [0u8; 4];
        if unsafe { libc::pread(executable.as_raw_fd(), magic.as_mut_ptr().cast(), 4, 0) } != 4
            || magic != *b"\x7fELF"
        {
            return Err(error("ELF executable required; no scripts"));
        }
        let cap = unsafe {
            libc::fgetxattr(
                executable.as_raw_fd(),
                c"security.capability".as_ptr(),
                std::ptr::null_mut(),
                0,
            )
        };
        if cap >= 0 || io::Error::last_os_error().raw_os_error() != Some(libc::ENODATA) {
            return Err(error(
                "executable capability check failed or capability present",
            ));
        }
        // No inherited loader/preload variables or credential-changing wrapper.
        let env = vec![CString::new("PATH=/usr/bin:/bin").unwrap()];
        Ok(Self {
            executable,
            argv,
            env,
        })
    }
}

pub struct Child {
    pidfd: OwnedFd,
    gate: Option<OwnedFd>,
}

fn pipe() -> io::Result<(OwnedFd, OwnedFd)> {
    let mut fds = [-1; 2];
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((owned(fds[0])?, owned(fds[1])?))
}

/// Called only in the single-threaded fork child, before any exec permission.
/// The CPU race adapter invokes this same primitive after its parent has died.
pub unsafe fn protect_parent(expected_parent: libc::pid_t) {
    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 || libc::getppid() != expected_parent
    {
        libc::_exit(75);
    }
}

impl Child {
    pub fn prepare(spec: Spec, old_mask: &libc::sigset_t, deadline: u64) -> io::Result<Self> {
        let (gate_read, gate_write) = pipe()?;
        let (ready_read, ready_write) = pipe()?;
        let argv: Vec<_> = spec
            .argv
            .iter()
            .map(|x| x.as_ptr())
            .chain([std::ptr::null()])
            .collect();
        let env: Vec<_> = spec
            .env
            .iter()
            .map(|x| x.as_ptr())
            .chain([std::ptr::null()])
            .collect();
        let expected_parent = unsafe { libc::getpid() };
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            return Err(io::Error::last_os_error());
        }
        if pid == 0 {
            // Single-threaded helper only. No allocation/unwind/destructors here.
            unsafe {
                libc::close(gate_write.as_raw_fd());
                libc::close(ready_read.as_raw_fd());
                protect_parent(expected_parent);
                if libc::write(ready_write.as_raw_fd(), b"R".as_ptr().cast(), 1) != 1 {
                    libc::_exit(75);
                }
                libc::close(ready_write.as_raw_fd());
                let mut byte = 0u8;
                loop {
                    let n = libc::read(gate_read.as_raw_fd(), (&mut byte as *mut u8).cast(), 1);
                    if n == 1 {
                        break;
                    }
                    if n < 0 && *libc::__errno_location() == libc::EINTR {
                        continue;
                    }
                    libc::_exit(75);
                }
                if byte != b'X' || libc::getppid() != expected_parent {
                    libc::_exit(75);
                }
                libc::close(gate_read.as_raw_fd());
                // Mark all unrelated inherited FDs CLOEXEC without a max-FD guess.
                if libc::syscall(
                    libc::SYS_close_range,
                    3u32,
                    u32::MAX,
                    libc::CLOSE_RANGE_CLOEXEC,
                ) != 0
                    || libc::sigprocmask(libc::SIG_SETMASK, old_mask, std::ptr::null_mut()) != 0
                {
                    libc::_exit(75);
                }
                libc::fexecve(spec.executable.as_raw_fd(), argv.as_ptr(), env.as_ptr());
                libc::_exit(76);
            }
        }
        drop(gate_read);
        drop(ready_write);
        // Normal SIGCHLD ownership, no other reaper, child cannot exec yet.
        let pidfd = owned(unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) as i32 })?;
        let child = Self {
            pidfd,
            gate: Some(gate_write),
        };
        Io::nonblocking(ready_read.as_raw_fd())?;
        loop {
            if Io::now()? >= deadline {
                return Err(error("child setup deadline"));
            }
            let mut byte = [0];
            match Io::read(ready_read.as_raw_fd(), &mut byte)? {
                Some(1) if byte == [b'R'] => return Ok(child),
                Some(_) => return Err(error("child setup failed")),
                None => Io::poll(
                    &mut [libc::pollfd {
                        fd: ready_read.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    }],
                    10,
                )?,
            }
        }
    }
    pub fn fd(&self) -> i32 {
        self.pidfd.as_raw_fd()
    }
    pub fn release(&mut self) -> io::Result<()> {
        let gate = self
            .gate
            .take()
            .ok_or_else(|| error("gate already released"))?;
        // One byte into an empty pipe; SIGPIPE is ignored by Rust startup.
        if unsafe { libc::write(gate.as_raw_fd(), b"X".as_ptr().cast(), 1) } != 1 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    pub fn terminate(&mut self) -> io::Result<()> {
        self.gate.take();
        let result = unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                self.fd(),
                libc::SIGKILL,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            )
        };
        if result != 0 && io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    pub fn reap(&self) -> io::Result<bool> {
        let mut info = unsafe { std::mem::zeroed::<libc::siginfo_t>() };
        if unsafe {
            libc::waitid(
                libc::P_PIDFD,
                self.fd() as _,
                &mut info,
                libc::WEXITED | libc::WNOHANG,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { info.si_pid() } != 0)
    }
}
