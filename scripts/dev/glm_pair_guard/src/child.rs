// SPDX-License-Identifier: AGPL-3.0-only

use crate::linux::{error, owned, Io};
use atlas_glm_pair_io::{Channel, Credentials};
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};

#[path = "child_spec.rs"]
mod spec;
pub use spec::Spec;

pub struct Child {
    pidfd: OwnedFd,
    gate: Option<OwnedFd>,
    credentials: Credentials,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExitStatus {
    pub code: i32,
    pub status: i32,
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
        if spec.explicit_env {
            return Err(error("explicit LIVE environment requires private channel"));
        }
        Self::prepare_inner(spec, old_mask, deadline, None).map(|(child, _)| child)
    }

    /// The caller owns recipe equality/authorization; this only launches the
    /// validated explicit environment with its private pre-created channel.
    pub fn prepare_live(
        spec: Spec,
        old_mask: &libc::sigset_t,
        deadline: u64,
        parent: Channel,
        child: Channel,
    ) -> io::Result<(Self, Channel)> {
        if !spec.explicit_env {
            return Err(error(
                "LIVE requires explicit environment, no PATH fallback",
            ));
        }
        let (child, parent) = Self::prepare_inner(spec, old_mask, deadline, Some((parent, child)))?;
        Ok((child, parent.expect("LIVE owns parent channel")))
    }

    fn prepare_inner(
        spec: Spec,
        old_mask: &libc::sigset_t,
        deadline: u64,
        live: Option<(Channel, Channel)>,
    ) -> io::Result<(Self, Option<Channel>)> {
        // Both descriptors are fixed before fork. FD3 may currently hold the
        // pinned ELF or either socket: dup3 must never overwrite our exec FD.
        let relocated = live
            .as_ref()
            .map(|(_, child)| -> io::Result<_> {
                Ok((
                    owned(unsafe {
                        libc::fcntl(spec.executable.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 4)
                    })?,
                    owned(unsafe { libc::fcntl(child.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 4) })?,
                ))
            })
            .transpose()?;
        let executable = relocated
            .as_ref()
            .map_or(spec.executable.as_raw_fd(), |(elf, _)| elf.as_raw_fd());
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
        let uid = unsafe { libc::getuid() };
        let gid = unsafe { libc::getgid() };
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            return Err(io::Error::last_os_error());
        }
        if pid == 0 {
            // Single-threaded helper only. No allocation/unwind/destructors here.
            unsafe {
                libc::close(gate_write.as_raw_fd());
                libc::close(ready_read.as_raw_fd());
                if let Some((parent, _)) = live.as_ref() {
                    libc::close(parent.as_raw_fd());
                }
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
                // Only FD3 survives the first exec. The actual server ingress
                // immediately consumes it into a CLOEXEC owner, closing FD3.
                if let Some((_, channel)) = relocated.as_ref() {
                    if libc::dup3(channel.as_raw_fd(), 3, 0) != 3 {
                        libc::_exit(75);
                    }
                }
                libc::fexecve(executable, argv.as_ptr(), env.as_ptr());
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
            credentials: Credentials { pid, uid, gid },
        };
        Io::nonblocking(ready_read.as_raw_fd())?;
        loop {
            if Io::now()? >= deadline {
                return Err(error("child setup deadline"));
            }
            let mut byte = [0];
            match Io::read(ready_read.as_raw_fd(), &mut byte)? {
                Some(1) if byte == [b'R'] => return Ok((child, live.map(|(parent, _)| parent))),
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
    pub fn pid(&self) -> libc::pid_t {
        self.credentials.pid
    }
    pub fn credentials(&self) -> Credentials {
        self.credentials
    }
    /// Observe the exact held child without consuming its status; `reap`
    /// remains the sole consuming operation used by the legacy runner.
    pub fn exit_status(&self) -> io::Result<Option<ExitStatus>> {
        let mut info = unsafe { std::mem::zeroed::<libc::siginfo_t>() };
        if unsafe {
            libc::waitid(
                libc::P_PIDFD,
                self.fd() as _,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        let pid = unsafe { info.si_pid() };
        if pid == 0 {
            return Ok(None);
        }
        if pid != self.pid() {
            return Err(error("pidfd exit identity mismatch"));
        }
        Ok(Some(ExitStatus {
            code: info.si_code,
            status: unsafe { info.si_status() },
        }))
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

#[cfg(test)]
#[path = "child_tests.rs"]
mod tests;
