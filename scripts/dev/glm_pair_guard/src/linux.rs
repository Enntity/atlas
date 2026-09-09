// SPDX-License-Identifier: AGPL-3.0-only

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

pub struct Io;
pub fn error(message: &'static str) -> io::Error {
    io::Error::other(message)
}
pub fn owned(fd: RawFd) -> io::Result<OwnedFd> {
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

impl Io {
    pub fn now() -> io::Result<u64> {
        let mut t = unsafe { std::mem::zeroed::<libc::timespec>() };
        if unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut t) } != 0 {
            return Err(io::Error::last_os_error());
        }
        u64::try_from(t.tv_sec)
            .ok()
            .and_then(|s| s.checked_mul(1000))
            .and_then(|s| s.checked_add((t.tv_nsec / 1_000_000) as u64))
            .ok_or_else(|| error("invalid boot clock"))
    }
    pub fn random() -> io::Result<[u8; 32]> {
        let mut bytes = [0; 32];
        let mut offset = 0;
        while offset < bytes.len() {
            let n = unsafe {
                libc::getrandom(
                    bytes[offset..].as_mut_ptr().cast(),
                    bytes.len() - offset,
                    libc::GRND_NONBLOCK,
                )
            };
            if n <= 0 {
                return Err(io::Error::last_os_error());
            }
            offset += n as usize;
        }
        Ok(bytes)
    }
    pub fn nonblocking(fd: RawFd) -> io::Result<()> {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    pub fn control(fd: RawFd) -> io::Result<OwnedFd> {
        if fd <= 2 {
            return Err(error("control must be inherited descriptor >2"));
        }
        for (option, expected) in [
            (libc::SO_TYPE, libc::SOCK_STREAM),
            (libc::SO_DOMAIN, libc::AF_UNIX),
        ] {
            let mut value = 0i32;
            let mut len = std::mem::size_of::<i32>() as libc::socklen_t;
            if unsafe {
                libc::getsockopt(
                    fd,
                    libc::SOL_SOCKET,
                    option,
                    (&mut value as *mut i32).cast(),
                    &mut len,
                )
            } != 0
                || value != expected
            {
                return Err(error("control is not local Unix stream"));
            }
        }
        let duplicate = owned(unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) })?;
        unsafe { libc::close(fd) };
        Self::nonblocking(duplicate.as_raw_fd())?;
        Ok(duplicate)
    }
    pub fn signals() -> io::Result<(OwnedFd, libc::sigset_t)> {
        let mut mask = unsafe { std::mem::zeroed::<libc::sigset_t>() };
        let mut old = unsafe { std::mem::zeroed::<libc::sigset_t>() };
        unsafe {
            libc::sigemptyset(&mut mask);
            for signal in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP, libc::SIGCHLD] {
                libc::sigaddset(&mut mask, signal);
            }
            let mut action = std::mem::zeroed::<libc::sigaction>();
            action.sa_sigaction = libc::SIG_DFL;
            libc::sigemptyset(&mut action.sa_mask);
            if libc::sigaction(libc::SIGCHLD, &action, std::ptr::null_mut()) != 0
                || libc::sigprocmask(libc::SIG_BLOCK, &mask, &mut old) != 0
            {
                return Err(io::Error::last_os_error());
            }
        }
        let fd =
            owned(unsafe { libc::signalfd(-1, &mask, libc::SFD_CLOEXEC | libc::SFD_NONBLOCK) })?;
        Ok((fd, old))
    }
    pub fn poll(fds: &mut [libc::pollfd], timeout: u64) -> io::Result<()> {
        let result = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, timeout as i32) };
        if result < 0 && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    pub fn read(fd: RawFd, bytes: &mut [u8]) -> io::Result<Option<usize>> {
        let n = unsafe { libc::read(fd, bytes.as_mut_ptr().cast(), bytes.len()) };
        if n >= 0 {
            return Ok(Some(n as usize));
        }
        let e = io::Error::last_os_error();
        if matches!(
            e.kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
        ) {
            Ok(None)
        } else {
            Err(e)
        }
    }
    pub fn write(fd: RawFd, bytes: &[u8]) -> io::Result<Option<usize>> {
        let n = unsafe { libc::send(fd, bytes.as_ptr().cast(), bytes.len(), libc::MSG_NOSIGNAL) };
        if n >= 0 {
            return Ok(Some(n as usize));
        }
        let e = io::Error::last_os_error();
        if matches!(
            e.kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
        ) {
            Ok(None)
        } else {
            Err(e)
        }
    }
}
