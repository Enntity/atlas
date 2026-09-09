// SPDX-License-Identifier: AGPL-3.0-only
//! Pipe-safe relay adapter. The guard's socket-only adapter is unchanged.
use super::*;
use std::os::fd::{FromRawFd, RawFd};
pub(super) struct Io;
impl Io {
    pub(super) fn owned(fd: RawFd) -> io::Result<OwnedFd> {
        if fd < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(unsafe { OwnedFd::from_raw_fd(fd) })
        }
    }
    pub(super) fn duplicate(fd: RawFd) -> io::Result<OwnedFd> {
        Self::owned(unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) })
    }
    pub(super) fn now() -> io::Result<u64> {
        let mut time = unsafe { std::mem::zeroed::<libc::timespec>() };
        if unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut time) } != 0 {
            return Err(io::Error::last_os_error());
        }
        u64::try_from(time.tv_sec)
            .ok()
            .and_then(|s| s.checked_mul(1000))
            .and_then(|s| s.checked_add((time.tv_nsec / 1_000_000) as u64))
            .ok_or_else(|| error("invalid boot clock"))
    }
    pub(super) fn nonblocking(fd: RawFd) -> io::Result<()> {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    pub(super) fn ignore_sigpipe() -> io::Result<()> {
        let mut action = unsafe { std::mem::zeroed::<libc::sigaction>() };
        action.sa_sigaction = libc::SIG_IGN;
        unsafe { libc::sigemptyset(&mut action.sa_mask) };
        if unsafe { libc::sigaction(libc::SIGPIPE, &action, std::ptr::null_mut()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    fn count(n: isize) -> io::Result<Option<usize>> {
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
    pub(super) fn read(fd: RawFd, bytes: &mut [u8]) -> io::Result<Option<usize>> {
        Self::count(unsafe { libc::read(fd, bytes.as_mut_ptr().cast(), bytes.len()) })
    }
    pub(super) fn write(fd: RawFd, bytes: &[u8]) -> io::Result<Option<usize>> {
        Self::count(unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) })
    }
    pub(super) fn poll(fds: &mut [libc::pollfd], timeout: u64) -> io::Result<()> {
        let timeout = i32::try_from(timeout).map_err(|_| error("poll overflow"))?;
        if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, timeout) } < 0
            && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}
