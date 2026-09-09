// SPDX-License-Identifier: AGPL-3.0-only

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

mod ancillary;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Credentials {
    pub pid: i32,
    pub uid: u32,
    pub gid: u32,
}

pub struct Channel(OwnedFd);

fn owned(fd: RawFd) -> io::Result<OwnedFd> {
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

impl AsRawFd for Channel {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

impl Channel {
    pub fn pair() -> io::Result<(Self, Self)> {
        let mut fds = [-1; 2];
        if unsafe {
            libc::socketpair(
                libc::AF_UNIX,
                libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
                0,
                fds.as_mut_ptr(),
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        let left = Self(owned(fds[0])?);
        let right = Self(owned(fds[1])?);
        for channel in [&left, &right] {
            let value = 1i32;
            if unsafe {
                libc::setsockopt(
                    channel.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_PASSCRED,
                    (&value as *const i32).cast(),
                    std::mem::size_of::<i32>() as _,
                )
            } != 0
            {
                return Err(io::Error::last_os_error());
            }
        }
        Ok((left, right))
    }

    /// Consume the single inherited FD3; the returned duplicate is CLOEXEC.
    ///
    /// # Safety
    /// The caller must exclusively control this descriptor slot, with no other
    /// Rust owner or user of an open descriptor in it. A missing FD is rejected.
    pub unsafe fn consume_inherited(fd: RawFd) -> io::Result<Self> {
        if fd != 3 {
            return Err(io::Error::other("paired child channel must be FD3"));
        }
        // from_raw_fd requires a valid open descriptor, not merely a positive
        // number. The single-threaded caller owns this slot across the check.
        if libc::fcntl(fd, libc::F_GETFD) < 0 {
            return Err(io::Error::last_os_error());
        }
        let original = owned(fd)?;
        for (option, expected) in [
            (libc::SO_TYPE, libc::SOCK_SEQPACKET),
            (libc::SO_DOMAIN, libc::AF_UNIX),
            (libc::SO_PASSCRED, 1),
        ] {
            let mut value = 0i32;
            let mut len = std::mem::size_of::<i32>() as libc::socklen_t;
            if libc::getsockopt(
                original.as_raw_fd(),
                libc::SOL_SOCKET,
                option,
                (&mut value as *mut i32).cast(),
                &mut len,
            ) != 0
                || len as usize != std::mem::size_of::<i32>()
                || value != expected
            {
                return Err(io::Error::other("invalid inherited paired socket"));
            }
        }
        let duplicate = owned(libc::fcntl(original.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 4))?;
        let flags = libc::fcntl(duplicate.as_raw_fd(), libc::F_GETFL);
        if flags < 0
            || libc::fcntl(
                duplicate.as_raw_fd(),
                libc::F_SETFL,
                flags | libc::O_NONBLOCK,
            ) < 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(duplicate))
    }

    pub fn send(&self, bytes: &[u8]) -> io::Result<bool> {
        if bytes.is_empty() || bytes.len() > 4096 {
            return Err(io::Error::other("invalid paired packet size"));
        }
        let n = unsafe {
            libc::send(
                self.as_raw_fd(),
                bytes.as_ptr().cast(),
                bytes.len(),
                libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
            )
        };
        if n < 0 {
            let e = io::Error::last_os_error();
            return if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) {
                Ok(false)
            } else {
                Err(e)
            };
        }
        if n as usize != bytes.len() {
            return Err(io::Error::other("short paired packet send"));
        }
        Ok(true)
    }

    /// Read at most one packet. Every delivered descriptor is closed before
    /// rejecting ancillary data, including packets with invalid credentials.
    pub fn receive(&self, expected: Credentials) -> io::Result<Option<Vec<u8>>> {
        let mut bytes = [0u8; 4096];
        // cmsghdr alignment, one ucred plus sixteen received descriptors.
        let mut control = ancillary::Storage::new();
        let mut iov = libc::iovec {
            iov_base: bytes.as_mut_ptr().cast(),
            iov_len: bytes.len(),
        };
        let mut message = unsafe { std::mem::zeroed::<libc::msghdr>() };
        message.msg_iov = &mut iov;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr();
        message.msg_controllen = ancillary::CAPACITY;
        let n = unsafe {
            libc::recvmsg(
                self.as_raw_fd(),
                &mut message,
                libc::MSG_DONTWAIT | libc::MSG_CMSG_CLOEXEC,
            )
        };
        if n < 0 {
            let e = io::Error::last_os_error();
            return if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) {
                Ok(None)
            } else {
                Err(e)
            };
        }
        // Never reject payload/EOF/truncation before disposing delivered FDs.
        let ancillary = unsafe { ancillary::validate_and_dispose(&message, expected) };
        // A closed peer returns no packet and no ancillary data. A zero-length
        // packet still carries credentials and must not masquerade as EOF.
        // Ancillary disposal always precedes this distinction.
        if n == 0
            && message.msg_controllen == 0
            && message.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) == 0
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "paired peer closed",
            ));
        }
        ancillary?;
        if n == 0 || n as usize > bytes.len() {
            return Err(io::Error::other("paired channel EOF or invalid packet"));
        }
        Ok(Some(bytes[..n as usize].to_vec()))
    }
}

#[cfg(test)]
mod tests;
