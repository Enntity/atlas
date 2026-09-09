// SPDX-License-Identifier: AGPL-3.0-only
//! The production wrapper has one fixed root-owned path, never a caller path.
use super::*;
use std::ffi::{CStr, CString};

fn stat(fd: i32) -> io::Result<libc::stat> {
    let mut value = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut value) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(value)
}
fn directory(parent: i32, name: &CStr, uid: u32, gid: u32, private: bool) -> io::Result<OwnedFd> {
    let fd = Io::owned(unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    })?;
    let s = stat(fd.as_raw_fd())?;
    if s.st_uid != uid
        || s.st_gid != gid
        || s.st_mode & 0o022 != 0
        || (private && s.st_mode & 0o7777 != 0o700)
    {
        return Err(error("relay directory ownership or permissions"));
    }
    Ok(fd)
}
fn socket_stat(parent: i32, uid: u32, gid: u32) -> io::Result<libc::stat> {
    let mut s = unsafe { std::mem::zeroed::<libc::stat>() };
    if unsafe {
        libc::fstatat(
            parent,
            c"control.sock".as_ptr(),
            &mut s,
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    if s.st_uid != uid
        || s.st_gid != gid
        || s.st_mode & libc::S_IFMT != libc::S_IFSOCK
        || s.st_mode & 0o7777 != 0o600
    {
        return Err(error("relay socket ownership, type or permissions"));
    }
    Ok(s)
}

pub(super) fn connect(config: &Config, started: u64) -> io::Result<OwnedFd> {
    let run = directory(libc::AT_FDCWD, c"/run", 0, 0, false)?;
    let base = directory(run.as_raw_fd(), c"atlas-glm-pairs", 0, 0, false)?;
    let session = directory(
        base.as_raw_fd(),
        &CString::new(config.session.as_str()).unwrap(),
        0,
        0,
        true,
    )?;
    let rank = directory(
        session.as_raw_fd(),
        &CString::new(format!("rank{}", config.rank)).unwrap(),
        0,
        0,
        true,
    )?;
    connect_at(&rank, config, started, 0, 0)
}

// Tests supply an actual private temporary directory and their real uid/gid;
// only the fixed wrapper above is reachable from the production CLI.
fn connect_at(
    rank: &OwnedFd,
    config: &Config,
    started: u64,
    uid: u32,
    gid: u32,
) -> io::Result<OwnedFd> {
    let before = socket_stat(rank.as_raw_fd(), uid, gid)?;
    let path = CString::new(format!("/proc/self/fd/{}/control.sock", rank.as_raw_fd())).unwrap();
    let mut address = unsafe { std::mem::zeroed::<libc::sockaddr_un>() };
    address.sun_family = libc::AF_UNIX as _;
    if path.as_bytes_with_nul().len() > address.sun_path.len() {
        return Err(error("socket path bound"));
    }
    for (dst, src) in address.sun_path.iter_mut().zip(path.as_bytes_with_nul()) {
        *dst = *src as _;
    }
    let socket = Io::owned(unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            0,
        )
    })?;
    config.check(started, Io::now()?)?;
    frame::check_deadline(started, Io::now()?, config.connect).map_err(error)?;
    let result = unsafe {
        libc::connect(
            socket.as_raw_fd(),
            (&address as *const libc::sockaddr_un).cast(),
            std::mem::size_of_val(&address) as _,
        )
    };
    if result != 0 {
        if io::Error::last_os_error().raw_os_error() != Some(libc::EINPROGRESS) {
            return Err(io::Error::last_os_error());
        }
        loop {
            config.check(started, Io::now()?)?;
            frame::check_deadline(started, Io::now()?, config.connect).map_err(error)?;
            let mut poll = [libc::pollfd {
                fd: socket.as_raw_fd(),
                events: libc::POLLOUT,
                revents: 0,
            }];
            Io::poll(&mut poll, config.poll)?;
            if poll[0].revents != 0 {
                break;
            }
        }
        let mut status: i32 = 0;
        let mut size = std::mem::size_of_val(&status) as libc::socklen_t;
        if unsafe {
            libc::getsockopt(
                socket.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_ERROR,
                (&mut status as *mut i32).cast(),
                &mut size,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
    }
    let after = socket_stat(rank.as_raw_fd(), uid, gid)?;
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) {
        return Err(error("socket replaced during connect"));
    }
    let mut peer = unsafe { std::mem::zeroed::<libc::ucred>() };
    let mut size = std::mem::size_of_val(&peer) as libc::socklen_t;
    if unsafe {
        libc::getsockopt(
            socket.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut peer as *mut libc::ucred).cast(),
            &mut size,
        )
    } != 0
        || size as usize != std::mem::size_of_val(&peer)
        || peer.uid != uid
        || peer.gid != gid
    {
        return Err(error("relay peer credentials"));
    }
    config.check(started, Io::now()?)?;
    frame::check_deadline(started, Io::now()?, config.connect).map_err(error)?;
    Ok(socket)
}

#[cfg(test)]
#[path = "path_tests.rs"]
mod tests;
