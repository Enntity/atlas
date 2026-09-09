// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::ffi::CString;
use std::fs::{File, Metadata};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;

fn cstring(value: &str) -> io::Result<CString> {
    CString::new(value).map_err(|_| error("NUL path"))
}
fn owned(fd: i32) -> io::Result<File> {
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}
pub(super) fn path_parts(path: &Path) -> io::Result<Vec<&str>> {
    let s = path
        .to_str()
        .ok_or_else(|| error("UTF8 absolute path required"))?;
    if !s.starts_with('/') || s.len() > 4096 || s.contains('\0') {
        return Err(error("absolute bounded path required"));
    }
    let parts: Vec<_> = s[1..].split('/').collect();
    if parts.iter().any(|p| matches!(*p, "" | "." | "..")) {
        return Err(error("noncanonical path component"));
    }
    Ok(parts)
}
fn open_dir(path: &Path) -> io::Result<File> {
    let mut file = owned(unsafe {
        libc::open(
            c"/".as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
        )
    })?;
    for part in path_parts(path)? {
        file = owned(unsafe {
            libc::openat(
                file.as_raw_fd(),
                cstring(part)?.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        })?;
    }
    Ok(file)
}
fn parent(path: &Path) -> io::Result<(File, CString)> {
    path_parts(path)?;
    let parent = path.parent().ok_or_else(|| error("missing parent"))?;
    let file = if parent == Path::new("/") {
        owned(unsafe {
            libc::open(
                c"/".as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        })?
    } else {
        open_dir(parent)?
    };
    Ok((
        file,
        cstring(
            path.file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| error("missing filename"))?,
        )?,
    ))
}
fn unchanged(a: &Metadata, b: &Metadata) -> bool {
    (
        a.dev(),
        a.ino(),
        a.len(),
        a.mode(),
        a.uid(),
        a.gid(),
        a.nlink(),
        a.mtime(),
        a.mtime_nsec(),
        a.ctime(),
        a.ctime_nsec(),
    ) == (
        b.dev(),
        b.ino(),
        b.len(),
        b.mode(),
        b.uid(),
        b.gid(),
        b.nlink(),
        b.mtime(),
        b.mtime_nsec(),
        b.ctime(),
        b.ctime_nsec(),
    )
}
fn read_fd(mut file: File, max: usize, private: bool) -> io::Result<Vec<u8>> {
    let before = file.metadata()?;
    if !before.is_file()
        || before.nlink() != 1
        || before.len() > max as u64
        || before.mode() & 0o022 != 0
        || (private
            && (before.mode() & 0o7777 != 0o600
                || before.uid() != unsafe { libc::geteuid() }
                || before.gid() != unsafe { libc::getegid() }))
    {
        return Err(error("unsafe or oversized input file"));
    }
    let mut bytes = Vec::new();
    (&mut file).take(max as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > max
        || bytes.len() as u64 != before.len()
        || !unchanged(&before, &file.metadata()?)
    {
        return Err(error("input changed while reading"));
    }
    Ok(bytes)
}
pub(super) fn read_input(path: &Path, max: usize) -> io::Result<Vec<u8>> {
    let (parent, name) = parent(path)?;
    read_fd(
        owned(unsafe {
            libc::openat(
                parent.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_NONBLOCK | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        })?,
        max,
        false,
    )
}
pub(super) struct Directory {
    file: File,
    path: PathBuf,
    device: u64,
    inode: u64,
}
impl Directory {
    pub fn create(path: &Path) -> io::Result<Self> {
        let (parent, name) = parent(path)?;
        if unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), 0o700) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let file = owned(unsafe {
            libc::openat(
                parent.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        })?;
        parent.sync_all()?;
        let directory = Self::from_file(file, path)?;
        directory.revalidate()?;
        Ok(directory)
    }
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::from_file(open_dir(path)?, path)
    }
    fn from_file(file: File, path: &Path) -> io::Result<Self> {
        let m = file.metadata()?;
        if !m.is_dir()
            || m.uid() != unsafe { libc::geteuid() }
            || m.gid() != unsafe { libc::getegid() }
            || m.mode() & 0o7777 != 0o700
        {
            return Err(error(
                "prepared directory requires current owner and mode0700",
            ));
        }
        Ok(Self {
            file,
            path: path.to_owned(),
            device: m.dev(),
            inode: m.ino(),
        })
    }
    pub fn revalidate(&self) -> io::Result<()> {
        let held = self.file.metadata()?;
        let current = Self::open(&self.path)?;
        if held.uid() != unsafe { libc::geteuid() }
            || held.gid() != unsafe { libc::getegid() }
            || held.mode() & 0o7777 != 0o700
            || (current.device, current.inode) != (self.device, self.inode)
        {
            return Err(error("prepared directory replaced or changed"));
        }
        Ok(())
    }
    pub fn read(&self, name: &str, max: usize) -> io::Result<Vec<u8>> {
        self.revalidate()?;
        let bytes = read_fd(
            owned(unsafe {
                libc::openat(
                    self.file.as_raw_fd(),
                    cstring(name)?.as_ptr(),
                    libc::O_RDONLY | libc::O_NONBLOCK | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                )
            })?,
            max,
            true,
        )?;
        self.revalidate()?;
        Ok(bytes)
    }
    pub fn write(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.revalidate()?;
        let mut file = owned(unsafe {
            libc::openat(
                self.file.as_raw_fd(),
                cstring(name)?.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        })?;
        let m = file.metadata()?;
        if !m.is_file()
            || m.nlink() != 1
            || m.mode() & 0o7777 != 0o600
            || m.uid() != unsafe { libc::geteuid() }
            || m.gid() != unsafe { libc::getegid() }
        {
            return Err(error("unsafe newly created prepared file"));
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        self.file.sync_all()?;
        self.revalidate()
    }
}
