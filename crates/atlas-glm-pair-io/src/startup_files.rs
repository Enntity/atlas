// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-path pinned startup input, not Docker assertions or recipe authority.

use std::ffi::CStr;
use std::fs::{File, Metadata};
use std::io::{self, Read};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;

fn invalid(message: &'static str) -> io::Error {
    io::Error::other(message)
}

fn owned_file(fd: i32) -> io::Result<File> {
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

fn validate(file: &File, directory: bool) -> io::Result<Metadata> {
    let m = file.metadata()?;
    if m.uid() != 0
        || m.gid() != 0
        || m.mode() & 0o7777 != if directory { 0o700 } else { 0o600 }
        || if directory {
            !m.is_dir()
        } else {
            !m.is_file() || m.nlink() != 1
        }
    {
        return Err(invalid(
            "startup requires root-owned private directory/files",
        ));
    }
    Ok(m)
}

fn same_file(a: &Metadata, b: &Metadata) -> bool {
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

pub struct PrivateDirectory {
    file: File,
    device: u64,
    inode: u64,
}

impl PrivateDirectory {
    fn open_path() -> io::Result<File> {
        owned_file(unsafe {
            libc::open(
                c"/run/atlas-pair".as_ptr(),
                libc::O_RDONLY
                    | libc::O_DIRECTORY
                    | libc::O_CLOEXEC
                    | libc::O_NOFOLLOW
                    | libc::O_NONBLOCK,
            )
        })
    }

    pub fn open() -> io::Result<Self> {
        let file = Self::open_path()?;
        let m = validate(&file, true)?;
        let value = Self {
            file,
            device: m.dev(),
            inode: m.ino(),
        };
        value.revalidate()?;
        Ok(value)
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    /// Detect replacement of the path or changes to the pinned private owner.
    /// Directory timestamps may change when the guard creates its marker/socket.
    pub fn revalidate(&self) -> io::Result<()> {
        let held = validate(&self.file, true)?;
        let current = validate(&Self::open_path()?, true)?;
        if (held.dev(), held.ino()) != (self.device, self.inode)
            || (current.dev(), current.ino()) != (self.device, self.inode)
        {
            return Err(invalid("startup directory pathname identity changed"));
        }
        Ok(())
    }

    pub fn read(&self, name: &CStr, max: usize) -> io::Result<Vec<u8>> {
        let component = name.to_bytes();
        if component.is_empty()
            || component.len() > 255
            || component.contains(&b'/')
            || component == b"."
            || component == b".."
            || max == 0
            || max > 65536
        {
            return Err(invalid("invalid bounded startup record name/size"));
        }
        self.revalidate()?;
        let open = || {
            owned_file(unsafe {
                libc::openat(
                    self.file.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
                )
            })
        };
        let mut file = open()?;
        let before = validate(&file, false)?;
        if before.len() == 0 || before.len() > max as u64 {
            return Err(invalid("startup record size"));
        }
        let mut bytes = Vec::with_capacity(before.len() as usize);
        (&mut file).take(max as u64 + 1).read_to_end(&mut bytes)?;
        let after = validate(&file, false)?;
        let current = validate(&open()?, false)?;
        if bytes.len() != before.len() as usize
            || !same_file(&before, &after)
            || !same_file(&before, &current)
        {
            return Err(invalid("startup record changed while reading"));
        }
        self.revalidate()?;
        Ok(bytes)
    }
}
