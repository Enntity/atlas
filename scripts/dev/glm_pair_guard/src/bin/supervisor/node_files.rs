// SPDX-License-Identifier: AGPL-3.0-only
//! Root-owned retained records. Failed operations never remove their evidence.
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;

fn error(message: &'static str) -> io::Error {
    io::Error::other(message)
}

pub(super) struct Directory {
    file: File,
    uid: u32,
    gid: u32,
    private: bool,
    parent: Option<Box<Directory>>,
    parent_fd: i32,
    name: CString,
}
impl Directory {
    pub(super) fn open(session: &str, rank: u8, create: bool) -> io::Result<Self> {
        let run = Self::at(libc::AT_FDCWD, c"/run", false, false, 0, 0)?;
        let base = run.child(c"atlas-glm-pairs", create, false)?;
        let session = base.child(&CString::new(session)?, create, true)?;
        let name = CString::new(format!("rank{rank}"))?;
        if create {
            // A partially prepared rank is retained evidence, never reopened as
            // a fresh launch. Only the shared base/session may already exist.
            if unsafe { libc::mkdirat(session.file.as_raw_fd(), name.as_ptr(), 0o700) } != 0 {
                return Err(io::Error::last_os_error());
            }
            session.file.sync_all()?;
        }
        session.child(&name, false, true)
    }
    fn child(self, name: &CStr, create: bool, private: bool) -> io::Result<Self> {
        self.revalidate()?;
        let mut child = Self::at(
            self.file.as_raw_fd(),
            name,
            create,
            private,
            self.uid,
            self.gid,
        )?;
        if create {
            self.file.sync_all()?;
        }
        child.parent = Some(Box::new(self));
        Ok(child)
    }
    fn at(
        parent: i32,
        name: &CStr,
        create: bool,
        private: bool,
        uid: u32,
        gid: u32,
    ) -> io::Result<Self> {
        if create && unsafe { libc::mkdirat(parent, name.as_ptr(), 0o700) } != 0 {
            let e = io::Error::last_os_error();
            // Only shared session/base creation uses this existing-dir case.
            // The rank directory is exclusively created by open above.
            if e.raw_os_error() != Some(libc::EEXIST) {
                return Err(e);
            }
        }
        let fd = unsafe {
            libc::openat(
                parent,
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let file = unsafe { File::from_raw_fd(fd) };
        let m = file.metadata()?;
        if m.uid() != uid
            || m.gid() != gid
            || m.mode() & 0o022 != 0
            || (private && m.mode() & 0o7777 != 0o700)
        {
            return Err(error("node directory ownership or mode"));
        }
        Ok(Self {
            file,
            uid,
            gid,
            private,
            parent: None,
            parent_fd: parent,
            name: name.to_owned(),
        })
    }
    fn revalidate(&self) -> io::Result<()> {
        if let Some(parent) = &self.parent {
            parent.revalidate()?;
        }
        let mut s = unsafe { std::mem::zeroed::<libc::stat>() };
        if unsafe {
            libc::fstatat(
                self.parent_fd,
                self.name.as_ptr(),
                &mut s,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        let m = self.file.metadata()?;
        if s.st_dev != m.dev()
            || s.st_ino != m.ino()
            || s.st_mode & libc::S_IFMT != libc::S_IFDIR
            || s.st_uid != self.uid
            || s.st_gid != self.gid
            || s.st_mode & 0o022 != 0
            || (self.private && s.st_mode & 0o7777 != 0o700)
            || s.st_mode != m.mode()
        {
            return Err(error("node directory identity changed"));
        }
        Ok(())
    }
    pub(super) fn write_new(&self, name: &CStr, bytes: &[u8]) -> io::Result<()> {
        self.revalidate()?;
        let fd = unsafe {
            libc::openat(
                self.file.as_raw_fd(),
                name.as_ptr(),
                libc::O_WRONLY
                    | libc::O_CREAT
                    | libc::O_EXCL
                    | libc::O_NOFOLLOW
                    | libc::O_NONBLOCK
                    | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let m = file.metadata()?;
        if !m.is_file()
            || m.uid() != self.uid
            || m.gid() != self.gid
            || m.nlink() != 1
            || m.mode() & 0o7777 != 0o600
        {
            return Err(error("new record ownership or mode"));
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        self.file.sync_all()?;
        self.revalidate()
    }
    pub(super) fn read(&self, name: &CStr, max: usize) -> io::Result<Vec<u8>> {
        self.revalidate()?;
        let fd = unsafe {
            libc::openat(
                self.file.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let before = file.metadata()?;
        if !before.is_file()
            || before.uid() != self.uid
            || before.gid() != self.gid
            || before.mode() & 0o7777 != 0o600
            || before.nlink() != 1
            || before.len() > max as u64
        {
            return Err(error("node record ownership, type, mode or bound"));
        }
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take(max as u64 + 1)
            .read_to_end(&mut bytes)?;
        let after = file.metadata()?;
        if bytes.len() as u64 != before.len()
            || (
                before.dev(),
                before.ino(),
                before.len(),
                before.mtime(),
                before.mtime_nsec(),
                before.ctime(),
                before.ctime_nsec(),
            ) != (
                after.dev(),
                after.ino(),
                after.len(),
                after.mtime(),
                after.mtime_nsec(),
                after.ctime(),
                after.ctime_nsec(),
            )
        {
            return Err(error("node record changed during read"));
        }
        self.revalidate()?;
        Ok(bytes)
    }
    pub(super) fn socket_ready(&self) -> io::Result<bool> {
        self.revalidate()?;
        let mut s = unsafe { std::mem::zeroed::<libc::stat>() };
        if unsafe {
            libc::fstatat(
                self.file.as_raw_fd(),
                c"control.sock".as_ptr(),
                &mut s,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } != 0
        {
            let e = io::Error::last_os_error();
            return if e.raw_os_error() == Some(libc::ENOENT) {
                Ok(false)
            } else {
                Err(e)
            };
        }
        if s.st_uid != self.uid
            || s.st_gid != self.gid
            || s.st_mode & libc::S_IFMT != libc::S_IFSOCK
            || s.st_mode & 0o7777 != 0o600
        {
            return Err(error("node control socket identity"));
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_exclusive_record_and_nofollow() {
        let temp = std::env::temp_dir().join(format!(
            "glm-node-record-{}-{}",
            std::process::id(),
            atlas_glm_pair_io::identity::boot_time_ms().unwrap()
        ));
        std::fs::create_dir(&temp).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o700)).unwrap();
        let dir = Directory::at(
            libc::AT_FDCWD,
            &CString::new(temp.to_str().unwrap()).unwrap(),
            false,
            true,
            unsafe { libc::getuid() },
            unsafe { libc::getgid() },
        )
        .unwrap();
        dir.write_new(c"receipt", b"actual full identity").unwrap();
        assert_eq!(dir.read(c"receipt", 64).unwrap(), b"actual full identity");
        assert!(dir.write_new(c"receipt", b"replacement").is_err());
        std::os::unix::fs::symlink("receipt", temp.join("alias")).unwrap();
        assert!(dir.read(c"alias", 64).is_err());
        assert!(dir.read(c"receipt", 1).is_err());
        std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(dir.read(c"receipt", 64).is_err());
        std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::remove_file(temp.join("alias")).unwrap();
        std::fs::remove_file(temp.join("receipt")).unwrap();
        std::fs::remove_dir(temp).unwrap();
    }
}
