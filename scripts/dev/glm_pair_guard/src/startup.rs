// SPDX-License-Identifier: AGPL-3.0-only

//! Pinned root-owned startup records. Docker assertions remain controller-owned.
use crate::{
    child::Spec,
    linux::{error, owned, Io},
};
use atlas_glm_pair_io::identity::PinnedExecutable;
use atlas_glm_pair_io::PrivateDirectory;
use atlas_glm_pair_wire as wire;
use std::fs::File;
use std::io::{self, Write};
use std::os::fd::{AsRawFd, OwnedFd};

pub struct Loaded {
    pub directory: PrivateDirectory,
    pub rank: u8,
    pub startup: wire::StartupRecord,
    pub recipe: wire::Recipe,
    pub spec: Spec,
    pub server: PinnedExecutable,
    pub guard: PinnedExecutable,
    pub started: u64,
}

impl Loaded {
    pub fn read() -> io::Result<Self> {
        let started = Io::now()?;
        let directory = PrivateDirectory::open()?;
        let frame = wire::Frame::decode(
            &directory.read(c"startup.bin", 288)?,
            wire::Direction::StartupFile,
        )
        .map_err(|e| error(e.0))?;
        let wire::Body::Startup(startup) = frame.body else {
            return Err(error("startup file kind"));
        };
        let recipe_bytes = directory.read(c"recipe.bin", wire::MAX_RECIPE_BYTES)?;
        let recipe = wire::Recipe::decode(&recipe_bytes).map_err(|e| error(e.0))?;
        if recipe.rank != frame.rank
            || recipe.world != 2
            || wire::recipe_digest(&recipe_bytes).map_err(|e| error(e.0))? != startup.recipe_digest
            || recipe.image_digest != startup.image_digest
            || recipe.guard_elf_digest != startup.guard_elf_digest
            || recipe.server_elf_digest != startup.server_elf_digest
        {
            return Err(error("recipe/startup identity mismatch"));
        }
        let resources = &recipe.resources;
        if resources.uid != 0
            || resources.gid != 0
            || resources.init
            || resources.pid_mode != "private"
            || resources.restart_policy != "no"
            || resources.memory == 0
            || resources.swap != resources.memory
        {
            return Err(error(
                "LIVE requires root private PID1/no restart/no swap recipe",
            ));
        }
        if !recipe.mounts.iter().any(|mount| {
            mount.destination == "/run/atlas-pair"
                && !mount.read_only
                && mount.propagation == "rprivate"
        }) {
            return Err(error("missing private writable startup mount"));
        }
        // This checks local loader paths; exact native Atlas/NCCL policy and
        // Docker inspect equality remain the deploying controller's obligation.
        for (key, value) in &recipe.environment {
            if key == "LD_LIBRARY_PATH" {
                use std::os::unix::fs::MetadataExt;
                for path in value.split(':') {
                    if !path.starts_with('/') {
                        return Err(error("absolute library path required"));
                    }
                    let metadata = std::fs::metadata(path)?;
                    if !metadata.is_dir() || metadata.uid() != 0 || metadata.mode() & 0o022 != 0 {
                        return Err(error(
                            "library path must be root-owned and not group/world writable",
                        ));
                    }
                }
            }
        }
        let deadline = started
            .checked_add(startup.policy.startup)
            .ok_or_else(|| error("startup overflow"))?;
        let spec = Spec::with_environment(&recipe.argv, &recipe.environment)?;
        // Protocol implementation bound, not a fallback model/resource setting.
        // Guard and server must both fit this pinned executable hashing budget.
        let server = spec.pin_executable(512 * 1024 * 1024, deadline)?;
        let guard = PinnedExecutable::open_process(1, 512 * 1024 * 1024, deadline)?;
        if server.digest() != startup.server_elf_digest
            || guard.digest() != startup.guard_elf_digest
        {
            return Err(error("pinned executable digest mismatch"));
        }
        let loaded = Self {
            directory,
            rank: frame.rank,
            startup,
            recipe,
            spec,
            server,
            guard,
            started,
        };
        loaded.revalidate()?;
        Ok(loaded)
    }

    pub fn consume(&self) -> io::Result<()> {
        self.directory.revalidate()?;
        let mut marker = File::from(owned(unsafe {
            libc::openat(
                self.directory.file().as_raw_fd(),
                c"consumed".as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                0o600,
            )
        })?);
        marker.write_all(&self.startup.pair_session)?;
        marker.sync_all()?;
        self.directory.file().sync_all()?;
        Ok(())
    }

    pub fn deadline(&self) -> io::Result<u64> {
        self.started
            .checked_add(self.startup.policy.startup)
            .ok_or_else(|| error("startup deadline overflow"))
    }

    pub fn revalidate(&self) -> io::Result<()> {
        self.guard.revalidate_process(1)?;
        self.server.revalidate()?;
        self.directory.revalidate()?;
        if Io::now()? >= self.deadline()? {
            return Err(error("startup expired"));
        }
        Ok(())
    }
}

/// The listener belongs to the pinned mounted directory and accepts one root
/// controller connection. Numeric peer PID is not cross-namespace authority.
pub fn accept_control(loaded: &Loaded, signals: &OwnedFd) -> io::Result<OwnedFd> {
    use std::os::unix::net::UnixListener;
    loaded.revalidate()?;
    let listener = UnixListener::bind("/run/atlas-pair/control.sock")?;
    // umask is set before any LIVE I/O in the dedicated single-threaded guard.
    if unsafe { libc::chmod(c"/run/atlas-pair/control.sock".as_ptr(), 0o600) } != 0 {
        return Err(io::Error::last_os_error());
    }
    listener.set_nonblocking(true)?;
    loop {
        loaded.revalidate()?;
        match listener.accept() {
            Ok((stream, _)) => {
                let mut credential = unsafe { std::mem::zeroed::<libc::ucred>() };
                let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
                if unsafe {
                    libc::getsockopt(
                        stream.as_raw_fd(),
                        libc::SOL_SOCKET,
                        libc::SO_PEERCRED,
                        (&mut credential as *mut libc::ucred).cast(),
                        &mut length,
                    )
                } != 0
                    || length as usize != std::mem::size_of::<libc::ucred>()
                    || credential.uid != 0
                    || credential.gid != 0
                {
                    return Err(error("LIVE controller must be root over private socket"));
                }
                stream.set_nonblocking(true)?;
                loaded.revalidate()?;
                return Ok(stream.into());
            }
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) => {}
            Err(e) => return Err(e),
        }
        let mut polls = [
            libc::pollfd {
                fd: listener.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: signals.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        Io::poll(&mut polls, loaded.startup.policy.poll)?;
        if polls[1].revents != 0 {
            return Err(error("startup signal"));
        }
    }
}
