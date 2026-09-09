// SPDX-License-Identifier: AGPL-3.0-only

//! Production one-shot startup and terminal exit ownership.
use crate::{
    child::Child,
    core::{Policy, State},
    linux::{error, Io},
    live, startup,
};
use atlas_glm_pair_io::{identity::LocalIdentity, Channel};
use atlas_glm_pair_wire::RankRecord;
use std::io;

pub fn run() -> io::Result<()> {
    if std::env::args_os().count() != 2
        || unsafe { libc::getpid() } != 1
        || unsafe { libc::getuid() } != 0
        || unsafe { libc::getgid() } != 0
    {
        return Err(error(
            "LIVE requires sole --live argument as private root PID1",
        ));
    }
    unsafe {
        libc::umask(0o077);
    }
    let (signals, mask) = Io::signals()?;
    let loaded = startup::Loaded::read()?;
    loaded.consume()?;
    let control = startup::accept_control(&loaded, &signals)?;
    let p = loaded.startup.policy;
    let policy = Policy {
        startup: p.startup,
        lease: p.lease,
        challenge: p.challenge,
        frame: p.frame,
        campaign: p.campaign,
        poll: p.poll,
        reap: p.reap,
    };
    let (mut state, hello) = State::new(
        loaded.started,
        policy,
        Io::random()?,
        Io::random()?,
        Io::random()?,
    )
    .map_err(error)?;
    let deadline = loaded.deadline()?;
    loaded.revalidate()?;
    let (parent, endpoint) = Channel::pair()?;
    // Keep both pinned ELF observations outside the live child/loop owners.
    let startup::Loaded {
        directory: _directory,
        rank,
        startup,
        recipe: _recipe,
        spec,
        server: _server,
        guard: _guard,
        started,
    } = loaded;
    let (mut child, channel) = Child::prepare_live(spec, &mask, deadline, parent, endpoint)?;
    let result = (|| {
        let observed = LocalIdentity::observe_child(child.pid() as u32)?;
        if observed.parent.pid != 1 || observed.child != child.credentials() {
            return Err(error("actual gated child identity mismatch"));
        }
        let record = RankRecord {
            container_id: startup.container_id,
            image_digest: startup.image_digest,
            recipe_digest: startup.recipe_digest,
            server_elf_digest: startup.server_elf_digest,
            guard_elf_digest: startup.guard_elf_digest,
            local_control_session: hello.session,
            guard_instance: hello.instance,
            original_startup_challenge: hello.challenge,
            process: atlas_glm_pair_wire::ProcessIdentity {
                boot_id: observed.boot_id,
                pid_namespace_device: observed.pid_namespace_device,
                pid_namespace_inode: observed.pid_namespace_inode,
                guard_pid: 1,
                child_pid: observed.child.pid as u32,
                guard_start_ticks: observed.parent_start_ticks,
                child_start_ticks: observed.child_start_ticks,
                child_instance: Io::random()?,
            },
        };
        live::drive(
            &control, &signals, &mut child, &channel, &mut state, hello, &startup, record, rank,
            started,
        )
    })();
    if result.is_ok() {
        // drive has observed and reaped the released exact child exit0.
        unsafe { libc::_exit(0) }
    }
    state.stop();
    child.terminate()?;
    let until = Io::now()?
        .checked_add(p.reap)
        .ok_or_else(|| error("reap deadline overflow"))?;
    loop {
        if child.reap()? {
            break;
        }
        if Io::now()? >= until {
            return Err(error("unconfirmed LIVE child exit"));
        }
        Io::poll(
            &mut [libc::pollfd {
                fd: child.fd(),
                events: libc::POLLIN,
                revents: 0,
            }],
            p.poll,
        )?;
    }
    result
}
