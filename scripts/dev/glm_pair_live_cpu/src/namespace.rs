// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::os::fd::{OwnedFd, RawFd};
use std::os::unix::process::CommandExt;
use std::process::{Child as Process, Command};

mod main_fixture;
#[cfg(feature = "model-test-support")]
pub(crate) use main_fixture::registered_mode;
pub(super) use main_fixture::{mount_guard, run as main_controller};

struct Namespace {
    process: Process,
    pidfd: OwnedFd,
    socket: UnixStream,
    finished: bool,
}
impl Namespace {
    fn start(rank: u8, mode: &str) -> Result<Self> {
        let (socket, inherited) = UnixStream::pair()?;
        socket.set_read_timeout(Some(Duration::from_secs(10)))?;
        socket.set_write_timeout(Some(Duration::from_secs(3)))?;
        let fd = inherited.as_raw_fd();
        let mut command = Command::new("/usr/bin/unshare");
        command
            .args([
                "--mount",
                "--pid",
                "--fork",
                "--kill-child=KILL",
                "--mount-proc",
            ])
            .arg(std::env::current_exe()?)
            .arg("--guard")
            .arg(rank.to_string())
            .arg(mode);
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(fd, 4) < 0 || libc::fcntl(4, libc::F_SETFD, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let process = command
            .spawn()
            .context("private PID/proc namespace prerequisite")?;
        let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, process.id(), 0) } as i32;
        ensure!(descriptor >= 0, "namespace wrapper pidfd required");
        Ok(Self {
            process,
            pidfd: unsafe { OwnedFd::from_raw_fd(descriptor) },
            socket,
            finished: false,
        })
    }
    fn finish(&mut self) -> Result<()> {
        self.finish_expected(0)
    }
    fn finish_expected(&mut self, expected: i32) -> Result<()> {
        let end = identity::boot_time_ms()? + 10000;
        loop {
            if let Some(status) = self.process.try_wait()? {
                self.finished = true;
                ensure!(
                    status.code() == Some(expected),
                    "namespace fixture status {status}, expected{expected}"
                );
                return Ok(());
            }
            identity::check_deadline(end)?;
            linux::Io::poll(
                &mut [libc::pollfd {
                    fd: self.pidfd.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                }],
                5,
            )?;
        }
    }
}
impl Drop for Namespace {
    fn drop(&mut self) {
        if !self.finished {
            // Only this retained exact wrapper; unshare --kill-child contains init.
            unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    self.pidfd.as_raw_fd(),
                    libc::SIGKILL,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                );
            }
        }
    }
}

pub(super) fn controller(mode: &str) -> Result<()> {
    ensure!(
        [
            "valid",
            "bad-echo",
            "bad-recipe",
            "stalled-ticket",
            "release"
        ]
        .contains(&mode),
        "explicit CPU fixture mode"
    );
    ensure!(
        unsafe { libc::geteuid() } == 0,
        "root is required for actual PID/proc namespaces"
    );
    let pair_session = identity::fresh_nonce()?;
    let binary = identity::PinnedExecutable::open_process(
        unsafe { libc::getpid() } as u32,
        512 * 1024 * 1024,
        identity::boot_time_ms()? + 10000,
    )?;
    let mut nodes = [Namespace::start(0, mode)?, Namespace::start(1, mode)?];
    for (rank, node) in nodes.iter_mut().enumerate() {
        write_frame(
            &mut node.socket,
            wire::Frame {
                rank: rank as u8,
                body: wire::Body::Startup(wire::StartupRecord {
                    pair_session,
                    container_id: identity::fresh_nonce()?,
                    image_digest: identity::fresh_nonce()?,
                    recipe_digest: identity::fresh_nonce()?,
                    server_elf_digest: binary.digest(),
                    guard_elf_digest: binary.digest(),
                    policy: policy(),
                }),
            },
        )?;
    }
    let mut reports = Vec::new();
    for (rank, node) in nodes.iter_mut().enumerate() {
        if mode == "release" {
            let mut bytes = [0; frame::LEN];
            node.socket.read_exact(&mut bytes)?;
            ensure!(
                frame::Frame::decode(&bytes).map_err(linux::error)?.kind == frame::HELLO,
                "actual legacy HELLO"
            );
        }
        let received = read_frame(&mut node.socket, wire::Direction::GuardToController)?;
        ensure!(received.rank == rank as u8, "report rank");
        let wire::Body::GatedReport(report) = received.body else {
            bail!("expected gated report")
        };
        ensure!(report.pair_session == pair_session, "report session");
        reports.push(report.record);
    }
    ensure!(
        reports[0].process.pid_namespace_inode != reports[1].process.pid_namespace_inode,
        "two distinct actual private PID namespaces"
    );
    let manifest = wire::Manifest {
        pair_session,
        policy_digest: wire::policy_digest(&policy())?,
        ranks: [reports[0], reports[1]],
    };
    for (rank, node) in nodes.iter_mut().enumerate() {
        write_frame(
            &mut node.socket,
            wire::Frame {
                rank: rank as u8,
                body: wire::Body::PairedStart(manifest),
            },
        )?;
    }
    if mode == "release" {
        let mut receipts = Vec::new();
        for (rank, node) in nodes.iter_mut().enumerate() {
            let frame = read_frame(&mut node.socket, wire::Direction::GuardToController)?;
            ensure!(frame.rank == rank as u8, "receipt rank");
            let wire::Body::Quiescent(receipt) = frame.body else {
                bail!("expected actual quiescence receipt")
            };
            receipt.validate(&manifest, rank as u8)?;
            receipts.push(wire::QuiescentFrame {
                rank: rank as u8,
                receipt,
            });
        }
        let release = wire::PairRelease {
            pair_digest: wire::manifest_digest(&manifest)?,
            epoch: wire::DRAIN_EPOCH,
            receipts: [receipts[0], receipts[1]],
            receipt_digests: [
                wire::quiescent_digest(&receipts[0])?,
                wire::quiescent_digest(&receipts[1])?,
            ],
        };
        for (rank, node) in nodes.iter_mut().enumerate() {
            write_frame(
                &mut node.socket,
                wire::Frame {
                    rank: rank as u8,
                    body: wire::Body::PairRelease(release),
                },
            )?;
        }
        for node in &mut nodes {
            node.finish()?;
        }
        println!("PASS: production LIVE loop and actual server quiescent-release transport under two PID1/proc namespaces; CPU only, NO Model/Docker/native proof");
        return Ok(());
    }
    for (rank, node) in nodes.iter_mut().enumerate() {
        let mut marker = [0u8; 1];
        node.socket.read_exact(&mut marker)?;
        ensure!(
            marker == [if mode == "valid" { b'H' } else { b'R' }],
            "actual server handshake witness rank{rank}"
        );
        node.finish()?;
    }
    println!("PASS ({mode}): both actual gated children, private PID1/proc, production inherited server consumer; CPU fixture only, NO Model/guard-main/lease/release/Docker/GPU qualification");
    Ok(())
}

fn process_record(observed: identity::LocalIdentity) -> Result<wire::ProcessIdentity> {
    Ok(wire::ProcessIdentity {
        boot_id: observed.boot_id,
        pid_namespace_device: observed.pid_namespace_device,
        pid_namespace_inode: observed.pid_namespace_inode,
        guard_pid: observed.parent.pid as u32,
        child_pid: observed.child.pid as u32,
        guard_start_ticks: observed.parent_start_ticks,
        child_start_ticks: observed.child_start_ticks,
        child_instance: identity::fresh_nonce()?,
    })
}

pub(super) fn guard(rank: u8, mode: &str) -> Result<()> {
    ensure!(
        unsafe { libc::getpid() } == 1 && rank < 2,
        "actual namespace PID1"
    );
    let mut socket = unsafe { UnixStream::from_raw_fd(4 as RawFd) };
    socket.set_read_timeout(Some(Duration::from_secs(10)))?;
    socket.set_write_timeout(Some(Duration::from_secs(3)))?;
    let startup_frame = read_frame(&mut socket, wire::Direction::StartupFile)?;
    ensure!(startup_frame.rank == rank, "startup rank");
    let wire::Body::Startup(startup) = startup_frame.body else {
        bail!("startup record")
    };
    let startup_hex = hex(startup_frame.encode()?.as_slice());
    let args = vec![
        std::env::current_exe()?
            .to_str()
            .context("UTF8 executable")?
            .to_owned(),
        "--consumer".to_owned(),
        startup_hex,
    ];
    let env = vec![
        ("ATLAS_GLM_PAIR_FD".to_owned(), "3".to_owned()),
        ("ATLAS_PAIR_CPU_MODE".to_owned(), mode.to_owned()),
        ("PATH".to_owned(), "/usr/bin:/bin".to_owned()),
    ];
    let spec = child::Spec::with_environment(&args, &env)?;
    let (signals, mask) = linux::Io::signals()?;
    let old_policy = core::Policy {
        startup: startup.policy.startup,
        lease: startup.policy.lease,
        challenge: startup.policy.challenge,
        frame: startup.policy.frame,
        campaign: startup.policy.campaign,
        poll: startup.policy.poll,
        reap: startup.policy.reap,
    };
    let now = identity::boot_time_ms()?;
    let deadline = now + 25000;
    let (mut state, hello) = core::State::new(
        now,
        old_policy,
        identity::fresh_nonce()?,
        identity::fresh_nonce()?,
        identity::fresh_nonce()?,
    )
    .map_err(linux::error)?;
    let (parent, child_channel) = Channel::pair()?;
    let (mut child, channel) =
        child::Child::prepare_live(spec, &mask, deadline, parent, child_channel)?;
    let observed = identity::LocalIdentity::observe_child(child.pid() as u32)?;
    ensure!(
        channel.receive(observed.child)?.is_none(),
        "child must remain gated"
    );
    let record = wire::RankRecord {
        container_id: startup.container_id,
        image_digest: startup.image_digest,
        recipe_digest: startup.recipe_digest,
        server_elf_digest: startup.server_elf_digest,
        guard_elf_digest: startup.guard_elf_digest,
        local_control_session: hello.session,
        guard_instance: hello.instance,
        original_startup_challenge: hello.challenge,
        process: process_record(observed)?,
    };
    if mode == "release" {
        let control: OwnedFd = socket.into();
        linux::Io::nonblocking(control.as_raw_fd())?;
        let result = live::drive(
            &control, &signals, &mut child, &channel, &mut state, hello, &startup, record, rank,
            now,
        );
        if result.is_err() {
            child.terminate()?;
        }
        result?;
        unsafe { libc::_exit(0) }
    }
    write_frame(
        &mut socket,
        wire::Frame {
            rank,
            body: wire::Body::GatedReport(wire::GatedReport {
                pair_session: startup.pair_session,
                record,
            }),
        },
    )?;
    let start = read_frame(&mut socket, wire::Direction::ControllerToGuard)?;
    ensure!(start.rank == rank, "start rank");
    let wire::Body::PairedStart(manifest) = start.body else {
        bail!("paired start")
    };
    ensure!(
        manifest.ranks[rank as usize] == record
            && manifest.pair_session == startup.pair_session
            && manifest.policy_digest == wire::policy_digest(&startup.policy)?,
        "start binding"
    );
    ensure!(
        identity::LocalIdentity::observe_child(child.pid() as u32)? == observed,
        "gated identity stable"
    );
    let mut response = hello.clone();
    response.kind = frame::START;
    ensure!(
        state
            .accept(identity::boot_time_ms()?, &response)
            .map_err(linux::error)?,
        "actual lease START"
    );
    child.release()?;
    let received = wire::Frame::decode(
        &packet(&channel, observed.child, deadline)?,
        wire::Direction::ChildToGuard,
    )?;
    ensure!(received.rank == rank, "child hello rank");
    let wire::Body::ChildHello(child_hello) = received.body else {
        bail!("child hello")
    };
    ensure!(
        child_hello.boot_id == observed.boot_id
            && child_hello.pid_namespace_device == observed.pid_namespace_device
            && child_hello.pid_namespace_inode == observed.pid_namespace_inode
            && child_hello.guard_pid == 1
            && child_hello.child_pid == child.pid() as u32
            && child_hello.guard_start_ticks == observed.parent_start_ticks
            && child_hello.child_start_ticks == observed.child_start_ticks,
        "actual postexec identity"
    );
    ensure!(
        identity::LocalIdentity::observe_child(child.pid() as u32)? == observed,
        "postexec identity stable"
    );
    let mut child_ticket = wire::ChildTicket {
        manifest,
        echoed_server_challenge: child_hello.server_challenge,
        guard_ticket_challenge: identity::fresh_nonce()?,
    };
    match mode {
        "valid" | "stalled-ticket" => {}
        "bad-echo" => child_ticket.echoed_server_challenge[0] ^= 128,
        "bad-recipe" => child_ticket.manifest.ranks[rank as usize].recipe_digest[0] ^= 128,
        _ => bail!("unknown fixture mode"),
    }
    let ticket = wire::Frame {
        rank,
        body: wire::Body::ChildTicket(child_ticket),
    };
    let waiting_since = identity::boot_time_ms()?;
    if mode != "stalled-ticket" {
        send_packet(&channel, ticket.encode()?.as_slice(), deadline)?;
    }
    loop {
        state
            .check(identity::boot_time_ms()?)
            .map_err(linux::error)?;
        identity::check_deadline(deadline)?;
        if let Some(exit) = child.exit_status()? {
            ensure!(
                exit.code == libc::CLD_EXITED
                    && exit.status == if mode == "valid" { 0 } else { 74 },
                "unexpected consumer status: {exit:?}"
            );
            ensure!(child.reap()?, "actual child reap");
            break;
        }
        if mode == "stalled-ticket" {
            ensure!(
                identity::boot_time_ms()? < waiting_since + startup.policy.frame + 1000,
                "consumer did not enforce the frame deadline"
            );
        }
        linux::Io::poll(
            &mut [libc::pollfd {
                fd: child.fd(),
                events: libc::POLLIN,
                revents: 0,
            }],
            5,
        )?;
    }
    // Test witness ONLY. Production guard must reject this unreleased exit0.
    // This fixture does not invoke or qualify the pending LIVE guard main loop.
    socket.write_all(if mode == "valid" { b"H" } else { b"R" })?;
    Ok(())
}

pub(super) fn consumer(bytes: &[u8]) -> Result<()> {
    let frame = wire::Frame::decode(bytes, wire::Direction::StartupFile)?;
    let wire::Body::Startup(startup) = frame.body else {
        bail!("consumer startup")
    };
    let expected = inherited::ExpectedSession {
        rank: frame.rank,
        pair_session: startup.pair_session,
        policy: startup.policy,
        container_id: startup.container_id,
        image_digest: startup.image_digest,
        recipe_digest: startup.recipe_digest,
        server_elf_digest: startup.server_elf_digest,
        guard_elf_digest: startup.guard_elf_digest,
        max_executable_bytes: 512 * 1024 * 1024,
    };
    let session = unsafe { inherited::InheritedSession::receive(expected, 20000) }?;
    // Closing FD3 permits later proc/ELF opens to reuse that numeric slot.
    // It must not remain an inheritable socket; retained pinned ELFs are valid.
    let flags = unsafe { libc::fcntl(3, libc::F_GETFD) };
    if flags >= 0 {
        let mut stat = unsafe { std::mem::zeroed::<libc::stat>() };
        ensure!(
            unsafe { libc::fstat(3, &mut stat) } == 0
                && stat.st_mode & libc::S_IFMT == libc::S_IFREG
                && flags & libc::FD_CLOEXEC != 0,
            "FD3 must not retain the inherited socket"
        );
    } else {
        ensure!(
            std::io::Error::last_os_error().raw_os_error() == Some(libc::EBADF),
            "FD3 status"
        );
    }
    if std::env::var("ATLAS_PAIR_CPU_MODE").as_deref() == Ok("release") {
        // CPU protocol fixture: no issued GPU work or Model capability is claimed.
        session.exit_after_quiescence();
    }
    // Handshake-only child witness. Not a successful paired quiescent release.
    unsafe { libc::_exit(0) }
}
