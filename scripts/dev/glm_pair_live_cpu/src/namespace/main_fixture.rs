// SPDX-License-Identifier: AGPL-3.0-only

//! Actual `glm-pair-guard --live` entry, pinned startup files and server consumer.
//! All Docker/resource IDs are explicitly CPU fixture assertions, not observations.
use super::*;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
mod main_exchange;

pub(crate) fn registered_mode(mode: &str) -> bool {
    matches!(
        mode,
        "registered-valid"
            | "registered-drain"
            | "registered-drain-foreign"
            | "registered-drain-replay"
            | "registered-drain-controller"
            | "registered-wrong-rank"
            | "registered-missing-capability"
            | "registered-unhealthy"
    )
}

pub(crate) fn mount_guard(source: &str, executable: &str) -> Result<()> {
    ensure!(
        unsafe { libc::getpid() } == 1,
        "mount adapter must become PID1"
    );
    let source = std::ffi::CString::new(source)?;
    // Only this private mount namespace sees the new small /run. The host's
    // mount table and /run contents are untouched; original /tmp source remains.
    ensure!(
        unsafe {
            libc::mount(
                c"tmpfs".as_ptr(),
                c"/run".as_ptr(),
                c"tmpfs".as_ptr(),
                libc::MS_NOSUID | libc::MS_NODEV,
                c"size=1048576,mode=0755".as_ptr().cast(),
            )
        } == 0,
        "private /run mount: {}",
        std::io::Error::last_os_error()
    );
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create("/run/atlas-pair")?;
    ensure!(
        unsafe {
            libc::mount(
                source.as_ptr(),
                c"/run/atlas-pair".as_ptr(),
                std::ptr::null(),
                libc::MS_BIND,
                std::ptr::null(),
            )
        } == 0,
        "private session bind"
    );
    let error = Command::new(executable).arg("--live").exec();
    Err(error.into())
}

fn directory() -> Result<PathBuf> {
    let mut template = b"/tmp/atlas-live-main-XXXXXX\0".to_vec();
    let pointer = unsafe { libc::mkdtemp(template.as_mut_ptr().cast()) };
    ensure!(!pointer.is_null(), "exclusive CPU fixture directory");
    Ok(PathBuf::from(
        unsafe { std::ffi::CStr::from_ptr(pointer) }.to_str()?,
    ))
}
fn write_record(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn fixture_recipe(
    rank: u8,
    source: &Path,
    server: [u8; 32],
    guard: [u8; 32],
    mode: &str,
) -> Result<wire::Recipe> {
    Ok(wire::Recipe {
        argv: vec![
            std::env::current_exe()?
                .to_str()
                .context("UTF8 ELF")?
                .to_owned(),
            if registered_mode(mode) {
                "--consumer-registered"
            } else {
                "--consumer-files"
            }
            .to_owned(),
        ],
        environment: {
            let mut environment = vec![
                ("ATLAS_GLM_PAIR_FD".to_owned(), "3".to_owned()),
                (
                    "ATLAS_PAIR_CPU_DELAY_MS".to_owned(),
                    if mode == "delayed" && rank == 1 {
                        "6000"
                    } else {
                        "0"
                    }
                    .to_owned(),
                ),
                ("ATLAS_PAIR_CPU_MODE".to_owned(), mode.to_owned()),
                ("PATH".to_owned(), "/usr/bin:/bin".to_owned()),
            ];
            if registered_mode(mode) {
                // Same literal policy table as real-controller CPU recipes;
                // ordinary namespace fixtures keep their unchanged values.
                environment = crate::controller_fixture::environment(mode, "0");
            }
            environment
        },
        mounts: vec![wire::Mount {
            source: source.to_str().context("UTF8 source")?.to_owned(),
            destination: "/run/atlas-pair".to_owned(),
            read_only: false,
            propagation: "rprivate".to_owned(),
        }],
        image_digest: identity::fresh_nonce()?,
        server_elf_digest: server,
        guard_elf_digest: guard,
        rank,
        world: 2,
        profile: wire::Profile {
            tp: 2,
            ep: 2,
            ep_protocol: 2,
            max_sequences: 2,
            context: 2044,
            prefill: 1024,
            drafts: 4,
            eager: true,
            kv_format: 1,
            cold_min: 2,
            cold_max: 1024,
        },
        resources: wire::Resources {
            devices: vec![],
            security_options: vec![],
            memory: 114 << 30,
            swap: 114 << 30,
            cpuset: "0-19".to_owned(),
            shm: 1 << 30,
            device_requests: vec![],
            ulimits: vec![],
            cap_add: vec![],
            cap_drop: vec![],
            uid: 0,
            gid: 0,
            network_mode: "host".to_owned(),
            pid_mode: "private".to_owned(),
            restart_policy: "no".to_owned(),
            init: false,
            no_new_privileges: true,
            ipc_mode: "private".to_owned(),
        },
    })
}
fn start(source: &Path, executable: &str) -> Result<Namespace> {
    let process = Command::new("/usr/bin/unshare")
        .args([
            "--mount",
            "--pid",
            "--fork",
            "--kill-child=KILL",
            "--mount-proc",
        ])
        .arg(std::env::current_exe()?)
        .arg("--mount-guard")
        .arg(source)
        .arg(executable)
        .spawn()?;
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, process.id(), 0) } as i32;
    ensure!(fd >= 0, "held namespace wrapper pidfd");
    let pidfd = unsafe { OwnedFd::from_raw_fd(fd) };
    // Own cleanup even if setup/connection fails before a Namespace exists.
    let (dummy, _) = UnixStream::pair()?;
    let mut node = Namespace {
        process,
        pidfd,
        socket: dummy,
        finished: false,
    };
    let deadline = identity::boot_time_ms()? + 10000;
    loop {
        ensure!(
            node.process.try_wait()?.is_none(),
            "actual --live guard exited before listener"
        );
        identity::check_deadline(deadline)?;
        match UnixStream::connect(source.join("control.sock")) {
            Ok(socket) => {
                socket.set_read_timeout(Some(Duration::from_secs(10)))?;
                socket.set_write_timeout(Some(Duration::from_secs(3)))?;
                node.socket = socket;
                return Ok(node);
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                ) => {}
            Err(error) => return Err(error.into()),
        }
        linux::Io::poll(
            &mut [libc::pollfd {
                fd: node.pidfd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            }],
            5,
        )?;
    }
}

pub(crate) fn run(executable: &str, mode: &str) -> Result<()> {
    ensure!(
        unsafe { libc::geteuid() } == 0 && executable.starts_with('/'),
        "root and absolute guard ELF required"
    );
    ensure!(
        [
            "valid",
            "delayed",
            "bad-release",
            "replayed-release",
            "unreleased-zero",
            "bad-environment",
            "reused-session",
            "hardlinked-record"
        ]
        .contains(&mode)
            || registered_mode(mode),
        "explicit main fixture mode"
    );
    ensure!(
        !registered_mode(mode) || cfg!(feature = "model-test-support"),
        "registered modes require the explicit model-test-support build"
    );
    let deadline = identity::boot_time_ms()? + 10000;
    let server = identity::PinnedExecutable::open_process(
        unsafe { libc::getpid() } as u32,
        512 << 20,
        deadline,
    )?;
    let guard =
        identity::PinnedExecutable::from_file(File::open(executable)?, 512 << 20, deadline)?;
    let root = directory()?;
    println!("retained CPU startup records: {}", root.display());
    let pair_session = identity::fresh_nonce()?;
    let mut recipes = Vec::new();
    let mut sources = Vec::new();
    for rank in 0..2 {
        let source = root.join(format!("rank{rank}"));
        std::fs::DirBuilder::new().mode(0o700).create(&source)?;
        let recipe = fixture_recipe(rank, &source, server.digest(), guard.digest(), mode)?;
        let startup = wire::StartupRecord {
            pair_session,
            container_id: identity::fresh_nonce()?,
            image_digest: recipe.image_digest,
            recipe_digest: recipe.digest()?,
            server_elf_digest: server.digest(),
            guard_elf_digest: guard.digest(),
            policy: policy(),
        };
        write_record(&source.join("recipe.bin"), &recipe.encode()?)?;
        write_record(
            &source.join("startup.bin"),
            wire::Frame {
                rank,
                body: wire::Body::Startup(startup),
            }
            .encode()?
            .as_slice(),
        )?;
        File::open(&source)?.sync_all()?;
        recipes.push(startup);
        sources.push(source);
    }
    if mode == "hardlinked-record" {
        for source in &sources {
            std::fs::hard_link(source.join("recipe.bin"), source.join("recipe-alias"))?;
            ensure!(
                start(source, executable).is_err(),
                "hardlinked startup input must refuse"
            );
            ensure!(
                !source.join("consumed").exists(),
                "invalid startup must refuse before consuming or forking"
            );
        }
        println!("PASS (hardlinked-record): actual --live rejects nonexclusive input before listener/child");
        return Ok(());
    }
    let mut nodes = [
        start(&sources[0], executable)?,
        start(&sources[1], executable)?,
    ];
    let mut reports = Vec::new();
    for (rank, node) in nodes.iter_mut().enumerate() {
        let mut hello = [0; frame::LEN];
        node.socket.read_exact(&mut hello)?;
        let hello = frame::Frame::decode(&hello).map_err(linux::error)?;
        ensure!(hello.kind == frame::HELLO, "guard legacy HELLO");
        let report = read_frame(&mut node.socket, wire::Direction::GuardToController)?;
        ensure!(report.rank as usize == rank, "report rank");
        let wire::Body::GatedReport(report) = report.body else {
            bail!("gated report")
        };
        ensure!(
            report.pair_session == pair_session
                && report.record.container_id == recipes[rank].container_id
                && report.record.local_control_session == hello.session
                && report.record.guard_instance == hello.instance
                && report.record.original_startup_challenge == hello.challenge,
            "actual report/startup/HELLO binding"
        );
        reports.push(report.record);
    }
    let manifest = wire::Manifest {
        pair_session,
        policy_digest: wire::policy_digest(&policy())?,
        ranks: [reports[0], reports[1]],
    };
    main_exchange::run(&mut nodes, &manifest, mode, &sources)?;
    if registered_mode(mode) {
        for (rank, source) in sources.iter().enumerate() {
            let expected =
                if matches!(mode, "registered-drain" | "registered-drain-replay") && rank == 0 {
                    b"before-register\nregistered\ndrain-signal\n".as_slice()
                } else if mode == "registered-valid" || mode.starts_with("registered-drain") {
                    b"before-register\nregistered\n".as_slice()
                } else {
                    b"before-register\n".as_slice()
                };
            ensure!(
                std::fs::read(source.join("registered-witness"))? == expected,
                "actual Model registration/terminal witness mismatch"
            );
        }
        println!("PASS ({mode}): actual inherited startup, Model registration and terminal owner; valid mode uses actual head/worker shutdown and local Model quiescence, with scripted model command replay, NOT NCCL/GPU/Docker qualification");
        return Ok(());
    }
    if mode == "reused-session" {
        for source in &sources {
            ensure!(
                std::fs::read(source.join("consumed"))? == pair_session,
                "exact durable consumed marker"
            );
            ensure!(
                start(source, executable).is_err(),
                "same consumed session must not restart"
            );
        }
    }
    println!("PASS ({mode}): actual --live guard entry, private startup files, inherited server ingress and bounded paired exit; CPU fixture, NO Model/Docker/resource/GPU qualification");
    Ok(())
}
