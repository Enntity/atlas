// SPDX-License-Identifier: AGPL-3.0-only

//! Actual server registration/worker/shutdown source with real CPU Model owners.
//! No fabricated session, capability, terminal key, or cross-process collective.
#[rustfmt::skip]
#[path = "../../../../crates/spark-server/src/glm_terminal_session/core.rs"]
mod core;
#[rustfmt::skip]
#[path = "../../../../crates/spark-server/src/glm_terminal_session/panic.rs"]
mod panic;
#[rustfmt::skip]
#[path = "../../../../crates/spark-server/src/glm_terminal_session/runtime.rs"]
mod runtime;
pub(crate) use runtime::terminate;
use runtime::CORE;
#[rustfmt::skip]
#[path = "../../../../crates/spark-server/src/glm_terminal_session/inherited.rs"]
mod inherited;
#[rustfmt::skip]
#[path = "../../../../crates/spark-server/src/glm_terminal_session/inherited_startup.rs"]
mod inherited_startup;
#[rustfmt::skip]
#[path = "../../../../crates/spark-server/src/glm_terminal_session/selected.rs"]
mod selected;
#[path = "registered_controller.rs"]
mod registered_controller;

use anyhow::{ensure, Result};
use spark_model::model::glm_c2_test_support::{Event, Fixture};
use spark_model::traits::Model;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicBool, Ordering};

static DRAIN_SIGNAL: AtomicBool = AtomicBool::new(false);
extern "C" fn drain_signal(_: i32) {
    DRAIN_SIGNAL.store(true, Ordering::Release);
}

struct Witness(File);
impl Witness {
    fn mark(&mut self, text: &[u8]) -> Result<()> {
        self.0.write_all(text)?;
        self.0.sync_all()?;
        Ok(())
    }
}
impl Drop for Witness {
    fn drop(&mut self) {
        let _ = self.0.write_all(b"unexpected-drop\n");
        let _ = self.0.sync_all();
    }
}

pub(crate) fn consumer() -> Result<()> {
    runtime::install_panic_ingress();
    // This runs before threads and before model construction, over real FD3.
    let received = unsafe { inherited_startup::receive(20000, 512 * 1024 * 1024) }?;
    let rank = received.recipe.rank;
    let mode = std::env::var("ATLAS_PAIR_CPU_MODE")?;
    ensure!(
        super::namespace::registered_mode(&mode),
        "registered fixture mode"
    );
    if mode.starts_with("registered-drain") && rank == 0 {
        let mut action = unsafe { std::mem::zeroed::<libc::sigaction>() };
        action.sa_sigaction = drain_signal as *const () as usize;
        unsafe { libc::sigemptyset(&mut action.sa_mask) };
        ensure!(
            unsafe { libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut()) } == 0,
            "install actual CPU drain signal"
        );
    }
    let mut witness = Witness(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open("/run/atlas-pair/registered-witness")?,
    );
    let model_rank = if mode == "registered-wrong-rank" {
        1 - rank
    } else {
        rank
    };
    let owners = usize::from(received.recipe.profile.max_sequences);
    let valid = matches!(
        mode.as_str(),
        "registered-valid" | "registered-valid3" | "registered-valid4"
    ) || mode.starts_with("registered-drain");
    let mut fixture = if mode == "registered-missing-capability" {
        Fixture::legacy(usize::from(model_rank))
    } else if mode == "registered-capacity-mismatch" {
        // Actual four-owner pool/target construction versus the received,
        // unchanged two-owner recipe; no fabricated capability or ticket.
        Fixture::paired_compute_with_owner_capacity(usize::from(model_rank), 4)
    } else if matches!(mode.as_str(), "registered-valid3" | "registered-valid4") {
        Fixture::paired_compute_with_owner_capacity(usize::from(model_rank), owners)
    } else {
        Fixture::paired(usize::from(model_rank))
    };
    let wire = fixture.install_wire();
    let (model, sequences, observer) = fixture.into_parts();
    let mut sequences = Vec::from(sequences);
    if valid && rank == 0 {
        while sequences.len() < owners {
            let sequence = model.alloc_sequence()?;
            ensure!(
                sequence.slot_idx == sequences.len(),
                "actual head fixture owner index"
            );
            sequences.push(sequence);
        }
    }
    if valid && rank == 1 {
        // The fixture starts with two real owners. Retire those before actual
        // run_worker allocates its own exact capacity; never fabricate SlotGuards.
        for sequence in &mut sequences {
            model.free_sequence(sequence)?;
        }
        wire.queue(&[vec![0], vec![atlas_glm_pair_wire::SHUTDOWN_COMMAND]]);
    }
    observer.clear();
    if mode == "registered-unhealthy" {
        observer.fail_at(1);
    }
    witness.mark(b"before-register\n")?;
    let owner = selected::SelectedModel::register(Box::new(model), received, rank);
    if valid {
        ensure!(
            owner.owner_capacity() == owners,
            "actual registered capacity"
        );
    }
    ensure!(
        observer.events() == [Event::Health(true)],
        "registration actual health only"
    );
    witness.mark(b"registered\n")?;
    if mode == "registered-capacity-mismatch" {
        // A false acceptance must fail the parent's exact witness oracle,
        // without proceeding into worker allocations or head shutdown.
        terminate();
    }
    if mode == "registered-drain-controller" {
        let operation = owner.begin();
        operation.require(registered_controller::wait(&owner, rank));
        if rank == 0 {
            operation.require(witness.mark(b"drain-signal\n"));
        }
        operation.complete();
    } else if mode.starts_with("registered-drain") && rank == 0 {
        let deadline = atlas_glm_pair_io::identity::boot_time_ms()? + 10_000;
        while !DRAIN_SIGNAL.load(Ordering::Acquire) {
            owner.check_health();
            atlas_glm_pair_io::identity::check_deadline(deadline)?;
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        witness.mark(b"drain-signal\n")?;
    }
    if mode == "registered-drain-threaded" {
        // Fixture scheduling only: let the controller observe both genuine
        // registrations and issue the actual rank0 drain before either new
        // thread can fail. This file confers no Model or release authority.
        let deadline = atlas_glm_pair_io::identity::boot_time_ms()? + 15000;
        loop {
            owner.check_health();
            atlas_glm_pair_io::identity::check_deadline(deadline)?;
            match std::fs::read("/run/atlas-pair/thread-handoff") {
                Ok(bytes) => {
                    ensure!(bytes == b"handoff\n", "thread fixture handoff marker");
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        std::thread::spawn(move || {
            // Retain actual sequence ownership alongside the moved actual
            // SelectedModel/InheritedSession, just as native serving does.
            let _sequences = sequences;
            (|| -> Result<()> {
                let mut signal = -1;
                ensure!(
                    unsafe { libc::prctl(libc::PR_GET_PDEATHSIG, &mut signal) } == 0,
                    "read actual thread parent-death signal"
                );
                ensure!(signal == 0, "new OS thread must start without PDEATHSIG");
                witness.mark(b"thread-pdeathsig=0\n")
            })()
            .unwrap_or_else(|_| terminate());
            if rank == 0 {
                // The real head scheduler calls this before its Model bind;
                // actual run_worker below performs its own same handoff.
                owner.bind_execution_thread();
                owner.shutdown_head()
            } else {
                owner.run_worker()
            }
        })
        .join()
        .unwrap_or_else(|_| terminate());
        // Both real production entry points are nonreturning. A join is never
        // interpreted as a quiescence certificate or permission to Drop.
        terminate();
    }
    // Both functions are the actual nonreturning production paths. Rank0 sends
    // the shutdown words; rank1 consumes that exact protocol via local replay.
    // Guard control remains a genuine two-process connected exchange.
    if rank == 0 {
        owner.shutdown_head()
    } else {
        owner.run_worker()
    }
}
