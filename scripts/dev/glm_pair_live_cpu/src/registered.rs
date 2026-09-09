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
    let mut fixture = if mode == "registered-missing-capability" {
        Fixture::legacy(usize::from(model_rank))
    } else {
        Fixture::paired(usize::from(model_rank))
    };
    let wire = fixture.install_wire();
    let (model, mut sequences, observer) = fixture.into_parts();
    if (mode == "registered-valid" || mode.starts_with("registered-drain")) && rank == 1 {
        // The fixture starts with two real owners. Retire those before actual
        // run_worker allocates its own two slots; never fabricate SlotGuards.
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
    ensure!(
        observer.events() == [Event::Health(true)],
        "registration actual health only"
    );
    witness.mark(b"registered\n")?;
    if mode.starts_with("registered-drain") && rank == 0 {
        let deadline = atlas_glm_pair_io::identity::boot_time_ms()? + 10_000;
        while !DRAIN_SIGNAL.load(Ordering::Acquire) {
            owner.check_health();
            atlas_glm_pair_io::identity::check_deadline(deadline)?;
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        witness.mark(b"drain-signal\n")?;
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
