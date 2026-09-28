// SPDX-License-Identifier: AGPL-3.0-only

//! Actual fixed-path startup input for the inherited handshake, not activation.
//! Exact argv/env binding does not itself validate parsed CLI/model settings.

use super::inherited::{ExpectedSession, InheritedSession};
use anyhow::{Context, Result, anyhow, ensure};
use atlas_glm_pair_io::{PrivateDirectory, identity::boot_time_ms};
use atlas_glm_pair_wire::{Body, Direction, Frame, MAX_RECIPE_BYTES, Recipe, recipe_digest};

/// Keep the validated launch settings with their actual inherited session.
pub(crate) struct ReceivedStartup {
    pub(crate) session: InheritedSession,
    pub(crate) recipe: Recipe,
}

fn remaining(deadline: u64, last: &mut u64) -> Result<u64> {
    let now = boot_time_ms()?;
    ensure!(
        now >= *last && now < deadline,
        "startup deadline/clock failure"
    );
    *last = now;
    Ok(deadline - now)
}

fn arguments() -> Result<Vec<String>> {
    let mut values = Vec::new();
    for value in std::env::args_os() {
        ensure!(values.len() < 64, "startup argv exceeds bound");
        let value = value
            .into_string()
            .map_err(|_| anyhow!("non-UTF8 startup argument"))?;
        ensure!(value.len() <= 4096, "startup argument exceeds bound");
        values.push(value);
    }
    Ok(values)
}

fn environment() -> Result<Vec<(String, String)>> {
    let mut values = Vec::new();
    for (key, value) in std::env::vars_os() {
        ensure!(values.len() < 128, "startup environment exceeds bound");
        let key = key
            .into_string()
            .map_err(|_| anyhow!("non-UTF8 startup environment key"))?;
        let value = value
            .into_string()
            .map_err(|_| anyhow!("non-UTF8 startup environment value"))?;
        ensure!(
            key.len() <= 4096 && value.len() <= 4096,
            "startup environment entry exceeds bound"
        );
        values.push((key, value));
    }
    // Preserve duplicates so they cannot be silently normalized into agreement.
    values.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(values)
}

/// # Safety
/// Single-threaded early ingress exclusively controls FD3 and the process
/// environment. Failure is terminal for selected startup, never a fallback.
pub(crate) unsafe fn receive(
    handshake_ms: u64,
    max_executable_bytes: u64,
) -> Result<ReceivedStartup> {
    // Before opening ANY record/directory: otherwise a missing inherited socket
    // could be replaced by a newly opened record at FD3 and mistaken for ownership.
    if unsafe { libc::fcntl(3, libc::F_GETFD) } < 0 {
        return Err(std::io::Error::last_os_error())
            .context("missing inherited FD3 before startup I/O");
    }
    ensure!(
        handshake_ms > 0 && handshake_ms <= 86_400_000,
        "invalid explicit startup deadline"
    );
    ensure!(max_executable_bytes >= 4, "invalid explicit ELF bound");
    let mut last = boot_time_ms()?;
    let deadline = last
        .checked_add(handshake_ms)
        .context("startup deadline overflow")?;
    let directory = PrivateDirectory::open()?;
    remaining(deadline, &mut last)?;
    let startup_bytes = directory.read(c"startup.bin", 288)?;
    remaining(deadline, &mut last)?;
    let frame = Frame::decode(&startup_bytes, Direction::StartupFile)?;
    let Body::Startup(startup) = frame.body else {
        anyhow::bail!("expected startup file record")
    };
    ensure!(
        handshake_ms <= startup.policy.child_handshake,
        "startup exceeds child handshake policy"
    );
    let recipe_bytes = directory.read(c"recipe.bin", MAX_RECIPE_BYTES)?;
    remaining(deadline, &mut last)?;
    let recipe = Recipe::decode(&recipe_bytes)?;
    ensure!(
        recipe_digest(&recipe_bytes)? == startup.recipe_digest,
        "startup recipe digest mismatch"
    );
    ensure!(
        recipe.rank == frame.rank && recipe.world == 2,
        "startup recipe rank/world mismatch"
    );
    ensure!(
        recipe.image_digest == startup.image_digest
            && recipe.server_elf_digest == startup.server_elf_digest
            && recipe.guard_elf_digest == startup.guard_elf_digest,
        "startup recipe image/ELF assertion mismatch"
    );
    ensure!(
        arguments()? == recipe.argv,
        "actual startup argv differs from recipe"
    );
    ensure!(
        environment()? == recipe.environment,
        "actual startup environment differs from recipe"
    );
    directory.revalidate()?;
    let expected = ExpectedSession {
        rank: frame.rank,
        pair_session: startup.pair_session,
        policy: startup.policy,
        container_id: startup.container_id,
        image_digest: startup.image_digest,
        recipe_digest: startup.recipe_digest,
        server_elf_digest: startup.server_elf_digest,
        guard_elf_digest: startup.guard_elf_digest,
        max_executable_bytes,
    };
    let left = remaining(deadline, &mut last)?;
    let session = unsafe { InheritedSession::receive(expected, left) }?;
    directory.revalidate()?;
    remaining(deadline, &mut last)?;
    Ok(ReceivedStartup { session, recipe })
}
