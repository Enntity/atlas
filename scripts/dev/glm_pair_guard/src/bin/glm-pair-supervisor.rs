// SPDX-License-Identifier: AGPL-3.0-only
//! Deployment-specific paired-serving controller adapters. No implicit launch.
#![recursion_limit = "256"]
use atlas_glm_pair_wire as wire;
use std::io::{self, Read, Write};

#[path = "supervisor/docker.rs"]
mod docker;
#[allow(dead_code)] // Existing legacy control codec; no alternate parser.
#[path = "../frame.rs"]
mod frame;
#[allow(dead_code)] // Shared bounded command adapter, connected by the run driver next.
#[path = "supervisor/process.rs"]
mod process;
#[allow(dead_code)] // Run-loop wiring follows the separately checked pure transition layer.
#[path = "supervisor/state.rs"]
mod state;

fn error(message: &'static str) -> io::Error {
    io::Error::other(message)
}

fn run() -> io::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !((args.len() == 4 && args[0] == "create-request")
        || (args.len() == 6 && args[0] == "inspect"))
    {
        return Err(error("usage: create-request RECIPE_FILE ABSOLUTE_GUARD_PATH runc | inspect RECIPE_FILE ABSOLUTE_GUARD_PATH runc FULL_CONTAINER_ID created|running|exited (JSON on stdin)"));
    }
    let mut bytes = Vec::new();
    std::fs::File::open(&args[1])?
        .take((wire::MAX_RECIPE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    let recipe = wire::Recipe::decode(&bytes).map_err(io::Error::other)?;
    if args[0] == "inspect" {
        let id = docker::parse_id(&args[4])?;
        let stage = match args[5].as_str() {
            "created" => docker::Stage::Created,
            "running" => docker::Stage::Running,
            "exited" => docker::Stage::Exited,
            _ => return Err(error("explicit inspect phase required")),
        };
        bytes.clear();
        io::stdin().lock().take(1_048_577).read_to_end(&mut bytes)?;
        if bytes.len() > 1_048_576 {
            return Err(error("Docker observation exceeds bound"));
        }
        let observed: serde_json::Value = serde_json::from_slice(&bytes)?;
        let pid = docker::inspect(&recipe, &args[2], &args[3], &id, stage, &observed)?;
        serde_json::to_writer(
            io::stdout().lock(),
            &serde_json::json!({"host_init_pid":pid}),
        )?;
    } else {
        let request = docker::create_request(&recipe, &args[2], &args[3])?;
        serde_json::to_writer(io::stdout().lock(), &request)?;
    }
    io::stdout().write_all(b"\n")
}

fn main() {
    if let Err(e) = run() {
        eprintln!("paired supervisor: {e}");
        std::process::exit(74);
    }
}
