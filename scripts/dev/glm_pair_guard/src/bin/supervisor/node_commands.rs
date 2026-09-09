// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use serde_json::{json, Value};

/// Called only after the latest full Docker snapshot passed exact validation.
/// A disappearing /proc entry is not health or exit evidence by itself.
pub(super) fn process_at_snapshot(
    stage: docker::Stage,
    process: io::Result<Option<NodeProcess>>,
) -> io::Result<Option<NodeProcess>> {
    match (stage, process) {
        (docker::Stage::Exited, Ok(_)) => Ok(None),
        (docker::Stage::Exited, Err(error))
            if error.kind() == io::ErrorKind::NotFound
                || error.raw_os_error() == Some(libc::ESRCH) =>
        {
            Ok(None)
        }
        (_, result) => result,
    }
}

pub(super) fn prepare(command: &Command, bytes: &[u8]) -> io::Result<Value> {
    let input: PrepareInput = serde_json::from_slice(bytes)?;
    input.metadata.validate()?;
    let encoded = decode_hex(&input.recipe_hex, wire::MAX_RECIPE_BYTES)?;
    let recipe = wire::Recipe::decode(&encoded).map_err(io::Error::other)?;
    validate_recipe(command, &recipe)?;
    // The exact create adapter applies native policy before any retained writes.
    docker::create_request(&recipe, &input.metadata.guard_container_path, "runc")?;
    let deadline = now()?
        .checked_add(input.metadata.command_ms)
        .ok_or_else(|| error("ELF deadline"))?;
    engine::executable(
        &input.metadata.relay_path,
        docker::parse_id(&input.metadata.relay_sha256)?,
        deadline,
    )?;
    let directory = files::Directory::open(&command.session, command.rank, true)?;
    directory.write_new(c"prepare.json", &serde_json::to_vec(&input.metadata)?)?;
    directory.write_new(c"recipe.bin", &encoded)?;
    Ok(json!({}))
}
pub(super) fn validate_recipe(command: &Command, recipe: &wire::Recipe) -> io::Result<()> {
    recipe.validate().map_err(io::Error::other)?;
    let mount = recipe
        .mounts
        .iter()
        .find(|m| m.destination == "/run/atlas-pair")
        .ok_or_else(|| error("missing exact startup mount"))?;
    if recipe.rank != command.rank
        || recipe.world != 2
        || mount.source
            != format!(
                "/run/atlas-glm-pairs/{}/rank{}",
                command.session, command.rank
            )
        || mount.read_only
        || mount.propagation != "rprivate"
    {
        return Err(error("node recipe session/rank/bind identity"));
    }
    Ok(())
}
pub(super) struct Loaded {
    pub directory: files::Directory,
    pub metadata: Metadata,
    pub recipe: wire::Recipe,
}
impl Loaded {
    pub fn open(command: &Command) -> io::Result<Self> {
        let directory = files::Directory::open(&command.session, command.rank, false)?;
        let metadata: Metadata =
            serde_json::from_slice(&directory.read(c"prepare.json", MAX_JSON)?)?;
        metadata.validate()?;
        let recipe = wire::Recipe::decode(&directory.read(c"recipe.bin", wire::MAX_RECIPE_BYTES)?)
            .map_err(io::Error::other)?;
        validate_recipe(command, &recipe)?;
        docker::create_request(&recipe, &metadata.guard_container_path, "runc")?;
        Ok(Self {
            directory,
            metadata,
            recipe,
        })
    }
    fn recorded(&self, command: &Command) -> io::Result<wire::Digest> {
        let raw = self.directory.read(c"container.id", 64)?;
        let text = std::str::from_utf8(&raw).map_err(|_| error("container ID encoding"))?;
        let id = docker::parse_id(text)?;
        if command.id.as_ref() != Some(&id) {
            return Err(error("node requested ID differs from retained create ID"));
        }
        Ok(id)
    }
    fn read_inspect(&self, id: &wire::Digest) -> io::Result<Value> {
        let bytes = engine::request(
            &self.metadata,
            "GET",
            &format!("/containers/{}/json", docker::hex(id)),
            vec![],
            200,
        )?;
        Ok(serde_json::from_slice(&bytes)?)
    }
    fn inspect(&self, id: &wire::Digest, stage: docker::Stage) -> io::Result<(Value, u32)> {
        let value = self.read_inspect(id)?;
        let pid = docker::inspect(
            &self.recipe,
            &self.metadata.guard_container_path,
            "runc",
            id,
            stage,
            &value,
        )?;
        Ok((value, pid))
    }
    pub fn create(&self) -> io::Result<Value> {
        let request =
            docker::create_request(&self.recipe, &self.metadata.guard_container_path, "runc")?;
        // An ambiguous failed create cannot be retried into a second container.
        self.directory.write_new(c"create-issued", b"1")?;
        let bytes = engine::request(
            &self.metadata,
            "POST",
            "/containers/create",
            serde_json::to_vec(&request)?,
            201,
        )?;
        let response: Value = serde_json::from_slice(&bytes)?;
        let text = response
            .get("Id")
            .and_then(Value::as_str)
            .ok_or_else(|| error("Docker create omitted ID"))?;
        let id = docker::parse_id(text)?;
        self.directory.write_new(c"container.id", text.as_bytes())?;
        let (inspect, _) = self.inspect(&id, docker::Stage::Created)?;
        Ok(json!({"container_id":text,"inspect":inspect}))
    }
    pub fn seal(&self, command: &Command, bytes: &[u8]) -> io::Result<Value> {
        let id = self.recorded(command)?;
        self.validate_startup(command, &id, bytes)?;
        self.inspect(&id, docker::Stage::Created)?;
        self.directory.write_new(c"startup.bin", bytes)?;
        Ok(json!({}))
    }
    fn validate_startup(
        &self,
        command: &Command,
        id: &wire::Digest,
        bytes: &[u8],
    ) -> io::Result<()> {
        let frame =
            wire::Frame::decode(bytes, wire::Direction::StartupFile).map_err(io::Error::other)?;
        let wire::Body::Startup(startup) = frame.body else {
            return Err(error("startup frame required"));
        };
        if frame.rank != command.rank
            || startup.pair_session != docker::parse_id(&command.session)?
            || startup.container_id != *id
            || startup.image_digest != self.recipe.image_digest
            || startup.recipe_digest != self.recipe.digest().map_err(io::Error::other)?
            || startup.guard_elf_digest != self.recipe.guard_elf_digest
            || startup.server_elf_digest != self.recipe.server_elf_digest
            || startup.policy != self.metadata.policy.to_wire()?
        {
            return Err(error(
                "sealed startup does not match retained recipe/ID/policy",
            ));
        }
        Ok(())
    }
    pub fn start(&self, command: &Command) -> io::Result<Value> {
        let id = self.recorded(command)?;
        self.validate_startup(command, &id, &self.directory.read(c"startup.bin", 288)?)?;
        self.inspect(&id, docker::Stage::Created)?;
        self.directory.write_new(c"start-issued", b"1")?;
        engine::request(
            &self.metadata,
            "POST",
            &format!("/containers/{}/start", docker::hex(&id)),
            vec![],
            204,
        )?;
        Ok(json!({}))
    }
    pub fn observation(&self, command: &Command, socket_only: bool) -> io::Result<NodeObservation> {
        let id = self.recorded(command)?;
        let inspect = self.read_inspect(&id)?;
        let stage = docker::observation_stage(&inspect, socket_only)?;
        let pid = docker::inspect(
            &self.recipe,
            &self.metadata.guard_container_path,
            "runc",
            &id,
            stage,
            &inspect,
        )?;
        // A child can have exited while PID1 is finishing matched release.
        // The controller permits None only pre-report or after local release;
        // it must never renew from this absence.
        let process = if pid == 0 {
            Ok(None)
        } else {
            proc::pair(pid, true)
        };
        let socket_ready = pid != 0 && self.directory.socket_ready()?;
        let (mem_available_kib, swap_used_kib) = proc::memory()?;
        // A second exact inspect closes the Docker PID/config read window.
        let after = self.read_inspect(&id)?;
        let (after_stage, _) = docker::observation_pair(
            &self.recipe,
            &self.metadata.guard_container_path,
            &id,
            socket_only,
            &inspect,
            &after,
        )?;
        Ok(NodeObservation {
            inspect: after,
            process: process_at_snapshot(after_stage, process)?,
            socket_ready: after_stage == docker::Stage::Running && socket_ready,
            mem_available_kib,
            swap_used_kib,
        })
    }
    pub fn relay(&self, command: &Command) -> io::Result<()> {
        let id = self.recorded(command)?;
        self.validate_startup(command, &id, &self.directory.read(c"startup.bin", 288)?)?;
        self.inspect(&id, docker::Stage::Running)?;
        if !self.directory.socket_ready()? {
            return Err(error("control socket not ready"));
        }
        self.directory.write_new(c"relay-issued", b"1")?;
        engine::relay(&self.metadata, &command.session, command.rank)
    }
    pub fn kill(&self, command: &Command) -> io::Result<Value> {
        let id = self.recorded(command)?;
        // Cleanup deliberately does not require a healthy recipe/state inspect;
        // the exact full ID from the retained create is the only target.
        engine::request(
            &self.metadata,
            "POST",
            &format!("/containers/{}/kill?signal=SIGKILL", docker::hex(&id)),
            vec![],
            204,
        )?;
        Ok(json!({}))
    }
}
