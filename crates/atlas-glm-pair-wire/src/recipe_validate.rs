// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::Error;

fn text(value: &str, nonempty: bool) -> Result<()> {
    if value.len() > 4096 || value.contains('\0') || nonempty && value.is_empty() {
        return Err(Error("invalid bounded recipe string"));
    }
    Ok(())
}
fn count(n: usize, max: usize) -> Result<()> {
    if n > max {
        return Err(Error("recipe vector exceeds bound"));
    }
    Ok(())
}
fn ordered<'a>(keys: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut previous = None;
    for key in keys {
        if previous.is_some_and(|p| p >= key) {
            return Err(Error("recipe keys not sorted unique"));
        }
        previous = Some(key);
    }
    Ok(())
}
fn strings(values: &[String], max: usize) -> Result<()> {
    count(values.len(), max)?;
    for v in values {
        text(v, true)?;
    }
    ordered(values.iter().map(String::as_str))
}
fn pairs(values: &[(String, String)], max: usize) -> Result<()> {
    count(values.len(), max)?;
    for (k, v) in values {
        text(k, true)?;
        text(v, false)?;
    }
    ordered(values.iter().map(|(k, _)| k.as_str()))
}

impl Profile {
    pub fn validate(&self) -> Result<()> {
        if self.tp != 2
            || self.ep != 2
            || self.ep_protocol != 2
            || self.max_sequences != 2
            || self.context != 2044
            || self.prefill != 1024
            || self.drafts != 4
            || !self.eager
            || self.kv_format != 1
            || self.cold_min != 2
            || self.cold_max != 1024
        {
            return Err(Error("recipe is not the fixed selected profile"));
        }
        Ok(())
    }
}

pub(super) fn recipe(v: &Recipe) -> Result<()> {
    if v.rank > 1 || v.world != 2 {
        return Err(Error("recipe rank/world"));
    }
    for digest in [&v.image_digest, &v.guard_elf_digest, &v.server_elf_digest] {
        if *digest == [0; 32] {
            return Err(Error("missing recipe digest"));
        }
    }
    v.profile.validate()?;
    count(v.argv.len(), 64)?;
    if v.argv.is_empty() {
        return Err(Error("missing recipe argv"));
    }
    for (i, arg) in v.argv.iter().enumerate() {
        text(arg, i == 0)?;
    }
    pairs(&v.environment, 128)?;
    if v.environment.is_empty() {
        return Err(Error("missing explicit recipe environment"));
    }
    for (key, _) in &v.environment {
        if key.as_bytes()[0].is_ascii_digit()
            || !key
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
        {
            return Err(Error("noncanonical environment key"));
        }
    }
    count(v.mounts.len(), 32)?;
    ordered(v.mounts.iter().map(|m| m.destination.as_str()))?;
    for m in &v.mounts {
        text(&m.source, true)?;
        text(&m.destination, true)?;
        text(&m.propagation, true)?;
    }
    resources(&v.resources)
}

fn resources(v: &Resources) -> Result<()> {
    for field in [
        &v.cpuset,
        &v.network_mode,
        &v.pid_mode,
        &v.restart_policy,
        &v.ipc_mode,
    ] {
        text(field, true)?;
    }
    count(v.device_requests.len(), 8)?;
    ordered(v.device_requests.iter().map(|d| d.driver.as_str()))?;
    for d in &v.device_requests {
        // Docker's explicit empty driver is permitted; it is still serialized.
        text(&d.driver, false)?;
        if d.count < -1 {
            return Err(Error("invalid Docker device count"));
        }
        strings(&d.device_ids, 8)?;
        pairs(&d.options, 32)?;
        count(d.capabilities.len(), 8)?;
        for c in &d.capabilities {
            strings(c, 16)?;
        }
        if d.capabilities.windows(2).any(|p| p[0] >= p[1]) {
            return Err(Error("capability groups not sorted unique"));
        }
    }
    count(v.ulimits.len(), 32)?;
    ordered(v.ulimits.iter().map(|u| u.name.as_str()))?;
    for u in &v.ulimits {
        text(&u.name, true)?;
        if u.soft < -1 || u.hard < -1 {
            return Err(Error("invalid Docker ulimit"));
        }
    }
    strings(&v.cap_add, 64)?;
    strings(&v.cap_drop, 64)?;
    count(v.devices.len(), 32)?;
    ordered(v.devices.iter().map(|d| d.path_in_container.as_str()))?;
    for d in &v.devices {
        text(&d.path_on_host, true)?;
        text(&d.path_in_container, true)?;
        if !matches!(
            d.cgroup_permissions.as_str(),
            "r" | "w" | "m" | "rw" | "rm" | "wm" | "rwm"
        ) {
            return Err(Error("noncanonical device permissions"));
        }
    }
    strings(&v.security_options, 32)?;
    for option in &v.security_options {
        let key = option.split(['=', ':']).next().unwrap_or("").trim();
        if key.eq_ignore_ascii_case("no-new-privileges") {
            return Err(Error("no-new-privileges has a separate boolean authority"));
        }
    }
    // Explicit resource values are data, not admission. Root separately checks
    // memory/swap, device policy, modes, capabilities and literal environment.
    Ok(())
}
