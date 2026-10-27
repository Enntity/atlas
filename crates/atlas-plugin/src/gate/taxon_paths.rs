// SPDX-License-Identifier: AGPL-3.0-only

//! Resolve the build's single-hop model and quant composition rules.
use std::path::{Component, Path, PathBuf};

/// Distinguish absent optional metadata from unreadable or malformed metadata.
fn reference(path: &Path, table: &str, key: &str) -> Option<Option<String>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(None),
        Err(_) => return None,
    };
    let value: toml::Value = toml::from_str(&text).ok()?;
    let Some(value) = value.get(table).and_then(|t| t.get(key)) else {
        return Some(None);
    };
    let name = value.as_str()?;
    let mut parts = Path::new(name).components();
    if !matches!(parts.next(), Some(Component::Normal(_))) || parts.next().is_some() {
        return None;
    }
    Some(Some(name.to_owned()))
}

pub(super) fn model_dir(hw_dir: &Path, model: &str) -> Option<PathBuf> {
    let model_dir = hw_dir.join(model);
    if !model_dir.join("MODEL.toml").is_file() {
        return None;
    }
    match reference(&model_dir.join("MODEL.toml"), "model", "kernel_source")? {
        None => Some(model_dir),
        Some(source) => {
            let dir = hw_dir.join(source);
            if !dir.join("MODEL.toml").is_file()
                || reference(&dir.join("MODEL.toml"), "model", "kernel_source")?.is_some()
            {
                return None;
            }
            Some(dir)
        }
    }
}

pub(super) fn overlays(model_dir: &Path, quant: &str) -> Option<Vec<PathBuf>> {
    let dir = model_dir.join(quant);
    let mut dirs = Vec::new();
    if let Some(base) = reference(&dir.join("KERNEL.toml"), "build", "base_quant")? {
        let base_dir = model_dir.join(base);
        if !base_dir.is_dir()
            || reference(&base_dir.join("KERNEL.toml"), "build", "base_quant")?.is_some()
        {
            return None;
        }
        dirs.push(base_dir);
    }
    dirs.push(dir);
    Some(dirs)
}
