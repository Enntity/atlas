// SPDX-License-Identifier: AGPL-3.0-only
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(crate) type ModuleFlags = BTreeMap<String, Vec<String>>;

pub(crate) fn effective_flags(
    base: &[String],
    module: &str,
    overrides: &ModuleFlags,
) -> Vec<String> {
    let Some(flags) = overrides.get(module) else {
        return base.to_vec();
    };
    let option = |flag: &str| flag.split('=').next().unwrap_or(flag).to_owned();
    let mut result: Vec<String> = base
        .iter()
        .filter(|old| !flags.iter().any(|new| option(old) == option(new)))
        .cloned()
        .collect();
    result.extend(flags.iter().cloned());
    result
}

pub(crate) fn compile_key(
    source: &Path,
    arch: &str,
    flags: &[String],
) -> (PathBuf, String, Vec<String>) {
    let mut flags = flags.to_vec();
    flags.sort();
    (source.to_path_buf(), arch.to_owned(), flags)
}

/// Optional per-module compiler flags, keyed by resolved module names. Values
/// replace conflicting `option=value` tokens from the target's base flags.
/// MODEL.toml can scope policy without editing a shared quant-directory symlink.
pub(crate) fn parse_module_compile_flags(path: &std::path::Path, vendor: &str) -> ModuleFlags {
    let mut result = ModuleFlags::new();
    if !path.is_file() {
        return result;
    }
    let document: toml::Value = toml::from_str(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
    let key = match vendor {
        "apple" | "metal" => "module_metal_flags",
        _ => "module_nvcc_flags",
    };
    let Some(value) = document.get("build").and_then(|build| build.get(key)) else {
        return result;
    };
    let table = value
        .as_table()
        .unwrap_or_else(|| panic!("{}: build.{key} must be a table", path.display()));
    for (module, value) in table {
        let values = value
            .as_array()
            .unwrap_or_else(|| panic!("{}: {module} flags must be an array", path.display()));
        let mut flags = Vec::new();
        let mut keys = std::collections::HashSet::new();
        for value in values {
            let flag = value
                .as_str()
                .filter(|flag| !flag.is_empty() && !flag.contains(char::is_whitespace))
                .unwrap_or_else(|| {
                    panic!(
                        "{}: {module} flags must be nonempty individual tokens",
                        path.display()
                    )
                });
            let option = flag.split('=').next().unwrap_or(flag);
            assert!(
                keys.insert(option),
                "{}: duplicate compiler option {option} for {module}",
                path.display()
            );
            flags.push(flag.to_owned());
        }
        result.insert(module.clone(), flags);
    }
    result
}

pub(crate) fn validate_modules(
    available: &std::collections::HashSet<String>,
    overrides: &ModuleFlags,
) -> Result<(), String> {
    for module in overrides.keys() {
        if !available.contains(module) {
            return Err(format!("unknown compiler-flag module {module}"));
        }
    }
    Ok(())
}
