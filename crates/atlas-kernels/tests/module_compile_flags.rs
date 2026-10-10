// SPDX-License-Identifier: AGPL-3.0-only
#[path = "../build_module_flags.rs"]
mod module_flags;
use module_flags::{ModuleFlags, compile_key, effective_flags};
use std::path::Path;

fn overrides() -> ModuleFlags {
    [("exl3_mhc".into(), vec!["--fmad=true".into()])].into()
}

#[test]
fn explicit_module_override_replaces_conflicting_option_and_keeps_other_flags() {
    let base = vec!["--fmad=false".into(), "-DTQ_PLUS_SIGNS".into()];
    assert_eq!(
        effective_flags(&base, "exl3_mhc", &overrides()),
        vec!["-DTQ_PLUS_SIGNS", "--fmad=true"]
    );
}

#[test]
fn absent_module_override_preserves_legacy_flags_and_order() {
    let base = vec!["--fmad=false".into(), "-DTQ_PLUS_SIGNS".into()];
    assert_eq!(
        effective_flags(&base, "hyper_connection", &overrides()),
        base
    );
    assert_eq!(
        effective_flags(&base, "exl3_mhc", &ModuleFlags::new()),
        base
    );
}

#[test]
fn effective_module_flags_are_part_of_the_real_compile_dedup_key() {
    let base = vec!["--fmad=false".into()];
    let changed = effective_flags(&base, "exl3_mhc", &overrides());
    assert_ne!(
        compile_key(Path::new("exl3_mhc.cu"), "compute_121", &base),
        compile_key(Path::new("exl3_mhc.cu"), "compute_121", &changed)
    );
}

#[test]
fn actual_exl3_model_declares_only_its_mhc_override() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/glm-5.3-flash-exl3/MODEL.toml");
    let flags = module_flags::parse_module_compile_flags(&path, "nvidia");
    assert_eq!(flags, overrides());
    assert!(module_flags::parse_module_compile_flags(&path, "metal").is_empty());
    let legacy = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/deepseek-v4-flash/nvfp4/KERNEL.toml");
    assert!(module_flags::parse_module_compile_flags(&legacy, "nvidia").is_empty());
}

#[test]
fn malformed_explicit_compiler_options_fail_instead_of_disappearing() {
    for (label, text) in [
        ("wrong-array", "[build.module_nvcc_flags]\nexl3_mhc=1"),
        (
            "wrong-token",
            "[build.module_nvcc_flags]\nexl3_mhc=[\"--fmad=true -O3\"]",
        ),
        (
            "duplicate",
            "[build.module_nvcc_flags]\nexl3_mhc=[\"--fmad=true\",\"--fmad=false\"]",
        ),
    ] {
        let path = std::env::temp_dir().join(format!(
            "atlas-module-flags-{}-{label}.toml",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        let result =
            std::panic::catch_unwind(|| module_flags::parse_module_compile_flags(&path, "nvidia"));
        std::fs::remove_file(path).unwrap();
        assert!(result.is_err(), "accepted malformed {label}");
    }
}

#[test]
fn unknown_module_flags_fail_and_resolved_aliases_are_accepted() {
    let available = ["norm".to_owned()].into();
    let known = [("norm".into(), vec!["--fmad=true".into()])].into();
    assert!(module_flags::validate_modules(&available, &known).is_ok());
    let wrong = [("rms_norm".into(), vec!["--fmad=true".into()])].into();
    assert_eq!(
        module_flags::validate_modules(&available, &wrong),
        Err("unknown compiler-flag module rms_norm".into())
    );
}
