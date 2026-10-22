// SPDX-License-Identifier: AGPL-3.0-only

use super::{ToolKind, classify_tool, extract_bash_final_action, primary_arg_for_tool};

#[test]
fn bash_final_action_returns_last_segment() {
    let out = extract_bash_final_action("mkdir -p /tmp/x/src && cd /tmp/x && cargo init --name a");
    assert!(out.starts_with("cargo init"), "got: {out}");
}

#[test]
fn bash_final_action_no_chain_returns_original() {
    let out = extract_bash_final_action("ls -la /tmp/x");
    assert!(out.starts_with("ls -la"));
}

#[test]
fn bash_final_action_empty_returns_empty() {
    assert_eq!(extract_bash_final_action(""), "");
}

#[test]
fn classify_tool_case_insensitive() {
    assert_eq!(classify_tool("Bash"), ToolKind::Bash);
    assert_eq!(classify_tool("bash"), ToolKind::Bash);
    assert_eq!(classify_tool("BASH"), ToolKind::Bash);
    assert_eq!(classify_tool("Write"), ToolKind::Write);
    assert_eq!(classify_tool("Edit"), ToolKind::Edit);
    assert_eq!(classify_tool("Read"), ToolKind::Read);
    assert_eq!(classify_tool("MultiEdit"), ToolKind::MultiEdit);
    assert_eq!(classify_tool("multiedit"), ToolKind::MultiEdit);
}

#[test]
fn classify_tool_unknown_is_other() {
    assert_eq!(classify_tool("GetWeather"), ToolKind::Other);
    assert_eq!(classify_tool(""), ToolKind::Other);
    assert_eq!(classify_tool("Bashly"), ToolKind::Other);
}

#[test]
fn primary_arg_write_snake_and_camel() {
    let out = primary_arg_for_tool("Write", r#"{"file_path":"/tmp/x.rs"}"#);
    assert_eq!(out.as_deref(), Some("/tmp/x.rs"));
    let out = primary_arg_for_tool("write", r#"{"filePath":"/tmp/y.rs"}"#);
    assert_eq!(out.as_deref(), Some("/tmp/y.rs"));
}

#[test]
fn primary_arg_bash_collapses_chain() {
    let out = primary_arg_for_tool("Bash", r#"{"command":"cd /tmp && cargo build"}"#);
    assert!(out.as_ref().is_some_and(|s| s.starts_with("cargo build")));
}

#[test]
fn primary_arg_unknown_tool_falls_back() {
    // ToolKind::Other has no well-known key but fallback path may
    // return the first non-empty string field.
    let out = primary_arg_for_tool("GetWeather", r#"{"location":"Paris"}"#);
    assert!(
        out.is_some(),
        "fallback path should return some(location=Paris)"
    );
}

#[test]
fn primary_arg_malformed_json_returns_none() {
    assert_eq!(primary_arg_for_tool("Write", "not json"), None);
}

#[test]
fn primary_arg_missing_key_returns_none() {
    let out = primary_arg_for_tool("Write", r#"{"content":"fn main(){}"}"#);
    assert_eq!(out, None);
}
