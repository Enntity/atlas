// SPDX-License-Identifier: AGPL-3.0-only

//! Deterministic, CPU-only screening of complete tool-call arguments.
//! No network, model inference, prompt scanning, or execution. This is a
//! defense-in-depth filter, not a sandbox or an arbitrary-program analyzer.
use serde_json::Value;
use std::collections::HashSet;

mod scan;
#[cfg(test)]
mod tests;

/// Maximum serialized arguments per call, and aggregate pending stream bytes.
pub const DEFAULT_MAX_BYTES: usize = 1024 * 1024;
/// Stable wire marker. Details never contain the rejected arguments.
pub const GUARD_STOP: &str = "tool_call_guard";
pub const REFUSAL: &str = "[atlas] A tool call was blocked by the tool-call guard. No blocked call was released. Review the server policy before retrying.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    MetadataAccess,
    BlockedHost,
    CredentialTransfer,
    InvalidArguments,
    ArgumentsTooLarge,
    IncompleteCall,
}
impl BlockReason {
    pub fn code(self) -> &'static str {
        match self {
            Self::MetadataAccess => "metadata_access",
            Self::BlockedHost => "blocked_host",
            Self::CredentialTransfer => "credential_transfer",
            Self::InvalidArguments => "invalid_arguments",
            Self::ArgumentsTooLarge => "arguments_too_large",
            Self::IncompleteCall => "incomplete_call",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Policy {
    pub enabled: bool,
    pub max_bytes: usize,
    allow_hosts: HashSet<String>,
    deny_hosts: HashSet<String>,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_bytes: DEFAULT_MAX_BYTES,
            allow_hosts: HashSet::new(),
            deny_hosts: HashSet::new(),
        }
    }
}
impl Policy {
    /// Exact hosts only: no wildcard domains, URLs, ports, or implicit suffixes.
    /// Explicit denials and metadata rules outrank host exceptions.
    pub fn new(
        enabled: bool,
        allow: &[String],
        deny: &[String],
        max_bytes: usize,
    ) -> Result<Self, String> {
        if max_bytes == 0 || max_bytes > 64 * DEFAULT_MAX_BYTES {
            return Err("tool-guard-max-bytes must be between 1 and 67108864".into());
        }
        let hosts = |items: &[String]| -> Result<HashSet<String>, String> {
            items.iter().map(|s| scan::config_host(s)).collect()
        };
        Ok(Self {
            enabled,
            max_bytes,
            allow_hosts: hosts(allow)?,
            deny_hosts: hosts(deny)?,
        })
    }

    /// Inspect once per complete call; bounded before JSON parsing. JSON escapes
    /// are decoded by serde, and URL parsing uses the standard URL crate.
    pub fn inspect(&self, arguments: &str) -> Result<(), BlockReason> {
        if !self.enabled {
            return Ok(());
        }
        if arguments.len() > self.max_bytes {
            return Err(BlockReason::ArgumentsTooLarge);
        }
        let value: Value =
            serde_json::from_str(arguments).map_err(|_| BlockReason::InvalidArguments)?;
        if !value.is_object() {
            return Err(BlockReason::InvalidArguments);
        }
        let mut strings = Vec::new();
        collect_strings(&value, &mut strings, true);
        // Check every literal destination, including nested batch tools. A tool
        // name or schema supplied by a client cannot exempt its arguments.
        let mut untrusted_destination = false;
        let mut transfer = structured_payload(&value);
        let mut sensitive = sensitive_payload(&value, false);
        for (text, scan_source) in strings {
            let decoded = scan::decode_percent(text);
            let lower = decoded.to_ascii_lowercase();
            for url in scan::urls(&lower) {
                let Some(host) = url.host_str() else {
                    continue;
                };
                let host = host.trim_end_matches('.');
                if scan::is_metadata(host) {
                    return Err(BlockReason::MetadataAccess);
                }
                if self.deny_hosts.contains(host) {
                    return Err(BlockReason::BlockedHost);
                }
                if self.allow_hosts.contains(host) {
                    continue;
                }
                if scan::BUILTIN_DENY_HOSTS.contains(&host) {
                    return Err(BlockReason::BlockedHost);
                }
                untrusted_destination = true;
                if scan::sensitive_url(&url) {
                    return Err(BlockReason::CredentialTransfer);
                }
            }
            transfer |= scan::is_transfer(&lower);
            sensitive |= scan_source && scan::sensitive_command(&lower);
            // Literal key material is sensitive even in a structured HTTP body.
            if scan::private_key(&lower) {
                transfer = true;
                sensitive = true;
            }
        }
        if untrusted_destination && transfer && sensitive {
            Err(BlockReason::CredentialTransfer)
        } else {
            Ok(())
        }
    }
}

fn collect_strings<'a>(value: &'a Value, out: &mut Vec<(&'a str, bool)>, scan_source: bool) {
    match value {
        Value::String(s) => out.push((s, scan_source)),
        Value::Array(v) => v.iter().for_each(|v| collect_strings(v, out, scan_source)),
        Value::Object(v) => v.iter().for_each(|(k, v)| {
            if k.eq_ignore_ascii_case("headers")
                && let Value::Object(headers) = v
            {
                for (name, value) in headers {
                    collect_strings(value, out, scan_source && !scan::auth_header(name));
                }
            } else {
                collect_strings(v, out, scan_source);
            }
        }),
        _ => {}
    }
}

// Payload fields describe data leaving a structured HTTP/upload tool. Normal
// authentication headers are deliberately not treated as exported payloads.
fn payload_key(key: &str) -> bool {
    matches!(
        key,
        "body" | "data" | "json" | "payload" | "query" | "params" | "form"
    )
}
fn structured_payload(v: &Value) -> bool {
    match v {
        Value::Object(o) => o.iter().any(|(k, v)| {
            (payload_key(&k.to_ascii_lowercase()) || k.eq_ignore_ascii_case("headers"))
                || structured_payload(v)
        }),
        Value::Array(a) => a.iter().any(structured_payload),
        _ => false,
    }
}
fn sensitive_payload(v: &Value, in_payload: bool) -> bool {
    match v {
        Value::Object(o) => o.iter().any(|(k, v)| {
            let k = k.to_ascii_lowercase();
            (in_payload
                && matches!(
                    k.as_str(),
                    "api_key" | "access_token" | "secret_key" | "private_key" | "password"
                )
                && !v.is_null())
                || sensitive_payload(v, in_payload || payload_key(&k))
        }),
        Value::Array(a) => a.iter().any(|v| sensitive_payload(v, in_payload)),
        Value::String(s) if in_payload => {
            let s = scan::decode_percent(s).to_ascii_lowercase();
            scan::sensitive_source(&s) || scan::private_key(&s) || scan::token_literal(&s)
        }
        _ => false,
    }
}
